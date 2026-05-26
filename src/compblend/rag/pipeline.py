"""RAG pipeline — compressed-doc blending on top of cacheblend-hf-v7.

Compared to CompBlend-old's `rag/pipeline.py`, this version:
    * has no `BlendingCache` — KVStore is the only KV container.
    * has no `backend.prepare_for_blend` step — pre-RoPE K is stored directly
      in KVStore entries, and `fuse_selective_compblend` applies RoPE at the
      fused position on-the-fly.
    * has no `SelectionContext` — importance + valid_mask + structural live
      directly on KVStore entries (CompBlend extensions).
    * is intentionally shorter (~170 LOC vs old ~330) — orchestration only.

Offline flow:
    backend.compress(input_ids, ...) → CompressedChunk → kv_store entry.

Online flow:
    1. Decide doc order for the fused prompt.
    2. Build [sys_chunk, *doc_chunks, query_chunk] as v7 `Chunk` objects.
    3. Precompute sys/query chunks into kv_store (cacheblend.precompute).
    4. fuse_selective_compblend(lw_model, chunks, kv_store, config).
    5. Greedy decode from the resulting LayerwiseOutput.

Stage 1 limitations (deliberate):
    * No disk persistence. Pipeline lives in-process; re-compress on restart.
    * No retriever. Caller passes `doc_ids` directly. A retriever is trivial
      to add (see CompBlend-old's `SimpleLexicalRetriever`); we keep this
      module focused on the swap path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from cacheblend.chunker import Chunk, chunk_texts, fused_input_ids
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseModel
from cacheblend.precompute import precompute_chunk_kv

from compblend.backends.base import (
    CompressedChunk,
    CompressionBudget,
    to_kvstore_entry,
)
from compblend.config import CompBlendConfig
from compblend.fuse_selective_compblend import fuse_selective_compblend


@dataclass
class BlendResult:
    """What `online_blend` returns."""

    text: str
    next_token_ids: list[int]
    ttft_seconds: float
    total_seconds: float
    blend_prompt_len: int
    top_k_selected: int


class RAGPipeline:
    """Glue between a CompressionBackend and v7's CacheBlend core.

    The HF model is loaded ONCE by the backend (e.g., `KVzipBackend.hf_model`)
    and shared with a `LayerwiseModel` view that adds the k_proj hooks needed
    for v7's precompute path. This avoids the double-load that would happen
    if we instantiated `LayerwiseModel(model_name)` independently.
    """

    def __init__(
        self,
        backend: Any,                       # CompressionBackend; must expose .hf_model + .tokenizer
        sys_prompt: str = "You are a helpful assistant.",
    ) -> None:
        self.backend = backend
        self.sys_prompt = sys_prompt
        # Share weights with the backend's HF model. LayerwiseModel.__new__
        # bypasses the standard constructor (which would load weights again);
        # we set the fields by hand and install the k_proj hooks LayerwiseModel
        # relies on for `precompute_chunk_kv`.
        hf_model = backend.hf_model
        lw = LayerwiseModel.__new__(LayerwiseModel)
        lw.model = hf_model
        lw.tokenizer = backend.tokenizer
        lw.device = next(hf_model.parameters()).device
        lw.dtype = next(hf_model.parameters()).dtype
        lw._inner = hf_model.model
        lw.num_layers = len(lw._inner.layers)
        lw._pre_rope_k = {}
        lw._hook_handles = []
        lw._install_k_proj_hooks()
        self.lw = lw

        self.kv_store = KVStore()
        # doc_id → (CompressedChunk, chunk_id-tied-to-kvstore)
        # We hold a reference so the chunk's tensors don't get GC'd while
        # the kv_store entry points at them.
        self._docs: dict[str, CompressedChunk] = {}

    # ──────────────────────────────────────────────────────────────────
    # Offline
    # ──────────────────────────────────────────────────────────────────

    def add_document(
        self,
        doc_id: str,
        text: str,
        ratio: float,
    ) -> CompressedChunk:
        """Compress one document and place its compressed cache in the KVStore.

        Returns the produced CompressedChunk; the entry is also indexed under
        `chunk.chunk_id` in the store so that `online_blend` can find it via
        the Chunk it places in the fused order.
        """
        ids = self.lw.tokenizer(
            text, add_special_tokens=False, return_tensors="pt",
        )["input_ids"].to(self.lw.device)

        compressed = self.backend.compress(
            ids, model=self.backend.hf_model,
            budget=CompressionBudget(ratio=ratio),
        )
        compressed = compressed.to(self.lw.device)

        # Bypass v7 KVStore.put (only takes K, V) and insert the full
        # CompBlend-extended entry directly. v7's `has`/`get` only look up
        # by chunk_id and return the dict unchanged.
        self.kv_store._cache[compressed.chunk_id] = to_kvstore_entry(compressed)
        # Hold a reference so the tensors stay alive.
        self._docs[doc_id] = compressed
        return compressed

    # ──────────────────────────────────────────────────────────────────
    # Online
    # ──────────────────────────────────────────────────────────────────

    def online_blend(
        self,
        query: str,
        doc_ids: list[str],
        config: CompBlendConfig,
        max_new_tokens: int = 32,
        prepend_bos: bool = True,
    ) -> BlendResult:
        """Blend `doc_ids` with `query` under `config`, return generated text.

        Args:
            query: user question text.
            doc_ids: ordered list of doc identifiers previously passed to
                `add_document`. Order determines fused-prompt placement.
            config: CompBlendConfig.
            max_new_tokens: greedy decode budget.
            prepend_bos: prepend tokenizer's BOS token to the first chunk's
                token_ids. Required for Llama-family models where attention
                conditioning expects BOS at position 0.
        """
        missing = [d for d in doc_ids if d not in self._docs]
        if missing:
            raise KeyError(f"unknown doc_ids: {missing}")

        # ── 1. Build chunks ────────────────────────────────────────────
        # Sys + query → fresh chunks (their KV is computed inside the
        # check-layer's fresh path OR precomputed below).
        sys_chunk, query_chunk = chunk_texts(self.lw.tokenizer, [self.sys_prompt, query])

        # BOS for the first chunk (matches v7's musique benchmark convention).
        if prepend_bos:
            bos_id = self.lw.tokenizer.bos_token_id
            if bos_id is None:
                raise RuntimeError(
                    "prepend_bos=True but tokenizer has no bos_token_id"
                )
            sys_chunk = Chunk(
                text=sys_chunk.text,
                token_ids=[bos_id] + list(sys_chunk.token_ids),
                chunk_id=f"sys-bos:{sys_chunk.chunk_id}",
            )

        # Doc chunks reuse the CompressedChunk's stored token_ids + chunk_id.
        doc_chunks: list[Chunk] = []
        for doc_id in doc_ids:
            compressed = self._docs[doc_id]
            doc_chunks.append(Chunk(
                text="",                              # not used by the fusor
                token_ids=list(compressed.token_ids),
                chunk_id=compressed.chunk_id,
            ))

        chunks_in_order: list[Chunk] = [sys_chunk, *doc_chunks, query_chunk]

        # ── 2. Precompute non-doc chunks into KVStore ──────────────────
        for c in chunks_in_order:
            if not self.kv_store.has(c.chunk_id):
                K, V = precompute_chunk_kv(self.lw, c)
                # v7-style entry: K, V only. Fusor defaults valid_mask /
                # importance / is_structural when those keys are absent.
                self.kv_store._cache[c.chunk_id] = {"K": K, "V": V}

        # ── 3. Fuse + decode ──────────────────────────────────────────
        if self.lw.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        out, top_indices = fuse_selective_compblend(
            self.lw, chunks_in_order, self.kv_store, config,
            return_layerwise_output=True, return_hkvd_indices=True,
        )

        if self.lw.device.type == "cuda":
            torch.cuda.synchronize()
        t_after_prefill = time.perf_counter()

        # Greedy decode from the LAST position of the fused prompt. The
        # last-position logits is valid because fuse_selective_compblend
        # always force-includes the last position into top_indices.
        prefill_logits = out.logits
        past_kv = out.past_key_values

        eos_id = getattr(self.lw.tokenizer, "eos_token_id", None)
        next_id = prefill_logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        if self.lw.device.type == "cuda":
            torch.cuda.synchronize()
        t_first_token = time.perf_counter()

        generated = [int(next_id.item())]
        for _ in range(max_new_tokens - 1):
            if eos_id is not None and generated[-1] == eos_id:
                break
            step = self.lw.model(
                input_ids=next_id,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = step.past_key_values
            next_id = step.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated.append(int(next_id.item()))

        if self.lw.device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        text = self.lw.tokenizer.decode(generated, skip_special_tokens=True)

        blend_prompt_len = sum(c.length for c in chunks_in_order)
        return BlendResult(
            text=text,
            next_token_ids=generated,
            ttft_seconds=t_first_token - t_start,
            total_seconds=t_end - t_start,
            blend_prompt_len=blend_prompt_len,
            top_k_selected=int(top_indices.shape[0]),
        )
