"""KVzip backend — pre-RoPE K capture for v7-compatible blending.

What changes vs CompBlend-old's KVzip adapter
─────────────────────────────────────────────
OLD: stored `kv.key_cache[layer]` directly. That tensor is the post-RoPE K
that KVzip's RetainCache holds. A separate `prepare_for_blend` step then had
to de-rotate at the chunk-local position and re-rotate at the fused-global
position (2-pass RoPE), which is the source of the de-rotate ambiguity that
v7 sidesteps.

NEW: install forward hooks on `model.model.layers[li].self_attn.k_proj` and
`v_proj` BEFORE calling `mk.prefill(text)`. The hook captures k_proj output
(shape `[1, seq, H_kv*D]`), which is pre-RoPE K by construction (RoPE is
applied AFTER projection inside HF's attention forward). At blend time, the
fusor just calls `apply_rotary_pos_emb` at the new fused position — same
operation, one direction.

Why this is correct
───────────────────
KVzip's `ModelKVzip` wraps a vanilla HF causal LM (`mk.model`). Its `prefill`
runs `mk.model.forward(prefill_ids)`. Every attention layer's k_proj is the
standard `nn.Linear`. KVzip's customizations are in `Cache.update()`
(retain/evict logic) and in the scoring task — neither affects whether
k_proj is called or what it returns. A forward-hook on k_proj therefore
captures exactly what HF would have computed.

What we keep from CompBlend-old's adapter
─────────────────────────────────────────
- Lazy loading of `ModelKVzip` (KVzip is on PYTHONPATH only on GPU pods).
- Slicing `[:, sink:sink+ctx_len, :]` to strip the sys-prompt prefix that
  KVzip prepends internally.
- `kv.valid` → `valid_mask`, `kv.score` → `importance` (per-(layer, head,
  position) salience). All carried as CompBlend extensions to the v7 layout.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from compblend.backends.base import (
    CompressedChunk,
    CompressionBackendBase,
    CompressionBudget,
)


@dataclass
class KVzipConfig:
    """User-facing KVzip knobs.

    Attributes:
        kv_type: "retain" (default, Stage-1-safe) or "evict". Stage 1 expects
            "retain" because then `kv.key_cache[layer]` is a regular 4D tensor
            and `kv.score` / `kv.valid` align with simple ctx-local indices.
        level: "pair" (default, per-head eviction) or "uniform" (per-layer
            eviction). Stage 1 supports both; "pair" produces meaningful
            `valid_mask` variance across heads.
        chunk_id_prefix: identifier prefix in the assembled CompressedChunk.chunk_id.
    """

    kv_type: str = "retain"
    level: str = "pair"
    chunk_id_prefix: str = "kvzip"


def kvzip_chunk_id(
    token_ids: list[int] | tuple[int, ...],
    ratio: float,
    level: str = "pair",
    prefix: str = "kvzip",
    *,
    model_id: str | None = None,
    tokenizer_id: str | None = None,
    dtype: str | None = None,
    n_layers: int | None = None,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
    algo_version: str = "v1",
) -> str:
    """Deterministic chunk_id from token_ids + compression params + model identity.

    Why all these keyword fields: the same token sequence at the same ratio
    produces COMPLETELY DIFFERENT K/V tensors under different models /
    tokenizers / dtypes. If we cache by `(token_ids, ratio, level)` alone,
    a precomputed corpus from one model can collide with another model's
    cache entries and the loader will silently return wrong KV.

    All extra fields are keyword-only and OPTIONAL for legacy compatibility —
    if not supplied, the key falls back to the original (token_ids, ratio,
    level) form. Production callers (KVzipBackend.compress, offline
    precompute scripts) SHOULD supply them; they will be filled in
    automatically by `compress()` below.

    Hardening tip: when sharing a cache directory across runs, always set
    `model_id` and `tokenizer_id` at minimum.
    """
    parts = [
        ",".join(str(t) for t in token_ids),
        f"ratio={ratio}",
        f"level={level}",
        f"algo={algo_version}",
    ]
    # Optional identity fields — only included when supplied, so older keys
    # remain reproducible. New callers should pass these.
    if model_id is not None:      parts.append(f"model={model_id}")
    if tokenizer_id is not None:  parts.append(f"tok={tokenizer_id}")
    if dtype is not None:         parts.append(f"dtype={dtype}")
    if n_layers is not None:      parts.append(f"L={n_layers}")
    if n_kv_heads is not None:    parts.append(f"Hkv={n_kv_heads}")
    if head_dim is not None:      parts.append(f"D={head_dim}")
    digest_input = "|".join(parts)
    return f"{prefix}:{hashlib.sha256(digest_input.encode('utf-8')).hexdigest()[:16]}"


class KVzipBackend(CompressionBackendBase):
    """KVzip → CompressedChunk adapter using forward-hook K capture.

    Threading note: `compress` installs forward hooks on the shared HF model,
    runs prefill, and removes them in a `finally`. The hooks are not
    re-entrant — do not call `compress` from multiple threads on the same
    backend instance.
    """

    algo_id = "kvzip"

    def __init__(
        self,
        model_id: str,
        kvzip_config: KVzipConfig | None = None,
    ) -> None:
        self.model_id = model_id
        self.kvzip_config = kvzip_config or KVzipConfig()
        self._mk: Any = None    # cached ModelKVzip — avoids double-loading the HF model

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    @property
    def hf_model(self) -> Any:
        """The underlying HF causal LM. Lazy-loads on first access.

        Downstream code (LayerwiseModel construction, RAG pipeline) shares
        this exact model to avoid double-loading the weights.
        """
        return self._get_model_kvzip().model

    @property
    def tokenizer(self) -> Any:
        return self._get_model_kvzip().tokenizer

    def compress(
        self,
        input_ids: torch.Tensor,
        model: Any,
        budget: CompressionBudget,
    ) -> CompressedChunk:
        """Compress one chunk.

        `model` is ignored — KVzip uses its own `ModelKVzip` wrapper exposed
        via `self.hf_model`. The signature matches the protocol so backends
        are swappable; the parameter is named `model` not `_model` to keep
        the abstract base honest. Pass `self.hf_model` when you need the
        actual weights externally.
        """
        del model  # KVzip carries its own ModelKVzip wrapper.
        if budget.ratio is None:
            raise ValueError(
                "KVzipBackend requires budget.ratio (kv.prune is ratio-based)."
            )

        mk = self._get_model_kvzip()

        # ----- 1. Install forward hooks on every layer's k_proj + v_proj -----
        # KVzip's `prefill()` runs the model TWICE: first the actual prefill
        # over [sys, content] (length = sink + ctx_len), then the scoring task
        # (context-reconstruction) over a shorter sequence. A naive "[idx] ="
        # hook would keep only the LAST capture (scoring) and miss prefill.
        # Collect ALL captures per layer into a list; post-filter by shape.
        pre_rope_k_all: dict[int, list[torch.Tensor]] = {}
        pre_v_all: dict[int, list[torch.Tensor]] = {}
        handles = self._install_hooks(mk, pre_rope_k_all, pre_v_all)

        try:
            # KVzip's `prefill(ctx_ids: Union[str, torch.Tensor])` accepts a
            # tensor directly — avoid the decode→encode roundtrip that broke
            # for tokenizer-unstable Loong docs.
            ids = input_ids
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            ids = ids.to(mk.model.device)
            kv = mk.prefill(ids, load_score=False, do_score=True)
        finally:
            for h in handles:
                h.remove()

        # ----- 2. Apply pruning to populate kv.valid + kv.score -----
        # ratio=1.0 → all-True mask, score still computed.
        kv.prune(ratio=budget.ratio, level=self.kvzip_config.level)

        # ----- 3. Pick the prefill capture (shape[1] == sink + ctx_len) -----
        sink_ctx = int(kv.sink) + int(kv.ctx_len)
        pre_rope_k = self._select_prefill_capture(pre_rope_k_all, sink_ctx, "k_proj")
        pre_v = self._select_prefill_capture(pre_v_all, sink_ctx, "v_proj")

        # ----- 4. Assemble CompressedChunk (slice out sys-prompt) -----
        return self._build_chunk(mk, kv, pre_rope_k, pre_v, input_ids, budget)

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _get_model_kvzip(self) -> Any:
        if self._mk is None:
            try:
                from model import ModelKVzip  # type: ignore  # KVzip's `model` package
            except ImportError as e:    # pragma: no cover — GPU pod has it
                raise ImportError(
                    "KVzipBackend requires the KVzip repo on PYTHONPATH. "
                    "Install via `git clone https://github.com/snu-mllab/KVzip` "
                    "and `export PYTHONPATH=$PYTHONPATH:/path/to/KVzip`."
                ) from e
            self._mk = ModelKVzip(self.model_id, kv_type=self.kvzip_config.kv_type)
        return self._mk

    @staticmethod
    def _ids_to_text(mk: Any, input_ids: torch.Tensor) -> str:
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)
        tok = mk.tokenizer
        if tok is None:
            raise RuntimeError("ModelKVzip.tokenizer is None — upstream API changed?")
        return tok.decode(input_ids.tolist(), skip_special_tokens=False)

    @staticmethod
    def _install_hooks(
        mk: Any,
        pre_rope_k_all: dict[int, list[torch.Tensor]],
        pre_v_all: dict[int, list[torch.Tensor]],
    ) -> list:
        """Hook k_proj and v_proj on every decoder layer, APPENDING every fire.

        Why append instead of overwrite: KVzip's prefill triggers multiple
        forwards (the actual prefill + a scoring task). Each forward fires
        every layer's k_proj. A naive `dict[idx] = output` keeps only the
        last one. We collect everything and let the caller pick the right
        shape post-hoc.

        Returns a flat list of hook handles. Caller MUST remove them in a
        finally block.
        """
        # KVzip stores the HF model at mk.model; the decoder body is at
        # mk.model.model (LlamaModel / MistralModel).
        base = getattr(mk.model, "model", mk.model)
        layers = base.layers
        handles = []
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn

            def make_k_hook(idx: int):
                def hook(_module, _inputs, output):
                    pre_rope_k_all.setdefault(idx, []).append(output.detach())
                return hook

            def make_v_hook(idx: int):
                def hook(_module, _inputs, output):
                    pre_v_all.setdefault(idx, []).append(output.detach())
                return hook

            handles.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
            handles.append(attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))
        return handles

    @staticmethod
    def _select_prefill_capture(
        captures_all: dict[int, list[torch.Tensor]],
        sink_ctx: int,
        proj_name: str,
    ) -> dict[int, torch.Tensor]:
        """Pick the prefill K/V per layer. Handles single AND chunked prefill.

        KVzip's prefill behavior depends on context length:
          * Short context (≤ chunk_size, typ. 16K): one capture with
            shape[1] == sink + ctx_len. We pick that single tensor.
          * Long context (> chunk_size): KVzip processes the prompt in
            multiple chunks (e.g. 16000, 16000, 7985 for 39985-token
            prefill). Each chunk's k_proj fires a separate hook. The
            scoring task that follows uses a different (typically
            shorter) sequence. We identify the prefill chunks as the
            FIRST `k` captures whose shape[1] sums to sink + ctx_len,
            then `torch.cat(..., dim=1)` them into a single tensor.

        Fails loudly if neither path can reconstruct sink_ctx — e.g. if
        the scoring task's captures got interleaved with prefill.
        """
        out: dict[int, torch.Tensor] = {}
        for layer_idx, caps in captures_all.items():
            # Path 1: single-capture match (short context).
            single = [t for t in caps if t.shape[1] == sink_ctx]
            if single:
                out[layer_idx] = single[0]
                continue

            # Path 2: chunked-prefill concat. Walk forward through captures
            # accumulating shape[1] until we hit sink_ctx exactly.
            running = 0
            chunks: list[torch.Tensor] = []
            for t in caps:
                chunks.append(t)
                running += int(t.shape[1])
                if running == sink_ctx:
                    break
                if running > sink_ctx:
                    chunks = []      # overshot — sequence didn't match
                    break

            if running == sink_ctx and chunks:
                out[layer_idx] = torch.cat(chunks, dim=1).contiguous()
                continue

            shapes = [tuple(t.shape) for t in caps]
            raise RuntimeError(
                f"Could not reconstruct {proj_name} prefill at layer {layer_idx}. "
                f"Target shape[1]={sink_ctx} (sink+ctx_len). Tried: single-match "
                f"(none); chunked-concat (running sum did not match). Captured "
                f"shapes: {shapes}"
            )
        return out

    def _build_chunk(
        self,
        mk: Any,
        kv: Any,
        pre_rope_k: dict[int, torch.Tensor],
        pre_v: dict[int, torch.Tensor],
        input_ids: torch.Tensor,
        budget: CompressionBudget,
    ) -> CompressedChunk:
        """Assemble the CompressedChunk from hooked tensors + KVzip metadata."""
        n_layers = kv.n_layers
        n_kv_heads = kv.n_heads_kv
        sink = int(kv.sink)            # sys-prompt length
        ctx_len = int(kv.ctx_len)      # user-content length

        # ── sanity: did we capture from every layer? ─────────────────────
        missing = [li for li in range(n_layers) if li not in pre_rope_k or li not in pre_v]
        if missing:
            raise RuntimeError(
                f"Forward hooks did not fire for layers {missing}. Did KVzip "
                f"swap the attention module to one that bypasses k_proj/v_proj?"
            )

        first_k = pre_rope_k[0]
        if first_k.dim() != 3:
            raise RuntimeError(
                f"Captured k_proj output has unexpected ndim {first_k.dim()}; "
                f"expected 3 (batch, seq, H_kv*D). Got shape {tuple(first_k.shape)}."
            )
        full_seq = first_k.shape[1]
        if full_seq < sink + ctx_len:
            raise RuntimeError(
                f"Captured seq_len ({full_seq}) < sink+ctx_len ({sink + ctx_len}). "
                f"Hook fired on the wrong forward pass?"
            )

        # ── 1. slice content region from captured K and V ────────────────
        # Some KVzip configurations append a probe-query suffix used for
        # scoring — slice strictly to [sink, sink+ctx_len).
        key_ctx = [
            pre_rope_k[li][:, sink:sink + ctx_len, :].contiguous()  # [1, ctx_len, H_kv*D]
            for li in range(n_layers)
        ]
        val_ctx = [
            pre_v[li][:, sink:sink + ctx_len, :].contiguous()
            for li in range(n_layers)
        ]

        # ── 2. importance — kv.score is [1, H_kv, ctx_len] per layer ─────
        # Stack across layers → [L, H_kv, ctx_len].
        importance = torch.stack(
            [kv.score[li].squeeze(0) for li in range(n_layers)],
            dim=0,
        ).to(torch.float32)

        # ── 3. valid_mask — kv.valid is [L, 1, H_kv, ctx_len] bool ───────
        if hasattr(kv, "valid") and kv.valid is not None:
            valid_full = kv.valid.detach().clone()
            if valid_full.dim() == 4:
                valid_mask = valid_full.squeeze(1).contiguous()      # [L, H_kv, ctx_len]
            else:
                valid_mask = valid_full.contiguous()
        else:
            # Defensive: treat as all-True (ratio=1.0 path).
            valid_mask = torch.ones(
                n_layers, n_kv_heads, ctx_len,
                dtype=torch.bool, device=first_k.device,
            )

        # Shape sanity.
        if valid_mask.shape != (n_layers, n_kv_heads, ctx_len):
            raise RuntimeError(
                f"valid_mask shape {tuple(valid_mask.shape)} doesn't match "
                f"expected (L={n_layers}, H={n_kv_heads}, T={ctx_len})."
            )
        if importance.shape != valid_mask.shape:
            raise RuntimeError(
                f"importance shape {tuple(importance.shape)} != valid_mask "
                f"{tuple(valid_mask.shape)}."
            )

        # ── 4. zero out K/V and importance at evicted (layer, head, pos) ─
        # Layout: key_ctx[li] is [1, ctx_len, H_kv*D]. Reshape to
        # [1, ctx_len, H_kv, D] for head-wise masking, then back.
        head_dim = key_ctx[0].shape[-1] // n_kv_heads
        for li in range(n_layers):
            k4 = key_ctx[li].view(1, ctx_len, n_kv_heads, head_dim)
            v4 = val_ctx[li].view(1, ctx_len, n_kv_heads, head_dim)
            # valid_mask[li] : [H_kv, ctx_len]  →  [1, ctx_len, H_kv, 1]
            m = valid_mask[li].t().unsqueeze(0).unsqueeze(-1).to(k4.dtype)   # [1, T, H, 1]
            k4 = k4 * m
            v4 = v4 * m
            key_ctx[li] = k4.reshape(1, ctx_len, n_kv_heads * head_dim).contiguous()
            val_ctx[li] = v4.reshape(1, ctx_len, n_kv_heads * head_dim).contiguous()
        importance = importance * valid_mask.to(importance.dtype)

        # ── 5. assemble chunk identity + tokens ──────────────────────────
        chunk_token_ids = (
            input_ids.squeeze(0).detach().cpu().tolist()
            if input_ids.dim() > 1
            else input_ids.detach().cpu().tolist()
        )
        if len(chunk_token_ids) != ctx_len:
            # ModelKVzip might apply chat-style wrapping; warn but don't fail.
            # Token_ids reflect the input the user passed; ctx_len reflects
            # what KVzip's tokenizer.encode produced. Carry the user's view.
            pass

        # Deterministic across processes: Python's built-in `hash()` is salted
        # per process (`PYTHONHASHSEED`) so cross-process disk caches would
        # miss. We delegate to `kvzip_chunk_id` so the offline precompute
        # script and the online benchmark agree on the cache key.
        #
        # Include model/tokenizer/dtype/shape identity in the cache key to
        # prevent cross-model contamination (see kvzip_chunk_id docstring).
        tokenizer_id = getattr(mk.tokenizer, "name_or_path", None) or self.model_id
        head_dim_meta = key_ctx[0].shape[-1] // n_kv_heads
        chunk_id = kvzip_chunk_id(
            chunk_token_ids,
            ratio=budget.ratio,
            level=self.kvzip_config.level,
            prefix=self.kvzip_config.chunk_id_prefix,
            model_id=self.model_id,
            tokenizer_id=tokenizer_id,
            dtype=str(first_k.dtype),
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim_meta,
        )

        # ── 6. structural — KVzip has no in-content sinks at this stage ──
        is_structural = torch.zeros(ctx_len, dtype=torch.bool, device=first_k.device)

        # ── 7. diagnostics ──────────────────────────────────────────────
        len_k_per_layer = [
            valid_mask[li].sum(dim=-1).tolist()
            for li in range(n_layers)
        ]

        return CompressedChunk(
            algo_id=self.algo_id,
            chunk_id=chunk_id,
            model_id=self.model_id,
            tokenizer_id=tokenizer_id,
            token_ids=chunk_token_ids,
            key_cache=key_ctx,                # list of [1, ctx_len, H_kv*D]   pre-RoPE
            value_cache=val_ctx,
            valid_mask=valid_mask,            # [L, H_kv, ctx_len]
            importance=importance,            # [L, H_kv, ctx_len]
            is_structural=is_structural,
            compression_rate=1.0 - float(valid_mask.float().mean().item()),
            backend_state={
                "kvzip_ratio": budget.ratio,
                "kvzip_level": self.kvzip_config.level,
                "kvzip_kv_type": self.kvzip_config.kv_type,
                "sys_prompt_len": sink,
                "ctx_len": ctx_len,
                "len_k_per_layer": len_k_per_layer,
            },
        )
