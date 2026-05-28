"""CompBlend's selective-recompute fusor — fork of cacheblend.fusor.fuse_selective.

Two extensions over v7:

1. **Gated HKVD selector** (paper §3) — `selectors.gated_top_k` runs over
   deviation+importance instead of v7's HKVD-only top-k. Gap positions
   (chunks not in KVStore) and the last position are passed as
   `forced_mask`. Structural (window) positions are passed as
   `structural_mask`.

2. **Per-head valid_mask** — when a backend provides `entry["valid_mask"]`
   of shape `[num_layers, H_kv, chunk_len]` (KVzip pair-level eviction
   produces this), the SDPA `attn_mask` is built as a `[1, H_q, Q, K]`
   additive float that masks out evicted (head, position) pairs IN ADDITION
   to the causal triangle. Fresh top-indices positions are unconditionally
   valid for every head.

The function falls back gracefully when KVStore entries are pure v7-style
(no valid_mask / no importance) — `valid_mask` defaults to all-True and
`gated_top_k` degenerates to HKVD-only when importance is uniform.

Reference for the unmodified algorithm — keep these line numbers in sync
when v7 changes:
    src/external/cacheblend-hf-v7/src/cacheblend/fusor.py:193-497
"""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as _sdpa
from transformers.cache_utils import DynamicCache


class _Timer:
    """Lightweight per-section timer using torch.cuda.synchronize.

    `mark(label)` adds the elapsed time since the previous mark/start to
    a running total under that label. Sections that occur inside loops
    (e.g. per-layer sparse forward) get summed automatically.

    Zero overhead when `enabled=False` — both start() and mark() short-circuit.
    """

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = enabled
        self.device = device
        self.events: dict[str, float] = {}
        self.last: float | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last = time.perf_counter()

    def mark(self, label: str) -> None:
        if not self.enabled or self.last is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.events[label] = self.events.get(label, 0.0) + (now - self.last) * 1000.0
        self.last = now

from cacheblend.chunker import Chunk, chunk_offsets, fused_input_ids
from cacheblend.hkvd import kv_deviation
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseOutput

from compblend.config import CompBlendConfig
from compblend.selectors import chunk_internal_rank, gated_top_k, select_topk_sorted


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _get_apply_rope(hf_model: Any):
    """Locate `apply_rotary_pos_emb` for the model's family.

    Llama / Mistral / Qwen2 each export their own copy of the function in
    their `modeling_*.py`. The math is identical (real-pair rotation), but
    the symbol lives in different modules. We pick by model class name.
    """
    cls_name = type(hf_model).__name__.lower()
    if "llama" in cls_name:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    elif "mistral" in cls_name:
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
    elif "qwen" in cls_name:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    else:
        # Fallback to mistral — every RoPE-using HF model uses identical math.
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
    return apply_rotary_pos_emb


def _select_recompute_indices(
    deviations: torch.Tensor,
    importance: torch.Tensor,
    recompute_k: int,
    config: CompBlendConfig,
    structural_mask: torch.Tensor,
    forced_mask: torch.Tensor,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch on `config.selector`. Returns sorted-ascending int64 indices.

    `eligible_mask` is the compression-aware "candidate pool" — typically the
    union over heads of `valid_mask` at the check layer (= positions that any
    head retained). It is applied ONLY to selectors that should respect
    compression's intent:
      - `hkvd_only`        : ignores eligible_mask (v7/LMCache baseline —
                              free to pick evicted positions, by design).
      - `importance_only`  : naturally avoids evicted (importance=0 there);
                              eligible_mask still applied for explicitness.
      - `gated_top_k`      : restricts candidates to eligible_mask.
                              Paper §3 alignment: "trust compression".
    """
    structural_eff = structural_mask if config.exempt_structural else None

    if config.selector == "hkvd_only":
        # naive HKVD — does NOT honor eligible_mask (by design).
        # Picks top-k by deviation regardless of compression state. At
        # heavy compression this tends to pick evicted positions (large
        # ‖K_fresh − 0‖²). Functionally that "recovers" them via sparse
        # forward, but it violates the paper's principle of trusting
        # compression. Kept as a baseline for comparison.
        dev = deviations.clone()
        if forced_mask is not None and bool(forced_mask.any().item()):
            dev[forced_mask] = float("inf")
        if structural_eff is not None and bool(structural_eff.any().item()):
            dev[structural_eff] = float("-inf")
        target_k = max(
            recompute_k,
            int(forced_mask.sum().item()) if forced_mask is not None else 0,
        )
        return select_topk_sorted(dev, target_k)

    if config.selector == "importance_only":
        scores = importance.clone()
        if forced_mask is not None and bool(forced_mask.any().item()):
            scores[forced_mask] = float("inf")
        if structural_eff is not None and bool(structural_eff.any().item()):
            scores[structural_eff] = float("-inf")
        # Explicit eligibility (importance=0 at evicted naturally, but
        # explicit guard prevents picking those when ties occur).
        if eligible_mask is not None:
            not_eligible = (~eligible_mask) & (
                forced_mask.logical_not()
                if forced_mask is not None
                else torch.ones_like(eligible_mask)
            )
            scores = torch.where(
                not_eligible, torch.full_like(scores, float("-inf")), scores,
            )
        target_k = max(
            recompute_k,
            int(forced_mask.sum().item()) if forced_mask is not None else 0,
        )
        return select_topk_sorted(scores, target_k)

    if config.selector == "gated_top_k":
        return gated_top_k(
            hkvd_scores=deviations,
            importance_scores=importance,
            recompute_k=recompute_k,
            gate_percentile=config.gate_percentile,
            structural_mask=structural_eff,
            forced_mask=forced_mask,
            eligible_mask=eligible_mask,
        )

    raise ValueError(f"unknown selector: {config.selector!r}")


def _build_attn_mask_with_per_head_valid(
    causal_mask_full: torch.Tensor,        # [1, 1, total_seq, total_seq] float
    top_indices: torch.Tensor,             # [Q] int64
    total_seq: int,
    valid_layer: torch.Tensor,             # [H_kv, total_seq] bool
    is_top: torch.Tensor,                  # [total_seq] bool — True at top_indices
    n_rep: int,                             # num_heads_q // num_heads_kv
    mask_dtype: torch.dtype,
    flags: dict | None = None,
) -> torch.Tensor:
    """Compose causal mask with per-(KV head, position) validity.

    Effective rule at (h_q, q, k):
        attend iff causal(q, k) AND (k ∈ top_indices OR valid_layer[map(h_q), k]).

    Returns `[1, H_q, Q, K]` float additive mask (0 where attend, -inf where
    masked). Caller passes this directly to SDPA's `attn_mask`.
    """
    # Step 1: slice causal to sparse Q rows.
    sparse_causal = causal_mask_full[:, :, top_indices, :total_seq]   # [1, 1, Q, K]

    # Step 2: effective per-(KV head, position) validity.
    #   True if any of: (a) position is in top_indices (fresh — valid for all
    #   heads), or (b) chunk's valid_mask says this (layer, head) kept it.
    valid_eff = valid_layer | is_top.unsqueeze(0)                     # [H_kv, K]
    if bool(valid_eff.all().item()):
        # No per-head eviction — sparse_causal alone is correct.
        if flags is not None:
            flags["per_head_mask_required"] = flags.get("per_head_mask_required", False)
        return sparse_causal

    # Memory guard: at long context the full [1, H_q, Q, K] mask exceeds GPU
    # memory. Mask size = H_q × Q × K × bytes. For Llama-3.1-8B (H_q=32),
    # recompute=0.15, total_seq=80K: 32 × 12000 × 80000 × 2 = 61 GB. OOM.
    #
    # Fall back to causal-only when the per-head mask would exceed
    # ATTN_MASK_MEMORY_CAP_BYTES. Quality cost: KVzip's per-head eviction
    # pattern is not enforced at SDPA. Mitigated because evicted positions
    # have K/V=0 (zero-filled at chunk build time), so they contribute very
    # little to attention output through softmax × V. Documented limitation
    # of Stage 1 — Stage 2 will use flash_attn_varlen to handle this exactly.
    ATTN_MASK_MEMORY_CAP_BYTES = 256 * 1024 ** 2       # 256 MB (aggressive cap for long ctx)
    bytes_per_elem = mask_dtype.itemsize if hasattr(mask_dtype, "itemsize") else 2
    Q = int(top_indices.numel())
    K = int(total_seq)
    H_q = int(valid_layer.shape[0]) * int(n_rep)
    mask_bytes = 1 * H_q * Q * K * bytes_per_elem
    if mask_bytes > ATTN_MASK_MEMORY_CAP_BYTES:
        # Fallback: causal-only mask. Evicted positions still contribute
        # near-zero (zero-filled K/V × softmax). Print fallback once-per-call
        # so the log shows we're not silently mis-attending.
        if flags is not None:
            flags["per_head_mask_required"] = True
            flags["per_head_mask_fallback_count"] = flags.get(
                "per_head_mask_fallback_count", 0
            ) + 1
            flags["per_head_mask_max_bytes"] = max(
                flags.get("per_head_mask_max_bytes", 0), mask_bytes
            )
        import sys as _sys
        print(
            f"[fuser] per-head mask too large "
            f"({mask_bytes / 1024 ** 2:.0f}MB > "
            f"{ATTN_MASK_MEMORY_CAP_BYTES / 1024 ** 2:.0f}MB cap) — "
            f"fallback to causal-only at Q={Q}, K={K}, H={H_q}",
            file=_sys.stderr, flush=True,
        )
        return sparse_causal

    # Exact per-head mask applied (no fallback).
    if flags is not None:
        flags["per_head_mask_required"] = True
        flags["per_head_mask_exact_count"] = flags.get(
            "per_head_mask_exact_count", 0
        ) + 1

    # Step 3: additive form. 0 where valid, -inf where evicted.
    # Cast carefully — sparse_causal is in model dtype (fp16/bf16/fp32).
    neg_inf = torch.tensor(float("-inf"), dtype=mask_dtype, device=valid_eff.device)
    zero = torch.tensor(0.0, dtype=mask_dtype, device=valid_eff.device)
    valid_additive_kv = torch.where(valid_eff, zero, neg_inf)         # [H_kv, K]

    # Step 4: GQA expansion to Q-heads.
    if n_rep > 1:
        valid_additive_q = valid_additive_kv.repeat_interleave(n_rep, dim=0)   # [H_q, K]
    else:
        valid_additive_q = valid_additive_kv

    # Step 5: reshape to broadcast → [1, H_q, 1, K] + [1, 1, Q, K] → [1, H_q, Q, K].
    valid_additive_4d = valid_additive_q.unsqueeze(0).unsqueeze(2)
    return sparse_causal + valid_additive_4d


def _full_fresh_layers(
    inner: Any,
    hidden_states: torch.Tensor,
    causal_mask_full: torch.Tensor,
    position_ids_full: torch.Tensor,
    position_embeddings_full: tuple[torch.Tensor, torch.Tensor],
    cache_position_full: torch.Tensor,
    past_key_values: DynamicCache,
    layer_range: range,
) -> torch.Tensor:
    """Run the unmodified HF decoder layer call for layers in `layer_range`.

    Matches v7 `fusor.py:313-323` verbatim — separated into a helper for
    readability. Returns updated hidden_states; past_key_values is mutated
    in-place via DynamicCache.update inside each layer.
    """
    for li in layer_range:
        out = inner.layers[li](
            hidden_states=hidden_states,
            attention_mask=causal_mask_full,
            position_ids=position_ids_full,
            past_key_value=past_key_values,
            use_cache=True,
            cache_position=cache_position_full,
            position_embeddings=position_embeddings_full,
        )
        hidden_states = out if not isinstance(out, tuple) else out[0]
    return hidden_states


def _load_chunk_extensions(
    entry: dict[str, Any],
    chunk_len: int,
    n_layers: int,
    n_kv_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read CompBlend extensions from a KVStore entry, defaulting safely.

    Returns:
        valid_mask:    [num_layers, H_kv, chunk_len] bool
        importance:    [num_layers, H_kv, chunk_len] fp32
        is_structural: [chunk_len] bool
        has_cache:     [chunk_len] bool — True everywhere (this entry covers
                       all positions of the chunk; gap-detection at fused
                       level is the caller's job)
    """
    if "valid_mask" in entry and entry["valid_mask"] is not None:
        valid_mask = entry["valid_mask"]
        if valid_mask.shape != (n_layers, n_kv_heads, chunk_len):
            raise ValueError(
                f"entry['valid_mask'] shape {tuple(valid_mask.shape)} != "
                f"expected ({n_layers}, {n_kv_heads}, {chunk_len})"
            )
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
    else:
        valid_mask = torch.ones(
            n_layers, n_kv_heads, chunk_len, dtype=torch.bool, device=device,
        )

    if "importance" in entry and entry["importance"] is not None:
        importance = entry["importance"]
        if importance.shape != (n_layers, n_kv_heads, chunk_len):
            raise ValueError(
                f"entry['importance'] shape {tuple(importance.shape)} != "
                f"expected ({n_layers}, {n_kv_heads}, {chunk_len})"
            )
        importance = importance.to(device=device, dtype=torch.float32)
    else:
        importance = torch.ones(
            n_layers, n_kv_heads, chunk_len, dtype=torch.float32, device=device,
        )

    if "is_structural" in entry and entry["is_structural"] is not None:
        is_structural = entry["is_structural"].to(device=device, dtype=torch.bool)
        if int(is_structural.numel()) != chunk_len:
            raise ValueError(
                f"entry['is_structural'].numel()={int(is_structural.numel())} "
                f"!= chunk_len={chunk_len}"
            )
    else:
        is_structural = torch.zeros(chunk_len, dtype=torch.bool, device=device)

    has_cache = torch.ones(chunk_len, dtype=torch.bool, device=device)
    return valid_mask, importance, is_structural, has_cache


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


def fuse_selective_compblend(
    layerwise_model,
    chunks: list[Chunk],
    kv_store: KVStore,
    config: CompBlendConfig,
    return_layerwise_output: bool = False,
    return_hkvd_indices: bool = False,
    timings: dict | None = None,
    last_logits_only: bool = False,
    flags: dict | None = None,
):
    """Paper §4 selective recompute + paper §3 Gated HKVD + per-head mask.

    Args:
        layerwise_model: `cacheblend.LayerwiseModel` (wraps the HF causal LM).
        chunks: list of `cacheblend.Chunk` covering the FUSED prompt in
            order. Every chunk_id MUST be present in `kv_store`. If you have
            uncached prefix/suffix (system prompt, query), precompute them
            into the store first with `precompute_chunk_kv`.
        kv_store: `cacheblend.KVStore`. Entries may carry CompBlend
            extensions (`valid_mask`, `importance`, `is_structural`) — if
            absent, defaults are used.
        config: `CompBlendConfig` with check_layer, recompute_ratio,
            selector, gate_percentile, exempt_structural.

    Returns:
        Logits tensor `[1, total_seq, vocab_size]` by default (non-top
        positions zero-filled — only the last position's logits are valid
        for greedy decoding). When `return_layerwise_output=True`, returns
        the `LayerwiseOutput(logits, past_key_values)` for autoregressive
        decode. When `return_hkvd_indices=True`, additionally returns the
        selected top_indices.

    Boundary safe-shortcuts (mirrored from v7):
      * recompute_ratio == 0   → `fuse_full_reuse` (KNOWN LIMITATION: ignores
                                  per-head eviction; documented). Use
                                  recompute_ratio = tiny epsilon if you have
                                  per-head eviction.
      * recompute_ratio >= 1   → `fuse_full_recompute` (no KVStore reads).
      * len(chunks) <= 1       → `fuse_full_recompute`.
    """
    from cacheblend.fusor import fuse_full_recompute, fuse_full_reuse

    # ── Boundary shortcuts ──────────────────────────────────────────────
    if config.recompute_ratio == 0:
        return fuse_full_reuse(
            layerwise_model, chunks, kv_store,
            return_layerwise_output=return_layerwise_output,
        )
    if config.recompute_ratio >= 1:
        return fuse_full_recompute(
            layerwise_model, chunks,
            return_layerwise_output=return_layerwise_output,
        )
    if len(chunks) <= 1:
        return fuse_full_recompute(
            layerwise_model, chunks,
            return_layerwise_output=return_layerwise_output,
        )

    # ── Model-family RoPE function ──────────────────────────────────────
    apply_rotary_pos_emb = _get_apply_rope(layerwise_model.model)

    # ── Setup ───────────────────────────────────────────────────────────
    inner = layerwise_model._inner
    n_layers = layerwise_model.num_layers
    device = layerwise_model.device
    dtype = layerwise_model.dtype
    attn0 = inner.layers[0].self_attn
    num_kv_heads = attn0.config.num_key_value_heads
    num_heads = attn0.config.num_attention_heads
    head_dim = attn0.head_dim
    hidden_kv = num_kv_heads * head_dim
    n_rep = num_heads // num_kv_heads

    # Instrumentation (zero overhead when timings is None)
    timer = _Timer(enabled=(timings is not None), device=device)
    timer.start()

    if not (0 <= config.check_layer < n_layers):
        raise ValueError(
            f"check_layer={config.check_layer} out of range [0, {n_layers})"
        )

    # ── Assemble full-length cached K_pre + V + CompBlend extensions ────
    offsets = chunk_offsets(chunks)
    total_seq = offsets[-1][1]

    K_stored_pre = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    V_stored = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    if flags is not None:
        bytes_per_elem = (
            dtype.itemsize if hasattr(dtype, "itemsize") else 2
        )
        flags["dense_workspace_bytes"] = (
            2 * n_layers * total_seq * hidden_kv * bytes_per_elem
        )
        flags["dense_workspace_note"] = (
            "Stage 1 allocates full [1, total_seq, H_kv*D] dense K_stored "
            "and V_stored per layer; evicted slots are zero-filled but the "
            "tensor itself is full-length. Stage 2 will replace with sparse "
            "per-head storage."
        )
        flags["total_seq"] = int(total_seq)
        flags["n_layers"] = int(n_layers)
        flags["selector"] = config.selector
        flags["gate_percentile"] = (
            float(config.gate_percentile)
            if config.selector == "gated_top_k"
            else None
        )
        flags["recompute_ratio"] = float(config.recompute_ratio)
        flags["check_layer"] = int(config.check_layer)
    valid_mask_full = torch.zeros(
        n_layers, num_kv_heads, total_seq, dtype=torch.bool, device=device,
    )
    # Importance at the check layer only — paper claim is about importance
    # GATING decisions made at the check layer's HKVD. Other layers' importance
    # is irrelevant to selector input.
    importance_full = torch.zeros(total_seq, dtype=torch.float32, device=device)
    structural_full = torch.zeros(total_seq, dtype=torch.bool, device=device)
    # Eligibility for the gated path: union over heads at the check layer
    # ("kept by any head"). Sys/query chunks (no valid_mask) default True
    # because they were never compressed.
    eligible_full = torch.ones(total_seq, dtype=torch.bool, device=device)

    for chunk, (start, end) in zip(chunks, offsets):
        if not kv_store.has(chunk.chunk_id):
            raise KeyError(
                f"fuse_selective_compblend: chunk {chunk.chunk_id!r} not in "
                f"KVStore. Stage 1 requires all chunks pre-cached. "
                f"Precompute sys/query chunks via precompute_chunk_kv before "
                f"calling fuse."
            )
        entry = kv_store.get(chunk.chunk_id)
        # K, V — required, layout matches v7
        for li in range(n_layers):
            K_stored_pre[li][:, start:end, :] = entry["K"][li]
            V_stored[li][:, start:end, :] = entry["V"][li]
        # CompBlend extensions — optional with safe defaults
        chunk_valid, chunk_imp, chunk_struct, _ = _load_chunk_extensions(
            entry, chunk_len=end - start, n_layers=n_layers,
            n_kv_heads=num_kv_heads, device=device,
        )
        valid_mask_full[:, :, start:end] = chunk_valid
        # Importance at the check layer, mean-pooled over heads → [chunk_len].
        chunk_imp_1d = chunk_imp[config.check_layer].mean(dim=0)
        if config.chunk_normalization == "rank":
            chunk_imp_1d = chunk_internal_rank(chunk_imp_1d)
        importance_full[start:end] = chunk_imp_1d
        structural_full[start:end] = chunk_struct
        # Eligible at check layer = union over heads of valid_mask. If the
        # chunk has no valid_mask key (sys/query precompute), `_load_chunk_
        # extensions` set it all-True → union is all-True too.
        eligible_full[start:end] = chunk_valid[config.check_layer].any(dim=0)
    timer.mark("kv_load")

    # ── Forward setup ──────────────────────────────────────────────────
    input_ids = fused_input_ids(chunks, device=device)
    position_ids_full = torch.arange(total_seq, device=device).unsqueeze(0)
    cache_position_full = torch.arange(total_seq, device=device)

    past_key_values = DynamicCache()
    # Reset pre-RoPE K capture (LayerwiseModel's hooks will fire during the
    # check_layer projection — we read the captured tensor for diagnostics
    # if needed; the algorithm itself uses k_full_pre directly).
    layerwise_model._pre_rope_k = {}

    with torch.inference_mode():
        hidden_states = inner.embed_tokens(input_ids)
        cos_full, sin_full = inner.rotary_emb(hidden_states, position_ids_full)
        position_embeddings_full = (cos_full, sin_full)
        timer.mark("io_embed_rotary")

        causal_mask_full = inner._update_causal_mask(
            attention_mask=None,
            input_tensor=hidden_states,
            cache_position=cache_position_full,
            past_key_values=past_key_values,
            output_attentions=False,
        )
        if causal_mask_full is None:
            # SDPA / FlashAttention2 return None — they build their own
            # causal internally. Our manual attention call below needs an
            # explicit additive mask. Construct (1, 1, S, S) -inf-above-diag.
            mask = torch.zeros((1, 1, total_seq, total_seq), dtype=dtype, device=device)
            triu = torch.triu(
                torch.ones(total_seq, total_seq, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask.masked_fill_(triu, float("-inf"))
            causal_mask_full = mask

        # ── Layers 0..check_layer-1: full fresh forward ─────────────────
        hidden_states = _full_fresh_layers(
            inner, hidden_states, causal_mask_full,
            position_ids_full, position_embeddings_full, cache_position_full,
            past_key_values, layer_range=range(config.check_layer),
        )
        timer.mark("full_prefix")

        # ── Layer check_layer: HKVD + selector + sparse slice ───────────
        layer_ck = inner.layers[config.check_layer]
        attn_ck = layer_ck.self_attn

        residual_full = hidden_states
        h_normed = layer_ck.input_layernorm(hidden_states)
        q_full = attn_ck.q_proj(h_normed)                # (1, S, num_heads*D)
        k_full_pre = attn_ck.k_proj(h_normed)            # (1, S, hidden_kv)
        v_full = attn_ck.v_proj(h_normed)

        # HKVD on pre-RoPE K (RoPE preserves L2 — same result as post-RoPE).
        deviations = kv_deviation(k_full_pre, K_stored_pre[config.check_layer])

        # Forced positions: last position (greedy decode needs valid logits).
        # In Stage 1, all chunks are required to be in KVStore, so there
        # are no "gap" positions from missing entries. The last-position
        # force is the only forced mask.
        forced_mask = torch.zeros(total_seq, dtype=torch.bool, device=device)
        forced_mask[-1] = True

        recompute_k = max(int(total_seq * config.recompute_ratio), 1)
        top_indices = _select_recompute_indices(
            deviations=deviations,
            importance=importance_full,
            recompute_k=recompute_k,
            config=config,
            structural_mask=structural_full,
            forced_mask=forced_mask,
            eligible_mask=eligible_full,
        )
        topk_num = int(top_indices.shape[0])
        if topk_num == 0:
            raise RuntimeError(
                "selector returned empty top_indices — should be unreachable "
                "since forced_mask always includes the last position."
            )

        is_top = torch.zeros(total_seq, dtype=torch.bool, device=device)
        is_top[top_indices] = True
        timer.mark("hkvd_select")

        # Mixed K_pre, V: cached at non-top, fresh at top.
        k_mixed_pre = k_full_pre.clone()
        v_mixed = v_full.clone()
        non_top_mask = ~is_top
        k_mixed_pre[:, non_top_mask, :] = K_stored_pre[config.check_layer][:, non_top_mask, :]
        v_mixed[:, non_top_mask, :] = V_stored[config.check_layer][:, non_top_mask, :]

        # Reshape to heads.
        hidden_shape_full = (1, total_seq, -1, head_dim)
        q_full_heads = q_full.view(hidden_shape_full).transpose(1, 2)
        k_mixed_heads_pre = k_mixed_pre.view(hidden_shape_full).transpose(1, 2)
        v_mixed_heads = v_mixed.view(hidden_shape_full).transpose(1, 2)

        # Apply RoPE on full Q and full mixed K at the full positions.
        q_full_heads_post, k_full_post = apply_rotary_pos_emb(
            q_full_heads, k_mixed_heads_pre, cos_full, sin_full,
        )

        # Slice Q to top_indices.
        q_sparse_post = q_full_heads_post[:, :, top_indices, :]
        residual_sparse = residual_full[:, top_indices, :]

        past_key_values.update(k_full_post, v_mixed_heads, config.check_layer)

        # CompBlend extension: per-head valid_mask in SDPA attn_mask.
        attn_mask_ck = _build_attn_mask_with_per_head_valid(
            causal_mask_full=causal_mask_full,
            top_indices=top_indices,
            total_seq=total_seq,
            valid_layer=valid_mask_full[config.check_layer],
            is_top=is_top,
            n_rep=n_rep,
            mask_dtype=causal_mask_full.dtype,
            flags=flags,
        )

        # GQA expansion of K/V.
        k_rep = k_full_post.repeat_interleave(n_rep, dim=1)
        v_rep = v_mixed_heads.repeat_interleave(n_rep, dim=1)
        attn_out_ck = _sdpa(
            q_sparse_post, k_rep, v_rep,
            attn_mask=attn_mask_ck,
            scale=attn_ck.scaling,
        )
        attn_out_ck = attn_out_ck.transpose(1, 2).reshape(
            1, topk_num, num_heads * head_dim,
        ).contiguous()
        attn_out_ck = attn_ck.o_proj(attn_out_ck)

        # Residual + FFN, sparse.
        h_sparse = residual_sparse + attn_out_ck
        residual_sparse2 = h_sparse
        h_sparse_normed = layer_ck.post_attention_layernorm(h_sparse)
        h_sparse = layer_ck.mlp(h_sparse_normed)
        h_sparse = residual_sparse2 + h_sparse
        timer.mark("check_layer")

        # ── Layers check_layer+1..n_layers-1: sparse hidden, mixed K/V ──
        cos_sparse = cos_full[:, top_indices, :]
        sin_sparse = sin_full[:, top_indices, :]

        for li in range(config.check_layer + 1, n_layers):
            layer_li = inner.layers[li]
            attn_li = layer_li.self_attn

            residual_sparse_in = h_sparse
            h_normed_sparse = layer_li.input_layernorm(h_sparse)

            q_sparse = attn_li.q_proj(h_normed_sparse)
            k_sparse_pre = attn_li.k_proj(h_normed_sparse)
            v_sparse = attn_li.v_proj(h_normed_sparse)

            sparse_shape = (1, topk_num, -1, head_dim)
            q_sparse_heads = q_sparse.view(sparse_shape).transpose(1, 2)
            k_sparse_heads_pre = k_sparse_pre.view(sparse_shape).transpose(1, 2)
            v_sparse_heads = v_sparse.view(sparse_shape).transpose(1, 2)

            q_sparse_post, k_sparse_post = apply_rotary_pos_emb(
                q_sparse_heads, k_sparse_heads_pre, cos_sparse, sin_sparse,
            )

            # Build full K cache: cached(RoPE-shifted) at non-top, fresh at top.
            k_cached_pre_full = K_stored_pre[li]                              # (1, S, hidden_kv)
            k_cached_pre_heads = k_cached_pre_full.view(
                1, total_seq, num_kv_heads, head_dim,
            ).transpose(1, 2)                                                  # (1, H_kv, S, D)
            dummy_q = torch.zeros_like(k_cached_pre_heads)
            _qd, k_cached_post_heads = apply_rotary_pos_emb(
                dummy_q, k_cached_pre_heads, cos_full, sin_full,
            )
            k_full_li = k_cached_post_heads.clone()
            k_full_li[:, :, top_indices, :] = k_sparse_post

            v_cached_heads = V_stored[li].view(
                1, total_seq, num_kv_heads, head_dim,
            ).transpose(1, 2)
            v_full_li = v_cached_heads.clone()
            v_full_li[:, :, top_indices, :] = v_sparse_heads

            past_key_values.update(k_full_li, v_full_li, li)

            # Per-head mask for this layer (mirrors check_layer).
            attn_mask_li = _build_attn_mask_with_per_head_valid(
                causal_mask_full=causal_mask_full,
                top_indices=top_indices,
                total_seq=total_seq,
                valid_layer=valid_mask_full[li],
                is_top=is_top,
                n_rep=n_rep,
                mask_dtype=causal_mask_full.dtype,
                flags=flags,
            )

            k_rep_li = k_full_li.repeat_interleave(n_rep, dim=1)
            v_rep_li = v_full_li.repeat_interleave(n_rep, dim=1)
            attn_out_li = _sdpa(
                q_sparse_post, k_rep_li, v_rep_li,
                attn_mask=attn_mask_li,
                scale=attn_li.scaling,
            )
            attn_out_li = attn_out_li.transpose(1, 2).reshape(
                1, topk_num, num_heads * head_dim,
            ).contiguous()
            attn_out_li = attn_li.o_proj(attn_out_li)

            h_sparse = residual_sparse_in + attn_out_li
            residual_sparse_in2 = h_sparse
            h_sparse_normed = layer_li.post_attention_layernorm(h_sparse)
            h_sparse = layer_li.mlp(h_sparse_normed)
            h_sparse = residual_sparse_in2 + h_sparse
        timer.mark("sparse_layers")

        # ── Final norm + lm_head on SPARSE hidden ──────────────────────
        h_sparse_normed = inner.norm(h_sparse)
        if last_logits_only:
            # top_indices is sorted ASC and total_seq-1 is forced via forced_mask,
            # so the last sparse row corresponds to the last sequence position.
            logits_full = layerwise_model.model.lm_head(h_sparse_normed[:, -1:, :])
        else:
            logits_sparse = layerwise_model.model.lm_head(h_sparse_normed)        # (1, Q, vocab)

            vocab_size = logits_sparse.shape[-1]
            logits_full = torch.zeros(
                (1, total_seq, vocab_size),
                dtype=logits_sparse.dtype, device=device,
            )
            logits_full[:, top_indices, :] = logits_sparse
        timer.mark("lmhead")

    # Stash timing events on the caller's dict (if provided)
    if timings is not None:
        timings.update(timer.events)
        timings["_total_ms"] = sum(timer.events.values())

    out_obj = LayerwiseOutput(logits=logits_full, past_key_values=past_key_values)
    result = out_obj if return_layerwise_output else logits_full
    if return_hkvd_indices:
        return result, top_indices
    return result
