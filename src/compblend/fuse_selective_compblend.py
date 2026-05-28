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

import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as _sdpa
from transformers.cache_utils import DynamicCache

# FlexAttention import is optional (PyTorch 2.5+ beta, 2.6+ stable).
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    _FLEX_AVAILABLE = True
except ImportError:
    _flex_attention = None
    _create_block_mask = None
    _FLEX_AVAILABLE = False


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


def _flex_attention_per_head_valid(
    q: torch.Tensor,              # [1, H_q, Q, D] (sparse Q)
    k: torch.Tensor,              # [1, H_q, K, D] (full K, evicted slots zero-filled)
    v: torch.Tensor,              # [1, H_q, K, D]
    *,
    top_indices: torch.Tensor,    # [Q] int64 — global positions of sparse Q
    valid_layer: torch.Tensor,    # [H_kv, K] bool — per-head per-position validity
    is_top: torch.Tensor,         # [K] bool — True at top_indices (always-valid override)
    n_rep: int,                   # H_q // H_kv (GQA)
    scale: float,
) -> torch.Tensor:
    """Per-head per-position valid mask via FlexAttention BlockMask.

    Trick: we project the sparse-Q causal+valid mask into a DENSE form
    indexed by (h_q, q_local, kv_global). Concretely we build a precomputed
    bool tensor of shape [H_q, Q, K] and let the mask_mod just index into it
    by (h, q, kv). The bool tensor itself is small (typically tens of MB)
    because FlexAttention only materializes blocks where the mask varies —
    OR the [H_q, Q, K] tensor must fit anyway. For the kept sparse Q this
    is feasible.

    Requires PyTorch 2.5+. Caller must check _FLEX_AVAILABLE.
    """
    if not _FLEX_AVAILABLE:
        raise RuntimeError("FlexAttention not available; install PyTorch 2.5+")

    B, H_q, Q, D = q.shape
    _, _, K, _ = k.shape
    device = q.device

    top_indices = top_indices.to(device, dtype=torch.long).contiguous()
    valid_layer = valid_layer.to(device, dtype=torch.bool).contiguous()
    is_top = is_top.to(device, dtype=torch.bool).contiguous()

    # Combine per-head valid + is_top override → [H_q, K] (GQA-expanded).
    valid_hq_k = valid_layer.repeat_interleave(n_rep, dim=0) | is_top.unsqueeze(0)

    # FlexAttention's mask_mod must accept scalar (b, h, q_local, kv_global)
    # and return a scalar bool. We use vmap-safe tensor indexing.
    def mask_mod(b, h, q_local, kv_global):
        q_global = top_indices[q_local]
        causal = q_global >= kv_global
        valid = valid_hq_k[h, kv_global]
        return causal & valid

    try:
        block_mask = _create_block_mask(
            mask_mod, B=B, H=H_q, Q_LEN=Q, KV_LEN=K, device=device,
        )
        return _flex_attention(q, k, v, block_mask=block_mask, scale=scale)
    except Exception as exc:
        # FlexAttention has known limitations with closure tensor lookups under vmap.
        # Fall back to SDPA additive mask path for this call.
        import sys as _sys
        print(f"[flex_attention fallback to SDPA] {type(exc).__name__}: {exc}",
              file=_sys.stderr, flush=True)
        raise


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
            flags.setdefault("per_head_mask_unnecessary_layers", []).append(
                int(flags.get("_check_layer_idx", flags.get("_sparse_layer_idx", -1)))
            )
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
    # Env override: set COMPBLEND_ATTN_MASK_CAP_MB to raise the cap (e.g. 32768
    # for 32 GB). Used to verify that per-head mask fallback is the root cause
    # of long-context F1 collapse. Production should use FlexAttention (Stage 2).
    import os as _os
    ATTN_MASK_MEMORY_CAP_BYTES = int(_os.environ.get("COMPBLEND_ATTN_MASK_CAP_MB", "256")) * 1024 ** 2
    bytes_per_elem = mask_dtype.itemsize if hasattr(mask_dtype, "itemsize") else 2
    Q = int(top_indices.numel())
    K = int(total_seq)
    H_q = int(valid_layer.shape[0]) * int(n_rep)
    mask_bytes = 1 * H_q * Q * K * bytes_per_elem
    cur_layer = -1
    if flags is not None:
        cur_layer = int(flags.get("_check_layer_idx", flags.get("_sparse_layer_idx", -1)))
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
            flags["attn_mask_memory_cap_bytes"] = ATTN_MASK_MEMORY_CAP_BYTES
            flags.setdefault("per_head_mask_fallback_layers", []).append(cur_layer)
            flags["fallback_reason"] = "mask_bytes > ATTN_MASK_MEMORY_CAP_BYTES"
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
        flags.setdefault("per_head_mask_exact_layers", []).append(cur_layer)
        flags["largest_attn_mask_estimated_bytes"] = max(
            flags.get("largest_attn_mask_estimated_bytes", 0), mask_bytes
        )

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
    selector_stats: dict | None = None,
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
        timer.mark("check_input_ln")

        q_full = attn_ck.q_proj(h_normed)                # (1, S, num_heads*D)
        k_full_pre = attn_ck.k_proj(h_normed)            # (1, S, hidden_kv)
        v_full = attn_ck.v_proj(h_normed)
        timer.mark("check_qkv_proj")

        # HKVD on pre-RoPE K (RoPE preserves L2 — same result as post-RoPE).
        deviations = kv_deviation(k_full_pre, K_stored_pre[config.check_layer])
        timer.mark("check_hkvd_score")

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
        timer.mark("check_selector")
        # Backward-compat alias for old "hkvd_select" key — sum of hkvd_score + selector.
        if timings is not None:
            timer.events["hkvd_select"] = (
                timer.events.get("check_hkvd_score", 0.0)
                + timer.events.get("check_selector", 0.0)
            )

        # ─── Selector statistics ────────────────────────────────────────
        if selector_stats is not None:
            structural_mask_cpu = structural_full.to(torch.bool)
            eligible_mask_cpu = eligible_full.to(torch.bool)
            forced_mask_cpu = forced_mask.to(torch.bool)
            is_top_cpu = is_top.to(torch.bool)
            n_total = int(total_seq)
            n_struct = int(structural_mask_cpu.sum().item())
            n_eligible = int(eligible_mask_cpu.sum().item())
            n_forced = int(forced_mask_cpu.sum().item())
            n_selected = int(is_top_cpu.sum().item())
            # Overlap with structural (selected positions that are structural)
            n_sel_from_struct = int((is_top_cpu & structural_mask_cpu).sum().item())
            # Selection diagnostics — how does this selector distribute its
            # choices across eligible / ineligible positions, and how do its
            # picked HKVD / importance values compare to the unpicked rest?
            sel_eligible_count = int((is_top_cpu & eligible_mask_cpu).sum().item())
            sel_ineligible_count = n_selected - sel_eligible_count
            # HKVD / importance distributions at selected vs not-selected
            sel_dev = deviations[top_indices].float()
            not_top = ~is_top
            not_top[forced_mask] = False  # exclude forced (selected anyway)
            unsel_dev = deviations[not_top].float() if bool(not_top.any().item()) else None
            sel_imp = importance_full[top_indices].float()
            # By eligibility — HKVD/importance at eligible vs ineligible
            elig = eligible_full
            inelig = ~eligible_full
            elig_dev_mean = float(deviations[elig].float().mean().item()) if bool(elig.any().item()) else None
            inelig_dev_mean = float(deviations[inelig].float().mean().item()) if bool(inelig.any().item()) else None
            elig_imp_mean = float(importance_full[elig].float().mean().item()) if bool(elig.any().item()) else None
            inelig_imp_mean = float(importance_full[inelig].float().mean().item()) if bool(inelig.any().item()) else None
            selector_stats.update({
                "total_tokens": n_total,
                "structural_tokens": n_struct,
                "compressible_tokens": n_total - n_struct,
                "eligible_tokens": n_eligible,
                "ineligible_tokens": n_total - n_eligible,
                "forced_tokens": n_forced,
                "selected_recompute_tokens": n_selected,
                "selected_from_structural_tokens": n_sel_from_struct,
                "selected_from_compressible_tokens": n_selected - n_sel_from_struct,
                "selected_from_eligible_tokens": sel_eligible_count,
                "selected_from_ineligible_tokens": sel_ineligible_count,
                "recompute_ratio_requested": float(config.recompute_ratio),
                "selected_ratio_actual": (n_selected / n_total) if n_total else 0.0,
                "eligible_ratio": (n_eligible / n_total) if n_total else 0.0,
                "selector": str(config.selector),
                "gate_percentile": (
                    float(config.gate_percentile)
                    if config.selector == "gated_top_k"
                    else None
                ),
                "gate_is_active": (
                    config.selector == "gated_top_k"
                    and 0.0 < float(config.gate_percentile) < 1.0
                ),
                # Distribution diagnostics
                "selected_hkvd_mean": float(sel_dev.mean().item()),
                "selected_hkvd_std":  float(sel_dev.std().item()) if sel_dev.numel() > 1 else 0.0,
                "selected_hkvd_min":  float(sel_dev.min().item()),
                "selected_hkvd_max":  float(sel_dev.max().item()),
                "not_selected_hkvd_mean": float(unsel_dev.mean().item()) if unsel_dev is not None else None,
                "selected_importance_mean": float(sel_imp.mean().item()),
                "selected_importance_std":  float(sel_imp.std().item()) if sel_imp.numel() > 1 else 0.0,
                "eligible_hkvd_mean":   elig_dev_mean,
                "ineligible_hkvd_mean": inelig_dev_mean,
                "eligible_importance_mean":   elig_imp_mean,
                "ineligible_importance_mean": inelig_imp_mean,
                # Selected indices themselves (for cross-selector overlap analysis).
                # ~4-5K ints per call → ~30KB JSON. Manageable.
                "selected_indices": top_indices.detach().cpu().tolist(),
            })

        # Mixed K_pre, V: cached at non-top, fresh at top.
        k_mixed_pre = k_full_pre.clone()
        v_mixed = v_full.clone()
        non_top_mask = ~is_top
        k_mixed_pre[:, non_top_mask, :] = K_stored_pre[config.check_layer][:, non_top_mask, :]
        v_mixed[:, non_top_mask, :] = V_stored[config.check_layer][:, non_top_mask, :]
        timer.mark("check_kv_mix")

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
        timer.mark("check_rope")

        past_key_values.update(k_full_post, v_mixed_heads, config.check_layer)

        # CompBlend extension: per-head valid_mask via SDPA additive mask OR FlexAttention.
        if flags is not None:
            flags["_check_layer_idx"] = int(config.check_layer)

        use_flex = (
            os.environ.get("COMPBLEND_USE_FLEX_ATTENTION", "0") == "1"
            and _FLEX_AVAILABLE
        )
        # GQA expansion of K/V (common to both paths).
        k_rep = k_full_post.repeat_interleave(n_rep, dim=1)
        v_rep = v_mixed_heads.repeat_interleave(n_rep, dim=1)

        if use_flex:
            if flags is not None:
                flags["attention_backend"] = "flex_attention"
                flags["per_head_mask_required"] = True
                flags.setdefault("per_head_mask_exact_layers", []).append(int(config.check_layer))
            timer.mark("check_attn_mask_build")
            attn_out_ck = _flex_attention_per_head_valid(
                q_sparse_post, k_rep, v_rep,
                top_indices=top_indices,
                valid_layer=valid_mask_full[config.check_layer],
                is_top=is_top, n_rep=n_rep,
                scale=attn_ck.scaling,
            )
        else:
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
            timer.mark("check_attn_mask_build")
            attn_out_ck = _sdpa(
                q_sparse_post, k_rep, v_rep,
                attn_mask=attn_mask_ck,
                scale=attn_ck.scaling,
            )
        attn_out_ck = attn_out_ck.transpose(1, 2).reshape(
            1, topk_num, num_heads * head_dim,
        ).contiguous()
        timer.mark("check_attention_sdpa")

        attn_out_ck = attn_ck.o_proj(attn_out_ck)
        timer.mark("check_o_proj")

        # Residual + FFN, sparse.
        h_sparse = residual_sparse + attn_out_ck
        residual_sparse2 = h_sparse
        h_sparse_normed = layer_ck.post_attention_layernorm(h_sparse)
        h_sparse = layer_ck.mlp(h_sparse_normed)
        h_sparse = residual_sparse2 + h_sparse
        timer.mark("check_ffn")

        # Roll-up: backward-compatible "check_layer" key = sum of all check_* events
        # for callers that still read the coarse name.
        if timings is not None:
            timer.events["check_layer"] = sum(
                v for k, v in timer.events.items() if k.startswith("check_")
                and k not in ("check_layer",)
            )

        # ── Layers check_layer+1..n_layers-1: sparse hidden, mixed K/V ──
        cos_sparse = cos_full[:, top_indices, :]
        sin_sparse = sin_full[:, top_indices, :]

        for li in range(config.check_layer + 1, n_layers):
            layer_li = inner.layers[li]
            attn_li = layer_li.self_attn

            residual_sparse_in = h_sparse
            h_normed_sparse = layer_li.input_layernorm(h_sparse)
            timer.mark("sparse_input_ln")

            q_sparse = attn_li.q_proj(h_normed_sparse)
            k_sparse_pre = attn_li.k_proj(h_normed_sparse)
            v_sparse = attn_li.v_proj(h_normed_sparse)
            timer.mark("sparse_qkv_proj")

            sparse_shape = (1, topk_num, -1, head_dim)
            q_sparse_heads = q_sparse.view(sparse_shape).transpose(1, 2)
            k_sparse_heads_pre = k_sparse_pre.view(sparse_shape).transpose(1, 2)
            v_sparse_heads = v_sparse.view(sparse_shape).transpose(1, 2)

            q_sparse_post, k_sparse_post = apply_rotary_pos_emb(
                q_sparse_heads, k_sparse_heads_pre, cos_sparse, sin_sparse,
            )
            timer.mark("sparse_rope")

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
            timer.mark("sparse_kv_mix")

            # Per-head mask for this layer (mirrors check_layer).
            if flags is not None:
                flags["_sparse_layer_idx"] = int(li)

            k_rep_li = k_full_li.repeat_interleave(n_rep, dim=1)
            v_rep_li = v_full_li.repeat_interleave(n_rep, dim=1)

            if use_flex:
                if flags is not None:
                    flags.setdefault("per_head_mask_exact_layers", []).append(int(li))
                timer.mark("sparse_attn_mask_build")
                attn_out_li = _flex_attention_per_head_valid(
                    q_sparse_post, k_rep_li, v_rep_li,
                    top_indices=top_indices,
                    valid_layer=valid_mask_full[li],
                    is_top=is_top, n_rep=n_rep,
                    scale=attn_li.scaling,
                )
            else:
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
                timer.mark("sparse_attn_mask_build")
                attn_out_li = _sdpa(
                    q_sparse_post, k_rep_li, v_rep_li,
                    attn_mask=attn_mask_li,
                    scale=attn_li.scaling,
                )
            attn_out_li = attn_out_li.transpose(1, 2).reshape(
                1, topk_num, num_heads * head_dim,
            ).contiguous()
            timer.mark("sparse_attention_sdpa")

            attn_out_li = attn_li.o_proj(attn_out_li)
            timer.mark("sparse_o_proj")

            h_sparse = residual_sparse_in + attn_out_li
            residual_sparse_in2 = h_sparse
            h_sparse_normed = layer_li.post_attention_layernorm(h_sparse)
            h_sparse = layer_li.mlp(h_sparse_normed)
            h_sparse = residual_sparse_in2 + h_sparse
            timer.mark("sparse_ffn")

        # Roll-up: backward-compatible "sparse_layers" key = sum of all sparse_* events.
        if timings is not None:
            timer.events["sparse_layers"] = sum(
                v for k, v in timer.events.items() if k.startswith("sparse_")
                and k not in ("sparse_layers",)
            )

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

    # Stash timing events on the caller's dict (if provided).
    # _total_ms excludes coarse rollup keys ("check_layer", "sparse_layers",
    # "hkvd_select") to avoid double-counting when fine-grained keys are present.
    _ROLLUP_KEYS = {"check_layer", "sparse_layers", "hkvd_select"}
    if timings is not None:
        timings.update(timer.events)
        timings["_total_ms"] = sum(
            v for k, v in timer.events.items() if k not in _ROLLUP_KEYS
        )

    # Compute actual KVzip retention ratio from the assembled valid_mask.
    # `valid_mask_full[L, H_kv, S]` was filled chunk-by-chunk during kv_load;
    # for chunks without a valid_mask entry (structural prefix/suffix) the
    # default was all-True (see _load_chunk_extensions). The "kvzip_ratio_actual"
    # measures the kept fraction over the full (L, H_kv, S) tensor — useful to
    # see how much KVzip's pair-level eviction actually dropped vs the
    # requested ratio.
    if flags is not None:
        try:
            stored_kv_valid_ratio = float(valid_mask_full.float().mean().item())
        except Exception:
            stored_kv_valid_ratio = -1.0
        flags["stored_kv_valid_ratio"] = stored_kv_valid_ratio
        flags["kvzip_ratio_actual"] = stored_kv_valid_ratio
        # Strip internal helper keys before returning to caller.
        flags.pop("_check_layer_idx", None)
        flags.pop("_sparse_layer_idx", None)

    out_obj = LayerwiseOutput(logits=logits_full, past_key_values=past_key_values)
    result = out_obj if return_layerwise_output else logits_full
    if return_hkvd_indices:
        return result, top_indices
    return result
