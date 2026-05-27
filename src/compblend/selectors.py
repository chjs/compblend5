"""Token selectors for CompBlend.

What's here (Stage 1, minimum viable):
    - `select_topk_sorted`     — pick top-k by score, return sorted-by-position
    - `gated_top_k`            — paper §3's Gated HKVD selector
    - `chunk_internal_rank`    — opt-in per-chunk normalization helper

What's NOT here (deferred to a later milestone):
    - additive_rank / rank_product / pareto_layered / per_chunk_gated_top_k
      — research alternatives from CompBlend-old. Bring them back only if a
      study needs them. The paper's defended algorithm is gated_top_k.

Fix vs CompBlend-old's gated_top_k
──────────────────────────────────
Old: the engine boosted hkvd[forced]=+inf and imp[forced]=+inf BEFORE
calling gated_top_k. When `forced_count > gate_percentile * N`, all +inf
positions saturated the importance gate, threshold became +inf, no
non-forced positions survived; then top-k by HKVD padded with -inf
positions (random non-eligibles). Documented in our CompBlend-old analysis.

New: `forced_mask` is a first-class parameter. Forced positions are
unconditionally included BEFORE the gate runs (they don't participate in
the gate's percentile computation). The remaining budget is filled by
gating + HKVD over the non-forced, non-structural candidates. No +inf hack
on score tensors; gate cardinality stays interpretable.
"""
from __future__ import annotations

import torch


# ──────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────


def select_topk_sorted(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k by score, returned as int64 indices sorted ASCENDING by position.

    Causal attention requires K positions to be monotonically increasing
    (matches LMCache's `top_indices, _ = torch.sort(top_indices)` step).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, int(scores.numel()))
    top = torch.topk(scores, k=k).indices
    top, _ = torch.sort(top)
    return top


def _rank_normalize(scores: torch.Tensor) -> torch.Tensor:
    """Percentile-rank transform. Returns values in [0, 1]; ties broken arbitrarily."""
    n = scores.numel()
    if n == 0:
        return scores
    order = torch.argsort(scores, descending=False)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks[order] = torch.arange(n, device=scores.device, dtype=torch.float32) / max(n - 1, 1)
    return ranks


def chunk_internal_rank(chunk_importance: torch.Tensor) -> torch.Tensor:
    """Per-chunk percentile rank — length-fair normalization across chunks.

    See CompBlend-old's 4a-1 finding (14× scale ratio between short/long
    SnapKV chunks): concatenating raw importance across chunks biases the
    selector toward short docs. Applying rank within each chunk before
    concatenation eliminates this.
    """
    return _rank_normalize(chunk_importance)


# ──────────────────────────────────────────────────────────────────────────
# Gated HKVD — paper §3
# ──────────────────────────────────────────────────────────────────────────


def gated_top_k(
    hkvd_scores: torch.Tensor,
    importance_scores: torch.Tensor,
    recompute_k: int,
    gate_percentile: float,
    structural_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paper §3 Gated HKVD: importance gates the candidate set, HKVD picks within it.

    Algorithm:
      1. Include all `forced_mask` positions unconditionally.
      2. Exclude all `structural_mask` positions.
      3. Restrict the candidate pool to `eligible_mask` (None = all).
      4. From the remaining positions, keep those with importance in the top
         `gate_percentile` fraction. With `gate_percentile=0.5`, keeps
         the top 50% by importance.
      5. From the gated set, pick `recompute_k - forced_count` by HKVD.

    Why `eligible_mask` (added 2026-05-27): compression's design intent is
    "evict the unimportant tokens; trust the choice". Recomputation must
    NOT recover evicted tokens — they were deliberately discarded. Pass
    `eligible_mask = (any-head retained at this position)` to restrict the
    selector to the kept set. forced_mask (e.g. last_pos for decode)
    overrides eligibility — those positions are included regardless.

    Args:
        hkvd_scores:        [N] per-token deviation.
        importance_scores:  [N] per-token compression importance.
        recompute_k:        target number of tokens to select.
        gate_percentile:    fraction of non-forced tokens to keep as gated
                            candidates. 1.0 = skip gate. Typical: 0.5.
        structural_mask:    optional [N] bool. True = exempt (never selected).
        forced_mask:        optional [N] bool. True = always selected.
        eligible_mask:      optional [N] bool. True = position is in the
                            candidate pool. None = all positions eligible
                            (backward compat). Compression-aware callers
                            pass `union-of-heads valid_mask` here.

    Returns:
        int64 tensor of selected positions, sorted ascending.
    """
    n = int(hkvd_scores.numel())
    if int(importance_scores.numel()) != n:
        raise ValueError(
            f"score length mismatch: hkvd={n}, importance={int(importance_scores.numel())}"
        )
    device = hkvd_scores.device

    # Coerce optional masks to bool [N] on the right device.
    if structural_mask is None:
        structural = torch.zeros(n, dtype=torch.bool, device=device)
    else:
        if int(structural_mask.numel()) != n:
            raise ValueError("structural_mask length mismatch")
        structural = structural_mask.to(device=device, dtype=torch.bool)

    if forced_mask is None:
        forced = torch.zeros(n, dtype=torch.bool, device=device)
    else:
        if int(forced_mask.numel()) != n:
            raise ValueError("forced_mask length mismatch")
        forced = forced_mask.to(device=device, dtype=torch.bool)

    if eligible_mask is None:
        eligible = torch.ones(n, dtype=torch.bool, device=device)
    else:
        if int(eligible_mask.numel()) != n:
            raise ValueError("eligible_mask length mismatch")
        eligible = eligible_mask.to(device=device, dtype=torch.bool)

    # Resolve conflicts: forced wins over structural (a gap that's also
    # window-protected still MUST recompute — no cached K to attend to).
    structural = structural & ~forced

    forced_idx = forced.nonzero(as_tuple=False).squeeze(-1)
    forced_count = int(forced_idx.numel())

    # Bump the effective budget if forced exceeds requested — caller's
    # `recompute_k` was a lower bound, not a ceiling.
    target_k = max(recompute_k, forced_count)

    remaining_slots = max(target_k - forced_count, 0)
    if remaining_slots == 0:
        # All slots are forced. Just return them sorted.
        out, _ = torch.sort(forced_idx)
        return out

    # Build the candidate pool: not-forced AND not-structural AND eligible.
    # `eligible` enforces "compression's choice is respected" — evicted
    # positions are not selectable by the gated path. forced positions are
    # included regardless (they override eligibility).
    cand = (~forced) & (~structural) & eligible
    if not bool(cand.any().item()):
        # Nothing to add beyond forced.
        out, _ = torch.sort(forced_idx)
        return out

    # ── Importance gate over the candidate pool ─────────────────────────
    # Use rank within the candidate pool only — that way the gate's
    # `gate_percentile` reads as "top fraction of candidates" regardless of
    # forced/structural composition.
    if 0.0 < gate_percentile < 1.0:
        cand_idx = cand.nonzero(as_tuple=False).squeeze(-1)
        cand_imp = importance_scores[cand_idx]
        n_cand = int(cand_idx.numel())
        n_keep_gate = max(int(round(n_cand * gate_percentile)), 1)
        # Threshold: the n_keep_gate-th largest importance in the candidate pool.
        sorted_imp, _ = torch.sort(cand_imp, descending=True)
        # Use the value AT index (n_keep_gate - 1) so >= keeps exactly that many
        # (or more, on ties).
        threshold = sorted_imp[min(n_keep_gate - 1, n_cand - 1)]
        passes_gate = cand & (importance_scores >= threshold)
    else:
        # gate_percentile = 0 (or anything outside (0, 1)): no gating.
        passes_gate = cand

    # ── HKVD top-k within gated pool ─────────────────────────────────────
    eligible_idx = passes_gate.nonzero(as_tuple=False).squeeze(-1)
    n_eligible = int(eligible_idx.numel())
    if n_eligible == 0:
        out, _ = torch.sort(forced_idx)
        return out

    take = min(remaining_slots, n_eligible)
    eligible_hkvd = hkvd_scores[eligible_idx]
    top_in_gate = torch.topk(eligible_hkvd, k=take).indices       # indices INTO eligible_idx
    picked_from_gate = eligible_idx[top_in_gate]

    # Union forced + picked_from_gate, sort ascending.
    union = torch.cat([forced_idx, picked_from_gate], dim=0).unique()
    union, _ = torch.sort(union)
    return union


__all__ = [
    "select_topk_sorted",
    "chunk_internal_rank",
    "gated_top_k",
]
