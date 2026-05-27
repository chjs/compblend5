"""M2 — selectors + per-head mask construction. CPU-only unit tests."""
from __future__ import annotations

import pytest
import torch

from compblend.config import CompBlendConfig
from compblend.selectors import (
    chunk_internal_rank,
    gated_top_k,
    select_topk_sorted,
)
from compblend.fuse_selective_compblend import (
    _build_attn_mask_with_per_head_valid,
    _select_recompute_indices,
)


# ──────────────────────────────────────────────────────────────────────────
# select_topk_sorted
# ──────────────────────────────────────────────────────────────────────────


def test_select_topk_sorted_returns_ascending_indices():
    scores = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5], dtype=torch.float32)
    idx = select_topk_sorted(scores, k=3)
    # Top 3 by value are positions [1, 3, 4] with scores [0.9, 0.7, 0.5].
    # Result MUST be sorted ascending (causal).
    assert idx.tolist() == [1, 3, 4]


def test_select_topk_sorted_clips_k_to_numel():
    scores = torch.tensor([1.0, 2.0])
    idx = select_topk_sorted(scores, k=10)
    assert idx.numel() == 2


def test_select_topk_sorted_rejects_non_positive_k():
    with pytest.raises(ValueError):
        select_topk_sorted(torch.zeros(5), k=0)


# ──────────────────────────────────────────────────────────────────────────
# gated_top_k — Gated HKVD core
# ──────────────────────────────────────────────────────────────────────────


def test_gated_top_k_basic_no_forced_no_structural():
    """Without forced/structural, gate picks top 50% by importance,
    then HKVD top-k among them."""
    # 8 tokens. Importance: [0,1,2,3,4,5,6,7]. HKVD: [7,6,5,4,3,2,1,0].
    # Gate keeps importance >= median → indices {4,5,6,7} (top 50%).
    # HKVD over {4,5,6,7} is [3,2,1,0] → top-2 by HKVD are positions 4,5.
    n = 8
    importance = torch.arange(n, dtype=torch.float32)
    hkvd = torch.arange(n, dtype=torch.float32).flip(0)        # 7..0
    idx = gated_top_k(
        hkvd_scores=hkvd, importance_scores=importance,
        recompute_k=2, gate_percentile=0.5,
    )
    assert sorted(idx.tolist()) == [4, 5]


def test_gated_top_k_forced_always_included():
    """forced_mask True positions are unconditionally selected."""
    n = 10
    importance = torch.zeros(n)            # uniform — gate is moot
    hkvd = torch.zeros(n)
    forced = torch.zeros(n, dtype=torch.bool)
    forced[2] = True
    forced[7] = True
    idx = gated_top_k(
        hkvd_scores=hkvd, importance_scores=importance,
        recompute_k=2, gate_percentile=0.5, forced_mask=forced,
    )
    # Forced count = 2, recompute_k = 2 → exactly the forced positions.
    assert sorted(idx.tolist()) == [2, 7]


def test_gated_top_k_forced_count_exceeds_recompute_k():
    """If forced_count > recompute_k, we bump the budget — never drop forced."""
    n = 10
    forced = torch.zeros(n, dtype=torch.bool)
    forced[[1, 3, 5, 7]] = True   # 4 forced
    importance = torch.zeros(n)
    hkvd = torch.zeros(n)
    idx = gated_top_k(
        hkvd_scores=hkvd, importance_scores=importance,
        recompute_k=2, gate_percentile=0.5, forced_mask=forced,
    )
    assert idx.tolist() == [1, 3, 5, 7]    # all 4 retained despite recompute_k=2


def test_gated_top_k_structural_never_selected():
    """structural_mask True positions are exempt."""
    n = 8
    structural = torch.zeros(n, dtype=torch.bool)
    structural[[0, 1]] = True
    # Importance/HKVD that would otherwise pick positions 0, 1.
    importance = torch.tensor([10., 10., 0., 0., 0., 0., 0., 0.])
    hkvd = torch.tensor([10., 10., 0., 0., 0., 0., 0., 0.])
    idx = gated_top_k(
        hkvd_scores=hkvd, importance_scores=importance,
        recompute_k=2, gate_percentile=0.5,
        structural_mask=structural,
    )
    assert 0 not in idx.tolist() and 1 not in idx.tolist()


def test_gated_top_k_forced_wins_over_structural():
    """Conflict resolution: forced beats structural (gap with no cache MUST
    recompute even if window-protected)."""
    n = 6
    forced = torch.zeros(n, dtype=torch.bool)
    structural = torch.zeros(n, dtype=torch.bool)
    forced[2] = True
    structural[2] = True
    idx = gated_top_k(
        hkvd_scores=torch.zeros(n),
        importance_scores=torch.zeros(n),
        recompute_k=1, gate_percentile=0.5,
        forced_mask=forced, structural_mask=structural,
    )
    assert 2 in idx.tolist()


def test_gated_top_k_gate_percentile_one_means_no_gating():
    """gate_percentile=1.0 disables the gate (everything passes)."""
    n = 6
    importance = torch.tensor([1., 2., 3., 4., 5., 6.])
    hkvd = torch.tensor([6., 5., 4., 3., 2., 1.])     # highest HKVD at 0
    idx = gated_top_k(
        hkvd_scores=hkvd, importance_scores=importance,
        recompute_k=2, gate_percentile=1.0,
    )
    # No gate → top-2 by HKVD = positions 0, 1.
    assert sorted(idx.tolist()) == [0, 1]


def test_gated_top_k_does_not_overflow_with_large_forced_pool():
    """Regression for CompBlend-old's gate-collapse bug: when forced
    occupies > gate_percentile fraction, old code padded with random
    non-eligibles. New code just returns forced + correctly-gated rest."""
    n = 20
    # 12 of 20 are forced (60%, more than gate_percentile=0.5 threshold).
    forced = torch.zeros(n, dtype=torch.bool)
    forced[:12] = True
    # Distinct importance/HKVD on the 8 non-forced positions.
    importance = torch.zeros(n)
    importance[12:] = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
    hkvd = torch.zeros(n)
    hkvd[12:] = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1.])
    # recompute_k=15: includes all 12 forced + 3 from non-forced.
    idx = gated_top_k(
        hkvd_scores=hkvd, importance_scores=importance,
        recompute_k=15, gate_percentile=0.5,
        forced_mask=forced,
    )
    chosen = idx.tolist()
    # All 12 forced present.
    for p in range(12):
        assert p in chosen, f"forced position {p} missing from {chosen}"
    # 3 extra from non-forced; gate picks top 50% of 8 = top 4 by importance,
    # which are positions {16, 17, 18, 19}. Then HKVD top-3 among those:
    # HKVD at {16,17,18,19} = {4,3,2,1} → top-3 are {16, 17, 18}.
    extras = [p for p in chosen if p >= 12]
    assert sorted(extras) == [16, 17, 18]


def test_gated_top_k_eligible_mask_restricts_to_kept():
    """eligible_mask=False positions must NEVER appear in the result (unless forced)."""
    n = 10
    # Make positions 0..4 ineligible (= evicted), 5..9 eligible (= kept)
    eligible = torch.tensor([False]*5 + [True]*5, dtype=torch.bool)
    # Importance / HKVD that, without eligibility, would prefer ineligible positions.
    importance = torch.tensor([10., 9., 8., 7., 6., 5., 4., 3., 2., 1.])
    hkvd = torch.tensor([10., 9., 8., 7., 6., 5., 4., 3., 2., 1.])
    idx = gated_top_k(
        hkvd_scores=hkvd, importance_scores=importance,
        recompute_k=3, gate_percentile=0.5,
        eligible_mask=eligible,
    )
    chosen = idx.tolist()
    # Without eligible_mask, top-3 by importance would be {0,1,2}.
    # With eligible_mask, candidates = {5..9}; gate top-50% = {5,6,7} by imp;
    # HKVD top-3 within = {5,6,7}.
    assert all(c >= 5 for c in chosen), (
        f"eligible_mask was bypassed; got {chosen} (expected all >= 5)"
    )


def test_gated_top_k_forced_overrides_eligible():
    """forced positions are included regardless of eligibility (last_pos for decode)."""
    n = 8
    eligible = torch.tensor([True]*6 + [False]*2, dtype=torch.bool)
    forced = torch.zeros(n, dtype=torch.bool)
    forced[7] = True  # last position, but ineligible — must still be picked
    idx = gated_top_k(
        hkvd_scores=torch.zeros(n),
        importance_scores=torch.zeros(n),
        recompute_k=1, gate_percentile=0.5,
        eligible_mask=eligible, forced_mask=forced,
    )
    assert 7 in idx.tolist(), (
        "forced position not picked despite being ineligible"
    )


def test_gated_top_k_eligible_default_none_unchanged():
    """eligible_mask=None preserves the pre-2026-05-27 behavior."""
    n = 6
    importance = torch.tensor([1., 2., 3., 4., 5., 6.])
    hkvd = torch.tensor([6., 5., 4., 3., 2., 1.])
    idx_with_none = gated_top_k(
        hkvd_scores=hkvd, importance_scores=importance,
        recompute_k=2, gate_percentile=0.5,
    )
    idx_with_all_true = gated_top_k(
        hkvd_scores=hkvd, importance_scores=importance,
        recompute_k=2, gate_percentile=0.5,
        eligible_mask=torch.ones(n, dtype=torch.bool),
    )
    assert idx_with_none.tolist() == idx_with_all_true.tolist()


def test_gated_top_k_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        gated_top_k(
            hkvd_scores=torch.zeros(5),
            importance_scores=torch.zeros(6),
            recompute_k=2, gate_percentile=0.5,
        )


# ──────────────────────────────────────────────────────────────────────────
# _select_recompute_indices — dispatch
# ──────────────────────────────────────────────────────────────────────────


def _baseline_select_inputs(n: int = 6):
    deviations = torch.arange(n, dtype=torch.float32).flip(0)     # 5..0
    importance = torch.arange(n, dtype=torch.float32)             # 0..5
    structural = torch.zeros(n, dtype=torch.bool)
    forced = torch.zeros(n, dtype=torch.bool)
    forced[-1] = True
    return deviations, importance, structural, forced


def test_dispatch_hkvd_only():
    dev, imp, struc, forced = _baseline_select_inputs()
    cfg = CompBlendConfig(selector="hkvd_only", recompute_ratio=0.5)
    idx = _select_recompute_indices(
        deviations=dev, importance=imp, recompute_k=3, config=cfg,
        structural_mask=struc, forced_mask=forced,
    )
    # +inf at last (forced); top-3 by HKVD: last (forced) + positions 0, 1.
    chosen = idx.tolist()
    assert 5 in chosen and 0 in chosen and 1 in chosen and len(chosen) == 3


def test_dispatch_importance_only():
    dev, imp, struc, forced = _baseline_select_inputs()
    cfg = CompBlendConfig(selector="importance_only", recompute_ratio=0.5)
    idx = _select_recompute_indices(
        deviations=dev, importance=imp, recompute_k=3, config=cfg,
        structural_mask=struc, forced_mask=forced,
    )
    # Forced + top importance: 5 (forced) is highest anyway, then 4, 3.
    chosen = idx.tolist()
    assert chosen == sorted([5, 4, 3])


def test_dispatch_gated_top_k():
    dev, imp, struc, forced = _baseline_select_inputs()
    cfg = CompBlendConfig(selector="gated_top_k", recompute_ratio=0.5,
                          gate_percentile=0.5)
    idx = _select_recompute_indices(
        deviations=dev, importance=imp, recompute_k=3, config=cfg,
        structural_mask=struc, forced_mask=forced,
    )
    # n=6, forced=[5]. Non-forced candidates: {0..4}, importance {0..4}.
    # Gate keeps top 50% by importance → indices {3, 4} (importance 3, 4).
    # HKVD over {3, 4} is {2, 1} → top-2: {3, 4} (need 3-1=2 from gate).
    # Result: forced + {3, 4} = {3, 4, 5}.
    assert sorted(idx.tolist()) == [3, 4, 5]


# ──────────────────────────────────────────────────────────────────────────
# _build_attn_mask_with_per_head_valid
# ──────────────────────────────────────────────────────────────────────────


def test_per_head_mask_all_valid_returns_causal_unchanged():
    """When all heads valid at every position, no per-head additive — just
    the causal slice is returned."""
    total_seq = 4
    Q = 2
    causal = torch.zeros((1, 1, total_seq, total_seq), dtype=torch.float32)
    causal.masked_fill_(
        torch.triu(torch.ones(total_seq, total_seq, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    top_idx = torch.tensor([1, 3], dtype=torch.int64)
    H_kv = 2
    valid_layer = torch.ones(H_kv, total_seq, dtype=torch.bool)
    is_top = torch.zeros(total_seq, dtype=torch.bool)
    is_top[top_idx] = True

    mask = _build_attn_mask_with_per_head_valid(
        causal_mask_full=causal, top_indices=top_idx, total_seq=total_seq,
        valid_layer=valid_layer, is_top=is_top, n_rep=1,
        mask_dtype=causal.dtype,
    )
    # All-valid path returns the causal slice unchanged, shape [1, 1, Q, K].
    assert mask.shape == (1, 1, Q, total_seq)
    expected = causal[:, :, top_idx, :total_seq]
    assert torch.equal(mask, expected)


def test_per_head_mask_evicted_position_becomes_neg_inf():
    """One head's evicted position → -inf at (that head, that position)."""
    total_seq = 4
    Q = 1
    causal = torch.zeros((1, 1, total_seq, total_seq), dtype=torch.float32)
    causal.masked_fill_(
        torch.triu(torch.ones(total_seq, total_seq, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    top_idx = torch.tensor([3], dtype=torch.int64)           # Q = last
    H_kv = 2
    valid_layer = torch.ones(H_kv, total_seq, dtype=torch.bool)
    # Head 0 evicted position 1; head 1 keeps everything.
    valid_layer[0, 1] = False
    is_top = torch.zeros(total_seq, dtype=torch.bool)
    is_top[top_idx] = True

    mask = _build_attn_mask_with_per_head_valid(
        causal_mask_full=causal, top_indices=top_idx, total_seq=total_seq,
        valid_layer=valid_layer, is_top=is_top, n_rep=1,
        mask_dtype=causal.dtype,
    )
    # Expected shape: [1, H_q=2, Q=1, K=4] because n_rep=1.
    assert mask.shape == (1, 2, 1, total_seq)
    # Head 0, q=0 (which is position 3), k=1 → -inf (evicted).
    assert mask[0, 0, 0, 1].item() == float("-inf")
    # Head 1, q=0, k=1 → 0 (not evicted; causal allows since k=1 < q=3).
    assert mask[0, 1, 0, 1].item() == 0.0
    # Causal still respected: any head, q=0 (=pos 3), k=anything > 3 → -inf
    # (in this test Q=last so this doesn't trip; left as note).


def test_per_head_mask_top_indices_always_valid_even_if_evicted():
    """Fresh top positions are valid for all heads, overriding eviction."""
    total_seq = 3
    causal = torch.zeros((1, 1, total_seq, total_seq), dtype=torch.float32)
    top_idx = torch.tensor([1], dtype=torch.int64)
    H_kv = 1
    valid_layer = torch.zeros(H_kv, total_seq, dtype=torch.bool)      # all evicted
    is_top = torch.zeros(total_seq, dtype=torch.bool)
    is_top[top_idx] = True

    mask = _build_attn_mask_with_per_head_valid(
        causal_mask_full=causal, top_indices=top_idx, total_seq=total_seq,
        valid_layer=valid_layer, is_top=is_top, n_rep=1,
        mask_dtype=causal.dtype,
    )
    # Q=0 attends to K=1 (its own position, fresh) → 0 (valid).
    assert mask[0, 0, 0, 1].item() == 0.0
    # Q=0 attends to K=0 → -inf (evicted at that head, not in top_indices).
    assert mask[0, 0, 0, 0].item() == float("-inf")


def test_per_head_mask_gqa_expansion():
    """num_heads_q = n_rep * num_heads_kv. Each Q-head inherits its KV-head's mask."""
    total_seq = 3
    causal = torch.zeros((1, 1, total_seq, total_seq), dtype=torch.float32)
    top_idx = torch.tensor([2], dtype=torch.int64)
    H_kv = 2
    valid_layer = torch.ones(H_kv, total_seq, dtype=torch.bool)
    valid_layer[0, 0] = False            # KV head 0 evicted pos 0
    is_top = torch.zeros(total_seq, dtype=torch.bool)
    is_top[top_idx] = True
    n_rep = 3            # 6 Q heads → KV head 0 maps to Q heads {0, 1, 2}

    mask = _build_attn_mask_with_per_head_valid(
        causal_mask_full=causal, top_indices=top_idx, total_seq=total_seq,
        valid_layer=valid_layer, is_top=is_top, n_rep=n_rep,
        mask_dtype=causal.dtype,
    )
    # 6 Q heads. The first 3 (mapped to KV head 0) see pos 0 as -inf.
    # The last 3 (mapped to KV head 1) see pos 0 as 0.
    assert mask.shape == (1, 6, 1, total_seq)
    for h_q in range(3):
        assert mask[0, h_q, 0, 0].item() == float("-inf"), f"Q-head {h_q}"
    for h_q in range(3, 6):
        assert mask[0, h_q, 0, 0].item() == 0.0, f"Q-head {h_q}"


# ──────────────────────────────────────────────────────────────────────────
# CompBlendConfig validation
# ──────────────────────────────────────────────────────────────────────────


def test_config_defaults_are_valid():
    cfg = CompBlendConfig()
    assert cfg.check_layer == 1
    assert cfg.selector == "gated_top_k"
    assert cfg.gate_percentile == 0.5


def test_config_rejects_bad_ratio():
    with pytest.raises(ValueError):
        CompBlendConfig(recompute_ratio=-0.1)
    with pytest.raises(ValueError):
        CompBlendConfig(recompute_ratio=1.5)


def test_config_rejects_bad_percentile():
    with pytest.raises(ValueError):
        CompBlendConfig(gate_percentile=-0.1)
    with pytest.raises(ValueError):
        CompBlendConfig(gate_percentile=1.5)


def test_config_rejects_unknown_selector():
    with pytest.raises(ValueError):
        CompBlendConfig(selector="magic")        # type: ignore[arg-type]


def test_config_rejects_negative_check_layer():
    with pytest.raises(ValueError):
        CompBlendConfig(check_layer=-1)


# ──────────────────────────────────────────────────────────────────────────
# chunk_internal_rank (smoke)
# ──────────────────────────────────────────────────────────────────────────


def test_chunk_internal_rank_in_unit_interval():
    imp = torch.tensor([3., 1., 4., 1., 5., 9.])
    ranks = chunk_internal_rank(imp)
    assert ranks.min().item() == 0.0
    assert ranks.max().item() == 1.0
    # Largest input value (9 at index 5) should map to 1.0.
    assert ranks[5].item() == 1.0
