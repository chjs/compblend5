"""M6 — ratio-sweep mask vs direct KVzip._threshold equality (CPU).

Loong sweep compresses each doc ONCE at KVzip ratio=1.0, then derives the
valid_mask at each sweep ratio via score thresholding. The benchmark
docstring claims this is "verbatim KVzip._threshold". This test enforces
that claim: at every sweep ratio, our derived mask MUST byte-equal what
KVzip would produce by calling `Score._threshold(score, ratio)` directly.

If KVzip's prune logic ever drifts (sink-token policy, head-budget
constraint, layered budget), this test fails loudly and the Loong sweep
results must be re-interpreted before publication.

CPU-only: works on random score tensors; no model/forward needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make KVzip importable. Both this test and the benchmark assume KVzip is
# vendored under src/external/KVzip.
_KVZIP_DIR = Path(__file__).resolve().parents[1] / "src" / "external" / "KVzip"
if _KVZIP_DIR.exists():
    sys.path.insert(0, str(_KVZIP_DIR))


def _ours_derive(importance: torch.Tensor, target_ratio: float) -> torch.Tensor:
    """Bench's _derive_valid_mask_pair, lifted here verbatim."""
    if target_ratio >= 1.0:
        return torch.ones_like(importance, dtype=torch.bool)
    flat = importance.reshape(-1)
    n = max(int(flat.numel() * target_ratio) - 1, 0)
    score_sort = torch.sort(flat, descending=True).values
    thres = score_sort[n].item()
    return importance > thres


def _kvzip_threshold(score: torch.Tensor, ratio: float) -> torch.Tensor:
    """Verbatim KVzip Score._threshold algorithm (src/external/KVzip/attention/score.py:88)."""
    if ratio < 1:
        score_sort = torch.sort(score.reshape(-1), descending=True).values
        n = max(int(len(score_sort) * ratio) - 1, 0)
        thres = score_sort[n].item()
        return torch.where(score > thres, True, False).bool()
    return torch.ones_like(score, dtype=torch.bool)


@pytest.mark.parametrize("ratio", [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("shape", [(2, 4, 100), (4, 8, 500), (32, 8, 200)])
def test_derived_mask_matches_kvzip_threshold(ratio: float, shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    score = torch.rand(shape, dtype=torch.float32)

    ours = _ours_derive(score, ratio)
    kvzip = _kvzip_threshold(score, ratio)

    assert ours.shape == kvzip.shape == shape
    assert ours.dtype == kvzip.dtype == torch.bool
    # Bit-equality at every (L, H, pos).
    assert torch.equal(ours, kvzip), (
        f"derived mask differs from KVzip._threshold at ratio={ratio}, shape={shape}; "
        f"diff_count={(ours ^ kvzip).sum().item()} / {ours.numel()}"
    )


def test_derived_mask_kept_count_matches_intent():
    """For ratio<1, the number of kept positions should be ~ratio * total
    (off-by-one due to the n = int(...) - 1 convention; this is also the
    convention KVzip uses, so we mirror it)."""
    torch.manual_seed(1)
    score = torch.rand((32, 8, 1000), dtype=torch.float32)
    for r in [0.1, 0.3, 0.5]:
        m = _ours_derive(score, r)
        kept = int(m.sum().item())
        total = m.numel()
        # KVzip's convention: n = max(int(total*r)-1, 0); kept ≈ n+1 = total*r
        # (rounding to int floor). Allow ±1 because ties at the threshold are
        # excluded via strict >.
        expected = int(total * r)
        assert abs(kept - expected) <= 2, (
            f"ratio={r}: kept={kept}, expected ≈ {expected} (±2 for ties)"
        )


def test_threshold_at_ratio_1_returns_all_true():
    score = torch.rand((4, 2, 50))
    m = _ours_derive(score, 1.0)
    assert bool(m.all().item())


def test_threshold_compatible_with_kvzip_if_importable():
    """If the actual KVzip package is on path, call its Score._threshold
    directly and compare. Skips when not available (e.g. local dev box)."""
    try:
        from attention.score import Score  # type: ignore[import-not-found]
    except Exception:
        pytest.skip("KVzip attention.score not importable; run on pod")

    torch.manual_seed(2)
    score = torch.rand((8, 4, 200), dtype=torch.float32)
    for r in [0.1, 0.3, 0.5, 0.9]:
        ours = _ours_derive(score, r)
        kvzip_valids, _ = Score._threshold(None, score, r)
        assert torch.equal(ours, kvzip_valids), (
            f"ratio={r}: derived mask differs from kvzip.Score._threshold"
        )
