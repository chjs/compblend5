"""CompBlend runtime configuration — slimmed to Stage-1 fields only.

Compared with CompBlend-old's CompBlendConfig, this drops:
    * `attention_kernel` / `varlen_threshold` — v7 fusor only uses SDPA;
      varlen support is Stage-2 work.
    * `check_layers` tuple + `recompute_ratios` cascade — Stage 1 uses
      a single check_layer. Cascade is Stage 2.
    * `hkvd_compute_dtype` — v7's `kv_deviation` already upcasts to fp32.
    * `boundary_force_count` — EPIC AttnLink is a research extension, not in
      paper §3. Re-add only when needed.
    * `alpha` — additive_rank selector deferred; not currently a default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SelectorKind = Literal[
    "hkvd_only",        # v7 baseline — top-k by deviation only
    "importance_only",  # ablation — top-k by importance only
    "gated_top_k",      # paper §3 default — importance gate then HKVD top-k
]


ChunkNormalization = Literal[
    "none",     # raw per-chunk importance, concatenated as-is (subject to
                # length bias for backends like SnapKV — see CompBlend-old's
                # 4a-1 finding of ~14× scale ratio between short/long docs).
    "rank",     # per-chunk percentile rank in [0, 1]. Length-fair across
                # chunks. Recommended when blending chunks of unequal length.
]


@dataclass
class CompBlendConfig:
    """Configuration consumed by `fuse_selective_compblend`."""

    # Layer at which HKVD selection runs. v7 default = 1 (one full forward
    # pass at layer 0; selection at layer 1; sparse from layer 1 onward).
    check_layer: int = 1

    # Fraction of fused-prompt tokens to recompute. 0 → fall back to
    # full_reuse, 1 → fall back to full_recompute (v7 boundary shortcuts).
    recompute_ratio: float = 0.15

    # Token selector. Default is paper §3 Gated HKVD.
    selector: SelectorKind = "gated_top_k"

    # For gated_top_k: importance percentile threshold over non-forced,
    # non-structural candidates. Keeps the top `gate_percentile` fraction.
    # 1.0 → no gating; 0.0 → no gating (everything passes); typical 0.5.
    gate_percentile: float = 0.5

    # Exempt structural (window / sink) tokens from selector pressure.
    # When True and a backend marks `is_structural[p] = True`, that position
    # is never returned as a selected token. Its cached K/V stays in use.
    exempt_structural: bool = True

    # How per-chunk importance vectors are combined into the fused-prompt
    # `importance_scores`. "none" concatenates raw; "rank" applies within-
    # chunk percentile rank before concat (length-fair).
    chunk_normalization: ChunkNormalization = "none"

    def __post_init__(self) -> None:
        if not (0.0 <= self.recompute_ratio <= 1.0):
            raise ValueError(
                f"recompute_ratio must be in [0, 1], got {self.recompute_ratio}"
            )
        if not (0.0 <= self.gate_percentile <= 1.0):
            raise ValueError(
                f"gate_percentile must be in [0, 1], got {self.gate_percentile}"
            )
        if self.check_layer < 0:
            raise ValueError(f"check_layer must be >= 0, got {self.check_layer}")
        if self.selector not in ("hkvd_only", "importance_only", "gated_top_k"):
            raise ValueError(f"unknown selector: {self.selector!r}")
        if self.chunk_normalization not in ("none", "rank"):
            raise ValueError(
                f"unknown chunk_normalization: {self.chunk_normalization!r}"
            )
