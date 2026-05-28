"""Selector comparison: only-HKVD vs Gated HKVD at a fixed (KVzip, recompute) cell.

Goal: isolate the selector's effect by holding KVzip ratio and recompute ratio
fixed and varying only the selector.

Selectors compared (paper §3):
    1. hkvd_only        — naive HKVD top-k over the ENTIRE sequence. May pick
                          evicted positions; their HKVD score is wildly
                          inflated (‖K_fresh‖² since K_cached=0 at evicted).
                          Baseline showing why eligibility matters.
    2. gated_top_k gate=1.0  — "no_gate" mode: importance gate disabled,
                          but eligible_mask still restricts to retained
                          positions (union over heads). This is the
                          compressed-aware HKVD baseline.
    3. gated_top_k gate=0.5  — Gated HKVD with medium importance gate.
                          Picks HKVD top-k within the top 50% importance
                          candidates.
    4. gated_top_k gate=0.3  — Stronger gate (top 30% importance only).

Output:
    logs/loong_selector_compare/
        raw_results.jsonl    — one line per (example, selector)
        summary.csv          — selector × {n_ok, mean_f1, std_f1, median_f1, ...}
        summary.json         — config + per-selector aggregates + selector stats
        bar_f1.png/pdf       — bar plot of F1 per selector
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "src" / "external" / "cacheblend-hf-v7" / "src"))

from compblend.verification import (
    TokenizationInvariantError,
    assert_compressed_storage_matches_tokens,
    assert_token_ids_equal,
)


# Reuse helpers from loong_f1_kvzip_recompute_grid.py (avoid copy-paste rot).
sys.path.insert(0, str(_REPO / "benchmarks"))
from loong_f1_kvzip_recompute_grid import (   # type: ignore
    _build_chunks, _compute_f1, _derive_valid_mask_pair,
    _greedy_decode, _load_loong_questions, _make_entry_at_ratio,
    _normalize_answer, _resolve_wrapper, MAX_DOC_TOKENS_FOR_KVZIP,
)


# ─── env config ──────────────────────────────────────────────────────────


MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LOONG_JSONL = Path(os.environ.get("LOONG_JSONL", "/root/Loong/data/loong.jsonl"))
LOONG_DATA_DIR = Path(os.environ.get("MODELSCOPE_LOONG_CACHE", "/root/loong_data/doc"))
LOONG_LEVELS = [int(x) for x in os.environ.get("LOONG_LEVELS", "1,2").split(",") if x.strip()]
LOONG_SETS = [int(x) for x in os.environ.get("LOONG_SETS", "1,2").split(",") if x.strip()]
LOONG_LANGUAGES = [x.strip() for x in os.environ.get("LOONG_LANGUAGES", "en").split(",") if x.strip()]
LOONG_MIN_LENGTH = int(os.environ.get("LOONG_MIN_LENGTH", "0"))
LOONG_MAX_LENGTH = int(os.environ.get("LOONG_MAX_LENGTH", "66000"))
N_MAX = int(os.environ.get("COMPBLEND_N", "10"))
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))
OUT_DIR = Path(os.environ.get("COMPBLEND_OUT_DIR", str(_REPO / "logs" / "loong_selector_compare")))
KVZIP_RATIO = float(os.environ.get("COMPBLEND_KVZIP_RATIO", "0.5"))
RECOMP_RATIO = float(os.environ.get("COMPBLEND_RECOMP_RATIO", "0.10"))
INCLUDE_ASSISTANT_OPEN = os.environ.get("COMPBLEND_INCLUDE_ASSISTANT_OPEN", "1") == "1"
MAX_NEW_TOKENS = int(os.environ.get("COMPBLEND_MAX_NEW_TOKENS", "64"))


# Selector arms — keyed by display name.
ARMS = [
    {"name": "hkvd_only",      "selector": "hkvd_only",    "gate_percentile": None},
    {"name": "no_gate",        "selector": "gated_top_k",  "gate_percentile": 1.0},
    {"name": "gated@0.5",      "selector": "gated_top_k",  "gate_percentile": 0.5},
    {"name": "gated@0.3",      "selector": "gated_top_k",  "gate_percentile": 0.3},
]


def main() -> int:
    from cacheblend import LayerwiseModel
    from cacheblend.fusor import fuse_full_recompute
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    from compblend.backends.base import CompressionBudget
    from compblend.backends.kvzip import KVzipBackend, KVzipConfig
    from compblend.config import CompBlendConfig
    from compblend.fuse_selective_compblend import fuse_selective_compblend

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "raw_results.jsonl"
    summary_csv_path = OUT_DIR / "summary.csv"
    summary_json_path = OUT_DIR / "summary.json"

    config = {
        "model": MODEL, "dataset": "Loong",
        "levels": LOONG_LEVELS, "sets": LOONG_SETS, "languages": LOONG_LANGUAGES,
        "max_length": LOONG_MAX_LENGTH, "n_max": N_MAX,
        "check_layer": CHECK_LAYER,
        "kvzip_ratio": KVZIP_RATIO,
        "recompute_ratio": RECOMP_RATIO,
        "selectors": ARMS,
        "include_assistant_open": INCLUDE_ASSISTANT_OPEN,
        "metric_note": "token_f1 substring (not Loong official LLM-judge)",
        "note": "Only KVzip ratio + recompute ratio held fixed. selector varies.",
    }
    (OUT_DIR / "config_snapshot.json").write_text(json.dumps(config, indent=2))
    print(f"[selector-compare] kvzip={KVZIP_RATIO} recompute={RECOMP_RATIO} arms={[a['name'] for a in ARMS]}", flush=True)

    questions, skip = _load_loong_questions(
        LOONG_JSONL, LOONG_DATA_DIR,
        LOONG_LEVELS, LOONG_SETS, LOONG_LANGUAGES,
        LOONG_MIN_LENGTH, LOONG_MAX_LENGTH, N_MAX,
    )
    print(f"[selector-compare] materialized {len(questions)} questions; skip={dict(skip)}", flush=True)
    if not questions:
        print("FATAL: no questions"); return 1

    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain", level="pair"))
    hf_model = backend.hf_model
    tokenizer = backend.tokenizer
    device = next(hf_model.parameters()).device
    cfg = hf_model.config
    n_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    lw = LayerwiseModel.__new__(LayerwiseModel)
    lw.model = hf_model
    lw.tokenizer = tokenizer
    lw.device = device
    lw.dtype = next(hf_model.parameters()).dtype
    lw._inner = hf_model.model
    lw.num_layers = len(lw._inner.layers)
    lw._pre_rope_k = {}
    lw._hook_handles = []
    lw._install_k_proj_hooks()

    user_open, assistant_open = _resolve_wrapper(MODEL)

    raw_fh = open(raw_path, "w", encoding="utf-8")

    def _emit(rec):
        raw_fh.write(json.dumps(rec) + "\n"); raw_fh.flush()

    for q_idx, ex in enumerate(questions):
        try:
            query_suffix = f"\n\n{ex['instruction']}\n\n{ex['question']}"
            if INCLUDE_ASSISTANT_OPEN:
                query_suffix = query_suffix + assistant_open
            chunks, structural_idx, compressible_idx, full_ids = _build_chunks(
                tokenizer, user_open, ex["doc_contents"], query_suffix,
            )

            actual_ids = [t for c in chunks for t in c.token_ids]
            if actual_ids != full_ids:
                # I1 warning only, not fatal.
                print(f"[Q{q_idx+1}] I1 boundary diff={len(actual_ids)-len(full_ids)}", flush=True)

            kv_base = KVStore()
            for c in chunks:
                K, V = precompute_chunk_kv(lw, c)
                kv_base.put(c.chunk_id, K, V)

            # Reference: full prefill F1
            out = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
            text_full = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device)
            f1_full, em_full = max(
                (_compute_f1(text_full, a, tokenizer) for a in ex["answers"]),
                key=lambda x: x[0],
            ) if "answers" in ex else (
                (_compute_f1(text_full, ex["answer"], tokenizer)
                 if isinstance(ex["answer"], str) else (0.0, 0)),
            )
            del out
            if device.type == "cuda": torch.cuda.empty_cache()

            # KVzip compress doc chunks once at the chosen ratio.
            compressed = {}
            skip_q = False
            for ci in compressible_idx:
                c = chunks[ci]
                ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
                try:
                    cmp = backend.compress(ids, model=hf_model,
                                           budget=CompressionBudget(ratio=1.0))
                    assert_compressed_storage_matches_tokens(
                        f"q{q_idx+1}_chunk{ci}", cmp, c.token_ids,
                    )
                except Exception as exc:
                    print(f"[Q{q_idx+1} compress fail] {type(exc).__name__}: {exc}", flush=True)
                    skip_q = True
                    break
                compressed[c.chunk_id] = cmp

            if skip_q:
                for arm in ARMS:
                    _emit({"status": "skip", "skip_stage": "kvzip_compress",
                           "example_id": ex["id"], "selector": arm["name"]})
                if device.type == "cuda": torch.cuda.empty_cache()
                continue

            # Build KVStore at the target KVzip ratio.
            kv_r = KVStore()
            for c in chunks:
                if c.chunk_id in compressed:
                    kv_r._cache[c.chunk_id] = _make_entry_at_ratio(
                        compressed[c.chunk_id], KVZIP_RATIO, n_kv_heads, head_dim,
                    )
                else:
                    kv_r._cache[c.chunk_id] = kv_base.get(c.chunk_id)

            # Per-arm: same KVStore, vary only the selector.
            for arm in ARMS:
                t0 = time.perf_counter()
                try:
                    cb_cfg = CompBlendConfig(
                        check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
                        selector=arm["selector"],
                        gate_percentile=arm["gate_percentile"] if arm["gate_percentile"] is not None else 1.0,
                        chunk_normalization="rank",
                    )
                    flags: dict = {}; sel_stats: dict = {}
                    out = fuse_selective_compblend(
                        lw, chunks, kv_r, cb_cfg,
                        return_layerwise_output=True, last_logits_only=True,
                        flags=flags, selector_stats=sel_stats,
                    )
                    text = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device)
                    if "answers" in ex:
                        f1, em = max((_compute_f1(text, a, tokenizer) for a in ex["answers"]),
                                     key=lambda x: x[0])
                    else:
                        f1, em = (_compute_f1(text, ex["answer"], tokenizer)
                                  if isinstance(ex["answer"], str) else (0.0, 0))
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    rec = {
                        "status": "ok",
                        "example_id": ex["id"], "question_idx": q_idx + 1,
                        "selector": arm["name"],
                        "selector_config": arm,
                        "kvzip_ratio": KVZIP_RATIO,
                        "recompute_ratio": RECOMP_RATIO,
                        "prediction": text,
                        "f1": f1, "exact_match": em,
                        "f1_full_prefill": f1_full,
                        "elapsed_ms": elapsed_ms,
                        "selector_stats": sel_stats,
                        "flags": {
                            k: v for k, v in flags.items()
                            if k in {"per_head_mask_fallback_count",
                                     "per_head_mask_exact_count",
                                     "kvzip_ratio_actual",
                                     "stored_kv_valid_ratio",
                                     "gate_is_active", "gate_percentile"}
                        },
                    }
                    _emit(rec)
                    print(f"[Q{q_idx+1} {arm['name']}] f1={f1:.3f} em={em} t={elapsed_ms:.0f}ms",
                          flush=True)
                    del out
                    if device.type == "cuda": torch.cuda.empty_cache()
                except Exception as exc:
                    print(f"[Q{q_idx+1} {arm['name']} FAIL] {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    _emit({"status": "fail",
                           "example_id": ex["id"], "selector": arm["name"],
                           "exception_type": type(exc).__name__,
                           "exception_message": str(exc)})
                    if device.type == "cuda": torch.cuda.empty_cache()

        except Exception as exc:
            print(f"[Q{q_idx+1} OUTER FAIL] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            for arm in ARMS:
                _emit({"status": "fail",
                       "example_id": ex.get("id"), "selector": arm["name"],
                       "exception_type": type(exc).__name__,
                       "exception_message": str(exc)})
            if device.type == "cuda": torch.cuda.empty_cache()

    raw_fh.close()

    # Aggregate.
    by_arm = {arm["name"]: [] for arm in ARMS}
    rows_all = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            rows_all.append(r)
            if r.get("status") == "ok":
                by_arm[r["selector"]].append(r)

    with open(summary_csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "selector", "n_ok", "n_fail",
            "mean_f1", "std_f1", "median_f1", "p10_f1", "p90_f1",
            "mean_em", "mean_elapsed_ms",
            "mean_selected_ratio_actual", "mean_eligible_ratio",
            "fallback_q",
        ])
        for arm in ARMS:
            recs = by_arm[arm["name"]]
            fails = sum(1 for r in rows_all if r.get("selector") == arm["name"] and r.get("status") != "ok")
            if not recs:
                wr.writerow([arm["name"], 0, fails] + [""] * 9); continue
            f1s = np.array([r["f1"] for r in recs])
            ems = np.array([r["exact_match"] for r in recs])
            ts = np.array([r["elapsed_ms"] for r in recs])
            sel_ratios = np.array([r["selector_stats"].get("selected_ratio_actual", -1) for r in recs])
            eli_ratios = np.array([r["selector_stats"].get("eligible_ratio", -1) for r in recs])
            fb = sum(1 for r in recs if r["flags"].get("per_head_mask_fallback_count", 0) > 0)
            wr.writerow([
                arm["name"], len(recs), fails,
                f"{f1s.mean():.4f}", f"{f1s.std():.4f}",
                f"{np.median(f1s):.4f}",
                f"{np.quantile(f1s, 0.10):.4f}", f"{np.quantile(f1s, 0.90):.4f}",
                f"{ems.mean():.4f}", f"{ts.mean():.1f}",
                f"{sel_ratios.mean():.4f}", f"{eli_ratios.mean():.4f}",
                fb,
            ])
    print(f"[selector-compare] wrote {summary_csv_path}", flush=True)

    per_arm = {}
    for arm in ARMS:
        recs = by_arm[arm["name"]]
        if not recs: per_arm[arm["name"]] = {"n": 0}; continue
        f1s = np.array([r["f1"] for r in recs])
        per_arm[arm["name"]] = {
            "n": len(recs),
            "mean_f1": float(f1s.mean()),
            "std_f1": float(f1s.std()),
            "median_f1": float(np.median(f1s)),
            "mean_elapsed_ms": float(np.mean([r["elapsed_ms"] for r in recs])),
            "selector_stats_mean": {
                k: float(np.mean([r["selector_stats"].get(k, 0) for r in recs]))
                for k in ["total_tokens", "eligible_tokens", "selected_recompute_tokens",
                          "selected_ratio_actual", "eligible_ratio",
                          "selected_from_structural_tokens",
                          "selected_from_compressible_tokens"]
            },
        }
    summary = {"config": config, "per_arm": per_arm}
    summary_json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[selector-compare] wrote {summary_json_path}", flush=True)

    # Bar plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[selector-compare] matplotlib missing; skipping plot"); print("SELECTOR_COMPARE_DONE", flush=True); return 0

    arm_names = [a["name"] for a in ARMS]
    means = [per_arm[n].get("mean_f1", 0) for n in arm_names]
    stds = [per_arm[n].get("std_f1", 0) for n in arm_names]
    ns = [per_arm[n].get("n", 0) for n in arm_names]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#888", "#1a5cff", "#e67e22", "#c0392b"]
    xs = np.arange(len(arm_names))
    bars = ax.bar(xs, means, yerr=stds, color=colors, alpha=0.85, capsize=5, edgecolor="black")
    for x, m, s, n in zip(xs, means, stds, ns):
        ax.text(x, m + s + 0.02, f"{m:.3f}\nn={n}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(arm_names, fontsize=11)
    ax.set_ylabel("Mean token F1", fontsize=12)
    ax.set_ylim(0, max(max(means) + max(stds) + 0.15, 0.5))
    ax.set_title(
        f"Loong: Selector comparison\n"
        f"KVzip={KVZIP_RATIO}, recompute={RECOMP_RATIO}, n={ns[0] if ns else 0}",
        fontsize=13,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"bar_f1_per_selector.{ext}", dpi=130)
    plt.close(fig)
    print(f"[selector-compare] wrote bar_f1_per_selector.png")

    print("SELECTOR_COMPARE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
