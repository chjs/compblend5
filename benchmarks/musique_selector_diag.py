"""MuSiQue selector diagnostic — only-HKVD vs Gated HKVD with rich stats.

Parallel to loong_selector_compare.py. MuSiQue's short contexts (3-7K) allow
n=20+ for stronger statistical signal.

Same 4 selectors:
  hkvd_only, no_gate, gated@0.5, gated@0.3

Records (via fuser's selector_stats):
  - F1, EM, latency
  - selected positions list (for cross-arm overlap analysis)
  - selected_from_eligible / selected_from_ineligible counts
  - HKVD / importance distributions at selected vs unselected
"""
from __future__ import annotations

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
sys.path.insert(0, str(_REPO / "benchmarks"))

from compblend.verification import (
    TokenizationInvariantError,
    assert_compressed_storage_matches_tokens,
    assert_token_ids_equal,
)

from musique_f1_kvzip_recompute_grid import (    # type: ignore
    _build_chunks, _compute_f1, _derive_valid_mask_pair,
    _greedy_decode, _load_musique_questions, _make_entry_at_ratio,
    _normalize_answer, _resolve_wrapper, MAX_DOC_TOKENS_FOR_KVZIP,
)

MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
MUSIQUE_PATH = Path(os.environ.get(
    "MUSIQUE_PATH",
    str(_REPO / "src" / "external" / "cacheblend-hf-v7" / "benchmarks" / "musique" / "inputs" / "musique_s.json"),
))
N_MAX = int(os.environ.get("COMPBLEND_N", "20"))
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))
OUT_DIR = Path(os.environ.get("COMPBLEND_OUT_DIR", str(_REPO / "logs" / "musique_selector_diag")))
KVZIP_RATIO = float(os.environ.get("COMPBLEND_KVZIP_RATIO", "0.5"))
RECOMP_RATIO = float(os.environ.get("COMPBLEND_RECOMP_RATIO", "0.10"))
INCLUDE_ASSISTANT_OPEN = os.environ.get("COMPBLEND_INCLUDE_ASSISTANT_OPEN", "1") == "1"
MAX_NEW_TOKENS = int(os.environ.get("COMPBLEND_MAX_NEW_TOKENS", "64"))

ARMS = [
    {"name": "hkvd_only",   "selector": "hkvd_only",    "gate_percentile": None},
    {"name": "no_gate",     "selector": "gated_top_k",  "gate_percentile": 1.0},
    {"name": "gated@0.5",   "selector": "gated_top_k",  "gate_percentile": 0.5},
    {"name": "gated@0.3",   "selector": "gated_top_k",  "gate_percentile": 0.3},
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
    config = {
        "model": MODEL, "dataset": "MuSiQue",
        "musique_path": str(MUSIQUE_PATH),
        "n_max": N_MAX, "check_layer": CHECK_LAYER,
        "kvzip_ratio": KVZIP_RATIO, "recompute_ratio": RECOMP_RATIO,
        "selectors": ARMS, "include_assistant_open": INCLUDE_ASSISTANT_OPEN,
    }
    (OUT_DIR / "config_snapshot.json").write_text(json.dumps(config, indent=2))
    print(f"[diag] MuSiQue kvzip={KVZIP_RATIO} recompute={RECOMP_RATIO}", flush=True)

    questions, skip = _load_musique_questions(MUSIQUE_PATH, N_MAX)
    print(f"[diag] {len(questions)} questions", flush=True)
    if not questions: print("FATAL"); return 1

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
    def _emit(rec): raw_fh.write(json.dumps(rec) + "\n"); raw_fh.flush()

    for q_idx, ex in enumerate(questions):
        try:
            doc_prompts = [f"{c['title']}\n\n{c['text']}\n\n" for c in ex["ctxs"]]
            q = ex["question"].strip()
            query_suffix = f"Question: {q}\nAnswer:"
            if INCLUDE_ASSISTANT_OPEN:
                query_suffix = query_suffix + assistant_open
            chunks, structural_idx, compressible_idx, full_ids = _build_chunks(
                tokenizer, user_open, doc_prompts, query_suffix,
            )
            actual_ids = [t for c in chunks for t in c.token_ids]
            if actual_ids != full_ids:
                print(f"[Q{q_idx+1}] I1 diff={len(actual_ids)-len(full_ids)}", flush=True)

            def _max_f1(text):
                return max((_compute_f1(text, a, tokenizer) for a in ex["answers"]),
                           key=lambda x: x[0])

            kv_base = KVStore()
            for c in chunks:
                K, V = precompute_chunk_kv(lw, c)
                kv_base.put(c.chunk_id, K, V)

            out = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
            text_full = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device)
            f1_full, em_full = _max_f1(text_full)
            del out
            if device.type == "cuda": torch.cuda.empty_cache()

            compressed = {}
            for ci in compressible_idx:
                c = chunks[ci]
                ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
                cmp = backend.compress(ids, model=hf_model,
                                       budget=CompressionBudget(ratio=1.0))
                assert_compressed_storage_matches_tokens(
                    f"q{q_idx+1}_chunk{ci}", cmp, c.token_ids,
                )
                compressed[c.chunk_id] = cmp

            kv_r = KVStore()
            for c in chunks:
                if c.chunk_id in compressed:
                    kv_r._cache[c.chunk_id] = _make_entry_at_ratio(
                        compressed[c.chunk_id], KVZIP_RATIO, n_kv_heads, head_dim,
                    )
                else:
                    kv_r._cache[c.chunk_id] = kv_base.get(c.chunk_id)

            for arm in ARMS:
                t0 = time.perf_counter()
                try:
                    cb_cfg = CompBlendConfig(
                        check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
                        selector=arm["selector"],
                        gate_percentile=arm["gate_percentile"] if arm["gate_percentile"] is not None else 1.0,
                        chunk_normalization="rank",
                    )
                    flags = {}; sel_stats = {}
                    out = fuse_selective_compblend(
                        lw, chunks, kv_r, cb_cfg,
                        return_layerwise_output=True, last_logits_only=True,
                        flags=flags, selector_stats=sel_stats,
                    )
                    text = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device)
                    f1, em = _max_f1(text)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    rec = {
                        "status": "ok",
                        "example_id": ex["id"], "question_idx": q_idx + 1,
                        "selector": arm["name"], "selector_config": arm,
                        "kvzip_ratio": KVZIP_RATIO, "recompute_ratio": RECOMP_RATIO,
                        "prediction": text, "f1": f1, "exact_match": em,
                        "f1_full_prefill": f1_full,
                        "elapsed_ms": elapsed_ms,
                        "selector_stats": sel_stats,
                        "flags": {k: v for k, v in flags.items()
                                  if k in {"per_head_mask_fallback_count",
                                           "per_head_mask_exact_count",
                                           "kvzip_ratio_actual"}},
                    }
                    _emit(rec)
                    sel_ineli = sel_stats.get("selected_from_ineligible_tokens", 0)
                    sel_eli = sel_stats.get("selected_from_eligible_tokens", 0)
                    print(f"[Q{q_idx+1} {arm['name']}] f1={f1:.3f} t={elapsed_ms:.0f}ms picked={sel_eli+sel_ineli} (evicted={sel_ineli})", flush=True)
                    del out
                    if device.type == "cuda": torch.cuda.empty_cache()
                except Exception as exc:
                    print(f"[Q{q_idx+1} {arm['name']} FAIL] {exc}", flush=True)
                    traceback.print_exc()
                    _emit({"status": "fail", "example_id": ex["id"], "selector": arm["name"],
                           "exception_type": type(exc).__name__, "exception_message": str(exc)})
                    if device.type == "cuda": torch.cuda.empty_cache()

        except Exception as exc:
            print(f"[Q{q_idx+1} OUTER FAIL] {exc}", flush=True)
            traceback.print_exc()
            for arm in ARMS:
                _emit({"status": "fail", "example_id": ex.get("id"), "selector": arm["name"],
                       "exception_type": type(exc).__name__, "exception_message": str(exc)})
            if device.type == "cuda": torch.cuda.empty_cache()

    raw_fh.close()
    print("MUSIQUE_DIAG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
