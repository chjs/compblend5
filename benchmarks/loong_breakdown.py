"""TTFT component breakdown — measure where time is spent in CompBlend vs CacheBlend.

For each Loong question, runs two configurations and collects per-section timings
via `fuse_selective_compblend(..., timings=...)`:

  A. CacheBlend baseline:  selector=hkvd_only, NO KVzip compression
                            KVStore from precompute_chunk_kv (full fp16)
  B. CompBlend:             selector=gated_top_k, KVzip r=0.10
                            KVStore entries with valid_mask + importance + eligible

The two configs share the SAME instrumented fuser, so identical timer breakpoints
give apples-to-apples comparison across components.

Output (per-question, aggregated to mean ± stdev):
  io_embed_rotary     — embed_tokens + rotary_emb (T_io part 1)
  kv_load             — copying chunk K/V into the fused-prompt buffer
  full_prefix         — layers 0..check_layer-1 (full forward, both arms identical)
  hkvd_select         — kv_deviation + selector dispatch
  check_layer         — mixed K/V build + per-head mask + SDPA + MLP at check_layer
  sparse_layers       — layers check_layer+1..end (sparse Q × full K)
  lmhead              — final norm + lm_head

Plus T_total (sum) and a separate T_full_recompute (no-blend reference).
"""
from __future__ import annotations

import json
import os
import sys
import time
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
    assert_kvzip_roundtrip_token_ids,
    assert_token_ids_equal,
)

MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DTYPE = os.environ.get("CACHEBLEND_DTYPE", "float16")
ATTN_IMPL = os.environ.get("CACHEBLEND_ATTN_IMPL", "sdpa")
RECOMP_RATIO = float(os.environ.get("COMPBLEND_RECOMP_RATIO", "0.15"))
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))
N_MAX = int(os.environ.get("COMPBLEND_N", "5"))
KVZIP_RATIO = float(os.environ.get("COMPBLEND_RATIO_KVZIP", "0.10"))
GATE_PCT = float(os.environ.get("COMPBLEND_GATE_PCT", "1.0"))   # best Gated config from sweep
LOONG_DATA_DIR = Path(os.environ.get("MODELSCOPE_LOONG_CACHE", "/root/loong_data/doc"))
LOONG_JSONL = Path(os.environ.get("LOONG_JSONL", "/root/Loong/data/loong.jsonl"))
LOONG_LEVELS = [int(x) for x in os.environ.get("LOONG_LEVELS", "1,2").split(",")]
LOONG_SETS = [int(x) for x in os.environ.get("LOONG_SETS", "2").split(",")]
LOONG_MAX_LENGTH = int(os.environ.get("LOONG_MAX_LENGTH", "30000"))
OUT_PATH = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "loong_breakdown.json")))

_WRAPPERS = {
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
}
PREFIX_PROMPT = ""
QUERY_PROMPT = ""


def _resolve_wrapper(model_id, tokenizer):
    mid = model_id.lower()
    for key, wrap in _WRAPPERS.items():
        if key in mid: return wrap
    raise RuntimeError(f"add wrapper for {model_id}")


def _build_chunks(
    tokenizer,
    doc_contents: list[str],
    query_suffix: str,
    user_open: str,
) -> tuple[list, list[int], list[int]]:
    """Loong chunk policy: structural prefix + doc chunks + structural suffix.

    Layout:
        chunks[0]      : structural prefix  = BOS + user_open   (uncompressed)
        chunks[1..N]   : doc chunks         = one per doc       (compressed)
        chunks[N+1]    : structural suffix  = query_suffix      (uncompressed)
    Returns (chunks, structural_idx, compressible_idx).
    """
    from cacheblend.chunker import Chunk, _stable_id
    bos = tokenizer.bos_token_id
    chunks = []

    pref_ids = tokenizer(user_open, add_special_tokens=False)["input_ids"]
    if bos is not None:
        pref_ids = [bos] + pref_ids
    chunks.append(Chunk(text=user_open, token_ids=pref_ids, chunk_id=_stable_id(user_open, pref_ids)))

    for d_text in doc_contents:
        d_ids = tokenizer(d_text, add_special_tokens=False)["input_ids"]
        chunks.append(Chunk(text=d_text, token_ids=d_ids, chunk_id=_stable_id(d_text, d_ids)))

    suf_ids = tokenizer(query_suffix, add_special_tokens=False)["input_ids"]
    chunks.append(Chunk(text=query_suffix, token_ids=suf_ids, chunk_id=_stable_id(query_suffix, suf_ids)))

    structural_idx = [0, len(chunks) - 1]
    compressible_idx = list(range(1, len(chunks) - 1))
    return chunks, structural_idx, compressible_idx


def _measure_full_recompute(lw, chunks, device) -> float:
    """Just total ms — no instrumentation. fuse_full_recompute is unmodified v7."""
    from cacheblend.fusor import fuse_full_recompute
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    fuse_full_recompute(lw, chunks, return_layerwise_output=True)
    if device.type == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


# Loong loader (copied from loong_set1_sweep.py — same logic)
def _load_loong_questions(jsonl_path, doc_path, levels, set_ids, language, max_length, max_n):
    import glob
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            if d["level"] not in levels: continue
            if d["set"] not in set_ids: continue
            if d["language"] != language: continue
            if not isinstance(d["answer"], str): continue
            if d.get("length", 0) > max_length: continue
            rows.append(d)
    print(f"[breakdown] {len(rows)} eligible rows", flush=True)

    materialized = []
    for r in rows:
        contents = []
        ok = True
        doc_type = r["type"]
        doc_level = r["level"]
        type_dir = doc_path / doc_type
        for idx, doc_name in enumerate(r["doc"]):
            if doc_type == "financial":
                if str(doc_level).strip() != "4":
                    matches = glob.glob(f"{type_dir}/*2024-{doc_name}*.txt")
                else:
                    matches = glob.glob(f"{type_dir}/*{doc_name}*.txt")
                if not matches: ok = False; break
                with open(matches[0]) as f:
                    stem = Path(matches[0]).stem.split("-")[-1]
                    contents.append(f"《{stem}》\n" + f.read() + "\n\n")
            elif doc_type == "paper":
                p = type_dir / doc_name
                if not p.exists(): ok = False; break
                with open(p) as f:
                    txt = f.read()
                title = txt.split("\n", 1)[0].strip("#").strip()
                contents.append(f"{title}\n{txt}\n\n")
            elif doc_type == "legal":
                lj = type_dir / "legal.json"
                if not lj.exists(): ok = False; break
                with open(lj) as f:
                    legal = json.load(f)
                if doc_name not in legal: ok = False; break
                e = legal[doc_name]
                contents.append(f"《判决文书{idx+1}》\n" + e["content"] + e.get("result", "") + "\n\n")
            else:
                ok = False; break
        if not ok: continue
        materialized.append({
            "id": r["id"], "question": r["question"],
            "instruction": r["instruction"], "doc_contents": contents,
            "answer": r["answer"], "length": r.get("length", -1),
        })
        if len(materialized) >= max_n: break
    print(f"[breakdown] materialized {len(materialized)}", flush=True)
    return materialized


def main() -> int:
    from cacheblend import LayerwiseModel
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv

    from compblend.backends.base import CompressionBudget, to_kvstore_entry
    from compblend.backends.kvzip import KVzipBackend, KVzipConfig
    from compblend.config import CompBlendConfig
    from compblend.fuse_selective_compblend import fuse_selective_compblend

    print(f"[breakdown] model={MODEL}  n={N_MAX}  kvzip_ratio={KVZIP_RATIO}  "
          f"gate={GATE_PCT}  recomp={RECOMP_RATIO}", flush=True)

    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain", level="pair"))
    hf_model = backend.hf_model
    tokenizer = backend.tokenizer
    device = next(hf_model.parameters()).device

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

    user_open, _assistant_open = _resolve_wrapper(MODEL, tokenizer)

    questions = _load_loong_questions(
        LOONG_JSONL, LOONG_DATA_DIR, LOONG_LEVELS, LOONG_SETS, "en", LOONG_MAX_LENGTH, N_MAX,
    )
    if not questions:
        print("FATAL: no questions", flush=True); return 1

    samples = []
    for idx, ex in enumerate(questions):
      try:
        query_suffix = f"\n\n{ex['instruction']}\n\n{ex['question']}"
        chunks, structural_idx, compressible_idx = _build_chunks(
            tokenizer, ex["doc_contents"], query_suffix, user_open,
        )

        # Invariant I1: full vs concat tokenization
        full_text = user_open + "".join(ex["doc_contents"]) + query_suffix
        expected_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if tokenizer.bos_token_id is not None:
            expected_ids = [tokenizer.bos_token_id] + expected_ids
        actual_ids = [t for c in chunks for t in c.token_ids]
        assert_token_ids_equal(
            f"loong_q{idx + 1}_full_vs_chunks",
            expected_ids, actual_ids, tokenizer,
        )

        # Precompute KV store (no compression — baseline)
        kv_nozip = KVStore()
        for c in chunks:
            K, V = precompute_chunk_kv(lw, c)
            kv_nozip.put(c.chunk_id, K, V)

        # Full prefill reference
        t_full = _measure_full_recompute(lw, chunks, device)

        # CacheBlend baseline (instrumented)
        timings_cb: dict = {}
        flags_cb: dict = {}
        cb_cfg = CompBlendConfig(
            check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
            selector="hkvd_only", chunk_normalization="none",
        )
        fuse_selective_compblend(
            lw, chunks, kv_nozip, cb_cfg,
            return_layerwise_output=True, timings=timings_cb,
            last_logits_only=True,
            flags=flags_cb,
        )
        if device.type == "cuda": torch.cuda.empty_cache()

        # CompBlend with KVzip — compress doc chunks (compressible_idx) only.
        # Structural prefix/suffix stay as plain precomputed fp16 K/V.
        kv_zip = KVStore()
        for i, c in enumerate(chunks):
            if i in structural_idx:
                kv_zip._cache[c.chunk_id] = kv_nozip.get(c.chunk_id)
                continue
            # Doc chunk: verify roundtrip then compress.
            assert_kvzip_roundtrip_token_ids(
                f"loong_q{idx + 1}_chunk{i}", c.token_ids, backend.tokenizer,
            )
            ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
            cmp = backend.compress(ids, model=hf_model, budget=CompressionBudget(ratio=KVZIP_RATIO))
            assert_compressed_storage_matches_tokens(
                f"loong_q{idx + 1}_chunk{i}", cmp, c.token_ids,
            )
            kv_zip._cache[c.chunk_id] = to_kvstore_entry(cmp)

        timings_compblend: dict = {}
        flags_compblend: dict = {}
        compblend_cfg = CompBlendConfig(
            check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
            selector="gated_top_k", gate_percentile=GATE_PCT,
            chunk_normalization="rank",
        )
        fuse_selective_compblend(
            lw, chunks, kv_zip, compblend_cfg,
            return_layerwise_output=True, timings=timings_compblend,
            last_logits_only=True,
            flags=flags_compblend,
        )
        if device.type == "cuda": torch.cuda.empty_cache()

        sample = {
            "id": ex["id"], "context_tokens": int(ex.get("length", -1)),
            "n_docs": len(ex["doc_contents"]),
            "full_recompute_ms": t_full,
            "cacheblend": timings_cb,
            "compblend": timings_compblend,
            "cacheblend_flags": flags_cb,
            "compblend_flags": flags_compblend,
        }
        samples.append(sample)
        # Compact log
        keys = ["io_embed_rotary", "kv_load", "full_prefix", "hkvd_select", "check_layer", "sparse_layers", "lmhead"]
        cb_parts = "/".join(f"{timings_cb.get(k, 0):.0f}" for k in keys)
        cp_parts = "/".join(f"{timings_compblend.get(k, 0):.0f}" for k in keys)
        print(f"[{idx+1}/{len(questions)}] full={t_full:.0f}  "
              f"CB(t={timings_cb.get('_total_ms', 0):.0f}: {cb_parts})  "
              f"CP(t={timings_compblend.get('_total_ms', 0):.0f}: {cp_parts})",
              flush=True)
      except Exception as exc:
        import traceback as _tb
        print(f"[Q{idx+1} FAILED] {type(exc).__name__}: {exc}", flush=True)
        _tb.print_exc()
        if device.type == "cuda": torch.cuda.empty_cache()
        continue

    # Aggregate
    def _stat(xs):
        if not xs: return {"mean": 0.0, "stdev": 0.0, "n": 0}
        arr = np.array(xs)
        return {"mean": float(arr.mean()), "stdev": float(arr.std()), "n": len(xs)}

    keys = ["io_embed_rotary", "kv_load", "full_prefix", "hkvd_select", "check_layer", "sparse_layers", "lmhead"]
    # Aggregate reliability flags across all samples.
    def _flag_count(samples_list, side: str, key: str) -> int:
        return sum(
            1 for s in samples_list
            if s.get(f"{side}_flags", {}).get(key, 0) > 0
        )

    cb_fallback_n = _flag_count(samples, "cacheblend", "per_head_mask_fallback_count")
    cp_fallback_n = _flag_count(samples, "compblend",  "per_head_mask_fallback_count")
    cp_exact_n    = _flag_count(samples, "compblend",  "per_head_mask_exact_count")
    cp_workspace_bytes = [
        s.get("compblend_flags", {}).get("dense_workspace_bytes", 0) for s in samples
    ]

    gate_label = "no_gate" if (GATE_PCT >= 1.0 or GATE_PCT <= 0.0) else f"gate={GATE_PCT}"

    summary = {
        "config": {
            "model": MODEL, "n": len(samples),
            "kvzip_ratio": KVZIP_RATIO, "recomp_ratio": RECOMP_RATIO,
            "gate_pct": GATE_PCT,
            "gate_label": gate_label,
            "gate_is_active": (gate_label != "no_gate"),
            "check_layer": CHECK_LAYER,
            "loong_levels": LOONG_LEVELS,
            "loong_sets": LOONG_SETS,
            "loong_max_length": LOONG_MAX_LENGTH,
            "dataset": f"Loong levels={LOONG_LEVELS} sets={LOONG_SETS} en len<={LOONG_MAX_LENGTH}",
            "stage": "Stage 1",
            "stage_notes": "Dense [1, total_seq, H_kv*D] workspace per layer; compressed-native KV is Stage 2.",
            "chunk_policy": "[BOS+user_open](structural) + [docs](compressed) + [query](structural)",
        },
        "full_recompute_ms": _stat([s["full_recompute_ms"] for s in samples]),
        "cacheblend_ms": {k: _stat([s["cacheblend"].get(k, 0) for s in samples]) for k in keys},
        "compblend_ms":  {k: _stat([s["compblend"].get(k, 0) for s in samples]) for k in keys},
        "cacheblend_total_ms": _stat([s["cacheblend"].get("_total_ms", 0) for s in samples]),
        "compblend_total_ms":  _stat([s["compblend"].get("_total_ms", 0) for s in samples]),
        "reliability_flags": {
            "cacheblend_fallback_questions": cb_fallback_n,
            "compblend_fallback_questions": cp_fallback_n,
            "compblend_exact_mask_questions": cp_exact_n,
            "compblend_dense_workspace_bytes_max": int(max(cp_workspace_bytes) if cp_workspace_bytes else 0),
            "compblend_dense_workspace_bytes_mean": float(
                sum(cp_workspace_bytes) / len(cp_workspace_bytes)
            ) if cp_workspace_bytes else 0.0,
            "note": (
                "fallback_questions = number of Qs where the per-head SDPA mask hit "
                "ATTN_MASK_MEMORY_CAP_BYTES and was downgraded to causal-only at one "
                "or more layers. If fallback_questions > 0, the timing/F1 numbers "
                "reflect causal-only attention with zero-filled K/V, NOT exact "
                "per-head valid_mask enforcement."
            ),
        },
        "per_sample": samples,
    }
    print("\n──────── BREAKDOWN SUMMARY ────────")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, indent=2))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[breakdown] wrote {OUT_PATH}")
    print("BREAKDOWN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
