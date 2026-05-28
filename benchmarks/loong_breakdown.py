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

def _floatlist(env: str, default: str) -> list[float]:
    raw = os.environ.get(env, default)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DTYPE = os.environ.get("CACHEBLEND_DTYPE", "float16")
ATTN_IMPL = os.environ.get("CACHEBLEND_ATTN_IMPL", "sdpa")
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))
N_MAX = int(os.environ.get("COMPBLEND_N", "5"))
LOONG_DATA_DIR = Path(os.environ.get("MODELSCOPE_LOONG_CACHE", "/root/loong_data/doc"))
LOONG_JSONL = Path(os.environ.get("LOONG_JSONL", "/root/Loong/data/loong.jsonl"))
LOONG_LEVELS = [int(x) for x in os.environ.get("LOONG_LEVELS", "1,2").split(",")]
LOONG_SETS = [int(x) for x in os.environ.get("LOONG_SETS", "2").split(",")]
LOONG_MAX_LENGTH = int(os.environ.get("LOONG_MAX_LENGTH", "30000"))
OUT_PATH = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "loong_breakdown.json")))

# Matrix-mode env lists. CompBlend arms are the cross product of:
#   recompute_ratio ∈ COMPBLEND_RECOMP_RATIO_LIST
#   gate_percentile ∈ COMPBLEND_GATE_PCT_LIST
#   KVzip ratio      ∈ COMPBLEND_RATIO_KVZIP_LIST
# Plus a "dense CacheBlend" baseline (no KVzip) at each recompute_ratio.
RECOMP_RATIO_LIST = _floatlist("COMPBLEND_RECOMP_RATIO_LIST", os.environ.get("COMPBLEND_RECOMP_RATIO", "0.15"))
GATE_PCT_LIST     = _floatlist("COMPBLEND_GATE_PCT_LIST",    os.environ.get("COMPBLEND_GATE_PCT", "1.0"))
RATIO_KVZIP_LIST  = _floatlist("COMPBLEND_RATIO_KVZIP_LIST", os.environ.get("COMPBLEND_RATIO_KVZIP", "0.10"))

# Optional per-kind recompute lists. If unset, both kinds use RECOMP_RATIO_LIST.
# Use these when CacheBlend and CompBlend are intended to run at DIFFERENT
# recompute ratios in the same benchmark run (e.g., paper's "same F1 at lower
# recompute" comparison: CB @ 0.15 vs CompBlend @ 0.10).
_DENSE_OVERRIDE = os.environ.get("COMPBLEND_DENSE_RECOMP_LIST")
_COMP_OVERRIDE  = os.environ.get("COMPBLEND_COMPRESSED_RECOMP_LIST")
DENSE_RECOMP_LIST = _floatlist("COMPBLEND_DENSE_RECOMP_LIST", ",".join(str(x) for x in RECOMP_RATIO_LIST)) if _DENSE_OVERRIDE else RECOMP_RATIO_LIST
COMP_RECOMP_LIST  = _floatlist("COMPBLEND_COMPRESSED_RECOMP_LIST", ",".join(str(x) for x in RECOMP_RATIO_LIST)) if _COMP_OVERRIDE else RECOMP_RATIO_LIST

# assistant_open inclusion — see §5 of the audit. Default True so the prompt
# ends in the assistant header (which is what reuse-time serving would do).
INCLUDE_ASSISTANT_OPEN = os.environ.get("COMPBLEND_INCLUDE_ASSISTANT_OPEN", "1") == "1"

# Whether to run dense CacheBlend baseline alongside CompBlend.
RUN_DENSE_BASELINE = os.environ.get("COMPBLEND_DENSE_BASELINE", "1") == "1"

# Whether to include layer breakdown lists (verbose; one entry per layer).
EMIT_PER_LAYER_DETAIL = os.environ.get("COMPBLEND_EMIT_PER_LAYER", "0") == "1"

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

    Tokenization policy:
      Each chunk is tokenized INDEPENDENTLY (add_special_tokens=False), BOS
      prepended to chunks[0] explicitly. This preserves KVzip's
      decode→encode roundtrip identity for each chunk individually — the
      cached K/V actually corresponds to the chunk's token sequence.

      Trade-off: concat(c.token_ids) may diverge from tokenize(full_prompt)
      at BPE-merge boundaries (e.g. "\\n\\n"+"\\n\\n" → 1 token in full,
      2 tokens in concat). The caller's I1 invariant flags such questions
      so they can be skipped rather than producing apples-to-oranges
      timing comparisons.
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


def _sync_perf(device) -> float:
    """torch.cuda.synchronize() if cuda, then perf_counter()."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def _measure_full_recompute(lw, chunks, device) -> dict:
    """Measure full_recompute wall time + crude breakdown.

    Returns dict with at least 'wall_total_ms'. v7's fuse_full_recompute is
    unmodified — we wrap it for wall timing. For a coarse component split we
    additionally time embed/lm_head segments around the model.forward call.
    """
    from cacheblend.fusor import fuse_full_recompute
    t0 = _sync_perf(device)
    out = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
    t1 = _sync_perf(device)
    wall_total_ms = (t1 - t0) * 1000.0
    return {
        "wall_total_ms": wall_total_ms,
        # full_recompute uses the standard HF forward — we don't split it
        # further to avoid touching v7 code. The summary explicitly notes
        # the measurement-boundary asymmetry.
        "instrumented_total_ms": None,
        "breakdown_available": False,
        "note": "Wall time only. fuse_full_recompute is unmodified v7; for component-level breakdown see CacheBlend/CompBlend arms.",
        "_out": out,
    }


def _stats(xs: list[float]) -> dict:
    """mean / stdev / median / p90 / p99 / n for a list of measurements."""
    if not xs:
        return {"mean": 0.0, "stdev": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "n": 0}
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "stdev": float(arr.std()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "p99": float(np.quantile(arr, 0.99)),
        "n": int(arr.size),
    }


def _percent_breakdown(per_sample_timings: list[dict], wall_means: dict) -> dict:
    """For each timing key, return mean(timing) / mean(wall_total_ms) as %."""
    if not per_sample_timings:
        return {}
    wall_mean = wall_means.get("mean", 0.0) or 1.0  # avoid div0
    keys = set()
    for s in per_sample_timings:
        keys.update(s.keys())
    out = {}
    for k in sorted(keys):
        if k.startswith("_") or k in {"wall_total_ms", "instrumented_total_ms", "unaccounted_ms"}:
            continue
        vals = [float(s.get(k, 0.0)) for s in per_sample_timings]
        out[k] = round(100.0 * float(np.mean(vals)) / wall_mean, 2)
    return out


def _call_arm_with_wall_timing(fuser_callable, device, *fuser_args, **fuser_kwargs) -> tuple:
    """Wrap a fuser call to capture wall_total_ms + unaccounted_ms.

    Returns (out, wall_total_ms). The fuser fills its passed `timings` dict
    with instrumented breakdown; we add wall_total + unaccounted here.
    """
    timings = fuser_kwargs.get("timings")
    t0 = _sync_perf(device)
    out = fuser_callable(*fuser_args, **fuser_kwargs)
    t1 = _sync_perf(device)
    wall_total_ms = (t1 - t0) * 1000.0
    if timings is not None:
        timings["wall_total_ms"] = wall_total_ms
        timings["unaccounted_ms"] = wall_total_ms - float(timings.get("_total_ms", 0.0))
        timings["instrumented_total_ms"] = float(timings.get("_total_ms", 0.0))
    return out, wall_total_ms


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


def _arm_name(kind: str, recomp: float, gate: float | None = None, kvzip: float | None = None) -> str:
    """Stable, sortable arm name used as dict key in results JSON."""
    if kind == "dense_cacheblend":
        return f"dense_cb_recomp={recomp:.2f}"
    if kind == "compblend":
        if gate is None or gate >= 1.0 or gate <= 0.0:
            glabel = "no_gate"
        else:
            glabel = f"gate={gate:.2f}"
        return f"compblend_kvzip={kvzip:.2f}_recomp={recomp:.2f}_{glabel}"
    return kind


def _gate_semantics(gate: float) -> dict:
    """Self-explaining gate semantics string + activeness flag."""
    if gate >= 1.0 or gate <= 0.0:
        return {
            "gate_is_active": False,
            "gate_label": "no_gate",
            "gate_semantics": (
                "selector=gated_top_k with gate_percentile=1.0 → importance "
                "gate disabled. Selects HKVD top-k among ELIGIBLE retained "
                "tokens (compressed-aware HKVD baseline). NOT Gated HKVD."
            ),
        }
    return {
        "gate_is_active": True,
        "gate_label": f"gate={gate:.2f}",
        "gate_semantics": (
            f"selector=gated_top_k with gate_percentile={gate:.2f} → "
            f"importance gate active. Selects HKVD top-k WITHIN the "
            f"importance-top-{int(gate*100)}% candidate set."
        ),
    }


def _run_fuser_arm(
    lw, chunks, kv_store, cfg, device, *, last_logits_only: bool = True,
):
    """Run fuse_selective_compblend with full instrumentation, return dict.

    Wraps the fuser to collect: timings (fine-grained), flags (mask/workspace/
    selector identity), selector_stats (candidate counts), and wall_total_ms
    (perf_counter around the call with cuda sync).
    """
    from compblend.fuse_selective_compblend import fuse_selective_compblend

    timings: dict = {}
    flags: dict = {}
    selector_stats: dict = {}
    out, wall_ms = _call_arm_with_wall_timing(
        fuse_selective_compblend, device,
        lw, chunks, kv_store, cfg,
        return_layerwise_output=True,
        timings=timings, flags=flags, selector_stats=selector_stats,
        last_logits_only=last_logits_only,
    )
    return {
        "timings": timings,
        "flags": flags,
        "selector_stats": selector_stats,
        "wall_total_ms": wall_ms,
        "_out": out,
    }


def main() -> int:
    from cacheblend import LayerwiseModel
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv

    from compblend.backends.base import CompressionBudget, to_kvstore_entry
    from compblend.backends.kvzip import KVzipBackend, KVzipConfig
    from compblend.config import CompBlendConfig

    print(
        f"[breakdown] model={MODEL}  n={N_MAX}\n"
        f"            recomp_list={RECOMP_RATIO_LIST}\n"
        f"            gate_list={GATE_PCT_LIST}\n"
        f"            kvzip_list={RATIO_KVZIP_LIST}\n"
        f"            include_assistant_open={INCLUDE_ASSISTANT_OPEN}\n"
        f"            run_dense_baseline={RUN_DENSE_BASELINE}",
        flush=True,
    )

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

    user_open, assistant_open = _resolve_wrapper(MODEL, tokenizer)

    questions = _load_loong_questions(
        LOONG_JSONL, LOONG_DATA_DIR, LOONG_LEVELS, LOONG_SETS, "en", LOONG_MAX_LENGTH, N_MAX,
    )
    if not questions:
        print("FATAL: no questions", flush=True); return 1

    # samples[idx] = dict with full_recompute + per-arm subdicts.
    samples: list[dict] = []
    failed_samples: list[dict] = []

    for idx, ex in enumerate(questions):
        stage = "build_chunks"
        try:
            query_suffix = f"\n\n{ex['instruction']}\n\n{ex['question']}"
            if INCLUDE_ASSISTANT_OPEN:
                query_suffix = query_suffix + assistant_open
            chunks, structural_idx, compressible_idx = _build_chunks(
                tokenizer, ex["doc_contents"], query_suffix, user_open,
            )

            stage = "tokenization_invariant"
            full_text = user_open + "".join(ex["doc_contents"]) + query_suffix
            expected_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if tokenizer.bos_token_id is not None:
                expected_ids = [tokenizer.bos_token_id] + expected_ids
            actual_ids = [t for c in chunks for t in c.token_ids]
            assert_token_ids_equal(
                f"loong_q{idx + 1}_full_vs_chunks",
                expected_ids, actual_ids, tokenizer,
            )

            stage = "precompute"
            kv_nozip = KVStore()
            for c in chunks:
                K, V = precompute_chunk_kv(lw, c)
                kv_nozip.put(c.chunk_id, K, V)

            stage = "full_recompute"
            full_result = _measure_full_recompute(lw, chunks, device)
            full_result.pop("_out", None)
            if device.type == "cuda": torch.cuda.empty_cache()

            stage = "kvzip_compress"
            # Compress doc chunks ONCE per (question, kvzip_ratio). Reuse-time
            # serving scenario: compression is offline, NOT charged to TTFT —
            # but we record offline_kvzip_compress_ms for transparency.
            kv_zip_by_ratio: dict[float, KVStore] = {}
            offline_kvzip_compress_ms_by_ratio: dict[float, float] = {}
            for kr in RATIO_KVZIP_LIST:
                kv_zip = KVStore()
                t_zip0 = _sync_perf(device)
                for i, c in enumerate(chunks):
                    if i in structural_idx:
                        kv_zip._cache[c.chunk_id] = kv_nozip.get(c.chunk_id)
                        continue
                    assert_kvzip_roundtrip_token_ids(
                        f"loong_q{idx + 1}_kr{kr}_chunk{i}",
                        c.token_ids, backend.tokenizer,
                    )
                    ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
                    cmp = backend.compress(
                        ids, model=hf_model, budget=CompressionBudget(ratio=kr),
                    )
                    assert_compressed_storage_matches_tokens(
                        f"loong_q{idx + 1}_kr{kr}_chunk{i}", cmp, c.token_ids,
                    )
                    kv_zip._cache[c.chunk_id] = to_kvstore_entry(cmp)
                t_zip1 = _sync_perf(device)
                kv_zip_by_ratio[kr] = kv_zip
                offline_kvzip_compress_ms_by_ratio[kr] = (t_zip1 - t_zip0) * 1000.0

            # ── ARMS ──────────────────────────────────────────────────────
            arms: dict[str, dict] = {}

            stage = "cacheblend_fuse"
            if RUN_DENSE_BASELINE:
                for rr in DENSE_RECOMP_LIST:
                    cb_cfg = CompBlendConfig(
                        check_layer=CHECK_LAYER, recompute_ratio=rr,
                        selector="hkvd_only", chunk_normalization="none",
                    )
                    arm_name = _arm_name("dense_cacheblend", rr)
                    res = _run_fuser_arm(lw, chunks, kv_nozip, cb_cfg, device)
                    arms[arm_name] = {
                        "kind": "dense_cacheblend",
                        "config": {
                            "selector": "hkvd_only",
                            "recompute_ratio": rr,
                            "kvzip_ratio": None,
                            "gate_percentile": None,
                            **_gate_semantics(1.0),
                        },
                        "timings": res["timings"],
                        "flags": res["flags"],
                        "selector_stats": res["selector_stats"],
                    }
                    if device.type == "cuda": torch.cuda.empty_cache()

            stage = "compblend_fuse"
            for kr in RATIO_KVZIP_LIST:
                kv_zip = kv_zip_by_ratio[kr]
                for rr in COMP_RECOMP_LIST:
                    for gp in GATE_PCT_LIST:
                        cp_cfg = CompBlendConfig(
                            check_layer=CHECK_LAYER, recompute_ratio=rr,
                            selector="gated_top_k", gate_percentile=gp,
                            chunk_normalization="rank",
                        )
                        arm_name = _arm_name("compblend", rr, gate=gp, kvzip=kr)
                        res = _run_fuser_arm(lw, chunks, kv_zip, cp_cfg, device)
                        arms[arm_name] = {
                            "kind": "compblend",
                            "config": {
                                "selector": "gated_top_k",
                                "recompute_ratio": rr,
                                "kvzip_ratio": kr,
                                "gate_percentile": gp,
                                **_gate_semantics(gp),
                            },
                            "timings": res["timings"],
                            "flags": res["flags"],
                            "selector_stats": res["selector_stats"],
                            "offline_kvzip_compress_ms": offline_kvzip_compress_ms_by_ratio[kr],
                        }
                        if device.type == "cuda": torch.cuda.empty_cache()

            samples.append({
                "idx": idx + 1,
                "id": ex["id"],
                "context_tokens": int(ex.get("length", -1)),
                "n_docs": len(ex["doc_contents"]),
                "n_chunks": len(chunks),
                "n_compressible_chunks": len(compressible_idx),
                "full_recompute": full_result,
                "arms": arms,
            })
            # Compact log
            compact_arms = [
                f"{name}:wall={a['timings'].get('wall_total_ms', 0):.0f}"
                for name, a in list(arms.items())[:6]
            ]
            print(
                f"[{idx + 1}/{len(questions)}] ctx={ex.get('length', -1)}  "
                f"full={full_result['wall_total_ms']:.0f}  "
                + "  ".join(compact_arms)
                + (f"  +{len(arms) - 6} more" if len(arms) > 6 else ""),
                flush=True,
            )
        except TokenizationInvariantError as exc:
            import traceback as _tb
            failed_samples.append({
                "idx": idx + 1, "id": ex.get("id"),
                "context_tokens": ex.get("length", -1),
                "stage": stage,
                "exception_type": "TokenizationInvariantError",
                "exception_message": str(exc).splitlines()[0],
                "traceback_summary": _tb.format_exc().splitlines()[-3:],
            })
            print(f"[Q{idx+1} FAILED stage={stage}] {exc.__class__.__name__}: {str(exc).splitlines()[0]}", flush=True)
            if device.type == "cuda": torch.cuda.empty_cache()
            continue
        except Exception as exc:
            import traceback as _tb
            failed_samples.append({
                "idx": idx + 1, "id": ex.get("id"),
                "context_tokens": ex.get("length", -1),
                "stage": stage,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback_summary": _tb.format_exc().splitlines()[-5:],
            })
            print(f"[Q{idx+1} FAILED stage={stage}] {type(exc).__name__}: {exc}", flush=True)
            _tb.print_exc()
            if device.type == "cuda": torch.cuda.empty_cache()
            continue

    # ── Aggregate ─────────────────────────────────────────────────────
    # Per-arm aggregation: collect every arm name that appears in any sample.
    all_arm_names = sorted({name for s in samples for name in s["arms"]})

    def _collect_arm_timings(name: str) -> list[dict]:
        return [s["arms"][name]["timings"] for s in samples if name in s["arms"]]

    def _collect_arm_flags(name: str) -> list[dict]:
        return [s["arms"][name]["flags"] for s in samples if name in s["arms"]]

    def _collect_arm_selstats(name: str) -> list[dict]:
        return [s["arms"][name]["selector_stats"] for s in samples if name in s["arms"]]

    full_wall = [s["full_recompute"]["wall_total_ms"] for s in samples]
    full_wall_stats = _stats(full_wall)

    arm_summaries: dict = {}
    for name in all_arm_names:
        ts = _collect_arm_timings(name)
        if not ts:
            continue
        fs = _collect_arm_flags(name)
        ss = _collect_arm_selstats(name)

        wall_stats = _stats([t.get("wall_total_ms", 0.0) for t in ts])
        instr_stats = _stats([t.get("instrumented_total_ms", t.get("_total_ms", 0.0)) for t in ts])
        unacc_stats = _stats([t.get("unaccounted_ms", 0.0) for t in ts])

        # Fine-grained component stats
        component_keys = sorted({
            k for t in ts for k in t.keys()
            if not k.startswith("_") and k not in {"wall_total_ms", "instrumented_total_ms", "unaccounted_ms"}
        })
        component_stats = {k: _stats([t.get(k, 0.0) for t in ts]) for k in component_keys}

        speedup_vs_full = (full_wall_stats["mean"] / wall_stats["mean"]) if wall_stats["mean"] else None

        # Reliability aggregation
        fallback_qs = sum(1 for f in fs if f.get("per_head_mask_fallback_count", 0) > 0)
        exact_qs = sum(1 for f in fs if f.get("per_head_mask_exact_count", 0) > 0)
        dense_workspace_bytes = [int(f.get("dense_workspace_bytes", 0)) for f in fs]
        kvzip_actual = [float(f.get("kvzip_ratio_actual", -1.0)) for f in fs if "kvzip_ratio_actual" in f]

        # Selector-stats means
        if ss:
            sel_keys = ["total_tokens", "structural_tokens", "compressible_tokens",
                        "eligible_tokens", "forced_tokens", "selected_recompute_tokens",
                        "selected_from_structural_tokens", "selected_from_compressible_tokens",
                        "selected_ratio_actual", "eligible_ratio"]
            selector_stats_agg = {
                k: float(np.mean([s.get(k, 0) for s in ss])) for k in sel_keys
            }
            selector_stats_agg["gate_is_active"] = bool(ss[0].get("gate_is_active", False))
        else:
            selector_stats_agg = {}

        arm_summaries[name] = {
            "config": (samples[0]["arms"][name]["config"] if samples and name in samples[0]["arms"] else {}),
            "wall_total_ms": wall_stats,
            "instrumented_total_ms": instr_stats,
            "unaccounted_ms": unacc_stats,
            "component_ms": component_stats,
            "percent_breakdown_vs_wall": _percent_breakdown(ts, wall_stats),
            "speedup_vs_full_recompute": speedup_vs_full,
            "reliability": {
                "per_head_mask_fallback_questions": fallback_qs,
                "per_head_mask_exact_questions": exact_qs,
                "dense_workspace_bytes_mean": float(np.mean(dense_workspace_bytes)) if dense_workspace_bytes else 0.0,
                "dense_workspace_bytes_max": int(max(dense_workspace_bytes)) if dense_workspace_bytes else 0,
                "kvzip_ratio_actual_mean": float(np.mean(kvzip_actual)) if kvzip_actual else None,
            },
            "selector_stats_mean": selector_stats_agg,
        }

    # Add speedup_vs_cacheblend (any dense_cb arm at same recomp counts).
    # We pick dense_cb at the FIRST recomp ratio as a generic reference. Per-arm
    # explicit comparisons are left to the analyst — the JSON has everything.
    cb_reference_name = None
    for name in all_arm_names:
        if name.startswith("dense_cb_recomp="):
            cb_reference_name = name; break
    if cb_reference_name is not None:
        cb_ref_mean = arm_summaries[cb_reference_name]["wall_total_ms"]["mean"]
        for name, a in arm_summaries.items():
            if cb_ref_mean and a["wall_total_ms"]["mean"]:
                a["speedup_vs_dense_cb_reference"] = cb_ref_mean / a["wall_total_ms"]["mean"]
                a["speedup_dense_cb_reference_arm"] = cb_reference_name

    summary = {
        "config": {
            "model": MODEL, "dtype": DTYPE, "attn_impl": ATTN_IMPL,
            "scenario": "reuse_time_cached_kv",
            "scenario_notes": (
                "Compressed KV is assumed to be already in cache. TTFT measurements "
                "do NOT include KVzip compression cost. 'offline_kvzip_compress_ms' "
                "per arm is informational only."
            ),
            "includes_kvzip_compression_cost": False,
            "includes_precompute_cost": False,
            "measured_path": "fuse / reuse path after KVStore entries are prepared",
            "stage": "Stage 1",
            "stage_notes": (
                "Dense [1, total_seq, H_kv*D] workspace per layer. KVzip compressed "
                "metadata is consumed for selection + zero-filled K/V values, but the "
                "underlying dense tensor allocation is unchanged. Compressed-native "
                "KV layout and bandwidth reduction are Stage 2."
            ),
            "chunk_policy": (
                "[BOS+user_open](structural) + [docs](compressed) + "
                "[\\n\\n+instruction+\\n\\n+question" +
                ("+assistant_open" if INCLUDE_ASSISTANT_OPEN else "") +
                "](structural)"
            ),
            "include_assistant_open": INCLUDE_ASSISTANT_OPEN,
            "check_layer": CHECK_LAYER,
            "recomp_ratio_list": RECOMP_RATIO_LIST,
            "dense_recomp_list": DENSE_RECOMP_LIST,
            "compressed_recomp_list": COMP_RECOMP_LIST,
            "gate_pct_list": GATE_PCT_LIST,
            "ratio_kvzip_list": RATIO_KVZIP_LIST,
            "run_dense_baseline": RUN_DENSE_BASELINE,
            "loong_levels": LOONG_LEVELS,
            "loong_sets": LOONG_SETS,
            "loong_max_length": LOONG_MAX_LENGTH,
            "language": "en",
            "answer_type_filter": "str",
            "n_loaded": len(questions),
            "n_successful": len(samples),
            "n_failed": len(failed_samples),
            "metric_for_quality": (
                "This breakdown measures TIMING only; F1 is in loong_set1_sweep.json."
            ),
        },
        "full_recompute": {
            "wall_total_ms": full_wall_stats,
            "instrumented_total_ms": None,
            "note": (
                "fuse_full_recompute is unmodified v7 — measured wall only. "
                "Per-component breakdown is NOT available for the full-recompute arm; "
                "use CacheBlend / CompBlend arms for component attribution."
            ),
        },
        "arms": arm_summaries,
        "failed_samples": failed_samples,
        "per_sample": samples,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))

    # Brief console summary.
    print("\n──────── BREAKDOWN SUMMARY ────────")
    print(f"full_recompute wall (mean / median / p90):  "
          f"{full_wall_stats['mean']:.1f} / {full_wall_stats['median']:.1f} / {full_wall_stats['p90']:.1f} ms"
          f"   n={full_wall_stats['n']}")
    for name in all_arm_names:
        a = arm_summaries[name]
        w = a["wall_total_ms"]
        sp = a.get("speedup_vs_full_recompute")
        if sp:
            print(f"  {name:55s}  wall={w['mean']:7.1f}±{w['stdev']:5.1f}ms  speedup×{sp:.2f}")
        else:
            print(f"  {name:55s}  wall={w['mean']:7.1f}±{w['stdev']:5.1f}ms")
    print(f"\n[breakdown] wrote {OUT_PATH}")
    print(f"[breakdown] {len(samples)} succeeded, {len(failed_samples)} failed", flush=True)
    print("BREAKDOWN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
