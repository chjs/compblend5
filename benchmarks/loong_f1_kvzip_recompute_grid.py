"""Loong F1 quality grid: KVzip ratio × Recompute ratio.

Purpose:
  Measure F1 trade-off curve over (KVzip ratio, recompute ratio) on Loong.
  This is a QUALITY sweep, not a TTFT breakdown. The selector is fixed at
  gated_top_k with gate_percentile=1.0 (importance gate disabled) so the
  variable factors are:
    * KVzip pair-level retention ratio (compression)
    * recompute ratio (selective recompute via HKVD over eligible tokens)

Grid (default):
  kvzip_ratio  ∈ [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  recomp_ratio ∈ [0.0, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
  = 80 cells per question

Outputs (under COMPBLEND_OUT_DIR, default logs/loong_f1_grid):
  raw_results.jsonl    — one line per (example, kvzip, recomp)
  summary.csv          — one row per (kvzip, recomp) with stats
  summary.json         — config + grid + skip reasons + env
  f1_vs_recompute_by_kvzip_ratio.{png,pdf}
  f1_vs_kvzip_by_recompute_ratio.{png,pdf}
  f1_heatmap_kvzip_recompute.{png,pdf}

Metric: token-F1 substring (NOT Loong's official LLM-judge). Same
implementation as loong_set1_sweep.py / MuSiQue. Use for selector/grid
comparison only; absolute scores are NOT directly comparable to Loong paper.

Env vars (env name → meaning, default):
  CACHEBLEND_MODEL            HF model id            (meta-llama/Llama-3.1-8B-Instruct)
  LOONG_JSONL                 Loong metadata jsonl   (/root/Loong/data/loong.jsonl)
  MODELSCOPE_LOONG_CACHE      Loong corpus dir       (/root/loong_data/doc)
  LOONG_LEVELS                comma-separated ints   (1,2)
  LOONG_SETS                  comma-separated ints   (1,2)
  LOONG_LANGUAGES             comma-separated strs   (en)
  LOONG_MIN_LENGTH            int                    (0)
  LOONG_MAX_LENGTH            int                    (100000)
  COMPBLEND_N                 int                    (10)
  COMPBLEND_CHECK_LAYER       int                    (1)
  COMPBLEND_OUT_DIR           dir path               (logs/loong_f1_grid)
  COMPBLEND_GATE_PCT          float                  (1.0)
  COMPBLEND_SELECTOR          str                    (gated_top_k)
  COMPBLEND_INCLUDE_ASSISTANT_OPEN  0/1              (1)
  COMPBLEND_KVZIP_RATIO_LIST  csv floats             (full grid above)
  COMPBLEND_RECOMP_RATIO_LIST csv floats             (full grid above)
  COMPBLEND_RESUME            0/1                    (1 — skip already-run cells)
  COMPBLEND_MAX_NEW_TOKENS    int                    (64)
"""
from __future__ import annotations

import collections
import csv
import json
import os
import platform
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
    assert_kvzip_roundtrip_token_ids,
    assert_token_ids_equal,
)


# ─── env config ──────────────────────────────────────────────────────────


def _intlist(env: str, default: str) -> list[int]:
    return [int(x.strip()) for x in os.environ.get(env, default).split(",") if x.strip()]


def _strlist(env: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(env, default).split(",") if x.strip()]


def _floatlist(env: str, default: str) -> list[float]:
    return [float(x.strip()) for x in os.environ.get(env, default).split(",") if x.strip()]


MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LOONG_JSONL = Path(os.environ.get("LOONG_JSONL", "/root/Loong/data/loong.jsonl"))
LOONG_DATA_DIR = Path(os.environ.get("MODELSCOPE_LOONG_CACHE", "/root/loong_data/doc"))
LOONG_LEVELS = _intlist("LOONG_LEVELS", "1,2")
LOONG_SETS = _intlist("LOONG_SETS", "1,2")
LOONG_LANGUAGES = _strlist("LOONG_LANGUAGES", "en")
LOONG_MIN_LENGTH = int(os.environ.get("LOONG_MIN_LENGTH", "0"))
LOONG_MAX_LENGTH = int(os.environ.get("LOONG_MAX_LENGTH", "100000"))
N_MAX = int(os.environ.get("COMPBLEND_N", "10"))
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))
OUT_DIR = Path(os.environ.get("COMPBLEND_OUT_DIR", str(_REPO / "logs" / "loong_f1_grid")))
GATE_PCT = float(os.environ.get("COMPBLEND_GATE_PCT", "1.0"))
SELECTOR = os.environ.get("COMPBLEND_SELECTOR", "gated_top_k")
INCLUDE_ASSISTANT_OPEN = os.environ.get("COMPBLEND_INCLUDE_ASSISTANT_OPEN", "1") == "1"
RESUME = os.environ.get("COMPBLEND_RESUME", "1") == "1"
MAX_NEW_TOKENS = int(os.environ.get("COMPBLEND_MAX_NEW_TOKENS", "64"))

KVZIP_RATIO_LIST = _floatlist(
    "COMPBLEND_KVZIP_RATIO_LIST",
    "1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1",
)
RECOMP_RATIO_LIST = _floatlist(
    "COMPBLEND_RECOMP_RATIO_LIST",
    "0.0,0.1,0.15,0.2,0.3,0.5,0.7,1.0",
)


# ─── Loong loader (reuses logic from loong_set1_sweep) ────────────────────


def _load_doc_content(item: dict, doc_name: str, idx: int, doc_path: Path) -> str | None:
    import glob
    doc_type = item["type"]
    doc_level = item["level"]
    type_dir = doc_path / doc_type
    if doc_type == "financial":
        if str(doc_level).strip() != "4":
            matches = glob.glob(f"{type_dir}/*2024-{doc_name}*.txt")
        else:
            matches = glob.glob(f"{type_dir}/*{doc_name}*.txt")
        if not matches:
            return None
        with open(matches[0]) as f:
            stem = Path(matches[0]).stem.split("-")[-1]
            return f"《{stem}》\n" + f.read() + "\n\n"
    elif doc_type == "paper":
        p = type_dir / doc_name
        if not p.exists():
            return None
        with open(p) as f:
            txt = f.read()
        title = txt.split("\n", 1)[0].strip("#").strip()
        return f"{title}\n{txt}\n\n"
    elif doc_type == "legal":
        legal_json = type_dir / "legal.json"
        if not legal_json.exists():
            return None
        with open(legal_json) as f:
            legal = json.load(f)
        if doc_name not in legal:
            return None
        entry = legal[doc_name]
        return f"《判决文书{idx + 1}》\n" + entry["content"] + entry.get("result", "") + "\n\n"
    return None


def _load_loong_questions(
    jsonl_path: Path, doc_path: Path,
    levels: list[int], set_ids: list[int], languages: list[str],
    min_length: int, max_length: int, max_n: int,
) -> tuple[list[dict], collections.Counter]:
    """Filter Loong jsonl + materialize doc texts. Returns (rows, skip_counter)."""
    rows = []
    skip = collections.Counter()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            if d["level"] not in levels:
                skip["level"] += 1; continue
            if d["set"] not in set_ids:
                skip["set"] += 1; continue
            if d["language"] not in languages:
                skip["language"] += 1; continue
            if not isinstance(d["answer"], str):
                skip["answer_not_str"] += 1; continue
            length = d.get("length", 0)
            if length < min_length:
                skip["below_min_length"] += 1; continue
            if length > max_length:
                skip["above_max_length"] += 1; continue
            rows.append(d)
    materialized = []
    for r in rows:
        contents = []
        ok = True
        for idx, doc_name in enumerate(r["doc"]):
            txt = _load_doc_content(r, doc_name, idx, doc_path)
            if txt is None:
                ok = False; break
            contents.append(txt)
        if not ok:
            skip["doc_missing"] += 1; continue
        materialized.append({
            "id": r["id"], "question": r["question"],
            "instruction": r["instruction"], "doc_contents": contents,
            "answer": r["answer"], "length": r.get("length", -1),
            "type": r.get("type", "?"),
        })
        if len(materialized) >= max_n: break
    return materialized, skip


# ─── chunk policy + tokenization invariant (same as loong_breakdown) ─────


_WRAPPERS = {
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
}


def _resolve_wrapper(model_id: str) -> tuple[str, str]:
    mid = model_id.lower()
    for key, wrap in _WRAPPERS.items():
        if key in mid: return wrap
    raise RuntimeError(f"add wrapper for {model_id}")


def _build_chunks(tokenizer, user_open: str, doc_contents: list[str], query_suffix: str):
    """Slice from full tokenization → concat(c.token_ids) == tokenize(full)."""
    from cacheblend.chunker import Chunk, _stable_id
    bos = tokenizer.bos_token_id
    segments = [user_open] + list(doc_contents) + [query_suffix]
    full_text = "".join(segments)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if bos is not None:
        full_ids = [bos] + full_ids
    bos_offset = 1 if bos is not None else 0

    boundaries = [bos_offset]
    cumulative = ""
    for seg in segments:
        cumulative += seg
        n = len(tokenizer(cumulative, add_special_tokens=False)["input_ids"])
        boundaries.append(bos_offset + n)

    chunks = []
    for i, seg in enumerate(segments):
        s = 0 if i == 0 else boundaries[i]
        e = boundaries[i + 1]
        ids = full_ids[s:e]
        chunks.append(Chunk(text=seg, token_ids=ids, chunk_id=_stable_id(seg, ids)))

    structural_idx = [0, len(chunks) - 1]
    compressible_idx = list(range(1, len(chunks) - 1))
    return chunks, structural_idx, compressible_idx, full_ids


# ─── F1 metric (same impl as loong_set1_sweep) ───────────────────────────


def _normalize_answer(s: str) -> str:
    import string, re
    def remove_articles(s): return re.sub(r"\b(a|an|the)\b", " ", s)
    def white_space_fix(s): return " ".join(s.split())
    def remove_punc(s): return "".join(ch for ch in s if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _compute_f1(a_pred: str, a_gold: str, tokenizer) -> tuple[float, int]:
    """Returns (f1, exact_match)."""
    a_pred = a_pred.strip().split("\n")[0]
    norm_pred = _normalize_answer(a_pred)
    norm_gold = _normalize_answer(a_gold)
    em = int(norm_pred == norm_gold)
    gold_toks = tokenizer.encode(norm_gold)[1:]
    pred_toks = tokenizer.encode(norm_pred)[1:]
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(int(gold_toks == pred_toks)), em
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, em
    p = num_same / len(pred_toks); r = num_same / len(gold_toks)
    return (2 * p * r) / (p + r), em


def _greedy_decode(model, tokenizer, prefill_logits, past_kv, device, max_new=MAX_NEW_TOKENS):
    eos = getattr(tokenizer, "eos_token_id", None)
    next_id = prefill_logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    generated = [int(next_id.item())]
    with torch.inference_mode():
        for _ in range(max_new - 1):
            if eos is not None and generated[-1] == eos: break
            step = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = step.past_key_values
            next_id = step.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated.append(int(next_id.item()))
    return tokenizer.decode(generated, skip_special_tokens=True)


def _derive_valid_mask_pair(importance: torch.Tensor, target_ratio: float) -> torch.Tensor:
    """Verbatim KVzip._threshold (pair-level)."""
    if target_ratio >= 1.0:
        return torch.ones_like(importance, dtype=torch.bool)
    flat = importance.reshape(-1)
    n = max(int(flat.numel() * target_ratio) - 1, 0)
    score_sort = torch.sort(flat, descending=True).values
    thres = score_sort[n].item()
    return importance > thres


def _make_entry_at_ratio(cmp_full, target_ratio: float, n_kv_heads: int, head_dim: int) -> dict:
    storage_len = cmp_full.key_cache[0].shape[1]
    n_layers = cmp_full.num_layers
    new_valid = _derive_valid_mask_pair(cmp_full.importance, target_ratio)
    new_K, new_V = [], []
    for li in range(n_layers):
        k = cmp_full.key_cache[li].view(1, storage_len, n_kv_heads, head_dim)
        v = cmp_full.value_cache[li].view(1, storage_len, n_kv_heads, head_dim)
        m = new_valid[li].t().unsqueeze(0).unsqueeze(-1).to(k.dtype)
        new_K.append((k * m).reshape(1, storage_len, n_kv_heads * head_dim).contiguous())
        new_V.append((v * m).reshape(1, storage_len, n_kv_heads * head_dim).contiguous())
    new_imp = cmp_full.importance * new_valid.to(cmp_full.importance.dtype)
    return {
        "K": new_K, "V": new_V,
        "valid_mask": new_valid, "importance": new_imp,
        "is_structural": cmp_full.is_structural,
        "algo_id": cmp_full.algo_id, "chunk_id": cmp_full.chunk_id,
    }


# ─── main grid run ───────────────────────────────────────────────────────


def _gate_semantics(gate: float) -> dict:
    if gate >= 1.0 or gate <= 0.0:
        return {
            "gate_is_active": False,
            "gate_label": "no_gate",
            "gate_semantics": "gate=1.0; importance gate disabled; HKVD top-k among retained eligible tokens",
        }
    return {
        "gate_is_active": True,
        "gate_label": f"gate={gate:.2f}",
        "gate_semantics": f"gate={gate:.2f}; importance gate active; HKVD top-k within importance-top-{int(gate*100)}% candidates",
    }


def _load_resume_set(raw_path: Path) -> set[tuple[str, float, float]]:
    """Read existing raw_results.jsonl and return set of completed cells."""
    done: set[tuple[str, float, float]] = set()
    if not raw_path.exists():
        return done
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                if d.get("status") == "ok":
                    done.add((d["example_id"], float(d["kvzip_ratio"]), float(d["recompute_ratio"])))
            except Exception:
                pass
    return done


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

    gate_info = _gate_semantics(GATE_PCT)
    config = {
        "model": MODEL,
        "loong_jsonl": str(LOONG_JSONL),
        "loong_corpus": str(LOONG_DATA_DIR),
        "levels": LOONG_LEVELS, "sets": LOONG_SETS, "languages": LOONG_LANGUAGES,
        "min_length": LOONG_MIN_LENGTH, "max_length": LOONG_MAX_LENGTH,
        "n_max": N_MAX,
        "check_layer": CHECK_LAYER,
        "selector": SELECTOR,
        "gate_percentile": GATE_PCT,
        **gate_info,
        "include_assistant_open": INCLUDE_ASSISTANT_OPEN,
        "kvzip_ratio_list": KVZIP_RATIO_LIST,
        "recompute_ratio_list": RECOMP_RATIO_LIST,
        "max_new_tokens": MAX_NEW_TOKENS,
        "metric_name": "token_f1",
        "official_loong_metric": False,
        "metric_note": "token-F1 substring on normalized answer; same as MuSiQue impl. NOT official Loong score.",
        "stage": "Stage 1",
        "stage_notes": "Dense [1, total_seq, H_kv*D] workspace per layer; compressed-native KV is Stage 2.",
        "chunk_policy": "[BOS+user_open](structural) + [docs](compressed) + [query+assistant_open](structural)",
        "ratio_1_handling": (
            "kvzip_ratio=1.0 → valid_mask all-True (no eviction); same code path as r<1 "
            "but with no zero-fill. Recompute_ratio=0.0 → fuse_full_reuse boundary shortcut; "
            "Recompute_ratio>=1.0 → fuse_full_recompute boundary shortcut."
        ),
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
        },
    }
    (OUT_DIR / "config_snapshot.json").write_text(json.dumps(config, indent=2))
    print(f"[grid] config saved to {OUT_DIR / 'config_snapshot.json'}", flush=True)

    print(f"[grid] loading Loong (levels={LOONG_LEVELS} sets={LOONG_SETS} lang={LOONG_LANGUAGES} "
          f"min={LOONG_MIN_LENGTH} max={LOONG_MAX_LENGTH} n={N_MAX})", flush=True)
    questions, skip = _load_loong_questions(
        LOONG_JSONL, LOONG_DATA_DIR,
        LOONG_LEVELS, LOONG_SETS, LOONG_LANGUAGES,
        LOONG_MIN_LENGTH, LOONG_MAX_LENGTH, N_MAX,
    )
    print(f"[grid] materialized {len(questions)} questions; pre-skip: {dict(skip)}", flush=True)
    if not questions:
        print("FATAL: no questions", flush=True); return 1

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

    completed = _load_resume_set(raw_path) if RESUME else set()
    if completed:
        print(f"[grid] resume: {len(completed)} cells already in raw_results.jsonl", flush=True)

    # Append mode so we can resume cleanly.
    raw_fh = open(raw_path, "a", encoding="utf-8")
    skip_runtime = collections.Counter(skip)

    def _emit(rec: dict) -> None:
        raw_fh.write(json.dumps(rec) + "\n")
        raw_fh.flush()

    # Per-question loop. Inside, we tokenize/build chunks ONCE, then iterate
    # KVzip ratios (compress doc chunks once per ratio), then recompute ratios.
    for q_idx, ex in enumerate(questions):
        stage = "build_chunks"
        try:
            query_suffix = f"\n\n{ex['instruction']}\n\n{ex['question']}"
            if INCLUDE_ASSISTANT_OPEN:
                query_suffix = query_suffix + assistant_open
            chunks, structural_idx, compressible_idx, full_ids = _build_chunks(
                tokenizer, user_open, ex["doc_contents"], query_suffix,
            )

            stage = "tokenization_check"
            actual_ids = [t for c in chunks for t in c.token_ids]
            assert_token_ids_equal(
                f"loong_q{q_idx+1}_full_vs_chunks", full_ids, actual_ids, tokenizer,
            )

            n_total = len(full_ids)
            n_struct = sum(len(chunks[i].token_ids) for i in structural_idx)
            n_doc = sum(len(chunks[i].token_ids) for i in compressible_idx)
            n_query = len(chunks[structural_idx[-1]].token_ids)
            # `n_struct` includes both prefix and suffix; split for clarity:
            n_prefix = len(chunks[structural_idx[0]].token_ids)

            stage = "precompute"
            kv_base = KVStore()
            for c in chunks:
                K, V = precompute_chunk_kv(lw, c)
                kv_base.put(c.chunk_id, K, V)

            stage = "kvzip_compress"
            # Compress doc chunks ONCE at ratio=1.0, then derive masks at each
            # sweep ratio. _derive_valid_mask_pair has been cross-checked
            # against KVzip._threshold (M6 tests, 26 cells bit-equal).
            compressed_1p0 = {}
            for ci in compressible_idx:
                c = chunks[ci]
                assert_kvzip_roundtrip_token_ids(
                    f"q{q_idx+1}_chunk{ci}", c.token_ids, backend.tokenizer,
                )
                ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
                cmp = backend.compress(ids, model=hf_model, budget=CompressionBudget(ratio=1.0))
                assert_compressed_storage_matches_tokens(f"q{q_idx+1}_chunk{ci}", cmp, c.token_ids)
                compressed_1p0[c.chunk_id] = cmp

        except TokenizationInvariantError as exc:
            reason = "tokenization_mismatch"
            print(f"[Q{q_idx+1} SKIP {reason}] {str(exc).splitlines()[0]}", flush=True)
            skip_runtime[reason] += 1
            for kr in KVZIP_RATIO_LIST:
                for rr in RECOMP_RATIO_LIST:
                    if (ex["id"], kr, rr) in completed: continue
                    _emit({
                        "status": "skip", "skip_reason": reason, "stage": stage,
                        "example_id": ex["id"], "question_idx": q_idx + 1,
                        "kvzip_ratio": kr, "recompute_ratio": rr,
                        "exception_type": "TokenizationInvariantError",
                        "exception_message": str(exc).splitlines()[0],
                    })
            if device.type == "cuda": torch.cuda.empty_cache()
            continue
        except Exception as exc:
            reason = f"{stage}_failed"
            print(f"[Q{q_idx+1} FAIL stage={stage}] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            skip_runtime[reason] += 1
            for kr in KVZIP_RATIO_LIST:
                for rr in RECOMP_RATIO_LIST:
                    if (ex["id"], kr, rr) in completed: continue
                    _emit({
                        "status": "fail", "stage": stage,
                        "example_id": ex["id"], "question_idx": q_idx + 1,
                        "kvzip_ratio": kr, "recompute_ratio": rr,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    })
            if device.type == "cuda": torch.cuda.empty_cache()
            continue

        # Per KVzip ratio
        for kr in KVZIP_RATIO_LIST:
            # Build kv_r for this ratio (shared across all recompute ratios).
            kv_r = KVStore()
            for c in chunks:
                if c.chunk_id in compressed_1p0:
                    kv_r._cache[c.chunk_id] = _make_entry_at_ratio(
                        compressed_1p0[c.chunk_id], kr, n_kv_heads, head_dim,
                    )
                else:
                    kv_r._cache[c.chunk_id] = kv_base.get(c.chunk_id)
            # Compute actual valid ratio from one entry's valid_mask.
            sample_entry = kv_r._cache[chunks[compressible_idx[0]].chunk_id]
            kvzip_valid_actual = float(sample_entry["valid_mask"].float().mean().item())

            for rr in RECOMP_RATIO_LIST:
                if (ex["id"], kr, rr) in completed:
                    continue
                t_start = time.perf_counter()
                inner_stage = "fuse_generate"
                try:
                    cb_cfg = CompBlendConfig(
                        check_layer=CHECK_LAYER, recompute_ratio=rr,
                        selector=SELECTOR, gate_percentile=GATE_PCT,
                        chunk_normalization="rank",
                    )
                    if device.type == "cuda": torch.cuda.synchronize()
                    flags: dict = {}
                    sel_stats: dict = {}
                    out = fuse_selective_compblend(
                        lw, chunks, kv_r, cb_cfg,
                        return_layerwise_output=True, last_logits_only=True,
                        flags=flags, selector_stats=sel_stats,
                    )
                    inner_stage = "decode"
                    text = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device)
                    inner_stage = "f1_metric"
                    f1, em = _compute_f1(text, ex["answer"], tokenizer)
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

                    n_selected = sel_stats.get("selected_recompute_tokens", 0)
                    n_eligible = sel_stats.get("eligible_tokens", 0)
                    n_forced = sel_stats.get("forced_tokens", 0)
                    n_sel_struct = sel_stats.get("selected_from_structural_tokens", 0)
                    rec = {
                        "status": "ok",
                        "example_id": ex["id"], "question_idx": q_idx + 1,
                        "type": ex.get("type", "?"),
                        "context_tokens": int(ex.get("length", -1)),
                        "kvzip_ratio": kr, "recompute_ratio": rr,
                        "selector": SELECTOR, "gate_percentile": GATE_PCT,
                        **gate_info,
                        "prediction": text, "gold_answer": ex["answer"],
                        "f1": f1, "exact_match": em,
                        "n_total_tokens": n_total,
                        "n_prefix_tokens": n_prefix,
                        "n_document_tokens": n_doc,
                        "n_query_tokens": n_query,
                        "n_structural_tokens": n_struct,
                        "kvzip_ratio_requested": kr,
                        "kvzip_valid_pair_ratio_actual": kvzip_valid_actual,
                        "eligible_token_ratio_actual": sel_stats.get("eligible_ratio", -1.0),
                        "recompute_ratio_requested": rr,
                        "recompute_tokens_selected": n_selected,
                        "recompute_ratio_actual": sel_stats.get("selected_ratio_actual", -1.0),
                        "forced_tokens_count": n_forced,
                        "eligible_tokens_count": n_eligible,
                        "selected_from_structural_tokens": n_sel_struct,
                        "selected_from_document_tokens": n_selected - n_sel_struct,
                        "per_head_mask_fallback_count": flags.get("per_head_mask_fallback_count", 0),
                        "per_head_mask_exact_count": flags.get("per_head_mask_exact_count", 0),
                        "elapsed_ms": elapsed_ms,
                    }
                    _emit(rec)
                    print(
                        f"[Q{q_idx+1}/{len(questions)} kr={kr:.2f} rr={rr:.2f}] "
                        f"f1={f1:.3f} em={em} t={elapsed_ms:.0f}ms",
                        flush=True,
                    )
                    del out
                    if device.type == "cuda": torch.cuda.empty_cache()
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    print(f"[Q{q_idx+1} kr={kr:.2f} rr={rr:.2f} FAIL stage={inner_stage}] "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    skip_runtime[f"{inner_stage}_failed"] += 1
                    _emit({
                        "status": "fail", "stage": inner_stage,
                        "example_id": ex["id"], "question_idx": q_idx + 1,
                        "kvzip_ratio": kr, "recompute_ratio": rr,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "elapsed_ms": elapsed_ms,
                    })
                    if device.type == "cuda": torch.cuda.empty_cache()

    raw_fh.close()

    # ── Aggregate ────────────────────────────────────────────────────────
    print("[grid] aggregating results", flush=True)
    rows = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))

    by_cell: dict[tuple[float, float], list[dict]] = {}
    for r in rows:
        if r.get("status") != "ok": continue
        key = (float(r["kvzip_ratio"]), float(r["recompute_ratio"]))
        by_cell.setdefault(key, []).append(r)

    # CSV
    with open(summary_csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "kvzip_ratio", "recompute_ratio",
            "n_success", "n_failed",
            "mean_f1", "std_f1", "median_f1", "p10_f1", "p90_f1",
            "mean_exact_match",
            "mean_context_tokens",
            "mean_kvzip_valid_pair_ratio_actual",
            "mean_eligible_token_ratio_actual",
            "mean_recompute_ratio_actual",
        ])
        for kr in KVZIP_RATIO_LIST:
            for rr in RECOMP_RATIO_LIST:
                key = (kr, rr)
                cell_rows = by_cell.get(key, [])
                fails = sum(
                    1 for r in rows
                    if r.get("status") != "ok"
                    and float(r.get("kvzip_ratio", -1)) == kr
                    and float(r.get("recompute_ratio", -1)) == rr
                )
                if not cell_rows:
                    wr.writerow([kr, rr, 0, fails, "", "", "", "", "", "", "", "", "", ""])
                    continue
                f1s = np.array([r["f1"] for r in cell_rows], dtype=float)
                ems = np.array([r["exact_match"] for r in cell_rows], dtype=float)
                ctx = np.array([r.get("context_tokens", 0) for r in cell_rows], dtype=float)
                kvz = np.array([r.get("kvzip_valid_pair_ratio_actual", 0) for r in cell_rows], dtype=float)
                eli = np.array([r.get("eligible_token_ratio_actual", 0) for r in cell_rows], dtype=float)
                rec = np.array([r.get("recompute_ratio_actual", 0) for r in cell_rows], dtype=float)
                wr.writerow([
                    kr, rr, len(cell_rows), fails,
                    f"{f1s.mean():.4f}", f"{f1s.std():.4f}",
                    f"{np.median(f1s):.4f}",
                    f"{np.quantile(f1s, 0.10):.4f}", f"{np.quantile(f1s, 0.90):.4f}",
                    f"{ems.mean():.4f}",
                    f"{ctx.mean():.0f}",
                    f"{kvz.mean():.4f}", f"{eli.mean():.4f}", f"{rec.mean():.4f}",
                ])
    print(f"[grid] wrote {summary_csv_path}", flush=True)

    # JSON
    grid_summary = {}
    for kr in KVZIP_RATIO_LIST:
        grid_summary[str(kr)] = {}
        for rr in RECOMP_RATIO_LIST:
            cell = by_cell.get((kr, rr), [])
            if cell:
                f1s = np.array([r["f1"] for r in cell], dtype=float)
                grid_summary[str(kr)][str(rr)] = {
                    "n": len(cell),
                    "mean_f1": float(f1s.mean()),
                    "std_f1": float(f1s.std()),
                    "median_f1": float(np.median(f1s)),
                }
            else:
                grid_summary[str(kr)][str(rr)] = {"n": 0}
    summary = {
        "config": config,
        "n_questions_loaded": len(questions),
        "skip_counter": dict(skip_runtime),
        "grid_summary": grid_summary,
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[grid] wrote {summary_json_path}", flush=True)

    # ── matplotlib plots ─────────────────────────────────────────────────
    print("[grid] drawing plots", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build a 2D matrix M[kvzip_idx, recomp_idx] = mean F1
    M = np.full((len(KVZIP_RATIO_LIST), len(RECOMP_RATIO_LIST)), np.nan)
    for i, kr in enumerate(KVZIP_RATIO_LIST):
        for j, rr in enumerate(RECOMP_RATIO_LIST):
            cell = by_cell.get((kr, rr), [])
            if cell:
                M[i, j] = float(np.mean([r["f1"] for r in cell]))

    # Plot 1: x=recompute, y=F1, line=kvzip
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.viridis
    for i, kr in enumerate(KVZIP_RATIO_LIST):
        color = cmap(i / max(len(KVZIP_RATIO_LIST) - 1, 1))
        ax.plot(RECOMP_RATIO_LIST, M[i], marker="o", label=f"kvzip={kr}", color=color)
    ax.set_xlabel("Recompute ratio")
    ax.set_ylabel("Mean token F1")
    ax.set_title("Loong F1 vs Recompute Ratio by KVzip Ratio")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"f1_vs_recompute_by_kvzip_ratio.{ext}", dpi=120)
    plt.close(fig)

    # Plot 2: x=kvzip, y=F1, line=recompute
    fig, ax = plt.subplots(figsize=(9, 6))
    for j, rr in enumerate(RECOMP_RATIO_LIST):
        color = cmap(j / max(len(RECOMP_RATIO_LIST) - 1, 1))
        ax.plot(KVZIP_RATIO_LIST, M[:, j], marker="o", label=f"recomp={rr}", color=color)
    ax.set_xlabel("KVzip ratio")
    ax.set_ylabel("Mean token F1")
    ax.set_title("Loong F1 vs KVzip Ratio by Recompute Ratio")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"f1_vs_kvzip_by_recompute_ratio.{ext}", dpi=120)
    plt.close(fig)

    # Plot 3: heatmap (x=recompute, y=kvzip)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(RECOMP_RATIO_LIST)))
    ax.set_xticklabels([f"{r}" for r in RECOMP_RATIO_LIST])
    ax.set_yticks(range(len(KVZIP_RATIO_LIST)))
    ax.set_yticklabels([f"{r}" for r in KVZIP_RATIO_LIST])
    ax.set_xlabel("Recompute ratio")
    ax.set_ylabel("KVzip ratio")
    ax.set_title("Loong F1 heatmap")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                color = "white" if M[i, j] < 0.5 else "black"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="Mean F1")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"f1_heatmap_kvzip_recompute.{ext}", dpi=120)
    plt.close(fig)

    print(f"[grid] plots written to {OUT_DIR}", flush=True)
    print("F1_GRID_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
