"""MuSiQue F1 quality grid — same 80-cell (kvzip × recompute) sweep as
loong_f1_kvzip_recompute_grid.py, but on MuSiQue's 3K-token contexts.

Purpose (Step 1 of CompBlend Stage 2 plan):
  Verify the diagnosis that the per-head SDPA mask fallback at long context
  is the root cause of F1 collapse. At ~3K context, the per-head mask fits
  inside the 256MB cap (mask ≈ 32 × 500 × 3000 × 2 = 96 MB), so NO fallback
  occurs. If F1 still collapses on this benchmark → some other bug.
  If F1 shows smooth proportional degradation across kvzip ratios →
  diagnosis CONFIRMED.

Identical grid:
  kvzip_ratio  ∈ [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  recomp_ratio ∈ [0.0, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
  selector = gated_top_k, gate_percentile=1.0 (no_gate)

Outputs:  same as Loong grid (raw_results.jsonl, summary.csv, summary.json,
          3 plots).

NOT Loong's official LLM-judge metric. token-F1 substring for relative
comparison only.
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
    assert_token_ids_equal,
)


# ─── env config (mirrors loong_f1_kvzip_recompute_grid) ───────────────────


def _floatlist(env: str, default: str) -> list[float]:
    return [float(x.strip()) for x in os.environ.get(env, default).split(",") if x.strip()]


MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
MUSIQUE_PATH = Path(os.environ.get(
    "MUSIQUE_PATH",
    str(_REPO / "src" / "external" / "cacheblend-hf-v7" / "benchmarks" / "musique" / "inputs" / "musique_s.json"),
))
N_MAX = int(os.environ.get("COMPBLEND_N", "30"))
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))
OUT_DIR = Path(os.environ.get("COMPBLEND_OUT_DIR", str(_REPO / "logs" / "musique_f1_grid")))
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


# ─── MuSiQue loader ──────────────────────────────────────────────────────


def _load_musique_questions(path: Path, max_n: int) -> tuple[list[dict], collections.Counter]:
    """Load MuSiQue from cacheblend-hf-v7's bundled json. Returns (rows, skip)."""
    skip = collections.Counter()
    with open(path) as f:
        data = json.load(f)
    rows = []
    for ex in data:
        if "answers" not in ex or "question" not in ex or "ctxs" not in ex:
            skip["missing_fields"] += 1; continue
        if not isinstance(ex["answers"], list) or not ex["answers"]:
            skip["no_answers"] += 1; continue
        rows.append({
            "id": ex.get("id", f"q{len(rows)}"),
            "question": ex["question"],
            "ctxs": ex["ctxs"],
            "answers": ex["answers"],
        })
        if len(rows) >= max_n:
            break
    return rows, skip


# ─── chunk + invariant ───────────────────────────────────────────────────


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


PREFIX_PROMPT = (
    "I will give you several documents. Read them carefully and answer "
    "the question at the end.\n\n"
)


def _build_chunks(
    tokenizer, user_open: str, doc_prompts: list[str], query_suffix: str,
):
    """Per-chunk independent tokenization. structural+docs+structural layout."""
    from cacheblend.chunker import Chunk, _stable_id
    bos = tokenizer.bos_token_id
    chunks = []

    pref_text = user_open + PREFIX_PROMPT
    pref_ids = tokenizer(pref_text, add_special_tokens=False)["input_ids"]
    if bos is not None:
        pref_ids = [bos] + pref_ids
    chunks.append(Chunk(text=pref_text, token_ids=pref_ids, chunk_id=_stable_id(pref_text, pref_ids)))

    for d_text in doc_prompts:
        d_ids = tokenizer(d_text, add_special_tokens=False)["input_ids"]
        chunks.append(Chunk(text=d_text, token_ids=d_ids, chunk_id=_stable_id(d_text, d_ids)))

    suf_ids = tokenizer(query_suffix, add_special_tokens=False)["input_ids"]
    chunks.append(Chunk(text=query_suffix, token_ids=suf_ids, chunk_id=_stable_id(query_suffix, suf_ids)))

    structural_idx = [0, len(chunks) - 1]
    compressible_idx = list(range(1, len(chunks) - 1))

    full_text = pref_text + "".join(doc_prompts) + query_suffix
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if bos is not None:
        full_ids = [bos] + full_ids
    return chunks, structural_idx, compressible_idx, full_ids


# ─── F1 metric (same as loong) ───────────────────────────────────────────


def _normalize_answer(s: str) -> str:
    import string, re
    def remove_articles(s): return re.sub(r"\b(a|an|the)\b", " ", s)
    def white_space_fix(s): return " ".join(s.split())
    def remove_punc(s): return "".join(ch for ch in s if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _compute_f1(a_pred: str, a_gold: str, tokenizer) -> tuple[float, int]:
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


def _derive_valid_mask_pair(importance, target_ratio):
    if target_ratio >= 1.0:
        return torch.ones_like(importance, dtype=torch.bool)
    flat = importance.reshape(-1)
    n = max(int(flat.numel() * target_ratio) - 1, 0)
    score_sort = torch.sort(flat, descending=True).values
    thres = score_sort[n].item()
    return importance > thres


def _make_entry_at_ratio(cmp_full, target_ratio, n_kv_heads, head_dim):
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


def _gate_semantics(gate):
    if gate >= 1.0 or gate <= 0.0:
        return {
            "gate_is_active": False, "gate_label": "no_gate",
            "gate_semantics": "gate=1.0; importance gate disabled; HKVD top-k among retained eligible tokens",
        }
    return {
        "gate_is_active": True, "gate_label": f"gate={gate:.2f}",
        "gate_semantics": f"gate={gate:.2f}; importance gate active; HKVD top-k within importance-top-{int(gate*100)}% candidates",
    }


def _load_resume_set(raw_path: Path) -> set:
    done = set()
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


# ─── main ────────────────────────────────────────────────────────────────


def main() -> int:
    from cacheblend import LayerwiseModel
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
        "dataset": "MuSiQue",
        "musique_path": str(MUSIQUE_PATH),
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
        "official_metric": False,
        "metric_note": "token-F1 substring; same impl as MuSiQue paper.",
        "stage": "Stage 1 — diagnosis verification",
        "stage_notes": (
            "MuSiQue ~3K context: per-head mask 32×500×3000×2 = 96 MB < 256 MB cap → "
            "NO fallback. If F1 collapses at kvzip<1 here, root cause is NOT mask fallback."
        ),
        "chunk_policy": "[BOS+user_open+PREFIX](structural) + [docs](compressed) + [query+assistant_open](structural)",
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
        },
    }
    (OUT_DIR / "config_snapshot.json").write_text(json.dumps(config, indent=2))
    print(f"[grid] config saved", flush=True)

    print(f"[grid] loading MuSiQue from {MUSIQUE_PATH}", flush=True)
    questions, skip = _load_musique_questions(MUSIQUE_PATH, N_MAX)
    print(f"[grid] {len(questions)} questions; pre-skip: {dict(skip)}", flush=True)
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

    completed = _load_resume_set(raw_path) if RESUME else set()
    if completed:
        print(f"[grid] resume: {len(completed)} cells already done", flush=True)
    raw_fh = open(raw_path, "a", encoding="utf-8")
    skip_runtime = collections.Counter(skip)

    def _emit(rec):
        raw_fh.write(json.dumps(rec) + "\n"); raw_fh.flush()

    for q_idx, ex in enumerate(questions):
        stage = "build_chunks"
        try:
            doc_prompts = [f"{c['title']}\n\n{c['text']}\n\n" for c in ex["ctxs"]]
            q = ex["question"].strip()
            query_suffix = f"Question: {q}\nAnswer:"
            if INCLUDE_ASSISTANT_OPEN:
                query_suffix = query_suffix + assistant_open
            chunks, structural_idx, compressible_idx, full_ids = _build_chunks(
                tokenizer, user_open, doc_prompts, query_suffix,
            )

            stage = "tokenization_check"
            actual_ids = [t for c in chunks for t in c.token_ids]
            if actual_ids != full_ids:
                print(f"[Q{q_idx+1}] I1 boundary divergence: full={len(full_ids)} concat={len(actual_ids)}", flush=True)

            n_total = len(actual_ids)
            n_struct = sum(len(chunks[i].token_ids) for i in structural_idx)

            stage = "precompute"
            kv_base = KVStore()
            for c in chunks:
                K, V = precompute_chunk_kv(lw, c)
                kv_base.put(c.chunk_id, K, V)

            stage = "kvzip_compress"
            compressed_1p0 = {}
            for ci in compressible_idx:
                c = chunks[ci]
                ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
                cmp = backend.compress(ids, model=hf_model, budget=CompressionBudget(ratio=1.0))
                assert_compressed_storage_matches_tokens(f"q{q_idx+1}_chunk{ci}", cmp, c.token_ids)
                compressed_1p0[c.chunk_id] = cmp

        except TokenizationInvariantError as exc:
            print(f"[Q{q_idx+1} SKIP tokenization_mismatch] {str(exc).splitlines()[0]}", flush=True)
            skip_runtime["tokenization_mismatch"] += 1
            for kr in KVZIP_RATIO_LIST:
                for rr in RECOMP_RATIO_LIST:
                    if (ex["id"], kr, rr) in completed: continue
                    _emit({
                        "status": "skip", "skip_reason": "tokenization_mismatch", "stage": stage,
                        "example_id": ex["id"], "question_idx": q_idx + 1,
                        "kvzip_ratio": kr, "recompute_ratio": rr,
                        "exception_message": str(exc).splitlines()[0],
                    })
            if device.type == "cuda": torch.cuda.empty_cache()
            continue
        except Exception as exc:
            print(f"[Q{q_idx+1} FAIL stage={stage}] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            skip_runtime[f"{stage}_failed"] += 1
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

        for kr in KVZIP_RATIO_LIST:
            kv_r = KVStore()
            for c in chunks:
                if c.chunk_id in compressed_1p0:
                    kv_r._cache[c.chunk_id] = _make_entry_at_ratio(
                        compressed_1p0[c.chunk_id], kr, n_kv_heads, head_dim,
                    )
                else:
                    kv_r._cache[c.chunk_id] = kv_base.get(c.chunk_id)

            sample_entry = kv_r._cache[chunks[compressible_idx[0]].chunk_id]
            kvzip_valid_actual = float(sample_entry["valid_mask"].float().mean().item())

            for rr in RECOMP_RATIO_LIST:
                if (ex["id"], kr, rr) in completed: continue
                t_start = time.perf_counter()
                inner_stage = "fuse_generate"
                try:
                    cb_cfg = CompBlendConfig(
                        check_layer=CHECK_LAYER, recompute_ratio=rr,
                        selector=SELECTOR, gate_percentile=GATE_PCT,
                        chunk_normalization="rank",
                    )
                    if device.type == "cuda": torch.cuda.synchronize()
                    flags = {}; sel_stats = {}
                    out = fuse_selective_compblend(
                        lw, chunks, kv_r, cb_cfg,
                        return_layerwise_output=True, last_logits_only=True,
                        flags=flags, selector_stats=sel_stats,
                    )
                    inner_stage = "decode"
                    text = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device)
                    inner_stage = "f1_metric"
                    # MuSiQue has multiple gold answers — take max F1
                    f1_per_gold = [_compute_f1(text, a, tokenizer) for a in ex["answers"]]
                    f1 = max(x[0] for x in f1_per_gold)
                    em = max(x[1] for x in f1_per_gold)
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

                    rec = {
                        "status": "ok",
                        "example_id": ex["id"], "question_idx": q_idx + 1,
                        "context_tokens": n_total,
                        "kvzip_ratio": kr, "recompute_ratio": rr,
                        "selector": SELECTOR, "gate_percentile": GATE_PCT,
                        **gate_info,
                        "prediction": text, "gold_answers": ex["answers"],
                        "f1": f1, "exact_match": em,
                        "n_total_tokens": n_total,
                        "n_structural_tokens": n_struct,
                        "kvzip_ratio_requested": kr,
                        "kvzip_valid_pair_ratio_actual": kvzip_valid_actual,
                        "eligible_token_ratio_actual": sel_stats.get("eligible_ratio", -1.0),
                        "recompute_ratio_requested": rr,
                        "recompute_tokens_selected": sel_stats.get("selected_recompute_tokens", 0),
                        "recompute_ratio_actual": sel_stats.get("selected_ratio_actual", -1.0),
                        "forced_tokens_count": sel_stats.get("forced_tokens", 0),
                        "eligible_tokens_count": sel_stats.get("eligible_tokens", 0),
                        "per_head_mask_fallback_count": flags.get("per_head_mask_fallback_count", 0),
                        "per_head_mask_exact_count": flags.get("per_head_mask_exact_count", 0),
                        "elapsed_ms": elapsed_ms,
                    }
                    _emit(rec)
                    fb = flags.get("per_head_mask_fallback_count", 0)
                    print(
                        f"[Q{q_idx+1}/{len(questions)} kr={kr:.2f} rr={rr:.2f}] "
                        f"f1={f1:.3f} em={em} t={elapsed_ms:.0f}ms fallback={fb}",
                        flush=True,
                    )
                    del out
                    if device.type == "cuda": torch.cuda.empty_cache()
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    print(f"[Q{q_idx+1} kr={kr:.2f} rr={rr:.2f} FAIL stage={inner_stage}] "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    skip_runtime[f"{inner_stage}_failed"] += 1
                    _emit({
                        "status": "fail", "stage": inner_stage,
                        "example_id": ex["id"], "question_idx": q_idx + 1,
                        "kvzip_ratio": kr, "recompute_ratio": rr,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    })
                    if device.type == "cuda": torch.cuda.empty_cache()

    raw_fh.close()

    # Aggregate
    print("[grid] aggregating", flush=True)
    rows = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))

    by_cell = {}
    for r in rows:
        if r.get("status") != "ok": continue
        key = (float(r["kvzip_ratio"]), float(r["recompute_ratio"]))
        by_cell.setdefault(key, []).append(r)

    with open(summary_csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "kvzip_ratio", "recompute_ratio",
            "n_success", "n_failed",
            "mean_f1", "std_f1", "median_f1", "p10_f1", "p90_f1",
            "mean_exact_match",
            "mean_context_tokens",
            "mean_kvzip_valid_pair_ratio_actual",
            "mean_recompute_ratio_actual",
            "mean_fallback_count",
        ])
        for kr in KVZIP_RATIO_LIST:
            for rr in RECOMP_RATIO_LIST:
                cell = by_cell.get((kr, rr), [])
                fails = sum(
                    1 for r in rows
                    if r.get("status") != "ok"
                    and float(r.get("kvzip_ratio", -1)) == kr
                    and float(r.get("recompute_ratio", -1)) == rr
                )
                if not cell:
                    wr.writerow([kr, rr, 0, fails] + [""] * 10)
                    continue
                f1s = np.array([r["f1"] for r in cell], dtype=float)
                ems = np.array([r["exact_match"] for r in cell], dtype=float)
                ctx = np.array([r.get("context_tokens", 0) for r in cell], dtype=float)
                kvz = np.array([r.get("kvzip_valid_pair_ratio_actual", 0) for r in cell], dtype=float)
                rec = np.array([r.get("recompute_ratio_actual", 0) for r in cell], dtype=float)
                fb = np.array([r.get("per_head_mask_fallback_count", 0) for r in cell], dtype=float)
                wr.writerow([
                    kr, rr, len(cell), fails,
                    f"{f1s.mean():.4f}", f"{f1s.std():.4f}",
                    f"{np.median(f1s):.4f}",
                    f"{np.quantile(f1s, 0.10):.4f}", f"{np.quantile(f1s, 0.90):.4f}",
                    f"{ems.mean():.4f}",
                    f"{ctx.mean():.0f}",
                    f"{kvz.mean():.4f}", f"{rec.mean():.4f}",
                    f"{fb.mean():.1f}",
                ])
    print(f"[grid] wrote {summary_csv_path}", flush=True)

    grid_summary = {}
    for kr in KVZIP_RATIO_LIST:
        grid_summary[str(kr)] = {}
        for rr in RECOMP_RATIO_LIST:
            cell = by_cell.get((kr, rr), [])
            if cell:
                f1s = np.array([r["f1"] for r in cell], dtype=float)
                fb = np.array([r.get("per_head_mask_fallback_count", 0) for r in cell], dtype=float)
                grid_summary[str(kr)][str(rr)] = {
                    "n": len(cell),
                    "mean_f1": float(f1s.mean()),
                    "std_f1": float(f1s.std()),
                    "median_f1": float(np.median(f1s)),
                    "mean_fallback_count": float(fb.mean()),
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

    # Plots
    print("[grid] drawing plots", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[grid] matplotlib not installed; skipping plots", flush=True)
        print("MUSIQUE_GRID_DONE", flush=True)
        return 0

    M = np.full((len(KVZIP_RATIO_LIST), len(RECOMP_RATIO_LIST)), np.nan)
    for i, kr in enumerate(KVZIP_RATIO_LIST):
        for j, rr in enumerate(RECOMP_RATIO_LIST):
            cell = by_cell.get((kr, rr), [])
            if cell:
                M[i, j] = float(np.mean([r["f1"] for r in cell]))

    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, kr in enumerate(KVZIP_RATIO_LIST):
        color = cmap(i / max(len(KVZIP_RATIO_LIST) - 1, 1))
        ax.plot(RECOMP_RATIO_LIST, M[i], marker="o", label=f"kvzip={kr}", color=color, linewidth=2)
    ax.set_xlabel("Recompute ratio", fontsize=12)
    ax.set_ylabel("Mean token F1", fontsize=12)
    ax.set_title(f"MuSiQue F1 vs Recompute Ratio by KVzip Ratio (Llama-3.1-8B)", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"f1_vs_recompute_by_kvzip_ratio.{ext}", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for j, rr in enumerate(RECOMP_RATIO_LIST):
        color = cmap(j / max(len(RECOMP_RATIO_LIST) - 1, 1))
        ax.plot(KVZIP_RATIO_LIST, M[:, j], marker="o", label=f"recomp={rr}", color=color, linewidth=2)
    ax.set_xlabel("KVzip ratio (kept fraction)", fontsize=12)
    ax.set_ylabel("Mean token F1", fontsize=12)
    ax.set_title("MuSiQue F1 vs KVzip Ratio by Recompute Ratio (Llama-3.1-8B)", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"f1_vs_kvzip_by_recompute_ratio.{ext}", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(RECOMP_RATIO_LIST)))
    ax.set_xticklabels([f"{r}" for r in RECOMP_RATIO_LIST])
    ax.set_yticks(range(len(KVZIP_RATIO_LIST)))
    ax.set_yticklabels([f"{r}" for r in KVZIP_RATIO_LIST])
    ax.set_xlabel("Recompute ratio", fontsize=12)
    ax.set_ylabel("KVzip ratio (kept fraction)", fontsize=12)
    ax.set_title("MuSiQue F1 heatmap (Llama-3.1-8B)", fontsize=13)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                color = "white" if M[i, j] < 0.5 else "black"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label="Mean F1")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"f1_heatmap_kvzip_recompute.{ext}", dpi=120)
    plt.close(fig)

    print(f"[grid] plots written to {OUT_DIR}", flush=True)
    print("MUSIQUE_GRID_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
