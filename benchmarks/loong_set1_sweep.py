"""Loong Set-1 (10K-50K context) ratio × gate sweep.

Why Loong: MuSiQue's ~7K context leaves only ~700 tokens after KVzip
r=0.10 — below the model's reasoning floor, causing F1 collapse to ~0.07.
Loong's level=1 set=1 contexts run 10K-50K, leaving 1K-5K kept tokens
under aggressive compression — enough headroom for CompBlend's compression
+ blending complementarity to actually show up.

Dataset:
  Source: https://github.com/MozerWang/Loong  (metadata jsonl)
         https://modelscope.cn/datasets/iic/Loong  (actual doc texts)
  Filter: level=1 (Spotlight Locating), set=1 (10K-50K tokens), language=en
  Expected rows: ~30-50 questions

Metric:
  Loong's official metric is LLM-as-judge (GPT-4 calls $$). We use
  substring/token-F1 instead — works for level=1's short numeric/string
  answers (e.g., "-$0.04"). Relative comparison across selectors still
  meaningful; absolute scores not directly comparable to Loong paper.

Required env on the pod:
    MODELSCOPE_LOONG_CACHE   path to ModelScope-downloaded Loong corpus
                              (default /root/loong_data)
                              Layout: financial/*.txt, paper/<name>, legal/legal.json
    CACHEBLEND_MODEL          Llama-3.1-8B-Instruct
    COMPBLEND_N               max questions (default 30)
    COMPBLEND_RECOMP_RATIO    0.15
"""
from __future__ import annotations

import glob
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


MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DTYPE = os.environ.get("CACHEBLEND_DTYPE", "float16")
ATTN_IMPL = os.environ.get("CACHEBLEND_ATTN_IMPL", "sdpa")
RECOMP_RATIO = float(os.environ.get("COMPBLEND_RECOMP_RATIO", "0.15"))
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))
N_MAX = int(os.environ.get("COMPBLEND_N", "30"))
LOONG_DATA_DIR = Path(os.environ.get("MODELSCOPE_LOONG_CACHE", "/root/loong_data/doc"))
LOONG_JSONL = Path(os.environ.get(
    "LOONG_JSONL", "/root/Loong/data/loong.jsonl",
))
LOONG_LEVELS = [int(x) for x in os.environ.get("LOONG_LEVELS", "1,2").split(",")]
LOONG_SETS = [int(x) for x in os.environ.get("LOONG_SETS", "2").split(",")]
LOONG_MAX_LENGTH = int(os.environ.get("LOONG_MAX_LENGTH", "32000"))  # safety cap for OOM
OUT_PATH = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "loong_set1_sweep.json")))
MAX_NEW_TOKENS = 64    # Loong answers can be slightly longer than MuSiQue's

# Phase 1 keeps the sweep modest — 6 ratios × 3 gates = 18 grid cells per question.
SWEEP_RATIOS = [0.10, 0.20, 0.30, 0.50, 0.70]
SWEEP_GATES = [0.3, 0.5, 1.0]


_WRAPPERS = {
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
}


# ── F1 metric (substring/token-level, same as MuSiQue) ────────────────────


def _normalize_answer(s: str) -> str:
    import re as _re
    import string as _string
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(_string.punctuation))
    s = _re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _compute_f1(a_pred: str, a_gold: str, tokenizer) -> float:
    """Token-level F1 over normalized answers. Same impl as MuSiQue."""
    import collections as _coll
    a_pred = a_pred.strip().split("\n")[0]
    gold_toks = tokenizer.encode(_normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(_normalize_answer(a_pred))[1:]
    common = _coll.Counter(gold_toks) & _coll.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(int(gold_toks == pred_toks))
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks); r = num_same / len(gold_toks)
    return (2 * p * r) / (p + r)


# ── Loong doc loader (ports MozerWang/Loong/src/utils/prompt.py:get_content) ──


def _load_doc_content(item: dict, doc_name: str, idx: int, doc_path: Path) -> str | None:
    """Read one document's text from the Loong corpus on disk.

    Returns None if not found (caller skips this question — rare).
    """
    doc_type = item["type"]
    doc_level = item["level"]
    type_dir = doc_path / doc_type

    if doc_type == "financial":
        # Filename pattern: "<...>-2024-<doc_name><...>.txt"
        if str(doc_level).strip() != "4":
            pattern = f"{type_dir}/*2024-{doc_name}*.txt"
        else:
            pattern = f"{type_dir}/*{doc_name}*.txt"
        matches = glob.glob(pattern)
        if not matches:
            return None
        with open(matches[0]) as f:
            stem = Path(matches[0]).stem.split("-")[-1]
            return f"《{stem}》\n" + f.read() + "\n\n"

    elif doc_type == "paper":
        path = type_dir / doc_name
        if not path.exists():
            return None
        with open(path) as f:
            content = f.read()
        name = content.split("\n", 1)[0].strip("#").strip()
        return f"{name}\n" + content + "\n\n"

    elif doc_type == "legal":
        legal_json = type_dir / "legal.json"
        if not legal_json.exists():
            return None
        with open(legal_json) as f:
            legal = json.load(f)
        if doc_name not in legal:
            return None
        entry = legal[doc_name]
        if doc_level == 4 and ("阅读以上判决文书" in item.get("instruction", "")):
            content = entry["content"]
        else:
            content = entry["content"] + entry["result"]
        return f"《判决文书{idx + 1}》\n" + content + "\n\n"

    return None


def _load_loong_questions(
    jsonl_path: Path, doc_path: Path,
    levels: list[int], set_ids: list[int], language: str,
    max_n: int,
) -> list[dict]:
    """Filter Loong jsonl + materialize doc texts. Skip rows with missing files.

    Accepts lists of levels and set ids so we can pool e.g. level=1+2, set=2
    when individual cells are too small for statistical comparison.
    Only str-typed answers are included (substring F1 won't work for dict/list).
    """
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
            if d.get("length", 0) > LOONG_MAX_LENGTH: continue
            rows.append(d)
    print(f"[loong] filtered to {len(rows)} rows "
          f"(levels={levels}, sets={set_ids}, lang={language}, answer=str)", flush=True)

    materialized = []
    skipped_count = 0
    for r in rows:
        contents = []
        all_loaded = True
        for idx, doc_name in enumerate(r["doc"]):
            txt = _load_doc_content(r, doc_name, idx, doc_path)
            if txt is None:
                all_loaded = False
                break
            contents.append(txt)
        if not all_loaded:
            skipped_count += 1
            continue
        materialized.append({
            "id": r["id"],
            "question": r["question"],
            "instruction": r["instruction"],
            "prompt_template": r["prompt_template"],
            "doc_contents": contents,
            "answer": r["answer"],
            "type": r["type"],
            "length": r.get("length", -1),
        })
        if len(materialized) >= max_n:
            break

    print(f"[loong] {len(materialized)} questions ready  ({skipped_count} skipped: docs not found)", flush=True)
    return materialized


# ── chunk construction (Loong-specific prompt format) ────────────────────


def _build_chunks(tokenizer, sys_text: str, doc_contents: list[str], query_text: str):
    """Build Chunk list matching Loong's prompt_template '{docs}\n\n{instruction}\n\n{question}'.

    Each doc is its own chunk (KV will be compressed per chunk).
    sys_text is empty for Loong (no system prompt convention).
    Query is the instruction + question concatenated.
    """
    from cacheblend.chunker import Chunk, _stable_id

    bos = tokenizer.bos_token_id
    chunk_texts = []
    if sys_text:
        chunk_texts.append(sys_text)
    chunk_texts.extend(doc_contents)
    chunk_texts.append(query_text)

    chunks = []
    for i, text in enumerate(chunk_texts):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if i == 0 and bos is not None:
            ids = [bos] + ids
        chunks.append(Chunk(text=text, token_ids=ids, chunk_id=_stable_id(text, ids)))
    return chunks


def _greedy_decode(model, tokenizer, prefill_logits, past_kv, device, t_start):
    eos = getattr(tokenizer, "eos_token_id", None)
    next_id = prefill_logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    if device.type == "cuda": torch.cuda.synchronize()
    t_first = time.perf_counter()
    generated = [int(next_id.item())]
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS - 1):
            if eos is not None and generated[-1] == eos: break
            step = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = step.past_key_values
            next_id = step.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated.append(int(next_id.item()))
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, t_first - t_start


def _resolve_wrapper(model_id, tokenizer):
    mid = model_id.lower()
    for key, wrap in _WRAPPERS.items():
        if key in mid: return wrap
    raise RuntimeError(f"add wrapper for {model_id}")


# ── KVzip pair-level derive (same as musique_ratio_gate_sweep.py) ────────


def _derive_valid_mask_pair(importance: torch.Tensor, target_ratio: float) -> torch.Tensor:
    if target_ratio >= 1.0:
        return torch.ones_like(importance, dtype=torch.bool)
    flat = importance.reshape(-1)
    n = max(int(flat.numel() * target_ratio) - 1, 0)
    score_sort = torch.sort(flat, descending=True).values
    thres = score_sort[n].item()
    return importance > thres


def _make_entry_at_ratio(cmp_full, target_ratio, n_kv_heads, head_dim):
    """Derive valid_mask at target_ratio + zero-fill K/V at evicted slots."""
    # Storage length (from the tensor itself) is the source of truth — KVzip's
    # tokenizer can produce a ctx_len that differs from len(token_ids) by a
    # special-token offset, and importance/valid_mask are sized to storage.
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


def main() -> int:
    from cacheblend import LayerwiseModel
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.fusor import fuse_full_recompute

    from compblend.backends.base import CompressionBudget
    from compblend.backends.kvzip import KVzipBackend, KVzipConfig
    from compblend.config import CompBlendConfig
    from compblend.fuse_selective_compblend import fuse_selective_compblend

    print(f"[loong-set1] model={MODEL}  recomp_ratio={RECOMP_RATIO}  n={N_MAX}", flush=True)
    print(f"[loong-set1] sweep ratios={SWEEP_RATIOS}  gates={SWEEP_GATES}", flush=True)
    print(f"[loong-set1] loong_data_dir={LOONG_DATA_DIR}  jsonl={LOONG_JSONL}", flush=True)

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

    user_open, assistant_open = _resolve_wrapper(MODEL, tokenizer)

    # Load Loong subset
    eval_dataset = _load_loong_questions(
        LOONG_JSONL, LOONG_DATA_DIR,
        levels=LOONG_LEVELS, set_ids=LOONG_SETS, language="en", max_n=N_MAX,
    )
    if len(eval_dataset) == 0:
        print("FATAL: no Loong questions loaded. Check LOONG_JSONL and MODELSCOPE_LOONG_CACHE.")
        return 1

    f1_full = []
    f1_arms: dict[tuple[float, str], list[float]] = {}
    for r in SWEEP_RATIOS:
        f1_arms[(r, "hkvd_only")] = []
        for g in SWEEP_GATES:
            f1_arms[(r, f"gate={g}")] = []

    for idx, ex in enumerate(eval_dataset):
      try:
        # Build prompt parts. Loong template = '{docs}\n\n{instruction}\n\n{question}'.
        # We chunk by: sys (empty) + each doc + query (instruction+question wrapped).
        query_text = f"\n\n{ex['instruction']}\n\n{ex['question']}{assistant_open}"
        # First chunk gets user_open prepended; rest are docs.
        first_doc = user_open + ex["doc_contents"][0]
        rest_docs = ex["doc_contents"][1:]
        chunks = _build_chunks(tokenizer, "", [first_doc, *rest_docs], query_text)
        doc_slice = slice(0, len(ex["doc_contents"]))

        # Precompute sys/query (we put doc chunks at indices 0..N-1, query at last)
        # But since "sys is empty" here, ALL doc chunks + query chunk need precomputing.
        kv_base = KVStore()
        for c in chunks:
            K, V = precompute_chunk_kv(lw, c)
            kv_base.put(c.chunk_id, K, V)

        # arm: full_recompute
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
        res, _ = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
        f1_full_q = _compute_f1(res, ex["answer"], tokenizer) if isinstance(ex["answer"], str) else 0.0
        f1_full.append(f1_full_q)
        del out
        if device.type == "cuda": torch.cuda.empty_cache()

        # Compress doc chunks once at ratio=1.0
        compressed_1p0 = {}
        skip_question = False
        for c in chunks[doc_slice]:
            ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
            cmp = backend.compress(
                ids, model=hf_model, budget=CompressionBudget(ratio=1.0),
            )
            # KVzip decode/encode roundtrip can shift token count by 1+ for some
            # texts (non-bijective tokenizer). The fuser overlays chunk KV onto
            # positions [start, start + len(token_ids)] so a length mismatch
            # corrupts position alignment. Skip the question rather than risk it.
            if cmp.key_cache[0].shape[1] != len(c.token_ids):
                print(
                    f"[skip Q{idx+1}] chunk storage={cmp.key_cache[0].shape[1]} "
                    f"!= len(token_ids)={len(c.token_ids)} (KVzip tokenizer drift)",
                    flush=True,
                )
                skip_question = True
                break
            compressed_1p0[c.chunk_id] = cmp
        if skip_question:
            if device.type == "cuda": torch.cuda.empty_cache()
            continue

        per_q_summary = {"full": f1_full_q}
        for r in SWEEP_RATIOS:
            kv_r = KVStore()
            for c in chunks:
                if c.chunk_id in compressed_1p0:
                    kv_r._cache[c.chunk_id] = _make_entry_at_ratio(
                        compressed_1p0[c.chunk_id], r, n_kv_heads, head_dim,
                    )
                else:
                    kv_r._cache[c.chunk_id] = kv_base.get(c.chunk_id)

            # arm: hkvd_only
            cb_cfg = CompBlendConfig(
                check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
                selector="hkvd_only", chunk_normalization="rank",
            )
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fuse_selective_compblend(
                lw, chunks, kv_r, cb_cfg,
                return_layerwise_output=True, last_logits_only=True,
            )
            res, _ = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
            f1_h = _compute_f1(res, ex["answer"], tokenizer) if isinstance(ex["answer"], str) else 0.0
            f1_arms[(r, "hkvd_only")].append(f1_h)
            per_q_summary[f"r{r}_h"] = f1_h
            del out
            if device.type == "cuda": torch.cuda.empty_cache()

            for g in SWEEP_GATES:
                cb_cfg = CompBlendConfig(
                    check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
                    selector="gated_top_k", gate_percentile=g,
                    chunk_normalization="rank",
                )
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fuse_selective_compblend(
                    lw, chunks, kv_r, cb_cfg,
                    return_layerwise_output=True, last_logits_only=True,
                )
                res, _ = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
                f1 = _compute_f1(res, ex["answer"], tokenizer) if isinstance(ex["answer"], str) else 0.0
                f1_arms[(r, f"gate={g}")].append(f1)
                per_q_summary[f"r{r}_g{g}"] = f1
                del out
                if device.type == "cuda": torch.cuda.empty_cache()

        print(
            f"[{idx + 1}/{len(eval_dataset)}] type={ex['type'][:3]} ctx_len={ex.get('length',-1):>6}  "
            f"full={f1_full_q:.2f}  " +
            "  ".join(
                f"r{r}:h={per_q_summary[f'r{r}_h']:.2f}/best={max(per_q_summary[f'r{r}_g{g}'] for g in SWEEP_GATES):.2f}"
                for r in SWEEP_RATIOS
            ),
            flush=True,
        )
      except Exception as exc:
        import traceback as _tb
        print(f"[Q{idx+1} FAILED] {type(exc).__name__}: {exc}", flush=True)
        _tb.print_exc()
        if device.type == "cuda": torch.cuda.empty_cache()
        continue

    # aggregate
    def _mean(xs): return float(np.mean(xs)) if xs else float("nan")
    def _boot_ci(deltas, n_boot=1000, seed=42):
        if len(deltas) == 0: return [0.0, 0.0], False
        rng = np.random.default_rng(seed)
        n = len(deltas)
        boot = np.array([np.mean(deltas[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        return [float(lo), float(hi)], bool(lo > 0 or hi < 0)

    summary: dict[str, Any] = {
        "config": {
            "model": MODEL, "dtype": DTYPE, "attn_impl": ATTN_IMPL,
            "recomp_ratio": RECOMP_RATIO, "check_layer": CHECK_LAYER,
            "n": len(eval_dataset),
            "sweep_ratios": SWEEP_RATIOS, "sweep_gates": SWEEP_GATES,
            "dataset": f"Loong levels={LOONG_LEVELS} sets={LOONG_SETS} lang=en (str answers only)",
            "kvzip_level": "pair",
            "valid_mask_source": "_derive_valid_mask_pair (verbatim from KVzip._threshold)",
            "metric": "token-F1 (substring) — NOT Loong's official LLM-judge",
        },
        "f1": {"full_mean": _mean(f1_full), "grid": {}},
        "n_per_arm": len(f1_full),
    }
    for r in SWEEP_RATIOS:
        hkvd = np.array(f1_arms[(r, "hkvd_only")])
        row = {"hkvd_only_mean": float(np.mean(hkvd)) if len(hkvd) else float("nan"), "gates": {}}
        for g in SWEEP_GATES:
            gated = np.array(f1_arms[(r, f"gate={g}")])
            deltas = gated - hkvd if len(gated) else np.zeros(0)
            ci, sig = _boot_ci(deltas) if len(deltas) else ([0.0, 0.0], False)
            row["gates"][str(g)] = {
                "mean": float(np.mean(gated)) if len(gated) else float("nan"),
                "delta_minus_hkvd_mean": float(np.mean(deltas)) if len(deltas) else 0.0,
                "delta_ci_95": ci, "delta_significant": sig,
            }
        summary["f1"]["grid"][str(r)] = row

    print("\n──────── Summary ────────")
    print(json.dumps(summary, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[loong-set1] wrote {OUT_PATH}")
    # Sentinel for polling — exact match.
    print("LOONG_BENCH_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
