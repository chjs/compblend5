"""Ratio sweep — F1 vs KVzip compression ratio at fixed recompute_ratio.

For each question:
  1. KVzipBackend.compress(ratio=1.0)  on every doc — captures full importance
  2. Precompute_chunk_kv on sys + query (uncompressed; for fuse path)
  3. For each target ratio r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
     a. Re-derive valid_mask = top-(r × M) per-(layer, head) by importance
     b. Run fuse_selective_compblend with selector ∈ {hkvd_only, gated_top_k}
     c. Greedy-decode → F1

Reuses M5's prompt layout and the same Llama-3.1-8B-Instruct setup, so the
ratio=0.10 result here is directly comparable to M5's measurement.

Output: a 16-row F1 matrix (1 full + 1 blend-no-kvzip + 7×2 kvzip×selector).
The 7×2 grid is the sweep; the 2 reference points let us anchor the curve.
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


MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DTYPE = os.environ.get("CACHEBLEND_DTYPE", "float16")
ATTN_IMPL = os.environ.get("CACHEBLEND_ATTN_IMPL", "sdpa")
RECOMP_RATIO = float(os.environ.get("COMPBLEND_RECOMP_RATIO", "0.15"))
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))
GATE_PCT = float(os.environ.get("COMPBLEND_GATE_PCT", "0.5"))
N_MAX = int(os.environ.get("COMPBLEND_N", "50"))
OUT_PATH = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "musique_ratio_sweep.json")))
MAX_NEW_TOKENS = 32

# The ratios to sweep. 1.0 means no eviction; 0.1 is M5's setting.
SWEEP_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
SWEEP_SELECTORS = ["hkvd_only", "gated_top_k"]

_WRAPPERS = {
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "mistral": ("[INST]", "[/INST]"),
}
PREFIX_PROMPT = (
    "You will be asked a question after reading several passages. Please "
    "directly answer the question based on the given passages. Do NOT repeat "
    "the question. The answer should be within 5 words..\nPassages:\n"
)
QUERY_PROMPT = (
    "\n\nAnswer the question directly based on the given passages. Do NOT "
    "repeat the question. The answer should be within 5 words. \nQuestion:"
)


# ── inlined musique helpers (see musique_selector_compare.py for rationale) ──


def _normalize_question(q: str) -> str:
    if not q.endswith("?"):
        q = q + "?"
    return q[0].lower() + q[1:]


def _parse_generation(s: str) -> str:
    s = s.lstrip("\n").split("\n")[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif s.split() and (s.split()[0]).startswith(("No", "no")):
        s = "No"
    return s


def _normalize_answer(s: str) -> str:
    import re as _re
    import string as _string
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(_string.punctuation))
    s = _re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _compute_f1(a_pred: str, a_gold: str, tokenizer) -> float:
    import collections as _coll
    a_pred = _parse_generation(a_pred)
    gold_toks = tokenizer.encode(_normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(_normalize_answer(a_pred))[1:]
    common = _coll.Counter(gold_toks) & _coll.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(int(gold_toks == pred_toks))
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def _resolve_wrapper(model_id: str, tokenizer):
    mid = model_id.lower()
    for key, wrap in _WRAPPERS.items():
        if key in mid:
            return wrap
    raise RuntimeError(f"add wrapper for {model_id}")


def _build_chunks(tokenizer, chunk_texts: list[str]):
    from cacheblend.chunker import Chunk, _stable_id
    bos = tokenizer.bos_token_id
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
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_first = time.perf_counter()
    generated = [int(next_id.item())]
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS - 1):
            if eos is not None and generated[-1] == eos:
                break
            step = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = step.past_key_values
            next_id = step.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated.append(int(next_id.item()))
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, t_first - t_start


def _derive_valid_mask_for_ratio(importance: torch.Tensor, target_ratio: float) -> torch.Tensor:
    """Per-(layer, head) top-(target_ratio × M) by importance.

    Replicates KVzip's `prune(ratio=r, level='pair')` semantics on a chunk
    that was originally compressed at ratio=1.0. Importance values were
    captured at the time of compress() so the score distribution is
    identical to a fresh prune.

    Args:
        importance: [L, H_kv, M] fp32 — per-(layer, head, position) salience.
        target_ratio: in (0, 1]. Fraction to KEEP.

    Returns:
        bool [L, H_kv, M] — True where (importance >= per-(L,H) threshold).
    """
    if not (0.0 < target_ratio <= 1.0):
        raise ValueError(f"target_ratio in (0, 1], got {target_ratio}")
    if target_ratio == 1.0:
        return torch.ones_like(importance, dtype=torch.bool)
    L, H, M = importance.shape
    k = max(int(round(M * target_ratio)), 1)
    # k-th largest per (L, H) = threshold
    sorted_imp, _ = torch.sort(importance, dim=-1, descending=True)
    threshold = sorted_imp[..., k - 1:k]                              # [L, H, 1]
    return importance >= threshold


def main() -> int:
    from cacheblend import LayerwiseModel
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.fusor import fuse_full_recompute, fuse_selective

    from compblend.backends.base import (
        CompressedChunk, CompressionBudget, to_kvstore_entry,
    )
    from compblend.backends.kvzip import KVzipBackend, KVzipConfig
    from compblend.config import CompBlendConfig
    from compblend.fuse_selective_compblend import fuse_selective_compblend

    print(f"[sweep] model={MODEL}  dtype={DTYPE}  attn={ATTN_IMPL}", flush=True)
    print(f"[sweep] recomp_ratio={RECOMP_RATIO}  check_layer={CHECK_LAYER}  "
          f"gate_pct={GATE_PCT}  n={N_MAX}", flush=True)
    print(f"[sweep] sweep_ratios={SWEEP_RATIOS}  selectors={SWEEP_SELECTORS}", flush=True)

    # KVzip backend loads its own model. We re-use its weights for LayerwiseModel
    # by sharing the HF model instance — avoids a second 16GB load.
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

    dataset_path = _REPO / "src" / "external" / "cacheblend-hf-v7" / "benchmarks" / "musique" / "inputs" / "musique_s.json"
    with open(dataset_path) as f:
        eval_dataset = json.load(f)[:N_MAX]
    print(f"[sweep] running {len(eval_dataset)} examples", flush=True)

    # Per-question F1 storage.
    # Keys: "full", "blend_no_kvzip", (ratio, selector) for kvzip arms.
    per_question: dict[Any, list[float]] = {
        "full": [], "blend_no_kvzip": [],
    }
    for r in SWEEP_RATIOS:
        for s in SWEEP_SELECTORS:
            per_question[(r, s)] = []
    answers_full = []

    for idx, ex in enumerate(eval_dataset):
        answers = ex["answers"]
        q = _normalize_question(ex["question"])
        doc_prompts = [f"{c['title']}\n\n{c['text']}\n\n" for c in ex["ctxs"]]
        q_prompt = f"{QUERY_PROMPT}{q}\nAnswer:"

        chunk_texts_list = [user_open + PREFIX_PROMPT, *doc_prompts, q_prompt + assistant_open]
        chunks = _build_chunks(tokenizer, chunk_texts_list)
        doc_slice = slice(1, 1 + len(doc_prompts))

        # ── precompute sys + query (uncompressed) ──────────────────────
        kv_base = KVStore()
        for c in chunks:
            K, V = precompute_chunk_kv(lw, c)
            kv_base.put(c.chunk_id, K, V)

        # ── arm: full prefill (reference; ratio=1.0 logical equivalent) ──
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
        res, _ = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
        f1_full = max(_compute_f1(res, a, tokenizer) for a in answers)
        per_question["full"].append(f1_full)
        answers_full.append(res)

        # ── arm: blend NO kvzip (v7 fuse_selective on uncompressed) ─────
        out = fuse_selective(
            lw, chunks, kv_base, recompute_ratio=RECOMP_RATIO,
            check_layer=CHECK_LAYER, return_layerwise_output=True,
        )
        res, _ = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
        f1_b = max(_compute_f1(res, a, tokenizer) for a in answers)
        per_question["blend_no_kvzip"].append(f1_b)

        # ── compress every doc at ratio=1.0 (full importance, no eviction) ──
        compressed_docs: dict[str, CompressedChunk] = {}
        for c in chunks[doc_slice]:
            ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
            cmp = backend.compress(
                ids, model=hf_model,
                budget=CompressionBudget(ratio=1.0),
            )
            compressed_docs[c.chunk_id] = cmp

        # ── sweep target ratios × selectors ──
        per_q_summary = {"full": f1_full, "blend_no_kvzip": f1_b}
        for r in SWEEP_RATIOS:
            # Re-derive valid_mask for this ratio (per-(L,H) top fraction r).
            kv_r = KVStore()
            # sys + query: uncompressed entries (re-use kv_base)
            for c in chunks:
                if c.chunk_id in compressed_docs:
                    continue
                kv_r._cache[c.chunk_id] = kv_base.get(c.chunk_id)
            # doc chunks: rebuild entry with masked valid for ratio r
            for c in chunks[doc_slice]:
                cmp = compressed_docs[c.chunk_id]
                entry = to_kvstore_entry(cmp)
                if r < 1.0:
                    new_valid = _derive_valid_mask_for_ratio(cmp.importance, r)
                    entry = dict(entry)
                    entry["valid_mask"] = new_valid
                kv_r._cache[c.chunk_id] = entry

            for sel in SWEEP_SELECTORS:
                cfg = CompBlendConfig(
                    check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
                    selector=sel, gate_percentile=GATE_PCT,
                    chunk_normalization="rank",
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fuse_selective_compblend(
                    lw, chunks, kv_r, cfg, return_layerwise_output=True,
                )
                res, _ = _greedy_decode(
                    lw.model, tokenizer, out.logits, out.past_key_values, device, t0,
                )
                f1 = max(_compute_f1(res, a, tokenizer) for a in answers)
                per_question[(r, sel)].append(f1)
                per_q_summary[f"r{r}_{sel[:4]}"] = f1

        # Compact per-question print
        line = f"[{idx + 1}/{len(eval_dataset)}] " + "  ".join(
            f"{k}={v:.2f}" for k, v in per_q_summary.items()
        )
        print(line, flush=True)

    # ── aggregate ──
    def _mean(xs): return float(np.mean(xs)) if xs else float("nan")
    summary: dict[str, Any] = {
        "config": {
            "model": MODEL, "dtype": DTYPE, "attn_impl": ATTN_IMPL,
            "recomp_ratio": RECOMP_RATIO, "check_layer": CHECK_LAYER,
            "gate_pct": GATE_PCT, "n": len(eval_dataset),
            "sweep_ratios": SWEEP_RATIOS,
            "sweep_selectors": SWEEP_SELECTORS,
            "kvzip_level": "pair",
            "rederive_strategy": "per-(L,H) top-k by importance, applied to ratio=1.0 corpus",
        },
        "f1": {
            "full_mean": _mean(per_question["full"]),
            "blend_no_kvzip_mean": _mean(per_question["blend_no_kvzip"]),
            "kvzip_grid": {},
        },
        "n_per_arm": len(eval_dataset),
    }
    # 7×2 grid: outer key = ratio, inner key = selector
    for r in SWEEP_RATIOS:
        row = {}
        for s in SWEEP_SELECTORS:
            row[s] = _mean(per_question[(r, s)])
        # paired Δ
        if per_question[(r, "hkvd_only")] and per_question[(r, "gated_top_k")]:
            d = np.array(per_question[(r, "gated_top_k")]) - np.array(per_question[(r, "hkvd_only")])
            row["delta_gated_minus_hkvd_mean"] = float(np.mean(d))
            # bootstrap CI
            rng = np.random.default_rng(seed=42)
            n = len(d)
            boot = np.array([np.mean(d[rng.integers(0, n, size=n)]) for _ in range(1000)])
            ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
            row["delta_ci_95"] = [float(ci_lo), float(ci_hi)]
            row["delta_significant"] = bool(ci_lo > 0 or ci_hi < 0)
        summary["f1"]["kvzip_grid"][str(r)] = row

    print("\n──────── Summary ────────")
    print(json.dumps(summary, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[sweep] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
