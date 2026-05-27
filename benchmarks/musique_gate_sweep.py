"""Gate-percentile sweep for Gated HKVD at fixed KVzip ratio.

Purpose
-------
After M5 and the (mis-derived) ratio sweep, the next open question is:
*what gate_percentile maximizes Gated HKVD's quality at our default
compression setting?*

The eligible_mask plumbing (added 2026-05-27) now strictly confines
gated_top_k to kept positions (union over heads). gate_percentile picks
the top fraction by importance WITHIN that kept set, then HKVD chooses
recompute_k tokens within the gated subset. Choosing the right gate is
purely empirical.

Workload
--------
Same MuSiQue setup as M5. For each question:
    1.  Precompute sys/query via v7 precompute_chunk_kv.
    2.  KVzip.compress each doc at ratio=0.10 (in memory).
    3.  Run each selector arm:
            - full_recompute                 (no blending, reference)
            - kvzip + hkvd_only              (naive baseline; ignores eligible_mask)
            - kvzip + gated_top_k @ gate ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
    4.  Collect F1 + TTFT per arm.

Output
------
F1 per (gate value), paired bootstrap CI vs hkvd_only. The best gate is
the one with the largest ΔF1 over hkvd_only with significant CI.

Env knobs
---------
    CACHEBLEND_MODEL          HF model id              (default Llama-3.1-8B-Instruct)
    COMPBLEND_RATIO_KVZIP     KVzip ratio              (default 0.10)
    COMPBLEND_RECOMP_RATIO    fuse recompute ratio     (default 0.15)
    COMPBLEND_N               # questions              (default 50)
    COMPBLEND_OUT             output path              (default logs/gate_sweep.json)
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
KVZIP_RATIO = float(os.environ.get("COMPBLEND_RATIO_KVZIP", "0.10"))
RECOMP_RATIO = float(os.environ.get("COMPBLEND_RECOMP_RATIO", "0.15"))
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))
N_MAX = int(os.environ.get("COMPBLEND_N", "50"))
OUT_PATH = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "gate_sweep.json")))
MAX_NEW_TOKENS = 32

SWEEP_GATES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

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


# inlined helpers (avoid musique utils name collision with KVzip's utils/)

def _normalize_question(q):
    if not q.endswith("?"): q = q + "?"
    return q[0].lower() + q[1:]

def _parse_generation(s):
    s = s.lstrip("\n").split("\n")[0]
    if s.startswith("Yes") or s.startswith("yes"): s = "Yes"
    elif s.split() and (s.split()[0]).startswith(("No", "no")): s = "No"
    return s

def _normalize_answer(s):
    import re as _re, string as _string
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(_string.punctuation))
    s = _re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def _compute_f1(a_pred, a_gold, tokenizer):
    import collections as _coll
    a_pred = _parse_generation(a_pred)
    gold_toks = tokenizer.encode(_normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(_normalize_answer(a_pred))[1:]
    common = _coll.Counter(gold_toks) & _coll.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(int(gold_toks == pred_toks))
    if num_same == 0: return 0.0
    p = num_same / len(pred_toks); r = num_same / len(gold_toks)
    return (2 * p * r) / (p + r)

def _resolve_wrapper(model_id, tokenizer):
    mid = model_id.lower()
    for key, wrap in _WRAPPERS.items():
        if key in mid: return wrap
    raise RuntimeError(f"add wrapper for {model_id}")

def _build_chunks(tokenizer, chunk_texts):
    from cacheblend.chunker import Chunk, _stable_id
    bos = tokenizer.bos_token_id
    chunks = []
    for i, text in enumerate(chunk_texts):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if i == 0 and bos is not None: ids = [bos] + ids
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


def main() -> int:
    from cacheblend import LayerwiseModel
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.fusor import fuse_full_recompute

    from compblend.backends.base import (
        CompressedChunk, CompressionBudget, to_kvstore_entry,
    )
    from compblend.backends.kvzip import KVzipBackend, KVzipConfig
    from compblend.config import CompBlendConfig
    from compblend.fuse_selective_compblend import fuse_selective_compblend

    print(f"[gate-sweep] model={MODEL}  kvzip_ratio={KVZIP_RATIO}  "
          f"recomp_ratio={RECOMP_RATIO}  n={N_MAX}", flush=True)
    print(f"[gate-sweep] gates={SWEEP_GATES}", flush=True)

    # Share weights with KVzip's wrapped model — avoids second 16GB load.
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
    print(f"[gate-sweep] running {len(eval_dataset)} examples", flush=True)

    f1_full = []
    f1_hkvd = []
    f1_gates = {g: [] for g in SWEEP_GATES}
    ttft_full, ttft_hkvd = [], []
    ttft_gates = {g: [] for g in SWEEP_GATES}

    for idx, ex in enumerate(eval_dataset):
        answers = ex["answers"]
        q = _normalize_question(ex["question"])
        doc_prompts = [f"{c['title']}\n\n{c['text']}\n\n" for c in ex["ctxs"]]
        q_prompt = f"{QUERY_PROMPT}{q}\nAnswer:"
        chunk_texts_list = [user_open + PREFIX_PROMPT, *doc_prompts, q_prompt + assistant_open]
        chunks = _build_chunks(tokenizer, chunk_texts_list)
        doc_slice = slice(1, 1 + len(doc_prompts))

        # ── precompute sys+query (v7 path) ─────────────────────────────
        kv_base = KVStore()
        for c in chunks:
            K, V = precompute_chunk_kv(lw, c)
            kv_base.put(c.chunk_id, K, V)

        # ── arm: full_recompute (reference) ───────────────────────────
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
        res, t = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
        f1_full.append(max(_compute_f1(res, a, tokenizer) for a in answers))
        ttft_full.append(t)

        # ── compress every doc once at the configured KVzip ratio ──────
        compressed = {}
        for c in chunks[doc_slice]:
            ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
            cmp = backend.compress(
                ids, model=hf_model, budget=CompressionBudget(ratio=KVZIP_RATIO),
            )
            compressed[c.chunk_id] = cmp

        # Build kv_store with kvzip-compressed docs + uncompressed sys/query.
        kv_kv = KVStore()
        for c in chunks:
            if c.chunk_id in compressed:
                kv_kv._cache[c.chunk_id] = to_kvstore_entry(compressed[c.chunk_id])
            else:
                kv_kv._cache[c.chunk_id] = kv_base.get(c.chunk_id)

        # ── arm: hkvd_only (naive baseline, ignores eligible_mask) ─────
        cfg = CompBlendConfig(
            check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
            selector="hkvd_only", chunk_normalization="rank",
        )
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fuse_selective_compblend(lw, chunks, kv_kv, cfg, return_layerwise_output=True)
        res, t = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
        f1_hkvd.append(max(_compute_f1(res, a, tokenizer) for a in answers))
        ttft_hkvd.append(t)

        # ── arms: gated_top_k @ each gate value ────────────────────────
        per_q_gates = {}
        for g in SWEEP_GATES:
            cfg = CompBlendConfig(
                check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
                selector="gated_top_k", gate_percentile=g,
                chunk_normalization="rank",
            )
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fuse_selective_compblend(lw, chunks, kv_kv, cfg, return_layerwise_output=True)
            res, t = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
            f1 = max(_compute_f1(res, a, tokenizer) for a in answers)
            f1_gates[g].append(f1)
            ttft_gates[g].append(t)
            per_q_gates[g] = f1

        print(
            f"[{idx + 1}/{len(eval_dataset)}] full={f1_full[-1]:.3f}  "
            f"hkvd={f1_hkvd[-1]:.3f}  " +
            "  ".join(f"g{g}={per_q_gates[g]:.3f}" for g in SWEEP_GATES),
            flush=True,
        )

    # ── aggregate + paired bootstrap vs hkvd_only ──────────────────────
    def _mean(xs): return float(np.mean(xs))
    def _bootstrap_ci(deltas, seed=42, n_boot=1000):
        rng = np.random.default_rng(seed)
        n = len(deltas)
        boot = np.array([np.mean(deltas[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        return [float(lo), float(hi)], bool(lo > 0 or hi < 0)

    summary = {
        "config": {
            "model": MODEL, "dtype": DTYPE, "attn_impl": ATTN_IMPL,
            "kvzip_ratio": KVZIP_RATIO, "recomp_ratio": RECOMP_RATIO,
            "check_layer": CHECK_LAYER, "n": len(eval_dataset),
            "sweep_gates": SWEEP_GATES,
            "kvzip_level": "pair",
            "eligible_mask_active_for_gated": True,
        },
        "f1": {
            "full_mean": _mean(f1_full),
            "hkvd_only_mean": _mean(f1_hkvd),
            "gates": {},
        },
        "ttft": {
            "full_mean": _mean(ttft_full),
            "hkvd_only_mean": _mean(ttft_hkvd),
            "gates": {},
        },
        "n_per_arm": len(f1_full),
    }
    hkvd_arr = np.array(f1_hkvd)
    for g in SWEEP_GATES:
        f1g_arr = np.array(f1_gates[g])
        deltas = f1g_arr - hkvd_arr
        ci, sig = _bootstrap_ci(deltas)
        summary["f1"]["gates"][str(g)] = {
            "mean": _mean(f1_gates[g]),
            "delta_minus_hkvd_mean": float(np.mean(deltas)),
            "delta_ci_95": ci,
            "delta_significant": sig,
        }
        summary["ttft"]["gates"][str(g)] = _mean(ttft_gates[g])

    print("\n──────── Summary ────────")
    print(json.dumps(summary, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[gate-sweep] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
