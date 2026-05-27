"""Diagnostic — fuse_selective on UNCOMPRESSED KV cache (no KVzip).

Purpose
-------
Isolate "CacheBlend overhead" from "KVzip compression damage" on MuSiQue:

    full_recompute     →  0.3156   (M5 result, no compression, no blending)
    arm_4 (THIS file)  →  ?         (no compression + blending)
    hkvd + KVzip       →  0.0694   (M5 result, compression + blending)
    gated + KVzip      →  0.0872   (M5 result, compression + blending)

If arm_4 ≈ full → KVzip is the dominant loss; the integration is fine.
If arm_4 ≈ KVzip arms → CacheBlend itself has a bug in our integration.

Workload
--------
Same MuSiQue questions as M5, same prompt format, same fuse settings.
For each question: precompute_chunk_kv for every chunk (sys + docs + query),
then v7's `fuse_selective(ratio=0.15, check_layer=1)` directly — no
fuse_selective_compblend wrapper, no per-head mask, no Gated HKVD.

Output
------
JSON + per-question TSV under logs/. The aggregate file is named so it
sits cleanly beside results/m5/n118_kvzip_r010_*.json for paper-figure
synthesis.

Env knobs
---------
    CACHEBLEND_MODEL          HF model id   (default Llama-3.1-8B-Instruct)
    COMPBLEND_RECOMP_RATIO    fuse ratio    (default 0.15 — matches M5)
    COMPBLEND_N               # questions   (default 118 — matches M5 subset)
    COMPBLEND_OUT             output path   (default logs/musique_no_kvzip.json)
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
N_MAX = int(os.environ.get("COMPBLEND_N", "118"))
OUT_PATH = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "musique_no_kvzip.json")))
MAX_NEW_TOKENS = 32


_WRAPPERS = {
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "mistral": ("[INST]", "[/INST]"),
    "qwen":    ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
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


def _load_dataset(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def _build_qa_prompt(example: dict) -> tuple[list[str], str]:
    q = _normalize_question(example["question"])
    doc_prompts = [f"{c['title']}\n\n{c['text']}\n\n" for c in example["ctxs"]]
    q_prompt = f"{QUERY_PROMPT}{q}\nAnswer:"
    return doc_prompts, q_prompt


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
    sentinel = "\x00CONTENT\x00"
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=True,
    )
    if sentinel not in templated:
        raise RuntimeError(f"can't derive wrapper for {model_id}")
    pre, post = templated.split(sentinel, 1)
    bos = tokenizer.bos_token or ""
    if bos and pre.startswith(bos):
        pre = pre[len(bos):]
    return pre, post


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


def main() -> int:
    from cacheblend import LayerwiseModel
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend.fusor import fuse_selective, fuse_full_recompute

    print(f"[no-kvzip] model={MODEL}  dtype={DTYPE}  attn={ATTN_IMPL}", flush=True)
    print(f"[no-kvzip] recomp_ratio={RECOMP_RATIO}  check_layer={CHECK_LAYER}  n={N_MAX}", flush=True)

    lw = LayerwiseModel(MODEL, dtype=DTYPE, attn_implementation=ATTN_IMPL)
    tokenizer, model, device = lw.tokenizer, lw.model, lw.device
    user_open, assistant_open = _resolve_wrapper(MODEL, tokenizer)

    dataset_path = _REPO / "src" / "external" / "cacheblend-hf-v7" / "benchmarks" / "musique" / "inputs" / "musique_s.json"
    eval_dataset = _load_dataset(str(dataset_path))[:N_MAX]
    print(f"[no-kvzip] running {len(eval_dataset)} examples", flush=True)

    f1_full, f1_blend = [], []
    ttft_full, ttft_blend = [], []
    ans_full, ans_blend = [], []

    for idx, ex in enumerate(eval_dataset):
        answers = ex["answers"]
        doc_prompts, q_prompt = _build_qa_prompt(ex)
        chunk_texts_list = [user_open + PREFIX_PROMPT, *doc_prompts, q_prompt + assistant_open]
        chunks = _build_chunks(tokenizer, chunk_texts_list)

        # Precompute ALL chunks fresh (no compression).
        kv = KVStore()
        for c in chunks:
            K, V = precompute_chunk_kv(lw, c)
            kv.put(c.chunk_id, K, V)

        # arm 1 — full prefill (matches M5's full_recompute)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_f = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
        res_f, t_f = _greedy_decode(model, tokenizer, out_f.logits, out_f.past_key_values, device, t0)
        f_f = max(_compute_f1(res_f, a, tokenizer) for a in answers)
        f1_full.append(f_f); ttft_full.append(t_f); ans_full.append(res_f)

        # arm 4 — v7 fuse_selective directly on uncompressed cache
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_b = fuse_selective(
            lw, chunks, kv, recompute_ratio=RECOMP_RATIO, check_layer=CHECK_LAYER,
            return_layerwise_output=True,
        )
        res_b, t_b = _greedy_decode(model, tokenizer, out_b.logits, out_b.past_key_values, device, t0)
        f_b = max(_compute_f1(res_b, a, tokenizer) for a in answers)
        f1_blend.append(f_b); ttft_blend.append(t_b); ans_blend.append(res_b)

        print(
            f"[{idx + 1}/{len(eval_dataset)}] full={f_f:.3f}  blend_no_kvzip={f_b:.3f}  "
            f"Δ(b-f)={f_b - f_f:+.3f}",
            flush=True,
        )

    deltas = np.array(f1_blend) - np.array(f1_full)
    rng = np.random.default_rng(seed=42)
    n = len(deltas)
    boot = np.array([np.mean(deltas[rng.integers(0, n, size=n)]) for _ in range(1000)])
    ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])

    summary: dict[str, Any] = {
        "config": {
            "model": MODEL, "dtype": DTYPE, "attn_impl": ATTN_IMPL,
            "recomp_ratio": RECOMP_RATIO, "check_layer": CHECK_LAYER,
            "n": len(eval_dataset),
            "compression": "none",
            "fuse_path": "v7 fuse_selective (NOT fuse_selective_compblend)",
        },
        "f1": {
            "full_mean": float(np.mean(f1_full)),
            "blend_no_kvzip_mean": float(np.mean(f1_blend)),
            "delta_blend_minus_full_mean": float(np.mean(deltas)),
            "delta_ci_95": [float(ci_lo), float(ci_hi)],
            "delta_significant": bool(ci_lo > 0 or ci_hi < 0),
            "n": len(f1_full),
        },
        "ttft": {
            "full_mean": float(np.mean(ttft_full)),
            "blend_no_kvzip_mean": float(np.mean(ttft_blend)),
        },
    }

    print("\n──────── Summary ────────")
    print(json.dumps(summary, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[no-kvzip] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
