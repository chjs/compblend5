"""M5 — quality regression. KVzip-compressed multi-doc QA across selectors.

What this measures
──────────────────
For each MuSiQue question (3+ supporting docs):

  * Run #1 — KVzip ratio=R, selector="hkvd_only"   →  F1_hkvd
  * Run #2 — KVzip ratio=R, selector="gated_top_k" →  F1_gated
  * Run #3 — fuse_full_recompute (no compression)  →  F1_full (reference)

Aggregates: mean F1 per arm + per-question paired ΔF1 (gated − hkvd) with
1k-sample bootstrap 95% CI. The CompBlend-old Phase 4b run 6 reference
(KVzip + r=0.10 + n=500 on 2WikiMQA) reported ΔF1 = +0.0348 [+0.0016,
+0.0694] ★ SIG. We expect the same DIRECTION here on MuSiQue (Phase 4b
run 17 measured +0.031 [-0.027, +0.092] at n=200).

When KVzip is not on PYTHONPATH
───────────────────────────────
The KVzip arms are skipped and only the FULL prefill baseline runs. That
still validates `fuse_selective_compblend` end-to-end on real data — useful
as a smoke check after a swap. Print a clear notice when this happens.

CLI / env
─────────
    CACHEBLEND_MODEL          HF model id              (default Llama-3.1-8B-Instruct)
    CACHEBLEND_DTYPE          float16 / bfloat16       (default float16)
    CACHEBLEND_ATTN_IMPL      eager / sdpa             (default sdpa)
    COMPBLEND_RATIO_KVZIP     KVzip ratio              (default 0.10)
    COMPBLEND_RECOMP_RATIO    fuse recompute ratio     (default 0.15)
    COMPBLEND_CHECK_LAYER     check_layer              (default 1)
    COMPBLEND_GATE_PCT        gated_top_k percentile   (default 0.5)
    COMPBLEND_N               # examples to run        (default 150 — full MuSiQue_s)
    COMPBLEND_OUT             output JSON path         (default ./logs/musique_compare.json)
    COMPBLEND_CACHE_DIR       offline KVzip cache dir  (optional; if set + the file
                              exists, load instead of compress online)

Launch (vast.ai A100):
    python benchmarks/musique_selector_compare.py 2>&1 | tee logs/m5-run.log
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

# Ensure src + v7 src + bench utils are on the path.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "src" / "external" / "cacheblend-hf-v7" / "src"))
sys.path.insert(0, str(_REPO / "src" / "external" / "cacheblend-hf-v7" / "benchmarks" / "musique"))


# ──────────────────────────────────────────────────────────────────────────
# Env knobs
# ──────────────────────────────────────────────────────────────────────────

MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DTYPE = os.environ.get("CACHEBLEND_DTYPE", "float16")
ATTN_IMPL = os.environ.get("CACHEBLEND_ATTN_IMPL", "sdpa")
KVZIP_RATIO = float(os.environ.get("COMPBLEND_RATIO_KVZIP", "0.10"))
RECOMP_RATIO = float(os.environ.get("COMPBLEND_RECOMP_RATIO", "0.15"))
CHECK_LAYER = int(os.environ.get("COMPBLEND_CHECK_LAYER", "1"))
GATE_PCT = float(os.environ.get("COMPBLEND_GATE_PCT", "0.5"))
N_MAX = int(os.environ.get("COMPBLEND_N", "150"))
OUT_PATH = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "musique_compare.json")))
CACHE_DIR_ENV = os.environ.get("COMPBLEND_CACHE_DIR", "")
CACHE_DIR = Path(CACHE_DIR_ENV) if CACHE_DIR_ENV else None
MAX_NEW_TOKENS = 32


# Per-family chat wrapper (subset of v7's blend_musique_generic._WRAPPERS).
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


def _resolve_wrapper(model_id: str, tokenizer):
    mid = model_id.lower()
    for key, wrap in _WRAPPERS.items():
        if key in mid:
            return wrap
    # Fallback: chat template
    sentinel = "\x00CONTENT\x00"
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=True,
    )
    if sentinel not in templated:
        raise RuntimeError(f"can't derive wrapper for {model_id!r}")
    pre, post = templated.split(sentinel, 1)
    bos = tokenizer.bos_token or ""
    if bos and pre.startswith(bos):
        pre = pre[len(bos):]
    return pre, post


def _build_chunks(tokenizer, chunk_texts: list[str]):
    """Tokenize once. BOS only on chunk 0."""
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


def _kvzip_available() -> bool:
    try:
        import model  # noqa
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Per-example arms
# ──────────────────────────────────────────────────────────────────────────


def _arm_full_recompute(lw, chunks, model, tokenizer, device):
    from cacheblend.fusor import fuse_full_recompute
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
    text, ttft = _greedy_decode(model, tokenizer, out.logits, out.past_key_values, device, t0)
    return text, ttft


def _arm_kvzip_selector(
    lw, chunks_with_doc_compressed_kv: list, kv_store,
    selector: str, model, tokenizer, device,
):
    from compblend.config import CompBlendConfig
    from compblend.fuse_selective_compblend import fuse_selective_compblend
    cfg = CompBlendConfig(
        check_layer=CHECK_LAYER,
        recompute_ratio=RECOMP_RATIO,
        selector=selector,
        gate_percentile=GATE_PCT,
        chunk_normalization="rank",        # length-fair across docs
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fuse_selective_compblend(
        lw, chunks_with_doc_compressed_kv, kv_store, cfg,
        return_layerwise_output=True,
    )
    text, ttft = _greedy_decode(model, tokenizer, out.logits, out.past_key_values, device, t0)
    return text, ttft


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    from cacheblend import LayerwiseModel
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    from utils import build_qa_prompt, compute_f1, load_dataset

    print(f"[m5] model={MODEL} dtype={DTYPE} attn={ATTN_IMPL}", flush=True)
    print(f"[m5] kvzip_ratio={KVZIP_RATIO}  recomp_ratio={RECOMP_RATIO}  "
          f"check_layer={CHECK_LAYER}  gate_pct={GATE_PCT}", flush=True)

    use_kvzip = _kvzip_available()
    if not use_kvzip:
        print("[m5] KVzip NOT on PYTHONPATH — running FULL baseline only "
              "(no compression). Set up KVzip to enable selector comparison.",
              flush=True)

    lw = LayerwiseModel(MODEL, dtype=DTYPE, attn_implementation=ATTN_IMPL)
    tokenizer, model, device = lw.tokenizer, lw.model, lw.device
    user_open, assistant_open = _resolve_wrapper(MODEL, tokenizer)

    # Use v7's MuSiQue dataset.
    dataset_path = _REPO / "src" / "external" / "cacheblend-hf-v7" / "benchmarks" / "musique" / "inputs" / "musique_s.json"
    eval_dataset = load_dataset(str(dataset_path))
    eval_dataset = eval_dataset[:N_MAX]
    print(f"[m5] running {len(eval_dataset)} examples", flush=True)

    # KVzip backend — one instance shared across questions (model already loaded).
    backend = None
    if use_kvzip:
        from compblend.backends.kvzip import KVzipBackend, KVzipConfig, kvzip_chunk_id
        from compblend.backends.base import (
            CompressedChunk, CompressionBudget, to_kvstore_entry,
        )
        # If a corpus cache dir is set AND populated, we'll load instead of
        # compressing online. Otherwise compress on demand (and optionally
        # populate the cache as a side effect when CACHE_DIR is writable).
        if CACHE_DIR is not None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[m5] offline cache dir = {CACHE_DIR}  "
                  f"(load if exists, save on miss)", flush=True)
        # Only load the heavy ModelKVzip when we actually need to compress
        # — i.e., when at least one doc isn't on disk. We defer the lazy load.
        backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain", level="pair"))

    f1_full, f1_hkvd, f1_gated = [], [], []
    ttft_full, ttft_hkvd, ttft_gated = [], [], []
    answers_full, answers_hkvd, answers_gated = [], [], []

    for idx, ex in enumerate(eval_dataset):
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)

        # Chunk layout: [user_open + prefix] [doc1]..[docN] [query + assistant_open]
        chunk_texts = [user_open + PREFIX_PROMPT, *doc_prompts, q_prompt + assistant_open]
        chunks = _build_chunks(tokenizer, chunk_texts)

        # KVStore for THIS example. Precompute every chunk via v7 first.
        kv_store = KVStore()
        for c in chunks:
            K, V = precompute_chunk_kv(lw, c)
            kv_store.put(c.chunk_id, K, V)

        # ── arm 1: full prefill (reference) ────────────────────────────
        res_full, t_full = _arm_full_recompute(lw, chunks, model, tokenizer, device)
        f_full = max(compute_f1(res_full, a, tokenizer) for a in answers)
        f1_full.append(f_full); ttft_full.append(t_full); answers_full.append(res_full)

        if use_kvzip:
            # Re-build kv_store: for the doc chunks, REPLACE precompute entries
            # with KVzip-compressed entries. sys + query stay v7-precomputed.
            doc_slice = slice(1, 1 + len(doc_prompts))   # chunks[1:1+N]
            for i, c in enumerate(chunks[doc_slice], start=1):
                # Disk-cache check first (deterministic id by token_ids).
                cid = kvzip_chunk_id(
                    c.token_ids, ratio=KVZIP_RATIO, level="pair",
                )
                cache_path = (
                    (CACHE_DIR / f"{cid}.pt") if CACHE_DIR is not None else None
                )
                if cache_path is not None and cache_path.exists():
                    compressed = CompressedChunk.load(cache_path).to(device)
                else:
                    # Compress online (KVzip prepends sys_prompt internally;
                    # the adapter slices it off).
                    ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
                    compressed = backend.compress(
                        ids, model=backend.hf_model,
                        budget=CompressionBudget(ratio=KVZIP_RATIO),
                    )
                    if cache_path is not None:
                        # Write to disk (CPU) for future runs.
                        compressed.to("cpu").save(cache_path)
                    compressed = compressed.to(device)
                # Bind by the v7-Chunk's chunk_id (NOT compressed.chunk_id —
                # the fusor's kv_store lookup uses chunk.chunk_id).
                kv_store._cache[c.chunk_id] = to_kvstore_entry(compressed)

            # ── arm 2: KVzip + hkvd_only ───────────────────────────────
            res_h, t_h = _arm_kvzip_selector(
                lw, chunks, kv_store, "hkvd_only", model, tokenizer, device,
            )
            f_h = max(compute_f1(res_h, a, tokenizer) for a in answers)
            f1_hkvd.append(f_h); ttft_hkvd.append(t_h); answers_hkvd.append(res_h)

            # ── arm 3: KVzip + gated_top_k ─────────────────────────────
            res_g, t_g = _arm_kvzip_selector(
                lw, chunks, kv_store, "gated_top_k", model, tokenizer, device,
            )
            f_g = max(compute_f1(res_g, a, tokenizer) for a in answers)
            f1_gated.append(f_g); ttft_gated.append(t_g); answers_gated.append(res_g)

            print(
                f"[{idx + 1}/{len(eval_dataset)}] "
                f"full={f_full:.3f}  kvzip_hkvd={f_h:.3f}  kvzip_gated={f_g:.3f}  "
                f"Δ(g-h)={f_g - f_h:+.3f}",
                flush=True,
            )
        else:
            print(
                f"[{idx + 1}/{len(eval_dataset)}] full={f_full:.3f}  "
                f"(kvzip arms skipped — install KVzip)",
                flush=True,
            )

    # ── aggregate + bootstrap CI on paired ΔF1(gated − hkvd) ────────────
    summary: dict[str, Any] = {
        "config": {
            "model": MODEL, "dtype": DTYPE, "attn_impl": ATTN_IMPL,
            "kvzip_ratio": KVZIP_RATIO, "recomp_ratio": RECOMP_RATIO,
            "check_layer": CHECK_LAYER, "gate_pct": GATE_PCT,
            "n": len(eval_dataset), "use_kvzip": use_kvzip,
        },
        "f1": {
            "full_mean": float(np.mean(f1_full)),
            "full_n": len(f1_full),
        },
        "ttft": {
            "full_mean": float(np.mean(ttft_full)),
        },
    }
    if use_kvzip:
        summary["f1"]["hkvd_mean"] = float(np.mean(f1_hkvd))
        summary["f1"]["gated_mean"] = float(np.mean(f1_gated))
        deltas = np.array(f1_gated) - np.array(f1_hkvd)
        summary["f1"]["delta_gated_minus_hkvd_mean"] = float(np.mean(deltas))
        # 1k-sample bootstrap on the paired ΔF1.
        rng = np.random.default_rng(seed=42)
        n = len(deltas)
        boot_means = np.array([
            np.mean(deltas[rng.integers(0, n, size=n)]) for _ in range(1000)
        ])
        ci_lo, ci_hi = np.quantile(boot_means, [0.025, 0.975])
        summary["f1"]["delta_ci_95"] = [float(ci_lo), float(ci_hi)]
        summary["f1"]["delta_significant"] = bool(ci_lo > 0 or ci_hi < 0)
        summary["ttft"]["hkvd_mean"] = float(np.mean(ttft_hkvd))
        summary["ttft"]["gated_mean"] = float(np.mean(ttft_gated))

    print("\n──────── Summary ────────")
    print(json.dumps(summary, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[m5] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
