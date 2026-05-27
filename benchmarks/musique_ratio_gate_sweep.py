"""Joint ratio × gate_percentile sweep for Gated HKVD.

Compresses each doc ONCE at ratio=1.0 (full importance, no eviction), then
in-memory derives valid_mask at each target ratio using a verbatim re-
implementation of KVzip's `_threshold` (level='pair', default). For each
ratio, sweeps gate_percentile and also records the hkvd_only baseline.

Why not call `KVzipBackend.compress(ratio=r)` 6 times per doc:
  - Slower: ~3000 compress calls vs ~500.
  - Same numerical result (verified against KVzip source — `score` from
    `mk.prefill(do_score=True)` is independent of ratio; ratio only enters
    the threshold step).

Output grid: F1 per (kvzip_ratio, selector) where selector ∈
{hkvd_only, gated_top_k @ each gate}. Plus full_recompute reference.

Env knobs (defaults match prior experiments):
    CACHEBLEND_MODEL          Llama-3.1-8B-Instruct
    COMPBLEND_RECOMP_RATIO    0.15
    COMPBLEND_N               50
    COMPBLEND_OUT             logs/ratio_gate_sweep.json
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
N_MAX = int(os.environ.get("COMPBLEND_N", "50"))
OUT_PATH = Path(os.environ.get("COMPBLEND_OUT", str(_REPO / "logs" / "ratio_gate_sweep.json")))
MAX_NEW_TOKENS = 32

# user-requested ratio grid
SWEEP_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
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


def _derive_valid_mask_pair(importance: torch.Tensor, target_ratio: float) -> torch.Tensor:
    """Verbatim re-implementation of KVzip._threshold (level='pair').

    See `src/external/KVzip/attention/score.py:88-104`. Importance is
    flattened across (L, H, M), a single global threshold is chosen at the
    (target_ratio × len)-th rank descending, and valids = importance > thres.

    Returns bool [L, H, M].
    """
    if target_ratio >= 1.0:
        return torch.ones_like(importance, dtype=torch.bool)
    flat = importance.reshape(-1)
    n = max(int(flat.numel() * target_ratio) - 1, 0)
    score_sort = torch.sort(flat, descending=True).values
    thres = score_sort[n].item()
    return importance > thres


def _make_entry_at_ratio(
    cmp_full,                          # CompressedChunk at ratio=1.0
    target_ratio: float,
    n_kv_heads: int,
    head_dim: int,
):
    """Produce a KVStore entry that simulates `KVzipBackend.compress(ratio=r)`.

    Steps:
      1. Derive new valid_mask via the verified pair-level threshold.
      2. Zero-fill K/V at the new evicted (layer, head, position) tuples.
      3. Zero-fill importance at the evicted entries.
      4. Build a dict matching `to_kvstore_entry()` output schema.

    All operations are out-of-place to keep cmp_full reusable for other ratios.
    """
    chunk_len = cmp_full.chunk_len
    n_layers = cmp_full.num_layers
    new_valid = _derive_valid_mask_pair(cmp_full.importance, target_ratio)   # [L, H, M]

    new_K: list[torch.Tensor] = []
    new_V: list[torch.Tensor] = []
    for li in range(n_layers):
        k = cmp_full.key_cache[li].view(1, chunk_len, n_kv_heads, head_dim)
        v = cmp_full.value_cache[li].view(1, chunk_len, n_kv_heads, head_dim)
        # new_valid[li]: [H_kv, M]  → [1, M, H_kv, 1]
        m = new_valid[li].t().unsqueeze(0).unsqueeze(-1).to(k.dtype)
        new_K.append((k * m).reshape(1, chunk_len, n_kv_heads * head_dim).contiguous())
        new_V.append((v * m).reshape(1, chunk_len, n_kv_heads * head_dim).contiguous())
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

    print(f"[ratio×gate] model={MODEL}  recomp_ratio={RECOMP_RATIO}  n={N_MAX}", flush=True)
    print(f"[ratio×gate] ratios={SWEEP_RATIOS}  gates={SWEEP_GATES}", flush=True)

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

    dataset_path = _REPO / "src" / "external" / "cacheblend-hf-v7" / "benchmarks" / "musique" / "inputs" / "musique_s.json"
    with open(dataset_path) as f:
        eval_dataset = json.load(f)[:N_MAX]
    print(f"[ratio×gate] running {len(eval_dataset)} examples", flush=True)

    f1_full = []
    # Keyed by (ratio, selector) where selector ∈ {"hkvd_only"} ∪ {"gate=<g>"}
    f1_arms: dict[tuple[float, str], list[float]] = {}
    for r in SWEEP_RATIOS:
        f1_arms[(r, "hkvd_only")] = []
        for g in SWEEP_GATES:
            f1_arms[(r, f"gate={g}")] = []

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

        # ── arm: full_recompute (reference) ────────────────────────────
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
        res, _ = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
        f1_full.append(max(_compute_f1(res, a, tokenizer) for a in answers))

        # ── compress every doc ONCE at ratio=1.0 ───────────────────────
        compressed_1p0: dict[str, Any] = {}
        for c in chunks[doc_slice]:
            ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
            cmp = backend.compress(
                ids, model=hf_model, budget=CompressionBudget(ratio=1.0),
            )
            compressed_1p0[c.chunk_id] = cmp

        # ── per-ratio: derive valid_mask + zero-fill + run all arms ────
        per_q_summary = {"full": f1_full[-1]}
        for r in SWEEP_RATIOS:
            # Build kv_store with simulated compression at this ratio
            kv_r = KVStore()
            for c in chunks:
                if c.chunk_id in compressed_1p0:
                    cmp = compressed_1p0[c.chunk_id]
                    kv_r._cache[c.chunk_id] = _make_entry_at_ratio(
                        cmp, r, n_kv_heads, head_dim,
                    )
                else:
                    kv_r._cache[c.chunk_id] = kv_base.get(c.chunk_id)

            # arm: hkvd_only at this ratio
            cb_cfg = CompBlendConfig(
                check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
                selector="hkvd_only", chunk_normalization="rank",
            )
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fuse_selective_compblend(lw, chunks, kv_r, cb_cfg, return_layerwise_output=True)
            res, _ = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
            f1_hkvd = max(_compute_f1(res, a, tokenizer) for a in answers)
            f1_arms[(r, "hkvd_only")].append(f1_hkvd)
            per_q_summary[f"r{r}_h"] = f1_hkvd

            # arms: gated_top_k at each gate
            for g in SWEEP_GATES:
                cb_cfg = CompBlendConfig(
                    check_layer=CHECK_LAYER, recompute_ratio=RECOMP_RATIO,
                    selector="gated_top_k", gate_percentile=g,
                    chunk_normalization="rank",
                )
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fuse_selective_compblend(lw, chunks, kv_r, cb_cfg, return_layerwise_output=True)
                res, _ = _greedy_decode(lw.model, tokenizer, out.logits, out.past_key_values, device, t0)
                f1 = max(_compute_f1(res, a, tokenizer) for a in answers)
                f1_arms[(r, f"gate={g}")].append(f1)
                per_q_summary[f"r{r}_g{g}"] = f1

        # Compact per-question log line (truncated to fit width)
        print(f"[{idx + 1}/{len(eval_dataset)}] full={per_q_summary['full']:.2f}  " +
              "  ".join(
                  f"r{r}:h={per_q_summary[f'r{r}_h']:.2f}/best={max(per_q_summary[f'r{r}_g{g}'] for g in SWEEP_GATES):.2f}"
                  for r in SWEEP_RATIOS
              ),
              flush=True)

    # ── aggregate + paired bootstrap on (gate vs hkvd) per ratio ───────
    def _mean(xs): return float(np.mean(xs))
    def _bootstrap_ci(deltas, seed=42, n_boot=1000):
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
            "sweep_ratios": SWEEP_RATIOS,
            "sweep_gates": SWEEP_GATES,
            "kvzip_level": "pair",
            "valid_mask_source": "_derive_valid_mask_pair (verbatim from KVzip._threshold)",
            "eligible_mask_active_for_gated": True,
        },
        "f1": {
            "full_mean": _mean(f1_full),
            "grid": {},
        },
        "n_per_arm": len(f1_full),
    }
    for r in SWEEP_RATIOS:
        hkvd = np.array(f1_arms[(r, "hkvd_only")])
        row: dict[str, Any] = {"hkvd_only_mean": float(np.mean(hkvd)), "gates": {}}
        for g in SWEEP_GATES:
            gated = np.array(f1_arms[(r, f"gate={g}")])
            deltas = gated - hkvd
            ci, sig = _bootstrap_ci(deltas)
            row["gates"][str(g)] = {
                "mean": float(np.mean(gated)),
                "delta_minus_hkvd_mean": float(np.mean(deltas)),
                "delta_ci_95": ci,
                "delta_significant": sig,
            }
        summary["f1"]["grid"][str(r)] = row

    print("\n──────── Summary ────────")
    print(json.dumps(summary, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[ratio×gate] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
