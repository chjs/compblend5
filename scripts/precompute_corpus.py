"""Offline corpus pre-compression — runs ONCE per (model, dataset, ratio).

Walks the MuSiQue dataset, encodes every doc paragraph, and writes a
KVzip-compressed CompressedChunk to `<cache_dir>/<chunk_id>.pt`. Duplicate
docs (across questions) are detected by their deterministic `chunk_id`
and skipped — the i-th encounter of a doc that's already on disk is a
zero-cost lookup.

This matches the paper's offline/online split:

  * Offline (this script, runs ONCE):
        for each unique doc in corpus:
            chunk = KVzipBackend.compress(doc, ratio=R)
            chunk.save(cache_dir / f"{chunk.chunk_id}.pt")

  * Online (benchmark, runs per-query):
        CompressedChunk.load(cache_dir / f"{chunk_id}.pt")  →  KVStore entry
        fuse_selective_compblend(...)

Time / cost on Llama-3.1-8B + MuSiQue_s (150 questions × ~10 ctxs ≈ 1500
paragraphs, of which a few hundred are unique):
    ~1-3 sec per unique compress on A100 80GB → 5-15 min total.
    Disk: ~3-7 GB at ratio=0.10.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

# Path setup — script lives under scripts/, repo root is parent.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "src" / "external" / "cacheblend-hf-v7" / "src"))
sys.path.insert(0, str(_REPO / "src" / "external" / "cacheblend-hf-v7" / "benchmarks" / "musique"))


MODEL = os.environ.get("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
KVZIP_RATIO = float(os.environ.get("COMPBLEND_RATIO_KVZIP", "0.10"))
KVZIP_LEVEL = os.environ.get("COMPBLEND_RATIO_LEVEL", "pair")
CACHE_DIR = Path(os.environ.get(
    "COMPBLEND_CACHE_DIR", str(_REPO / "cache" / "kvzip_musique"),
))


def _build_doc_text(ctx: dict) -> str:
    """Same as utils.build_qa_prompt's doc_prompt format."""
    return f"{ctx['title']}\n\n{ctx['text']}\n\n"


def main() -> int:
    from compblend.backends.base import CompressionBudget
    from compblend.backends.kvzip import (
        KVzipBackend, KVzipConfig, kvzip_chunk_id,
    )
    from utils import load_dataset

    print(f"[precompute] model={MODEL}  ratio={KVZIP_RATIO}  level={KVZIP_LEVEL}",
          flush=True)
    print(f"[precompute] cache_dir={CACHE_DIR}", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = _REPO / "src" / "external" / "cacheblend-hf-v7" / "benchmarks" / "musique" / "inputs" / "musique_s.json"
    eval_dataset = load_dataset(str(dataset_path))
    print(f"[precompute] {len(eval_dataset)} questions in dataset", flush=True)

    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain", level=KVZIP_LEVEL))
    tokenizer = backend.tokenizer
    device = next(backend.hf_model.parameters()).device

    n_attempted = 0
    n_already = 0
    n_compressed = 0

    for q_idx, ex in enumerate(eval_dataset):
        for c_idx, ctx in enumerate(ex.get("ctxs", [])):
            n_attempted += 1
            text = _build_doc_text(ctx)
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            # Predict chunk_id BEFORE compressing — skip if already on disk.
            chunk_id = kvzip_chunk_id(token_ids, ratio=KVZIP_RATIO, level=KVZIP_LEVEL)
            cache_path = CACHE_DIR / f"{chunk_id}.pt"
            if cache_path.exists():
                n_already += 1
                continue

            ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
            compressed = backend.compress(
                ids_t, model=backend.hf_model,
                budget=CompressionBudget(ratio=KVZIP_RATIO),
            )
            # Sanity: the predicted id must match what compress emitted.
            if compressed.chunk_id != chunk_id:
                raise RuntimeError(
                    f"chunk_id mismatch: predicted {chunk_id} got "
                    f"{compressed.chunk_id}. The kvzip_chunk_id helper must "
                    f"stay in sync with KVzipBackend.compress."
                )
            # Move to CPU before saving to keep files portable.
            compressed.to("cpu").save(cache_path)
            n_compressed += 1

            if n_compressed % 20 == 0:
                print(
                    f"[precompute] q={q_idx + 1}/{len(eval_dataset)}  "
                    f"new={n_compressed}  hit={n_already}  attempted={n_attempted}",
                    flush=True,
                )

    print()
    print("[precompute] done", flush=True)
    print(f"[precompute]   attempted   = {n_attempted}")
    print(f"[precompute]   cache hits  = {n_already}")
    print(f"[precompute]   new chunks  = {n_compressed}")
    print(f"[precompute]   cache dir   = {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
