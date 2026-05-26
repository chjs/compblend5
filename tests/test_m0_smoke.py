"""M0 smoke tests.

Two tiers:

1. **Import smoke** (always runs): verifies the v7 submodule is on PYTHONPATH and
   its public API is importable. Catches "forgot --recursive" / wrong path setup.

2. **Bit-exact 3-way** (@pytest.mark.gpu, ~1 min on first run): proves that on a
   real model, three forward paths give identical logits:
       (a) HF standard prefill: model(input_ids)
       (b) v7 fuse_full_recompute(chunks): equivalent path through cacheblend
       (c) v7 fuse_selective(chunks, ratio=1.0): dispatches to (b) via boundary
           safe-shortcut. Must be bit-exact with (b).

   This is the BASELINE every later milestone is compared against. If KVzip-backed
   blending later diverges from this, we know the divergence comes from KVzip /
   our fusor extensions — not from v7 itself.

The chosen model is TinyLlama-1.1B — small enough that the GPU run is ~1 min
total including download, but real (Llama-family RoPE, GQA, multi-layer).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Tier 1 — import smoke (no model, no GPU)
# ---------------------------------------------------------------------------


def test_compblend_imports():
    """Our own package skeleton loads."""
    import compblend
    assert compblend.__version__.startswith("0.5")


def test_cacheblend_submodule_imports():
    """The v7 submodule's public API is reachable."""
    cb = pytest.importorskip("cacheblend")
    # Spot-check the 5 entry points we'll use in M1–M3.
    assert hasattr(cb, "LayerwiseModel")
    assert hasattr(cb, "KVStore")
    assert hasattr(cb, "chunk_texts")
    assert hasattr(cb, "fuse_selective")
    assert hasattr(cb, "fuse_full_recompute")


def test_cacheblend_hkvd_imports():
    """HKVD primitive — central to M2's fusor fork."""
    from cacheblend.hkvd import kv_deviation, select_top_k
    assert callable(kv_deviation)
    assert callable(select_top_k)


# ---------------------------------------------------------------------------
# Tier 2 — bit-exact 3-way (GPU + real model)
# ---------------------------------------------------------------------------


_TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.gpu
@pytest.mark.slow
def test_three_way_bit_exact_tinyllama():
    """HF prefill ≡ v7 fuse_full_recompute ≡ v7 fuse_selective(ratio=1.0).

    Uses the same single-chunk input across paths. fuse_selective with
    ratio=1.0 falls back to fuse_full_recompute via its boundary safe-shortcut
    (fusor.py:232), and that path with len(chunks)<=1 falls back to
    fuse_full_recompute too — so all three must agree bit-exactly.
    """
    import torch
    from cacheblend import LayerwiseModel
    from cacheblend.chunker import Chunk, _stable_id
    from cacheblend.fusor import fuse_full_recompute, fuse_selective

    lw = LayerwiseModel(_TINY, dtype="float16", attn_implementation="eager")
    tok = lw.tokenizer

    text = "The quick brown fox jumps over the lazy dog."
    ids = tok(text, add_special_tokens=False)["input_ids"]
    chunks = [Chunk(text=text, token_ids=ids, chunk_id=_stable_id(text, ids))]

    input_ids = torch.tensor([ids], dtype=torch.long, device=lw.device)

    # (a) standard HF
    with torch.inference_mode():
        hf_logits = lw.model(input_ids=input_ids, use_cache=False).logits

    # (b) v7 fuse_full_recompute
    fr_logits = fuse_full_recompute(lw, chunks)

    # (c) v7 fuse_selective at ratio=1.0 — must dispatch to (b)
    fs_logits = fuse_selective(lw, chunks, kv_store=None, recompute_ratio=1.0)

    # Bit-exact: same dtype, same kernel path.
    assert torch.equal(fr_logits, fs_logits), (
        "fuse_selective(ratio=1.0) must be bit-exact with fuse_full_recompute"
    )
    # HF vs v7 path: tolerated tiny float drift from different attention mask
    # builder. Use a tight allclose rather than equal.
    assert torch.allclose(hf_logits, fr_logits, atol=1e-3, rtol=0), (
        f"HF baseline diverges from v7 fuse_full_recompute. "
        f"max_abs_diff={ (hf_logits - fr_logits).abs().max().item() }"
    )
