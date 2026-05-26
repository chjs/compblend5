"""M4 — `fuse_selective_compblend` end-to-end sanity (GPU + real model).

All tests use TinyLlama-1.1B-Chat-v1.0 (downloads ~2.2GB on first run).

The chain of guarantees we want from M4:

  T1. **Boundary shortcut**: `fuse_selective_compblend` at `ratio=1.0` must
      dispatch to `fuse_full_recompute` and therefore equal it bit-exactly.
      Establishes that the entry point glue is correct.

  T2. **No-extension parity**: when KVStore entries are pure v7-style (no
      `valid_mask`, no `importance`, no `is_structural`) and selector is
      `hkvd_only`, our fusor must produce the SAME logits as v7's
      `fuse_selective` on the same chunks. Establishes that the surgical
      fork didn't accidentally change behavior on the no-extension path.

  T3. **Per-head mask wiring**: two runs that differ ONLY in `valid_mask`
      must produce different last-position logits. Establishes that
      eviction actually flows through to attention.

  T4. **Gated HKVD ≠ hkvd_only**: when importance is non-uniform and
      `gate_percentile=0.5`, the selected top_indices differ from
      hkvd_only on the same inputs. Establishes that selector dispatch
      reaches the gated branch.
"""
from __future__ import annotations

import pytest


_TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def lw_and_chunks():
    """Two-chunk fused prompt against TinyLlama with v7-precomputed KVStore."""
    import torch
    from cacheblend import LayerwiseModel
    from cacheblend.chunker import Chunk, _stable_id
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv

    lw = LayerwiseModel(_TINY, dtype="float16", attn_implementation="eager")
    tok = lw.tokenizer

    texts = [
        "Document 1: Paris is the capital of France.",
        "Document 2: Berlin is the capital of Germany.",
    ]
    chunks = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        chunks.append(Chunk(text=t, token_ids=ids, chunk_id=_stable_id(t, ids)))

    kv = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw, c)
        kv.put(c.chunk_id, K, V)

    return {"lw": lw, "chunks": chunks, "kv": kv}


# ──────────────────────────────────────────────────────────────────────────
# T1 — ratio=1.0 dispatches to fuse_full_recompute (bit-exact)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.slow
def test_ratio_one_matches_full_recompute(lw_and_chunks):
    import torch
    from cacheblend.fusor import fuse_full_recompute
    from compblend.config import CompBlendConfig
    from compblend.fuse_selective_compblend import fuse_selective_compblend

    lw, chunks, kv = lw_and_chunks["lw"], lw_and_chunks["chunks"], lw_and_chunks["kv"]
    fr = fuse_full_recompute(lw, chunks)
    cb = fuse_selective_compblend(
        lw, chunks, kv,
        CompBlendConfig(check_layer=1, recompute_ratio=1.0),
    )
    assert torch.equal(fr, cb), (
        "ratio=1.0 must dispatch to fuse_full_recompute and be bit-exact. "
        f"max_abs_diff={(fr - cb).abs().max().item()}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T2 — no-extension parity with v7 fuse_selective (hkvd_only)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.slow
def test_no_extension_parity_with_v7(lw_and_chunks):
    """Plain v7-style KVStore + hkvd_only selector reproduces v7 logits."""
    import torch
    from cacheblend.fusor import fuse_selective as v7_fuse_selective
    from compblend.config import CompBlendConfig
    from compblend.fuse_selective_compblend import fuse_selective_compblend

    lw, chunks, kv = lw_and_chunks["lw"], lw_and_chunks["chunks"], lw_and_chunks["kv"]
    ratio = 0.30
    v7_logits = v7_fuse_selective(
        lw, chunks, kv, recompute_ratio=ratio, check_layer=1,
    )
    cb_logits = fuse_selective_compblend(
        lw, chunks, kv,
        CompBlendConfig(
            check_layer=1, recompute_ratio=ratio, selector="hkvd_only",
            exempt_structural=True,
        ),
    )
    # With no eviction (valid_mask all-True via default) and identical
    # selector behavior on the chosen top_indices, SDPA inputs are identical.
    # Allow tiny float drift from intermediate ops; assert tight tolerance.
    max_diff = (v7_logits - cb_logits).abs().max().item()
    assert max_diff < 1e-3, (
        f"hkvd_only on v7-style entries must match v7 fuse_selective; "
        f"observed max_abs_diff={max_diff} at last position"
    )


# ──────────────────────────────────────────────────────────────────────────
# T3 — per-head mask wiring (eviction changes last-position logits)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.slow
def test_per_head_eviction_changes_logits(lw_and_chunks):
    """Toggling valid_mask from all-True to half-evicted MUST change logits.

    If logits don't change, the per-head mask is not flowing through to SDPA.
    """
    import torch
    from compblend.config import CompBlendConfig
    from compblend.fuse_selective_compblend import fuse_selective_compblend

    lw, chunks, kv = lw_and_chunks["lw"], lw_and_chunks["chunks"], lw_and_chunks["kv"]
    cfg = lw.model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads

    # Baseline: kv entries have NO valid_mask → defaults to all-True.
    baseline_logits = fuse_selective_compblend(
        lw, chunks, kv,
        CompBlendConfig(check_layer=1, recompute_ratio=0.30, selector="hkvd_only"),
    )

    # Make a separate KVStore where each chunk carries a valid_mask that
    # evicts roughly half of the (layer, head, position) tuples — half of
    # the heads at every position. Same K, V tensors (cheaper than re-prefilling).
    from cacheblend.kv_store import KVStore
    kv_evict = KVStore()
    for c in chunks:
        entry = kv.get(c.chunk_id)
        chunk_len = entry["K"][0].shape[1]
        # Heads with index < n_kv_heads/2 keep all positions; rest evict all.
        valid = torch.ones(n_layers, n_kv_heads, chunk_len,
                           dtype=torch.bool, device=lw.device)
        valid[:, n_kv_heads // 2:, :] = False
        kv_evict._cache[c.chunk_id] = {
            "K": entry["K"],
            "V": entry["V"],
            "valid_mask": valid,
        }

    evicted_logits = fuse_selective_compblend(
        lw, chunks, kv_evict,
        CompBlendConfig(check_layer=1, recompute_ratio=0.30, selector="hkvd_only"),
    )

    # The last position's logits MUST differ — attention at later layers
    # touched the now-masked half of the heads in the baseline run.
    last_diff = (baseline_logits[0, -1] - evicted_logits[0, -1]).abs().max().item()
    assert last_diff > 1e-2, (
        f"Per-head eviction did not change last-position logits. "
        f"max_abs_diff = {last_diff}. The valid_mask is not reaching SDPA."
    )


# ──────────────────────────────────────────────────────────────────────────
# T4 — Gated HKVD selects different positions from hkvd_only
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.slow
def test_gated_hkvd_differs_from_hkvd_only(lw_and_chunks):
    """When importance is non-trivial, Gated HKVD picks different top_indices.

    Constructed importance is large at positions where HKVD is small (anti-
    correlated) — so the gate filters out the natural HKVD top-k, forcing
    a different selection. Different selection → different logits.
    """
    import torch
    from cacheblend.kv_store import KVStore
    from compblend.config import CompBlendConfig
    from compblend.fuse_selective_compblend import fuse_selective_compblend

    lw, chunks, kv = lw_and_chunks["lw"], lw_and_chunks["chunks"], lw_and_chunks["kv"]
    cfg = lw.model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads

    # Inject importance: increasing in position. Combined with the natural
    # HKVD distribution (which is roughly uniform/decaying with position),
    # the gate's top-50% by importance will be the LATER half of each
    # chunk; HKVD top-k within that half differs from global HKVD top-k.
    kv_with_imp = KVStore()
    for c in chunks:
        entry = kv.get(c.chunk_id)
        chunk_len = entry["K"][0].shape[1]
        imp = torch.arange(chunk_len, dtype=torch.float32, device=lw.device)
        imp = imp.unsqueeze(0).unsqueeze(0).expand(n_layers, n_kv_heads, chunk_len).contiguous()
        kv_with_imp._cache[c.chunk_id] = {
            "K": entry["K"],
            "V": entry["V"],
            "importance": imp,
        }

    base_cfg = CompBlendConfig(
        check_layer=1, recompute_ratio=0.30, selector="hkvd_only",
    )
    gated_cfg = CompBlendConfig(
        check_layer=1, recompute_ratio=0.30, selector="gated_top_k",
        gate_percentile=0.5,
    )

    _, top_h = fuse_selective_compblend(
        lw, chunks, kv_with_imp, base_cfg,
        return_layerwise_output=True, return_hkvd_indices=True,
    )
    _, top_g = fuse_selective_compblend(
        lw, chunks, kv_with_imp, gated_cfg,
        return_layerwise_output=True, return_hkvd_indices=True,
    )

    set_h = set(top_h.tolist())
    set_g = set(top_g.tolist())
    assert set_h != set_g, (
        f"hkvd_only and gated_top_k produced identical top_indices "
        f"({sorted(set_h)}) — the gate isn't filtering. Check that "
        f"importance flows through the chunk-loading loop."
    )
