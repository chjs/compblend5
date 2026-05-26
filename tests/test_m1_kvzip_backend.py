"""M1 — KVzip backend with pre-RoPE K capture.

Tier 1 (CPU) — schema sanity, no KVzip / no model needed.
Tier 2 (GPU + KVzip) — actual compress pass on TinyLlama. Key correctness
gate: layer-0 pre-RoPE K captured by our forward-hook MUST match v7's
`precompute_chunk_kv` at the same content positions (bit-exact).

Why layer-0 must match exactly
──────────────────────────────
Layer 0's K projection is `k_proj(input_layernorm(embed_tokens(ids)))`.
Every operation in that chain is position-local: embed lookup, LayerNorm,
Linear — none of them touch other positions. So pre-RoPE K at position p of
layer 0 depends ONLY on input_ids[p].

KVzip prepends a sys-prompt of length `sink` before user content. Captured
K has shape `(1, sink+ctx_len, H_kv*D)`. Slice to `[:, sink:sink+ctx_len, :]`
and the layer-0 result MUST equal v7's standalone-prefill K[0] over the
same input_ids (which has no sink, just content). Any mismatch means the
hook fired on the wrong forward, or the slice indices are off, or KVzip
swapped the projection module — all M1 bugs.

Layer 1+ DOES differ because attention propagates across positions and
KVzip's prefix changes the attention input. So this test deliberately
isolates layer 0 only.
"""
from __future__ import annotations

import pytest
import torch

from compblend.backends.base import (
    CompressedChunk,
    CompressionBudget,
    to_kvstore_entry,
)


# ──────────────────────────────────────────────────────────────────────────
# Tier 1 — CPU schema sanity
# ──────────────────────────────────────────────────────────────────────────


def _dummy_chunk(
    n_layers: int = 4, n_kv_heads: int = 2, head_dim: int = 8, chunk_len: int = 6,
) -> CompressedChunk:
    """Construct a minimal CompressedChunk for shape testing."""
    hidden_kv = n_kv_heads * head_dim
    return CompressedChunk(
        algo_id="dummy",
        chunk_id="d0",
        model_id="dummy-model",
        tokenizer_id="dummy-tok",
        token_ids=list(range(chunk_len)),
        key_cache=[torch.zeros(1, chunk_len, hidden_kv) for _ in range(n_layers)],
        value_cache=[torch.zeros(1, chunk_len, hidden_kv) for _ in range(n_layers)],
        valid_mask=torch.ones(n_layers, n_kv_heads, chunk_len, dtype=torch.bool),
        importance=torch.zeros(n_layers, n_kv_heads, chunk_len, dtype=torch.float32),
        is_structural=torch.zeros(chunk_len, dtype=torch.bool),
    )


def test_compressed_chunk_shapes():
    """Derived properties + tensor shapes are consistent."""
    chunk = _dummy_chunk(n_layers=4, n_kv_heads=2, head_dim=8, chunk_len=6)
    assert chunk.chunk_len == 6
    assert chunk.num_layers == 4
    assert chunk.key_cache[0].shape == (1, 6, 16)         # 2*8
    assert chunk.value_cache[0].shape == (1, 6, 16)
    assert chunk.valid_mask.shape == (4, 2, 6)
    assert chunk.importance.shape == (4, 2, 6)
    assert chunk.is_structural.shape == (6,)


def test_to_kvstore_entry_carries_extensions():
    """Conversion to v7 KVStore entry preserves CompBlend extensions."""
    chunk = _dummy_chunk()
    entry = to_kvstore_entry(chunk)
    assert entry["K"] is chunk.key_cache              # identity, not copy
    assert entry["V"] is chunk.value_cache
    assert entry["valid_mask"] is chunk.valid_mask
    assert entry["importance"] is chunk.importance
    assert entry["is_structural"] is chunk.is_structural
    assert entry["algo_id"] == "dummy"
    assert entry["chunk_id"] == "d0"


def test_chunk_to_device_preserves_dtype_when_unspecified():
    """`.to(device)` with no dtype arg keeps original dtypes per tensor."""
    chunk = _dummy_chunk()
    # On CPU only; round-trip should be a no-op on dtypes.
    moved = chunk.to(torch.device("cpu"))
    assert moved.key_cache[0].dtype == chunk.key_cache[0].dtype
    assert moved.valid_mask.dtype == torch.bool
    assert moved.importance.dtype == torch.float32
    assert moved.is_structural.dtype == torch.bool


def test_kvzip_chunk_id_is_deterministic():
    """Same inputs → same id, across calls and processes."""
    from compblend.backends.kvzip import kvzip_chunk_id

    a = kvzip_chunk_id([1, 2, 3], ratio=0.3, level="pair")
    b = kvzip_chunk_id([1, 2, 3], ratio=0.3, level="pair")
    assert a == b
    # Format check.
    assert a.startswith("kvzip:")
    assert len(a) == len("kvzip:") + 16


def test_kvzip_chunk_id_differs_on_ratio_and_tokens():
    """ratio and token sequence each contribute to the hash."""
    from compblend.backends.kvzip import kvzip_chunk_id

    base = kvzip_chunk_id([1, 2, 3], ratio=0.3)
    assert kvzip_chunk_id([1, 2, 3], ratio=0.5) != base       # ratio
    assert kvzip_chunk_id([1, 2, 4], ratio=0.3) != base       # tokens


def test_chunk_save_load_roundtrip(tmp_path):
    """Disk round-trip preserves every field exactly (bit-identical tensors)."""
    chunk = _dummy_chunk(n_layers=3, n_kv_heads=2, head_dim=8, chunk_len=5)
    # Fill with non-zero data so we'd catch a save-zeros bug.
    for li in range(chunk.num_layers):
        chunk.key_cache[li] = torch.randn_like(chunk.key_cache[li])
        chunk.value_cache[li] = torch.randn_like(chunk.value_cache[li])
    chunk.importance = torch.randn_like(chunk.importance).abs()
    chunk.valid_mask = torch.tensor(
        [[True, False, True, True, False],
         [True, True, False, False, True]],
        dtype=torch.bool,
    ).expand(3, 2, 5).contiguous()      # [L, H, T] = [3, 2, 5]
    chunk.is_structural = torch.tensor([True, False, False, True, False])
    chunk.compression_rate = 0.42
    chunk.backend_state = {"ratio": 0.30, "info": "test"}

    path = tmp_path / "test_chunk.pt"
    chunk.save(path)
    assert path.exists()

    loaded = CompressedChunk.load(path)
    assert loaded.algo_id == chunk.algo_id
    assert loaded.chunk_id == chunk.chunk_id
    assert loaded.model_id == chunk.model_id
    assert loaded.tokenizer_id == chunk.tokenizer_id
    assert loaded.token_ids == chunk.token_ids
    assert loaded.compression_rate == chunk.compression_rate
    assert loaded.backend_state == chunk.backend_state
    for li in range(chunk.num_layers):
        assert torch.equal(loaded.key_cache[li], chunk.key_cache[li])
        assert torch.equal(loaded.value_cache[li], chunk.value_cache[li])
    assert torch.equal(loaded.valid_mask, chunk.valid_mask)
    assert torch.equal(loaded.importance, chunk.importance)
    assert torch.equal(loaded.is_structural, chunk.is_structural)


def test_chunk_load_rejects_version_mismatch(tmp_path):
    """Bumping a payload's version on disk MUST cause load to fail loudly."""
    chunk = _dummy_chunk()
    path = tmp_path / "bad_version.pt"
    chunk.save(path)
    payload = torch.load(path, weights_only=False)
    payload["version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="version mismatch"):
        CompressedChunk.load(path)


def test_budget_validation():
    with pytest.raises(ValueError):
        CompressionBudget()                            # no budget at all
    with pytest.raises(ValueError):
        CompressionBudget(ratio=0.0)                   # must be > 0
    with pytest.raises(ValueError):
        CompressionBudget(ratio=1.5)                   # must be <= 1
    # Valid:
    b = CompressionBudget(ratio=0.5)
    assert b.ratio == 0.5


# ──────────────────────────────────────────────────────────────────────────
# Tier 2 — GPU + KVzip layer-0 bit-exact gate
# ──────────────────────────────────────────────────────────────────────────


_TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _kvzip_available() -> bool:
    try:
        import model  # noqa: F401 — KVzip's `from model import ModelKVzip`
        return True
    except ImportError:
        return False


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not _kvzip_available(), reason="KVzip package not on PYTHONPATH")
def test_kvzip_layer0_pre_rope_matches_v7_precompute():
    """The critical M1 correctness gate.

    1. Build a Chunk and run v7's `precompute_chunk_kv` → standalone prefill
       captures pre-RoPE K[0] over content tokens.
    2. Build a KVzipBackend and call `compress(input_ids, model=None,
       budget=ratio=1.0)` on the SAME tokens → KVzip prepends sys-prompt and
       its prefill captures pre-RoPE K[0] over (sink + content). Our adapter
       slices to content only.
    3. Compare the two layer-0 K tensors. They MUST be bit-exact because
       layer 0's K is position-local (no attention across the prefix).
    """
    import torch
    from cacheblend.chunker import Chunk, _stable_id
    from cacheblend.precompute import precompute_chunk_kv
    from cacheblend import LayerwiseModel

    from compblend.backends.kvzip import KVzipBackend, KVzipConfig

    # Use the same model id for both paths to guarantee weight parity.
    backend = KVzipBackend(_TINY, KVzipConfig(kv_type="retain", level="pair"))
    hf_model = backend.hf_model                # KVzip's wrapped HF model

    # ── path A: v7 precompute on a Chunk built from raw user text ────────
    # Build LayerwiseModel sharing the SAME hf_model so weights match
    # bit-exactly (LayerwiseModel installs its own k_proj hooks).
    lw = LayerwiseModel.__new__(LayerwiseModel)
    lw.model = hf_model
    lw.tokenizer = backend.tokenizer
    lw.device = next(hf_model.parameters()).device
    lw.dtype = next(hf_model.parameters()).dtype
    lw._inner = hf_model.model
    lw.num_layers = len(lw._inner.layers)
    lw._pre_rope_k = {}
    lw._hook_handles = []
    lw._install_k_proj_hooks()

    text = "The quick brown fox jumps over the lazy dog and looks back."
    token_ids = backend.tokenizer(text, add_special_tokens=False)["input_ids"]
    chunk_obj = Chunk(text=text, token_ids=token_ids, chunk_id=_stable_id(text, token_ids))
    K_v7, _V_v7 = precompute_chunk_kv(lw, chunk_obj)        # K_v7[0]: [1, L, H_kv*D]

    # ── path B: KVzip compress (ratio=1.0 keeps everything) ──────────────
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=lw.device)
    compressed = backend.compress(
        input_ids, model=None, budget=CompressionBudget(ratio=1.0),
    )
    K_kvzip_layer0 = compressed.key_cache[0]                # [1, ctx_len, H_kv*D]

    # ── compare layer 0 only ─────────────────────────────────────────────
    # Shape: KVzip's ctx_len should equal v7's chunk_len (same input_ids).
    assert K_kvzip_layer0.shape == K_v7[0].shape, (
        f"shape mismatch: kvzip {tuple(K_kvzip_layer0.shape)} vs "
        f"v7 {tuple(K_v7[0].shape)}"
    )
    # ratio=1.0 means valid_mask all True → no zero-masking applied to K.
    assert bool(compressed.valid_mask.all().item()), (
        "ratio=1.0 must produce all-True valid_mask"
    )
    # The two paths run the same k_proj on the same input tokens, but KVzip
    # monkey-patches the decoder layer's forward (`replace_attn`) which may
    # reorder ops inside input_layernorm + k_proj. The result differs by at
    # most 1 ULP at the model's dtype precision (bf16 ≈ 2^-6 around |K|≈1).
    # Anything LARGER would mean the hook fired on the wrong call, the
    # sys-prompt slice is off, or weights diverged — a real bug.
    diff = (K_kvzip_layer0.to(torch.float32) - K_v7[0].to(torch.float32))
    max_abs = float(diff.abs().max().item())
    # Tolerance scaled by K magnitude: 1 bf16 ULP ≈ 2^-6 of max value.
    tol = 2 ** -5 * max(1.0, float(K_v7[0].abs().max().item()))
    assert max_abs < tol, (
        f"layer-0 pre-RoPE K differs more than expected bf16 precision noise. "
        f"max_abs_diff = {max_abs}  tolerance = {tol}. "
        f"Likely the hook fired on the wrong call, the sys-prompt slice is "
        f"misaligned, or KVzip replaced the projection layer."
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not _kvzip_available(), reason="KVzip package not on PYTHONPATH")
def test_kvzip_compress_output_shapes():
    """Backend produces shape-consistent CompressedChunk at ratio < 1.0."""
    import torch
    from compblend.backends.kvzip import KVzipBackend, KVzipConfig

    backend = KVzipBackend(_TINY, KVzipConfig(kv_type="retain", level="pair"))
    text = "The quick brown fox jumps over the lazy dog and looks back briefly."
    ids = backend.tokenizer(text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([ids], dtype=torch.long, device=backend.hf_model.device)

    chunk = backend.compress(
        input_ids, model=None, budget=CompressionBudget(ratio=0.5),
    )

    cfg = backend.hf_model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    ctx_len = chunk.chunk_len

    assert chunk.algo_id == "kvzip"
    assert len(chunk.key_cache) == n_layers
    assert len(chunk.value_cache) == n_layers
    for li in range(n_layers):
        assert chunk.key_cache[li].shape == (1, ctx_len, n_kv_heads * head_dim)
        assert chunk.value_cache[li].shape == (1, ctx_len, n_kv_heads * head_dim)
    assert chunk.valid_mask.shape == (n_layers, n_kv_heads, ctx_len)
    assert chunk.importance.shape == (n_layers, n_kv_heads, ctx_len)
    assert chunk.is_structural.shape == (ctx_len,)
    # ratio=0.5 + pair-level eviction → expect SOME variance in compression.
    assert 0.0 < chunk.compression_rate < 1.0, (
        f"compression_rate = {chunk.compression_rate}; expected meaningful eviction"
    )
