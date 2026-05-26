"""M3 — RAG pipeline plumbing (CPU). Full end-to-end is M4 (GPU)."""
from __future__ import annotations

import pytest
import torch

from cacheblend.kv_store import KVStore

from compblend.backends.base import (
    CompressedChunk,
    to_kvstore_entry,
)
from compblend.config import CompBlendConfig


def _minimal_chunk(chunk_id: str = "c0", chunk_len: int = 4) -> CompressedChunk:
    n_layers, n_kv_heads, head_dim = 2, 2, 4
    hidden_kv = n_kv_heads * head_dim
    return CompressedChunk(
        algo_id="test",
        chunk_id=chunk_id,
        model_id="m",
        tokenizer_id="t",
        token_ids=list(range(chunk_len)),
        key_cache=[torch.randn(1, chunk_len, hidden_kv) for _ in range(n_layers)],
        value_cache=[torch.randn(1, chunk_len, hidden_kv) for _ in range(n_layers)],
        valid_mask=torch.ones(n_layers, n_kv_heads, chunk_len, dtype=torch.bool),
        importance=torch.zeros(n_layers, n_kv_heads, chunk_len, dtype=torch.float32),
        is_structural=torch.zeros(chunk_len, dtype=torch.bool),
    )


# ──────────────────────────────────────────────────────────────────────────
# Direct KVStore insertion via to_kvstore_entry
# ──────────────────────────────────────────────────────────────────────────


def test_to_kvstore_entry_roundtrip_via_kvstore():
    """A chunk's extensions survive insertion into v7's KVStore + retrieval."""
    chunk = _minimal_chunk()
    store = KVStore()
    store._cache[chunk.chunk_id] = to_kvstore_entry(chunk)
    assert store.has(chunk.chunk_id)
    entry = store.get(chunk.chunk_id)
    # K and V are reachable.
    assert len(entry["K"]) == 2
    assert entry["K"][0].shape == (1, 4, 8)
    # CompBlend extensions retained as-is.
    assert torch.equal(entry["valid_mask"], chunk.valid_mask)
    assert torch.equal(entry["importance"], chunk.importance)
    assert torch.equal(entry["is_structural"], chunk.is_structural)


def test_chunk_normalization_config_roundtrip():
    """CompBlendConfig accepts chunk_normalization='rank' as a valid setting."""
    cfg = CompBlendConfig(chunk_normalization="rank")
    assert cfg.chunk_normalization == "rank"


def test_chunk_normalization_rejects_unknown():
    with pytest.raises(ValueError):
        CompBlendConfig(chunk_normalization="zscore")    # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────
# RAGPipeline import smoke
# ──────────────────────────────────────────────────────────────────────────


def test_rag_pipeline_importable():
    """Importing RAGPipeline pulls in fusor + selectors + base via the rag
    package — catches any circular/path issue early."""
    from compblend.rag import RAGPipeline, BlendResult
    assert RAGPipeline is not None
    assert BlendResult is not None
