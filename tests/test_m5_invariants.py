"""M5 — tokenization / KV-cache invariants (CPU-only).

These tests exercise the verification helpers used by Loong benchmarks to
guarantee that:
    * full-prompt tokenization equals chunk-concat tokenization (I1),
    * decode→encode roundtrip preserves token ids (I2),
    * compressed K storage seq dim matches len(token_ids) (I3).

If any of these flips silently in the future, F1/TTFT numbers would compare
different prompt sequences against each other. The tests are cheap; run them
on every CI.
"""
from __future__ import annotations

import pytest

from compblend.verification import (
    TokenizationInvariantError,
    assert_compressed_storage_matches_tokens,
    assert_kvzip_roundtrip_token_ids,
    assert_token_ids_equal,
)


class _StubTokenizer:
    """Tiny deterministic tokenizer for unit tests (no HF dependency)."""

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return " ".join(f"<{int(i)}>" for i in ids)

    def __call__(self, text: str, add_special_tokens: bool = False):
        toks = [int(t.strip("<>")) for t in text.split() if t.startswith("<")]
        return {"input_ids": toks}


def test_assert_token_ids_equal_passes_on_equal_lists():
    tok = _StubTokenizer()
    assert_token_ids_equal("same", [1, 2, 3], [1, 2, 3], tok)


def test_assert_token_ids_equal_reports_first_diff_index():
    tok = _StubTokenizer()
    with pytest.raises(TokenizationInvariantError) as exc:
        assert_token_ids_equal("name", [1, 2, 3, 4], [1, 2, 9, 4], tok)
    msg = str(exc.value)
    assert "first divergence at index: 2" in msg
    assert "expected id: 3" in msg
    assert "actual   id: 9" in msg


def test_assert_token_ids_equal_reports_length_mismatch():
    tok = _StubTokenizer()
    with pytest.raises(TokenizationInvariantError) as exc:
        assert_token_ids_equal("name", [1, 2, 3], [1, 2, 3, 4], tok)
    msg = str(exc.value)
    assert "expected length: 3" in msg
    assert "actual   length: 4" in msg


class _DriftingTokenizer:
    """decode→encode is NOT identity (mimics rare KVzip drift)."""

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return " ".join(f"<{int(i)}>" for i in ids)

    def __call__(self, text: str, add_special_tokens: bool = False):
        toks = [int(t.strip("<>")) for t in text.split() if t.startswith("<")]
        # Simulate drift: drop a token in the middle.
        if len(toks) > 3:
            toks = toks[: len(toks) // 2] + toks[len(toks) // 2 + 1 :]
        return {"input_ids": toks}


def test_kvzip_roundtrip_detects_drift():
    tok = _DriftingTokenizer()
    with pytest.raises(TokenizationInvariantError) as exc:
        assert_kvzip_roundtrip_token_ids("test_chunk", [1, 2, 3, 4, 5, 6], tok)
    assert "KVzip decode→encode roundtrip" in str(exc.value)


def test_kvzip_roundtrip_passes_when_identity():
    class _IdentityTokenizer(_StubTokenizer):
        pass

    tok = _IdentityTokenizer()
    assert_kvzip_roundtrip_token_ids("test_chunk", [1, 2, 3], tok)


class _FakeCmp:
    def __init__(self, storage_len: int):
        import torch

        self.key_cache = [torch.zeros((1, storage_len, 1024))]


def test_storage_length_mismatch_detected():
    with pytest.raises(TokenizationInvariantError) as exc:
        assert_compressed_storage_matches_tokens(
            "chunk_x", _FakeCmp(storage_len=20201), [0] * 20202
        )
    assert "20201" in str(exc.value) and "20202" in str(exc.value)


def test_storage_length_match_passes():
    assert_compressed_storage_matches_tokens(
        "chunk_x", _FakeCmp(storage_len=100), [0] * 100
    )
