"""Tokenization / KV-cache invariants for CompBlend experiments.

These helpers exist so a failed invariant is a LOUD error with diagnostics,
not a silent skip or a misleading benchmark number.

Invariants enforced:

    I1 (full vs chunk concat tokenization)
        tokenizer(full_prompt).input_ids == concat(c.token_ids for c in chunks)

    I2 (KVzip prefill roundtrip)
        For every doc chunk c that goes through KVzipBackend.compress(c.token_ids),
        the tokens KVzip actually prefills with (decode→encode roundtrip inside
        the backend) must equal c.token_ids exactly.

    I3 (storage length matches token count)
        compressed_chunk.key_cache[0].shape[1] == len(c.token_ids)

If any of these fail, downstream F1/TTFT measurements compare apples to oranges.
The benchmark scripts should call these BEFORE running any expensive forward.
"""
from __future__ import annotations

from typing import Sequence


class TokenizationInvariantError(ValueError):
    """Raised when the full-prompt and chunk-concat token sequences disagree."""


def assert_token_ids_equal(
    name: str,
    expected_ids: Sequence[int],
    actual_ids: Sequence[int],
    tokenizer,
) -> None:
    """Verify two token-id sequences are byte-for-byte equal.

    On mismatch raises with the first divergent index, the surrounding
    context, and a single-token decode for each side so the cause (BOS,
    chat-template, whitespace, BPE boundary) is visible at a glance.
    """
    expected = [int(t) for t in expected_ids]
    actual = [int(t) for t in actual_ids]
    if expected == actual:
        return

    lines = [f"[TOKENIZATION MISMATCH] {name}"]
    lines.append(f"  expected length: {len(expected)}")
    lines.append(f"  actual   length: {len(actual)}")

    common = min(len(expected), len(actual))
    first_diff = None
    for i in range(common):
        if expected[i] != actual[i]:
            first_diff = i
            break
    if first_diff is None and len(expected) != len(actual):
        first_diff = common  # divergence is at the length boundary

    if first_diff is not None:
        lo = max(0, first_diff - 3)
        hi_e = min(len(expected), first_diff + 4)
        hi_a = min(len(actual), first_diff + 4)
        lines.append(f"  first divergence at index: {first_diff}")
        if first_diff < len(expected):
            tok_e = expected[first_diff]
            lines.append(
                f"  expected id: {tok_e}  decode={tokenizer.decode([tok_e])!r}"
            )
        else:
            lines.append("  expected ran out at this index")
        if first_diff < len(actual):
            tok_a = actual[first_diff]
            lines.append(
                f"  actual   id: {tok_a}  decode={tokenizer.decode([tok_a])!r}"
            )
        else:
            lines.append("  actual ran out at this index")
        lines.append(f"  context expected[{lo}:{hi_e}]: {expected[lo:hi_e]}")
        lines.append(f"  context actual  [{lo}:{hi_a}]: {actual[lo:hi_a]}")

    raise TokenizationInvariantError("\n".join(lines))


def assert_full_equals_chunk_concat(
    name: str,
    full_text: str,
    chunks,
    tokenizer,
    add_special_tokens: bool = False,
) -> None:
    """Invariant I1.

    `full_text` is what the model would see if no chunking happened.
    `chunks` is the chunk list whose token_ids will be concatenated to feed
    the fuser. Pass the SAME add_special_tokens used to tokenize each chunk
    (default False — chunk benchmarks here prepend BOS manually).
    """
    full_ids = tokenizer(full_text, add_special_tokens=add_special_tokens)[
        "input_ids"
    ]
    concat_ids: list[int] = []
    for c in chunks:
        concat_ids.extend(c.token_ids)
    assert_token_ids_equal(name, full_ids, concat_ids, tokenizer)


def assert_kvzip_roundtrip_token_ids(
    name: str,
    chunk_token_ids: Sequence[int],
    kvzip_tokenizer,
) -> None:
    """Invariant I2.

    KVzip currently consumes text (its `mk.prefill(text)`), so the backend
    decodes the caller's ids and re-encodes them inside KVzip. If that
    roundtrip is not the identity, the K/V actually stored corresponds to a
    DIFFERENT prompt than the one the caller intended — fatal.

    `kvzip_tokenizer` is `mk.tokenizer` (the one KVzip uses internally).
    """
    text = kvzip_tokenizer.decode(list(chunk_token_ids), skip_special_tokens=False)
    reencoded = kvzip_tokenizer(text, add_special_tokens=False)["input_ids"]
    assert_token_ids_equal(
        f"{name} [KVzip decode→encode roundtrip]",
        chunk_token_ids,
        reencoded,
        kvzip_tokenizer,
    )


def assert_compressed_storage_matches_tokens(
    name: str,
    cmp_full,
    chunk_token_ids: Sequence[int],
) -> None:
    """Invariant I3.

    KVzip's internal ctx_len occasionally drifts from len(token_ids) by a
    special-token offset (driven by the decode→encode roundtrip above).
    Even if I2 passes, the captured K/V tensor's seq dim must match the
    caller's token count or the fuser will overlay onto wrong positions.
    """
    storage_len = int(cmp_full.key_cache[0].shape[1])
    if storage_len != len(chunk_token_ids):
        raise TokenizationInvariantError(
            f"[STORAGE LENGTH MISMATCH] {name}: "
            f"compressed K storage seq_len = {storage_len}, "
            f"but len(token_ids) = {len(chunk_token_ids)}. "
            f"Fuser would overlay onto positions [start, start+{len(chunk_token_ids)}) "
            f"with only {storage_len} stored entries."
        )
