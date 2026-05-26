# compblend5

Compressed KV cache blending. Uses `cacheblend-hf-v7` as the CacheBlend core
(vendored as a git submodule), and adds:

- KVzip / SnapKV / PyramidKV / H2O / KVzip+KIVI compression backends
- Gated HKVD selector (paper §3) — importance gate then HKVD top-k within
- per-(layer, head) eviction support carried through to attention masking

## Project status

**Stage 1 — accuracy first.** Goal: reproduce paper Phase 4b results with
v7-correct CacheBlend internals. BlendingCache is still dense at this stage.

**Stage 2 — compressed-native.** Goal: drop dense `[1, H_kv, N, D]`
workspace. Per-head varlen storage end-to-end + on-demand dequant for KIVI.
Paper claim "no full-cache decompression" only becomes true at this stage.

## Layout

```
src/
  compblend/           — this project's code (selectors, backends, fusor extensions)
  external/
    cacheblend-hf-v7/  — submodule. Verbatim CacheBlend core (LayerwiseModel,
                         KVStore, precompute, fuse_selective).
tests/                 — pytest. CPU smoke tests run by default; GPU tests
                         marked @pytest.mark.gpu and skipped without CUDA.
```

## Install (local, CPU — smoke testing only)

```bash
git clone --recursive <this-repo>
cd compblend5
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1
pip install -r src/external/cacheblend-hf-v7/requirements.txt
pip install -e src/external/cacheblend-hf-v7
pip install -e .
pytest tests/ -m 'not gpu'
```

## Install (GPU pod — vast.ai/Lambda)

Use the `pytorch:2.4.1-cuda12.4` base image. Then:

```bash
git clone --recursive <this-repo>
cd compblend5
grep -v -E '^torch(\s|=|$)' src/external/cacheblend-hf-v7/requirements.txt > /tmp/reqs.txt
pip install -r /tmp/reqs.txt
pip install -e src/external/cacheblend-hf-v7
pip install -e .
pytest tests/ -m gpu        # M0 bit-exact sanity test
```

## Provenance

- `src/external/cacheblend-hf-v7/` — pinned to a specific commit of
  <https://github.com/chjs/cacheblend-hf-v7>.
- Predecessor: `CompBlend-old` (github.com/chjs/CompBlend-old). Its engine,
  BlendingCache, dispatch, and engine_varlen were retired in favor of v7's
  `fuse_selective`. Selector logic and backend adapters are being ported
  incrementally as the milestones below complete.

## Milestones

| | Stage | Status |
|---|---|---|
| M0 | Scaffold + v7 submodule + smoke test | **done (3 CPU tests)** |
| M1 | KVzip backend — pre-RoPE K capture via k_proj forward-hook | **done (CPU); GPU bit-exact deferred to pod** |
| M2 | `fuse_selective_compblend` — v7 fork + per-head mask + Gated HKVD | **done (24 CPU unit tests)** |
| M3 | RAG pipeline rewire (KVStore + Chunks, no BlendingCache) | **done (CPU plumbing tests)** |
| M4 | Bit-exact sanity: KVzip ratio=1.0 ↔ v7 fuse_full_recompute | **code ready; runs on GPU pod** |
| M5 | KVzip selector regression (offline compress + online blend) | **code + launch script ready; overnight GPU run** |
| M6 | Migrate remaining backends (SnapKV, PyramidKV, H2O) | pending |
| M7 | KIVI on-demand layer-wise dequant (no full materialization) | pending |
| S2 | Stage 2 — varlen per-head storage, compressed-native end to end | pending |

## Stage 1 — how to run on a GPU pod

```bash
# 1. M0+M1+M4 sanity (TinyLlama; KVzip optional)
bash scripts/run_gpu_sanity.sh

# 2. M5 quality regression (Llama-3.1-8B + KVzip + MuSiQue)
#    Step A: offline corpus pre-compression (one-time)
#    Step B: online blend with two selectors + bootstrap CI
#    Both wrapped by run_m5_regression.sh:
bash scripts/run_m5_regression.sh
```

The M5 script runs `scripts/precompute_corpus.py` once (offline phase) to
populate `cache/kvzip_musique_r0.10/{chunk_id}.pt`, then launches
`benchmarks/musique_selector_compare.py` (online phase) which loads from
that cache. Both arms (`hkvd_only` vs `gated_top_k`) read the same disk
artifacts — compression is paid exactly once per unique doc, matching the
paper's offline/online split.
