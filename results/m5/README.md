# M5 — quality regression results

## n118_kvzip_r010_2026-05-27 — Llama-3.1-8B-Instruct × MuSiQue × KVzip ratio=0.10

Reproducibility setup:

| Knob | Value |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` (fp16) |
| Dataset | `cacheblend-hf-v7/benchmarks/musique/inputs/musique_s.json` (first 118 of 150) |
| KVzip | `ratio=0.10`, `level=pair`, `kv_type=retain` |
| Fusor | `check_layer=1`, `recompute_ratio=0.15`, `gate_percentile=0.5`, `chunk_normalization=rank` |
| Hardware | A100 SXM4 80GB (vast.ai) |
| Commits | `compblend5 @ f225c9d`, `cacheblend-hf-v7 @ 3be92b7e`, `KVzip @ main 2026-05-26` |
| Disk-cache | offline-precompressed via `scripts/precompute_corpus.py` |

### Headline

```
ΔF1 (gated_top_k − hkvd_only) = +0.0177   CI95 [+0.0004, +0.0361]   ★ SIG@95
```

The 95% bootstrap CI lower bound is **above zero**, so the recovery effect
is statistically significant at α=0.05 on this sample. Direction +
significance match CompBlend-old's Phase 4b run 6 finding (+0.0348 ★ at
2WikiMQA n=500) but with smaller magnitude — see "discussion" below.

### Full numbers

```
Arm                      F1       TTFT (s)
─────────────────────────────────────────────
full_recompute           0.3156   0.513
KVzip + hkvd_only        0.0694   0.212
KVzip + gated_top_k      0.0872   0.211
```

- Compression cost: F1 dropped from 0.3156 → 0.0694 (−0.246, ~78% relative).
- Gated HKVD recovery: 7.2% of the compression-induced drop.
- TTFT speedup of the compressed arms: **~2.42×** over full prefill. Selector
  choice (hkvd vs gated) has no measurable TTFT difference.

### Files

- `n118_kvzip_r010_2026-05-27.json` — aggregate metrics + config
- `n118_per_question.tsv`            — per-question F1 lines (re-analysis input)

### Why the recovery is smaller than CompBlend-old's run 6

1. **Benchmark**: MuSiQue is harder than 2WikiMultihopQA (typically more
   docs per question, more hops). KVzip's compression damage is larger
   here (−0.246 vs −0.10 for 2WikiMQA at similar ratio), and the
   regime-dependence of Gated HKVD's gain is documented in CompBlend-old's
   Phase 4b sweet-spot analysis.
2. **N**: 118 vs 500. Wider CI (the lower bound just barely clears zero
   here, vs CI95 lower of +0.0016 there).
3. **F1 scale**: hkvd F1 is 0.07 here — very low absolute level. Many
   questions produce non-answer or refusal under heavy compression; Gated
   HKVD's incremental "right token kept" doesn't translate into many F1
   points when the floor is this low.

### Reproducing

```bash
# After cloning and installing per top-level README:
export PYTHONPATH=/path/to/KVzip:$PYTHONPATH
huggingface-cli login --token <hf_token>

# Offline (one-time per (model, ratio)) — populates cache_dir
COMPBLEND_RATIO_KVZIP=0.10 \
  python scripts/precompute_corpus.py

# Online — run the 3-arm benchmark
COMPBLEND_N=118 \
COMPBLEND_RATIO_KVZIP=0.10 \
COMPBLEND_CACHE_DIR=$PWD/cache/kvzip_musique_r0.10 \
  python benchmarks/musique_selector_compare.py
```
