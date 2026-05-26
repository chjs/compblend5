#!/usr/bin/env bash
# scripts/run_m5_regression.sh — overnight M5 quality regression on a GPU pod.
#
# Prereqs (same as scripts/run_gpu_sanity.sh + KVzip + HF token):
#   * Repo cloned --recursive
#   * `bash scripts/run_gpu_sanity.sh` has passed (M0/M1/M4 green)
#   * KVzip on PYTHONPATH:
#       git clone https://github.com/snu-mllab/KVzip /opt/KVzip
#       export PYTHONPATH=/opt/KVzip:$PYTHONPATH
#   * HF_TOKEN exported (Llama-3.1-8B-Instruct is gated)
#       huggingface-cli login --token $HF_TOKEN
#
# Time budget (rough, A100 80GB, n=150 MuSiQue):
#   Model load: ~2 min  (Llama-3.1-8B fp16)
#   KVzip load: ~2 min  (separate model copy via ModelKVzip)
#   Per question: ~25-40s (3 arms: full + KVzip-hkvd + KVzip-gated)
#                  ↑ KVzip compress dominates: ~3-5s per doc × 3 docs × 1 question
#   Total: ~1.5-2.5h for n=150.
#
# For n=500 (Phase 4b run 6 target), bump COMPBLEND_N=500. Time ~5-8h.

set -euo pipefail

# Sanity guards before launching the long run.
if ! python -c "import model" 2>/dev/null; then
    echo "FATAL: KVzip not importable. PYTHONPATH=$PYTHONPATH"
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "FATAL: not authenticated with Hugging Face (Llama-3.1 is gated)."
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# Defaults match CompBlend-old's Phase 4b run 6 EXCEPT n (set via env).
: "${COMPBLEND_N:=150}"
: "${COMPBLEND_RATIO_KVZIP:=0.10}"
: "${COMPBLEND_RECOMP_RATIO:=0.15}"
: "${COMPBLEND_GATE_PCT:=0.5}"
: "${COMPBLEND_CHECK_LAYER:=1}"
export COMPBLEND_N COMPBLEND_RATIO_KVZIP COMPBLEND_RECOMP_RATIO COMPBLEND_GATE_PCT COMPBLEND_CHECK_LAYER

echo "[m5] launching n=$COMPBLEND_N  kvzip_ratio=$COMPBLEND_RATIO_KVZIP  "
echo "     recomp_ratio=$COMPBLEND_RECOMP_RATIO  gate_pct=$COMPBLEND_GATE_PCT"

# Offline corpus pre-compression. This is the paper's "offline phase" — every
# unique doc gets compressed ONCE and saved to disk, then both selector arms
# in the online phase load the same artifact. Skip if cache_dir is already
# populated (precompute is idempotent).
: "${COMPBLEND_CACHE_DIR:=$PWD/cache/kvzip_musique_r${COMPBLEND_RATIO_KVZIP}}"
export COMPBLEND_CACHE_DIR
if [[ ! -d "$COMPBLEND_CACHE_DIR" ]] || [[ -z "$(ls -A "$COMPBLEND_CACHE_DIR" 2>/dev/null)" ]]; then
    echo "[m5] precompute pass first (one-time)..."
    python scripts/precompute_corpus.py 2>&1 | tee "logs/m5-precompute-$(date +%Y%m%d-%H%M%S).log"
else
    echo "[m5] reusing cache at $COMPBLEND_CACHE_DIR"
fi

# Use nohup so SSH disconnect doesn't kill the run.
LOG=logs/m5-n${COMPBLEND_N}-r${COMPBLEND_RATIO_KVZIP}-$(date +%Y%m%d-%H%M%S).log
nohup python benchmarks/musique_selector_compare.py > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID  LOG=$LOG"
echo "Tail progress:  tail -f $LOG"
echo "Summary JSON will land at: logs/musique_compare.json (or COMPBLEND_OUT override)"
