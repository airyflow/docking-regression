#!/bin/bash
# Fusion across budgets -- the gap flagged in IV.4b.
#
# Feature-level: the winning pair (Uni-Mol2 + MHG-GED) under both variants at
# every budget that already has unimodal runs. The all-six arms are left at
# 200k only: native concatenation is 6,740 dims, ~51 GB at full budget, and
# needs a memory-aware implementation first.
#
# Prediction-level: recomputed from saved predictions, no training needed.
set -u
cd "$(dirname "$0")/.." || exit 1
PAIR="unimol2,mhgged"
for n in 250 1000 5000 50000 1000000; do
  for mode in concat pca; do
    d="work/enhits/runs_featurefuse/pair-${mode}__seed0_v0.05_t0.05__n${n}__s0"
    [ -f "$d/metrics.json" ] && { echo "[skip] pair-$mode n=$n"; continue; }
    echo "=== pair-$mode n=$n  $(date +%H:%M:%S)"
    python -m dockreg.featurefuse --pool enhits --mode "$mode" \
        --backbones "$PAIR" --max-train "$n" --epochs 40 --patience 6 \
        --tag "pair-${mode}" 2>&1 | grep -E "^  test:" | sed 's/^/  /'
  done
done
echo "=== feature-level sweep done $(date +%H:%M:%S)"
