#!/bin/bash
# Three cache-hit jobs, run sequentially to avoid GPU contention.
#   1. GROVER seeds at full budget -- its numbers are currently one draw from
#      a distribution spanning 0.37 Pearson (F11), and they appear in F5, the
#      head comparison and every fusion table.
#   2. d4 Uni-Mol v1 vs v2 -- caches built, test never ran.
#   3. Feature-level fusion on enhits -- implemented, never evaluated.
set -u
cd "$(dirname "$0")/.." || exit 1
log(){ echo "=== $* $(date +%H:%M:%S)"; }

log "1/3 GROVER seeds, full budget"
for s in 1 2 3 4; do
  d="work/enhits/runs/grover3400__seed0_v0.05_t0.05__n1893872__s${s}"
  [ -f "$d/metrics.json" ] && { echo "  skip s=$s"; continue; }
  python -m dockreg.train --pool enhits --backbone grover3400 --seed "$s" \
      --epochs 40 --patience 6 2>&1 | grep -E "^  test:" | sed "s/^/  s=$s /"
done
# unimol2 full-budget seeds too, as the comparison point
for s in 1 2; do
  d="work/enhits/runs/unimol2__seed0_v0.05_t0.05__n1893872__s${s}"
  [ -f "$d/metrics.json" ] && continue
  python -m dockreg.train --pool enhits --backbone unimol2 --seed "$s" \
      --epochs 40 --patience 6 2>&1 | grep -E "^  test:" | sed "s/^/  unimol2 s=$s /"
done

log "2/3 d4 Uni-Mol v1 vs v2"
for bb in unimol unimol2; do for n in 50000 200000 500000; do
  python -m dockreg.train --pool d4 --backbone "$bb" --max-train "$n" \
      --cache-budget 500000 --max-eval 200000 --epochs 40 --patience 6 2>&1 \
      | grep -E "^  test:" | sed "s/^/  $bb n=$n /"
done; done

log "3/3 feature-level fusion, enhits (200k budget to bound memory)"
for mode in concat pca; do
  python -m dockreg.featurefuse --pool enhits --mode "$mode" --max-train 200000 \
      --backbones unimol2,mhgged --epochs 40 --patience 6 2>&1 \
      | grep -E "^  test:|block widths" | sed "s/^/  pair-$mode /"
  python -m dockreg.featurefuse --pool enhits --mode "$mode" --max-train 200000 \
      --backbones descriptors,unimol2,mhgged,grover3400,smited,molformer \
      --epochs 40 --patience 6 2>&1 \
      | grep -E "^  test:|block widths" | sed "s/^/  all6-$mode /"
done
log "batch complete"
