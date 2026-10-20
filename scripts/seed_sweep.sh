#!/bin/bash
# Replication seeds for the Story A headline: does the budget-dependent rank
# reversal survive resampling?
#
# Varies --seed, which changes both the model init and the shuffling, while
# the split stays fixed. NOTE: this does NOT resample which rows land in the
# training subset -- at 250 rows that is the dominant source of variability,
# and it needs --split-seed variation, which would rebuild caches. Treat these
# as init/optimisation seeds and say so.
set -u
cd "$(dirname "$0")/.." || exit 1
POOL=${POOL:-enhits}
BUDGETS=${BUDGETS:-"250 1000 5000"}
BACKBONES=${BACKBONES:-"descriptors unimol2 mhgged grover3400 smited molformer"}
SEEDS=${SEEDS:-"1 2 3 4"}
for s in $SEEDS; do for n in $BUDGETS; do for bb in $BACKBONES; do
  d="work/$POOL/runs/${bb}__seed0_v0.05_t0.05__n${n}__s${s}"
  [ -f "$d/metrics.json" ] && continue
  python -m dockreg.train --pool "$POOL" --backbone "$bb" --max-train "$n" \
      --seed "$s" --epochs 40 --patience 6 >/dev/null 2>&1 \
      || echo "FAIL $bb n=$n s=$s"
done; done; echo "seeds through s=$s done $(date +%H:%M:%S)"; done
