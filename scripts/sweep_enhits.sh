#!/bin/bash
# Learning-curve sweep over every enhits backbone.
#
# Backbones run small-dim first so the cheap ones finish before GROVER's 28 GB
# gather starts. Within a backbone, budgets run ascending: the first one builds
# the full-split cache, the rest are prefix reads of it.
#
# Resumable -- a run whose metrics.json already exists is skipped.

set -u
cd "$(dirname "$0")/.." || exit 1

POOL=enhits
EPOCHS=40
PATIENCE=6
LOGDIR=work/$POOL/logs
mkdir -p "$LOGDIR"

BACKBONES=(descriptors molformer smited unimol2 mhgged grover3400)
BUDGETS=(50000 200000 1000000 0)

TAG="seed0_v0.05_t0.05"

for bb in "${BACKBONES[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    if [ "$budget" = "0" ]; then
      n=$(python - <<'PY'
import numpy as np
z = np.load("work/enhits/split_seed0_v0.05_t0.05.npz")
print(z["train"].size)
PY
)
    else
      n=$budget
    fi
    run="${bb}__${TAG}__n${n}__s0"
    if [ -f "work/$POOL/runs/$run/metrics.json" ]; then
      echo "[skip] $run"
      continue
    fi
    echo "[run ] $bb budget=$n  $(date +%H:%M:%S)"
    python -m dockreg.train --pool "$POOL" --backbone "$bb" \
        --max-train "$budget" --epochs "$EPOCHS" --patience "$PATIENCE" \
        > "$LOGDIR/${run}.log" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
      echo "[FAIL] $run (exit $status) -- see $LOGDIR/${run}.log"
      tail -5 "$LOGDIR/${run}.log"
      continue
    fi
    tail -3 "$LOGDIR/${run}.log" | grep -E "^  (val|test):" | sed 's/^/       /'
  done
done

echo
echo "sweep done $(date +%H:%M:%S)"
