#!/bin/bash
# Refit every backbone with two non-MLP surrogate families, on the same
# cached embeddings and the same split.
#
# Purpose: decide whether the backbone ordering in the main results is a
# property of the representation or of the prediction head. Ridge is linear;
# LightGBM splits on axes. If the ordering survives both, the head is not
# driving it.
#
# Budget note: ridge runs at the full training split (closed form from the
# Gram matrix, so cost is linear in rows). GBDT is capped, since histogram
# building scales with rows x features and GROVER has 3400 of them.

set -u
cd "$(dirname "$0")/.." || exit 1

POOL=${POOL:-enhits}
GBDT_TRAIN=${GBDT_TRAIN:-200000}
RIDGE_TRAIN=${RIDGE_TRAIN:-0}      # 0 = full split
BACKBONES=${BACKBONES:-"descriptors unimol2 mhgged grover3400 smited molformer"}

LOGDIR=work/$POOL/logs
mkdir -p "$LOGDIR"

for bb in $BACKBONES; do
  for fam in ridge gbdt; do
    if [ "$fam" = "ridge" ]; then n=$RIDGE_TRAIN; else n=$GBDT_TRAIN; fi
    log="$LOGDIR/alt_${bb}_${fam}.log"
    echo "[run ] $bb / $fam (train=$n)  $(date +%H:%M:%S)"
    python -m dockreg.surrogates --pool "$POOL" --backbone "$bb" \
        --family "$fam" --max-train "$n" > "$log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[FAIL] $bb/$fam -- see $log"; tail -3 "$log"; continue
    fi
    grep -E "^  test:" "$log" | sed 's/^/       /'
  done
done

echo
echo "surrogate check done $(date +%H:%M:%S)"
