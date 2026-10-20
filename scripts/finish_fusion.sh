#!/bin/bash
# The 3 feature-level fusion arms left unfinished when the batch was stopped.
# pair-concat already completed (pearson 0.9533).
set -u
cd "$(dirname "$0")/.." || exit 1
ALL="descriptors,unimol2,mhgged,grover3400,smited,molformer"
run(){ # mode, backbones, label
  echo "=== $3  $(date +%H:%M:%S)"
  python -m dockreg.featurefuse --pool enhits --mode "$1" --backbones "$2" \
      --max-train 200000 --epochs 40 --patience 6 --tag "$3" 2>&1 \
      | grep -E "block widths|^  params|^  test:" | sed "s/^/  /"
}
run concat "$ALL"              all6-concat
run pca    "unimol2,mhgged"    pair-pca
run pca    "$ALL"              all6-pca
echo "=== fusion arms complete $(date +%H:%M:%S)"
