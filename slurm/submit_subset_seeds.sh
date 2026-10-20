#!/bin/bash

#SBATCH -J dr_subset
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -p general
#SBATCH -o logs/dr_subset_%j.txt
#SBATCH -e logs/dr_subset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
#SBATCH --mem=200G
#SBATCH -A r00939
#
# Training-subset resampling: the control Story A needs and does not have.
#
# WHY THIS AND NOT --seed. The five seeds already run vary model
# initialisation and batch shuffling while training on the SAME rows. At 250
# labels the dominant source of variability is *which* 250 molecules were
# drawn, which those seeds never touch. --subset-seed redraws the rows while
# holding val/test and the embedding cache fixed, separating the two variance
# sources instead of conflating them.
#
# WHAT IT DECIDES. The spread at 250 rows is 0.51 Pearson and will almost
# certainly survive. The question is the middle of the ranking: descriptors
# and GROVER sit 0.005 apart at 250 rows, well inside plausible subset noise.
# If that ordering shuffles across draws, the "rank reversal with budget"
# claim -- the sharpest element of Story A -- must be restated as a claim
# about the extremes only.
#
# ---------------------------------------------------------------------------
# WHY ONE FAT NODE RATHER THAN A 90-TASK ARRAY
#
# These are 250-5,000 row fits of a ~0.5M-parameter MLP: seconds of compute
# each, no GPU needed. An array would spend more time queueing than running.
# One node with many cores runs them concurrently via xargs -P.
#
# Memory is modest despite the large caches. --subset-seed materialises only
# the drawn rows (<= 5,000 x 3,400 x 4B = 68 MB); the larger cost is val/test,
# held fully in RAM at ~1.4 GB each for GROVER. Budget ~3 GB per concurrent
# worker, so 16 workers fit comfortably in 200 GB.
#
# CLUSTER CONSISTENCY. Every existing docking-regression run was produced on
# the shared VS Code node's A100. This job deliberately RE-RUNS subset draw 0
# rather than reusing those numbers, so all five draws come from one machine.
# Mixing them would confound hardware with treatment -- the trap documented in
# al-molecular/adaptive_acq/slurm/submit.sh, which matters here because the
# orderings in question differ by ~0.005. CPU also removes GPU nondeterminism
# from the comparison entirely.
# ---------------------------------------------------------------------------
#
# Submit (from the repo root, on a login node):
#   sbatch slurm/submit_subset_seeds.sh
#   PARALLEL=24 sbatch slurm/submit_subset_seeds.sh      # more workers

set -euo pipefail
# Repo-relative, not $0-relative: sbatch copies this script to the spool dir
# but starts it in SLURM_SUBMIT_DIR, so $0 points somewhere useless while a
# repo-relative path resolves. Same convention as al-molecular's scripts.
# Submit from the repo root.
source slurm/_preamble.sh

PARALLEL=${PARALLEL:-16}
# Each worker gets a slice of the allocation; leave headroom for the parent.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-64} / PARALLEL ))
[ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

BACKBONES="descriptors unimol2 mhgged grover3400 smited molformer"
BUDGETS="250 1000 5000"
DRAWS="0 1 2 3 4"

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} workers=$PARALLEL " \
     "threads/worker=$OMP_NUM_THREADS  start=$(date)"

# Emit one command per cell, then run them PARALLEL at a time.
for bb in $BACKBONES; do for n in $BUDGETS; do for u in $DRAWS; do
    name="${bb}__subset__n${n}__u${u}"
    [ -f "work/enhits/runs/${name}/metrics.json" ] && continue
    printf '%s\0' "python -m dockreg.train --pool enhits --backbone $bb \
--max-train $n --subset-seed $u --seed 0 --epochs 40 --patience 6 --cpu \
--run-name $name > logs/${name}.log 2>&1 || echo FAILED $name"
done; done; done | xargs -0 -P "$PARALLEL" -I{} bash -c '{}'

echo "done=$(date)"
echo "completed runs: $(ls -d work/enhits/runs/*__subset__* 2>/dev/null | wc -l) / 90"
