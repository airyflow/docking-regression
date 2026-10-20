# Sourced by every docking-regression slurm/*.sh: cd to the repo root, load
# config.env, activate the conda env, ensure logs/ exists.
#
# Mirrors al-molecular/adaptive_acq/slurm/_preamble.sh, including why
# BASH_SOURCE is not trusted under sbatch (the submitted copy may live
# elsewhere, so SLURM_SUBMIT_DIR is authoritative when set).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
fi
set -a
[ -f config.env ] && source config.env
set +a
: "${CONDA_SH:=/N/slate/mengjing/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=py310}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
mkdir -p logs

# Keep CPU runs from oversubscribing the allocation.
: "${OMP_NUM_THREADS:=${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
