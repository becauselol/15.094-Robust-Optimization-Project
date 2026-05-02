#!/bin/bash
#SBATCH -J subsample_40
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-12
#SBATCH --time=01:00:00
#SBATCH -o experiments/2026-05-01_sparse_subsample_40_v3/slurm_logs/job-%A_%a.out
#SBATCH -e experiments/2026-05-01_sparse_subsample_40_v3/slurm_logs/job-%A_%a.err

EXPERIMENT="experiments/2026-05-01_sparse_subsample_40_v3"

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
EXP_DIR="$PROJECT_ROOT/$EXPERIMENT"

echo "===== Subsample-40 Station Selection Sweep ====="
echo "Experiment:   $EXP_DIR"
echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURM_NODELIST"
echo "Start time:   $(date)"
echo ""

module load julia/1.12
module load gurobi/12
julia --version
echo ""

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo ""

JOB_FILE="$EXP_DIR/config/jobs.txt"
if [ ! -f "$JOB_FILE" ]; then
    echo "ERROR: job file not found: $JOB_FILE"
    echo "Run: julia scripts/01_setup.jl $EXPERIMENT"
    exit 1
fi

JOB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_FILE")
if [ -z "$JOB_ID" ]; then
    echo "ERROR: could not read job ID for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "===== Running job $JOB_ID ====="
mkdir -p "$EXP_DIR/slurm_logs"
srun --unbuffered julia scripts/03_run_job.jl "$EXP_DIR" "$JOB_ID"

EXIT_CODE=$?
echo ""
echo "===== Done — exit code $EXIT_CODE — $(date) ====="
exit $EXIT_CODE
