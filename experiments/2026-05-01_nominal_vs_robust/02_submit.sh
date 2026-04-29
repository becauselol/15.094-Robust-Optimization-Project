#!/bin/bash
#SBATCH -J nominal_vs_robust
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-648
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/job-%A_%a.out
#SBATCH -e slurm_logs/job-%A_%a.err

EXPERIMENT="experiments/2026-05-01_nominal_vs_robust"

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
EXP_DIR="$PROJECT_ROOT/$EXPERIMENT"

echo "===== Budget-Robust Station Selection Sweep ====="
echo "Experiment:   $EXP_DIR"
echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURM_NODELIST"
echo "Start time:   $(date)"
echo ""

module load julia/1.10.4
module load gurobi/12.0.3
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
julia scripts/03_run_job.jl "$EXP_DIR" "$JOB_ID"

EXIT_CODE=$?
echo ""
echo "===== Done — exit code $EXIT_CODE — $(date) ====="
exit $EXIT_CODE
