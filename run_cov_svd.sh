#!/bin/bash
JOB_ID=$(date +%s)

# Usage:
#   bash run_cov_svd.sh <dataset> [<view1> <view2>] [--cov] [--corr]
# Example:
#   bash run_cov_svd.sh qm9
#   bash run_cov_svd.sh handwritten 1 3 --cov

DATASET=$1
if [ -z "$DATASET" ]; then
    echo "Usage: bash run_cov_svd.sh <dataset> [<view1> <view2>] [--cov] [--corr]"
    exit 1
fi
shift

# Collect remaining arguments
ARGS=""
while (( "$#" )); do
    ARGS="$ARGS $1"
    shift
done

OUTDIR="outs/cov_svd"
mkdir -p "$OUTDIR"
# Use dataset name for log file if not running under slurm
LOG_NAME=${JOB_ID:-$DATASET}
exec > "${OUTDIR}/${LOG_NAME}.out" 2>&1

# Setup paths
PROJECT_ROOT=$(pwd)

echo "Host:   $(hostname)"
echo "Job ID: ${JOB_ID:-local}"
echo "Start:  $(date)"
echo "Dataset: $DATASET"
echo "Args:    $ARGS"

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Activate Environment
export PYTHONUNBUFFERED=1

echo "----------------------------------------------------------------"
echo "Running Cov SVD Analysis..."
echo "----------------------------------------------------------------"

python analysis/cov_svd/cov_svd.py "$DATASET" $ARGS

echo "----------------------------------------------------------------"
echo "Plotting Results..."
echo "----------------------------------------------------------------"

python analysis/cov_svd/plot_cov_svd.py "$DATASET" $ARGS

echo "----------------------------------------------------------------"
echo "Done."
echo "End:    $(date)"
echo "----------------------------------------------------------------"
