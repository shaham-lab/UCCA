#!/bin/bash

# Usage:
#   bash run_exp_total_correlation.sh --method ucca --data handwritten13

# Setup environment

export PYTHONUNBUFFERED=1

# Parameters
DATA=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)
            DATA="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$DATA" ]]; then
    echo "Usage: bash run_exp_total_correlation.sh --data <dataset> [options]"
    exit 1
fi

OUTDIR="outs/total_correlation/${DATA}/ucca"
mkdir -p "$OUTDIR"
exec > "${OUTDIR}/${JOB_ID:-local}.out" 2>&1

python exps/exp_total_correlation.py --data "$DATA" "${EXTRA_ARGS[@]}"
