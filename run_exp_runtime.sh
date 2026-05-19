#!/bin/bash
JOB_ID=$(date +%s)

# Usage:
#   bash run_exp_runtime.sh --data handwritten23 [--qap] [--values 1 5 10]

# --- Parameters ---
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
    echo "Usage: bash run_exp_runtime.sh --data <dataset> [--qap] [--values ...]"
    exit 1
fi

# Environment

# Execution
OUTDIR="outs/runtime/${DATA}"
mkdir -p "$OUTDIR"
exec > "${OUTDIR}/${JOB_ID:-$$}.out" 2>&1

python exps/exp_runtime.py --data "$DATA" "${EXTRA_ARGS[@]}"
