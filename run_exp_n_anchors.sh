#!/bin/bash
JOB_ID=$(date +%s)

# Usage:
#   bash run_exp_n_anchors.sh --data handwritten23

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
    echo "Usage: bash run_exp_n_anchors.sh --data <dataset> [--new]"
    exit 1
fi

# Environment

# Execution
OUTDIR="outs/n_anchors/${DATA}/qap_cca"
mkdir -p "$OUTDIR"
exec > "${OUTDIR}/${JOB_ID:-local}.out" 2>&1

python exps/exp_n_anchors.py --data "$DATA" "${EXTRA_ARGS[@]}"
