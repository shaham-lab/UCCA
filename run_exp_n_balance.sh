#!/bin/bash
JOB_ID=$(date +%s)

# Usage:
#   bash run_exp_n_balance.sh --method sca --data handwritten13 --subset_view 1

# --- Parameters ---
METHOD=""
DATA=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            shift 2
            ;;
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

if [[ -z "$METHOD" || -z "$DATA" ]]; then
    echo "Usage: bash run_exp_n_balance.sh --method <method> --data <dataset> [EXTRA_ARGS]"
    exit 1
fi

# Environment

# Execution
OUTDIR="outs/n_balance/${DATA}/${METHOD}"
mkdir -p "$OUTDIR"
exec > "${OUTDIR}/${JOB_ID:-local}.out" 2>&1

python exps/exp_n_balance.py --method "$METHOD" --data "$DATA" "${EXTRA_ARGS[@]}"
