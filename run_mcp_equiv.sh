#!/bin/bash
JOB_ID=${SLURM_JOB_ID:-$(date +%s)}

# Usage:
#   bash run_mcp_equiv.sh --data handwritten13
#   bash run_mcp_equiv.sh --random

# --- Parameters ---
DATA=""
RANDOM_FLAG=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)
            DATA="$2"
            shift 2
            ;;
        --random)
            RANDOM_FLAG=1
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$DATA" && "$RANDOM_FLAG" -eq 0 ]]; then
    echo "Usage: bash run_mcp_equiv.sh --data <data_kind> [options]"
    echo "       bash run_mcp_equiv.sh --random [options]"
    exit 1
fi

OUTDIR="outs/mcp_equiv"
if [[ "$RANDOM_FLAG" -eq 1 ]]; then
    OUTDIR="${OUTDIR}/random"
elif [[ -n "$DATA" ]]; then
    DATA_NAME="ucca_${DATA}"
    OUTDIR="${OUTDIR}/${DATA_NAME}"
fi

mkdir -p "$OUTDIR"
exec > "${OUTDIR}/${JOB_ID}.out" 2>&1

if [[ -n "$DATA" ]]; then
    python analysis/mcp_equiv/mcp_equiv.py --data "$DATA" "${EXTRA_ARGS[@]}"
else
    python analysis/mcp_equiv/mcp_equiv.py "${EXTRA_ARGS[@]}"
fi
