#!/bin/bash
JOB_ID=$(date +%s)

# Usage:
#   bash run_exp_n_unpairs.sh --method qap_cca --data handwritten23

EXP_SCRIPT="exps/exp_n_unpairs.py"
METHOD=""
DATA=""
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --data)
            DATA="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$METHOD" || -z "$DATA" ]]; then
    echo "Usage: bash run_exp_n_unpairs.sh --method <method> --data <data>"
    exit 1
fi

# Redirect output to outs/n_unpairs/data/method/<jobid>.out
OUTDIR="outs/n_unpairs/${DATA}/${METHOD}"
mkdir -p "$OUTDIR"
exec > "${OUTDIR}/${JOB_ID:-$$}.out" 2>&1

echo "Host: $(hostname)"
echo "Job ID: ${JOB_ID:-local}"
echo "Script: $EXP_SCRIPT"
echo "Args:   ${FORWARD_ARGS[*]}"
echo "Start:  $(date)"

nvidia-smi 2>/dev/null || true

export PYTHONUNBUFFERED=1

python "$EXP_SCRIPT" "${FORWARD_ARGS[@]}"
