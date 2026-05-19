#!/bin/bash
JOB_ID=${SLURM_JOB_ID:-$(date +%s)}

# Usage:
#   bash run_exp_hyperparam_tuning.sh --method sca --data handwritten13

export PYTHONUNBUFFERED=1

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
    echo "Usage: bash run_exp_hyperparam_tuning.sh --method <method> --data <dataset> [options]"
    exit 1
fi

OUTDIR="outs/tuning/${DATA}/${METHOD}"
mkdir -p "$OUTDIR"
exec > "${OUTDIR}/${JOB_ID}.out" 2>&1

echo "Host:   $(hostname)"
echo "Job ID: ${JOB_ID}"
echo "Start:  $(date)"
echo "Method: ${METHOD}"
echo "Data:   ${DATA}"

python exps/exp_hyperparam_tuning.py --method "$METHOD" --data "$DATA" "${EXTRA_ARGS[@]}"

echo "Done."
echo "End:    $(date)"
