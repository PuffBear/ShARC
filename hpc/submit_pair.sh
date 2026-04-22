#!/bin/bash
# ============================================================================
# submit_pair.sh — submit Run A (rn_shift) and Run B (cvar_shift) in parallel
#
# Cluster : 10.1.4.95  (ssh sharc)
# User    : agriya.yadav_ug2023
#
# Both jobs are identical except use_cvar. Submit from the project root:
#   bash hpc/submit_pair.sh
#
# After both complete, run the severity sweep locally:
#   python -m evaluation.severity_sweep \
#       --eval_dir  data/eval_dataset \
#       --cvar_ckpt experiments/results/cvar_shift/best.pt \
#       --rn_ckpt   experiments/results/rn_shift/best.pt \
#       --rn_use_shift True \
#       --out_csv   results/severity_sweep_v2.csv
# ============================================================================

set -euo pipefail

PROJECT="cc"   # change if needed

if [ ! -f "training/train.py" ]; then
    echo "ERROR: run from the ShARC project root (cd ~/ShARC first)"
    exit 1
fi

mkdir -p logs results

echo "Submitting training pair as agriya.yadav_ug2023 (project: $PROJECT)"
echo ""

JOB_A=$(qsub -P "$PROJECT" hpc/run_A_rn_shift.pbs)
echo "Submitted Run A  rn_shift:   $JOB_A"

JOB_B=$(qsub -P "$PROJECT" hpc/run_B_cvar_shift.pbs)
echo "Submitted Run B  cvar_shift: $JOB_B"

echo ""
echo "Both jobs running in parallel (no dependency). Monitor:"
echo "  qstat -a"
echo "  tail -f logs/run_A_rn_shift.out"
echo "  tail -f logs/run_B_cvar_shift.out"
