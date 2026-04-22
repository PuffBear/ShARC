#!/bin/bash
# ============================================================================
# submit_v2.sh — submit the v2 experiment (matched configs, gradient fix)
#
# Cluster : 10.1.4.95  (ssh sharc)
# User    : agriya.yadav_ug2023
#
# Run A (rn_v2)   + Run B (cvar_v2)   → parallel
# Run C (sweep)                       → after both A and B complete
#
# Usage (on HPC login node, from project root):
#   bash hpc/submit_v2.sh
# ============================================================================

set -euo pipefail

PROJECT="cc"

if [ ! -f "training/train.py" ]; then
    echo "ERROR: run from the ShARC project root (cd ~/ShARC first)"
    exit 1
fi

mkdir -p logs results

echo "Submitting v2 experiments as agriya.yadav_ug2023 (project: $PROJECT)"
echo ""

# ── Training (parallel) ─────────────────────────────────────────────────────
JOB_A=$(qsub -P "$PROJECT" hpc/run_A_rn_v2.pbs)
echo "Submitted Run A  rn_v2:     $JOB_A"

JOB_B=$(qsub -P "$PROJECT" hpc/run_B_cvar_v2.pbs)
echo "Submitted Run B  cvar_v2:   $JOB_B"

# ── Severity sweep (wait for both training jobs) ────────────────────────────
JOB_C=$(qsub -P "$PROJECT" -W depend=afterok:${JOB_A}:${JOB_B} hpc/run_C_sweep_v2.pbs)
echo "Submitted Run C  sweep_v2:  $JOB_C  [depends on $JOB_A, $JOB_B]"

echo ""
echo "Monitor:"
echo "  qstat -a"
echo "  tail -f logs/run_A_rn_v2.out"
echo "  tail -f logs/run_B_cvar_v2.out"
echo ""
echo "Results: results/severity_sweep_v2.csv"
