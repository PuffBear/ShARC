#!/bin/bash
# ============================================================================
# submit_all.sh — submit all ShARC experiment steps to PBS
#
# Cluster : 10.1.4.95  (ssh sharc)
# User    : agriya.yadav_ug2023
#
# Usage (on the HPC login node, from the project root):
#   bash hpc/submit_all.sh
#
# Dependency graph:
#   step1 (baselines)   — independent, CPU queue
#   step2a (cvar_nom)   — independent, GPU queue
#   step2b (rn_nom)     —┐ both independent
#   step3  (cvar_shift) —┘ but step4 waits for both
#   step4  (sweep)      — afterok:step2b + afterok:step3
#
# Monitor:  qstat -a
#           qstat -f <jobid>
#           tail -f logs/step3_cvar_shift.out
# ============================================================================

set -euo pipefail

# ── Project code ────────────────────────────────────────────────────────────
# VERIFY THIS FIRST: run `qstat -Q` after `ssh sharc` and check what projects
# are listed for your account. Using a wrong code will silently reject jobs.
# The cluster docs use "cc" as their example — yours may differ.
PROJECT="cc"

# ── Sanity check: must run from project root ─────────────────────────────────
if [ ! -f "run_all_baselines.py" ]; then
    echo "ERROR: run this script from the ShARC project root, not from hpc/"
    echo "  cd ~/ShARC && bash hpc/submit_all.sh
# Venv on HPC is ShARCvenv (no underscore) — PBS scripts already use this name."
    exit 1
fi

mkdir -p logs results

echo "Submitting ShARC experiments as agriya.yadav_ug2023 (project: $PROJECT)"
echo ""

# ── Step 1: baselines (CPU, independent) ─────────────────────────────────────
JOB1=$(qsub -P "$PROJECT" hpc/step1_baselines.pbs)
echo "Submitted step1  baselines:    $JOB1"

# ── Step 2a: cvar_nominal (GPU, independent) ─────────────────────────────────
JOB2A=$(qsub -P "$PROJECT" hpc/step2a_cvar_nominal.pbs)
echo "Submitted step2a cvar_nominal: $JOB2A"

# ── Step 2b: rn_nominal (GPU, independent) ───────────────────────────────────
JOB2B=$(qsub -P "$PROJECT" hpc/step2b_rn_nominal.pbs)
echo "Submitted step2b rn_nominal:   $JOB2B"

# ── Step 3: cvar_shift (GPU, independent) ────────────────────────────────────
JOB3=$(qsub -P "$PROJECT" hpc/step3_cvar_shift.pbs)
echo "Submitted step3  cvar_shift:   $JOB3"

# ── Step 4: severity sweep — waits for step2b AND step3 ──────────────────────
JOB4=$(qsub -P "$PROJECT" -W depend=afterok:${JOB2B}:${JOB3} hpc/step4_sweep.pbs)
echo "Submitted step4  sweep:        $JOB4  [depends on $JOB2B, $JOB3]"

echo ""
echo "All submitted. Track progress:"
echo "  qstat -a"
echo "  tail -f logs/step3_cvar_shift.out"
echo "  tail -f logs/step4_sweep.out"
echo ""
echo "Results land in:  results/"
