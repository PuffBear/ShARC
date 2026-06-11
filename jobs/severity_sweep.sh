#!/bin/bash
#PBS -N ShARC_severity_sweep
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/severity_sweep_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/severity_sweep_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC

mkdir -p logs results

python -m evaluation.severity_sweep \
  --eval_dir  /home/agriya.yadav_ug2023/ShARC/data/eval_dataset \
  --cvar_ckpt /home/agriya.yadav_ug2023/ShARC/experiments/results/cvar_shift/best.pt \
  --rn_ckpt   /home/agriya.yadav_ug2023/ShARC/experiments/results/rn_nominal/best.pt \
  --out_csv   /home/agriya.yadav_ug2023/ShARC/results/severity_sweep.csv \
  --severities 0.0 0.2 0.4 0.6 0.8 1.0 \
  --n_seeds 5 \
  --device cuda
