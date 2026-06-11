#!/bin/bash
#PBS -N ShARC_shifted_baselines
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/shifted_baselines_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/shifted_baselines_err.log
#PBS -l select=1:ncpus=8
#PBS -q cpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC
mkdir -p logs results

for BASELINE in ILS ACO EA; do
  for SEVERITY in 0.0 0.2 0.4 0.6 0.8 1.0; do
    python baseline/run_baseline_at_shift.py \
      --baseline $BASELINE \
      --eval_dir /home/agriya.yadav_ug2023/ShARC/data/eval_dataset \
      --severities $SEVERITY \
      --n_seeds 5 \
      --seed 42 \
      --out_csv /home/agriya.yadav_ug2023/ShARC/results/shifted_baselines.csv
  done
done
