#!/bin/bash
#PBS -N ShARC_ablation_sweep
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/ablation_sweep_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/ablation_sweep_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd "$PBS_O_WORKDIR"
mkdir -p logs results

python -m evaluation.ablation_sweep \
  --eval_dir  data/eval_dataset \
  --ckpt_dir  experiments/results \
  --out_csv   results/ablation_sweep.csv \
  --severities 0.0 0.2 0.4 0.6 0.8 1.0 \
  --n_seeds 5 \
  --seed_start 42 \
  --batch_size 32 \
  --ils_csv results/shifted_baselines.csv \
  --device cuda
