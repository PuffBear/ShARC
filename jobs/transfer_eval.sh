#!/bin/bash
#PBS -N ShARC_transfer_eval
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/transfer_eval_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/transfer_eval_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd "$PBS_O_WORKDIR"
mkdir -p logs results

python -m evaluation.transfer_eval \
  --eval_dir_50   data/cvrp50/eval \
  --eval_dir_100  data/cvrp100/eval \
  --cvar_ckpt_50  experiments/results/cvrp50_cvar_shift/best.pt \
  --cvar_ckpt_100 experiments/results/cvrp100_cvar_shift/best.pt \
  --rn_ckpt_50    experiments/results/cvrp50_rn/best.pt \
  --rn_ckpt_100   experiments/results/cvrp100_rn/best.pt \
  --out_csv       results/transfer_eval.csv \
  --severities 0.0 0.2 0.4 0.6 0.8 1.0 \
  --n_seeds 5 \
  --seed_start 42 \
  --batch_size 32 \
  --device cuda
