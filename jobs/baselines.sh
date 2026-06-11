#!/bin/bash
#PBS -N ShARC_baselines
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/baselines_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/baselines_err.log
#PBS -l select=1:ncpus=8
#PBS -q cpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC

mkdir -p logs results

python run_all_baselines.py \
  --path /home/agriya.yadav_ug2023/ShARC/data/eval_dataset \
  --out_csv /home/agriya.yadav_ug2023/ShARC/results/baselines_unshifted.csv \
  --ils_num_sample 20 \
  --ea_epochs 100 \
  --aco_epochs 100 \
  --seed 6868
