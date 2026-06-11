#!/bin/bash
#PBS -N ShARC_abl_rn
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/abl_rn_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/abl_rn_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC
mkdir -p logs

python -m training.train \
  --run_name ablation_rn \
  --use_cvar False \
  --use_shift True \
  --shift_mode curriculum \
  --use_budget_signal True \
  --d_shift 8 \
  --device cuda
