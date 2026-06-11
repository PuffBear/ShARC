#!/bin/bash
#PBS -N ShARC_abl_full
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/abl_full_sharc_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/abl_full_sharc_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC
mkdir -p logs

python -m training.train \
  --run_name ablation_full_sharc \
  --use_cvar True \
  --alpha 0.1 \
  --use_shift True \
  --shift_mode curriculum \
  --use_budget_signal True \
  --d_shift 8 \
  --device cuda
