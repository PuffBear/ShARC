#!/bin/bash
#PBS -N ShARC_abl_no_shift_ctx
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/abl_no_shift_ctx_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/abl_no_shift_ctx_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC
mkdir -p logs

python -m training.train \
  --run_name ablation_no_shift_ctx \
  --use_cvar True \
  --alpha 0.1 \
  --use_shift True \
  --shift_mode curriculum \
  --use_budget_signal True \
  --d_shift 0 \
  --device cuda
