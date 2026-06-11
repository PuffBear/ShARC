#!/bin/bash
#PBS -N ShARC_abl_no_curr
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/abl_no_curriculum_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/abl_no_curriculum_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC
mkdir -p logs

python -m training.train \
  --run_name ablation_no_curriculum \
  --use_cvar True \
  --alpha 0.1 \
  --use_shift True \
  --shift_mode uniform \
  --use_budget_signal True \
  --d_shift 8 \
  --device cuda
