#!/bin/bash
#PBS -N ShARC_cvrp50
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/cvrp50_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/cvrp50_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC
mkdir -p logs

# Generate CVRP-50 instances first if not already done
if [ ! -d "/home/agriya.yadav_ug2023/ShARC/data/cvrp50" ]; then
  python data/gen_cvrp.py \
    --n_nodes 50 \
    --n_instances 1000 \
    --out_dir /home/agriya.yadav_ug2023/ShARC/data/cvrp50 \
    --seed 42
fi

python -m training.train \
  --run_name cvrp50_cvar_shift \
  --data_dir /home/agriya.yadav_ug2023/ShARC/data/cvrp50 \
  --use_cvar True \
  --alpha 0.1 \
  --use_shift True \
  --shift_mode curriculum \
  --use_budget_signal True \
  --d_shift 8 \
  --device cuda
