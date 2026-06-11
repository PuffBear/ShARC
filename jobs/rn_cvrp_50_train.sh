#!/bin/bash
#PBS -N ShARC_rn_cvrp50
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/rn_cvrp50_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/rn_cvrp50_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd "$PBS_O_WORKDIR"
mkdir -p logs

# Generate CVRP-50 instances if not already done
if [ ! -d "data/cvrp50" ]; then
  python data/gen_cvrp.py \
    --n_nodes 50 \
    --n_instances 1000 \
    --out_dir data/cvrp50 \
    --seed 42
fi

python -m training.train \
  --problem cvrp \
  --run_name cvrp50_rn \
  --data_dir data/cvrp50 \
  --use_cvar False \
  --use_shift False \
  --device cuda
