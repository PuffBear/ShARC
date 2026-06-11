# ShARC — Missing Script Specifications

Specs for every script that must be written before all queued PBS jobs can complete and the paper draft can be assembled. Scripts are ordered by dependency (earlier entries unblock later ones).

---

## 1. `data/gen_cvrp.py`

**Needed by:** `jobs/cvrp_50_train.sh`, `jobs/cvrp_100_train.sh`

**Purpose:** Generate CVRP instances with demand/cost/availability shift support, serialised as `.npz` files compatible with the existing `training.train` data loader.

**CLI**
```
python data/gen_cvrp.py \
  --n_nodes 50 \
  --n_instances 1000 \
  --out_dir data/cvrp50 \
  --seed 42
```

**Arguments**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--n_nodes` | int | required | Number of customer nodes |
| `--n_instances` | int | 1000 | Training + eval instances to generate |
| `--out_dir` | str | required | Output directory |
| `--seed` | int | 42 | RNG seed |
| `--capacity` | float | 1.0 | Vehicle capacity (normalised) |
| `--demand_scale` | float | 0.1 | Mean demand per node |

**Output format per instance** — `.npz` with keys matching what `training/train.py` already expects:

| Key | Shape | Description |
|---|---|---|
| `coords` | `(N, 2)` | Node xy coordinates in [0,1] |
| `demands` | `(N,)` | Normalised demands |
| `depot` | `(2,)` | Depot coordinates |
| `capacity` | scalar | Vehicle capacity |

**Notes**
- Depot is node 0; customer nodes are 1..N.
- Coordinates sampled uniformly in [0,1]².
- Demands sampled from Uniform(0, `demand_scale`).
- The shift machinery (`env/shift.py`) is applied at training time, not generation time — gen_cvrp.py produces clean nominal instances only.
- Split: 90% train, 10% eval; write separate `train/` and `eval/` subdirectories inside `out_dir`.

---

## 2. `baseline/run_baseline_at_shift.py`

**Needed by:** `jobs/shifted_baselines.sh`

**Purpose:** Run a single classical baseline (ILS, ACO, or EA) on the eval dataset under a specified shift severity, write results to a CSV row. `shifted_baselines.sh` calls this in a nested loop over baselines × severity levels.

**CLI**
```
python baseline/run_baseline_at_shift.py \
  --baseline ILS \
  --eval_dir data/eval_dataset \
  --severity 0.4 \
  --n_seeds 5 \
  --seed 42 \
  --out_csv results/shifted_baselines.csv
```

**Arguments**

| Flag | Type | Description |
|---|---|---|
| `--baseline` | str | One of `ILS`, `ACO`, `EA` |
| `--eval_dir` | str | Path to eval .npz instances |
| `--severity` | float | Shift severity in [0,1] — maps directly to `ShiftConfig.severity` |
| `--n_seeds` | int | Number of random seeds to average over |
| `--seed` | int | Base seed (seed+i for seed i) |
| `--out_csv` | str | Append-mode CSV output path |

**Logic**
1. Load all `.npz` files from `eval_dir`.
2. For each instance, apply `ShiftScheduler` with `mode="adversarial"` and the given severity to produce shifted demands/costs.
3. Run the specified baseline solver on the shifted instance.
4. Compute `makespan`, `cvar_10` (CVaR at α=0.1 over seeds), `worst_case` across seeds.
5. Append one row per (baseline, severity, instance) to `out_csv`. CSV must be safe for concurrent writes — use file locking or write to a temp file and append atomically.

**Output CSV columns:** `baseline, severity, instance_id, seed, makespan_T1, makespan_T2, makespan_T3, cvar_10, worst_case`

**Notes**
- Reuse solver classes from `baseline/ils.py`, `baseline/aco.py`, `baseline/ea.py` — do not re-implement solvers here.
- If `out_csv` already has a row for this (baseline, severity, instance_id, seed), skip and continue (idempotent).

---

## 3. `evaluation/ablation_sweep.py`

**Needed by:** `jobs/ablation_sweep.sh` (to be written — see §7)

**Purpose:** Load all ablation checkpoints, run each on the eval set across severity levels, and produce the ablation comparison table used in Section 5 of the paper.

**CLI**
```
python -m evaluation.ablation_sweep \
  --eval_dir data/eval_dataset \
  --ckpt_dir experiments/results \
  --out_csv results/ablation_sweep.csv \
  --severities 0.0 0.2 0.4 0.6 0.8 1.0 \
  --n_seeds 5 \
  --device cuda
```

**Arguments**

| Flag | Type | Description |
|---|---|---|
| `--eval_dir` | str | Path to HCARP eval instances |
| `--ckpt_dir` | str | Directory containing ablation run subdirs |
| `--out_csv` | str | Output path |
| `--severities` | list[float] | Severity levels to evaluate |
| `--n_seeds` | int | Seeds per severity |
| `--device` | str | `cuda` or `cpu` |

**Expected checkpoint layout inside `ckpt_dir`:**
```
experiments/results/
  ablation_full_sharc/best.pt
  ablation_rn/best.pt
  ablation_no_shift_ctx/best.pt
  ablation_no_budget_sig/best.pt
  ablation_alpha_005/best.pt
  ablation_alpha_020/best.pt
  ablation_no_curriculum/best.pt
```

**Logic**
1. Glob `ckpt_dir/**/best.pt` to discover all ablation runs.
2. For each checkpoint × severity × seed: roll out the policy on the eval set using `evaluation/shift_evaluator.py`'s `sweep()` method.
3. Compute mean makespan, CVaR@0.1, worst-case makespan, and gap-to-baseline (ILS unshifted as oracle).
4. Write one row per (run_name, severity) to `out_csv`.

**Output CSV columns:** `run_name, severity, mean_makespan, cvar_10, worst_case, gap_to_ils`

---

## 4. `evaluation/transfer_eval.py`

**Needed by:** `jobs/transfer_eval.sh` (to be written — see §8)

**Purpose:** Evaluate ShARC trained on CVRP-50 and CVRP-100 against a risk-neutral (RN) CVRP baseline across shift severities. This is the primary generality experiment Prof. Cao requested.

**CLI**
```
python -m evaluation.transfer_eval \
  --eval_dir_50  data/cvrp50/eval \
  --eval_dir_100 data/cvrp100/eval \
  --cvar_ckpt_50  experiments/results/cvrp50_cvar_shift/best.pt \
  --cvar_ckpt_100 experiments/results/cvrp100_cvar_shift/best.pt \
  --rn_ckpt_50    experiments/results/cvrp50_rn/best.pt \
  --rn_ckpt_100   experiments/results/cvrp100_rn/best.pt \
  --out_csv results/transfer_eval.csv \
  --severities 0.0 0.2 0.4 0.6 0.8 1.0 \
  --n_seeds 5 \
  --device cuda
```

**Arguments** — all paths above, plus `--n_seeds`, `--severities`, `--device`.

**Logic**
1. For each problem size (50, 100):
   - Load CVaR checkpoint and RN checkpoint.
   - For each severity × seed: roll out both policies on the corresponding eval set.
   - Record mean makespan, CVaR@0.1, worst-case, and `delta_cvar = cvar_rn - cvar_sharc` (positive = ShARC better).
2. Write results to `out_csv`.

**Output CSV columns:** `problem, n_nodes, method, severity, mean_makespan, cvar_10, worst_case, delta_cvar`

**Notes**
- RN checkpoints come from `jobs/rn_cvrp_50_train.sh` and `jobs/rn_cvrp_100_train.sh` (see §5–6 below).
- If either RN checkpoint is missing, warn and skip that row rather than crashing.
- This script must be runnable on CPU (for quick sanity checks) — guard `--device cuda` with a `torch.cuda.is_available()` fallback.

---

## 5. `jobs/rn_cvrp_50_train.sh`

**Purpose:** Train a risk-neutral CVRP-50 baseline for comparison in transfer_eval.

```bash
#!/bin/bash
#PBS -N ShARC_rn_cvrp50
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/rn_cvrp50_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/rn_cvrp50_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC
mkdir -p logs

if [ ! -d "/home/agriya.yadav_ug2023/ShARC/data/cvrp50" ]; then
  python data/gen_cvrp.py \
    --n_nodes 50 \
    --n_instances 1000 \
    --out_dir /home/agriya.yadav_ug2023/ShARC/data/cvrp50 \
    --seed 42
fi

python -m training.train \
  --run_name cvrp50_rn \
  --data_dir /home/agriya.yadav_ug2023/ShARC/data/cvrp50 \
  --use_cvar False \
  --use_shift False \
  --use_budget_signal False \
  --device cuda
```

---

## 6. `jobs/rn_cvrp_100_train.sh`

Same as §5 but for CVRP-100:

```bash
#!/bin/bash
#PBS -N ShARC_rn_cvrp100
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/rn_cvrp100_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/rn_cvrp100_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC
mkdir -p logs

if [ ! -d "/home/agriya.yadav_ug2023/ShARC/data/cvrp100" ]; then
  python data/gen_cvrp.py \
    --n_nodes 100 \
    --n_instances 1000 \
    --out_dir /home/agriya.yadav_ug2023/ShARC/data/cvrp100 \
    --seed 42
fi

python -m training.train \
  --run_name cvrp100_rn \
  --data_dir /home/agriya.yadav_ug2023/ShARC/data/cvrp100 \
  --use_cvar False \
  --use_shift False \
  --use_budget_signal False \
  --device cuda
```

---

## 7. `jobs/ablation_sweep.sh`

**Run after:** all 7 `jobs/ablation_*.sh` training jobs complete.

```bash
#!/bin/bash
#PBS -N ShARC_ablation_sweep
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/ablation_sweep_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/ablation_sweep_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC
mkdir -p logs results

python -m evaluation.ablation_sweep \
  --eval_dir  /home/agriya.yadav_ug2023/ShARC/data/eval_dataset \
  --ckpt_dir  /home/agriya.yadav_ug2023/ShARC/experiments/results \
  --out_csv   /home/agriya.yadav_ug2023/ShARC/results/ablation_sweep.csv \
  --severities 0.0 0.2 0.4 0.6 0.8 1.0 \
  --n_seeds 5 \
  --device cuda
```

---

## 8. `jobs/transfer_eval.sh`

**Run after:** `cvrp_50_train.sh`, `cvrp_100_train.sh`, `rn_cvrp_50_train.sh`, `rn_cvrp_100_train.sh` all complete.

```bash
#!/bin/bash
#PBS -N ShARC_transfer_eval
#PBS -o /home/agriya.yadav_ug2023/ShARC/logs/transfer_eval_out.log
#PBS -e /home/agriya.yadav_ug2023/ShARC/logs/transfer_eval_err.log
#PBS -l select=1:ncpus=4
#PBS -q gpu

module load compiler/anaconda3
source /home/agriya.yadav_ug2023/ShARC/ShARCvenv/bin/activate

cd /home/agriya.yadav_ug2023/ShARC
mkdir -p logs results

python -m evaluation.transfer_eval \
  --eval_dir_50  /home/agriya.yadav_ug2023/ShARC/data/cvrp50/eval \
  --eval_dir_100 /home/agriya.yadav_ug2023/ShARC/data/cvrp100/eval \
  --cvar_ckpt_50  /home/agriya.yadav_ug2023/ShARC/experiments/results/cvrp50_cvar_shift/best.pt \
  --cvar_ckpt_100 /home/agriya.yadav_ug2023/ShARC/experiments/results/cvrp100_cvar_shift/best.pt \
  --rn_ckpt_50    /home/agriya.yadav_ug2023/ShARC/experiments/results/cvrp50_rn/best.pt \
  --rn_ckpt_100   /home/agriya.yadav_ug2023/ShARC/experiments/results/cvrp100_rn/best.pt \
  --out_csv /home/agriya.yadav_ug2023/ShARC/results/transfer_eval.csv \
  --severities 0.0 0.2 0.4 0.6 0.8 1.0 \
  --n_seeds 5 \
  --device cuda
```

---

## 9. `evaluation/make_paper_figures.py`

**Needed by:** manual run after all CSVs are produced.

**Purpose:** Reads all result CSVs and generates the exact figures for the AAAI paper (PDF/PNG, publication quality).

**CLI**
```
python -m evaluation.make_paper_figures \
  --results_dir results/ \
  --out_dir paper/figures/
```

**Figures to produce**

| Figure | Source CSV | Type | Description |
|---|---|---|---|
| Fig 1 | `severity_sweep.csv` | Line plot | CVaR@0.1 vs severity: ShARC vs RN-RL |
| Fig 2 | `ablation_sweep.csv` | Bar chart | CVaR@0.1 at severity=0.8 for each ablation variant |
| Fig 3 | `shifted_baselines.csv` + `severity_sweep.csv` | Line plot | Worst-case makespan vs severity: ShARC vs ILS/ACO/EA |
| Fig 4 | `transfer_eval.csv` | Grouped bar | delta_cvar for CVRP-50 and CVRP-100 at severity=0.6 |
| Table 1 | `baselines_unshifted.csv` | LaTeX table | Nominal performance: ShARC vs ILS/ACO/EA |
| Table 2 | `ablation_sweep.csv` | LaTeX table | Full ablation table across all severities |

**Notes**
- Use `matplotlib` with `rcParams` set for AAAI double-column width (3.5 inches per figure).
- Save both `.pdf` (for paper) and `.png` (for email to Prof. Cao) for every figure.
- Error bars: ±1 std across seeds.
- Colour scheme: ShARC = blue, RN-RL = orange, ILS = green, ACO = red, EA = purple — consistent across all figures.
- LaTeX tables: use `\toprule/\midrule/\bottomrule`, bold the ShARC row.

---

## Dependency Order

```
gen_cvrp.py
  └── cvrp_50_train.sh  ──────────────────────────────────┐
  └── cvrp_100_train.sh  ─────────────────────────────────┤
  └── rn_cvrp_50_train.sh  ───────────────────────────────┤
  └── rn_cvrp_100_train.sh  ──────────────────────────────┤
                                                           ▼
run_baseline_at_shift.py                        transfer_eval.py ──┐
  └── shifted_baselines.sh                         └── transfer_eval.sh
                                                                    │
ablation_sweep.py                                                   │
  └── ablation_sweep.sh (after ablation training)                  │
                                                                    │
                                      make_paper_figures.py  ◄─────┘
                                       (reads all CSVs)
```
