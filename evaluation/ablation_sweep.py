"""
Ablation sweep — load every ablation checkpoint, run ShiftEvaluator.sweep(),
and produce the comparison table for Section 5 of the paper.

Expected checkpoint layout under --ckpt_dir:
    ablation_full_sharc/best.pt
    ablation_rn/best.pt
    ablation_no_shift_ctx/best.pt
    ablation_no_budget_sig/best.pt
    ablation_alpha_005/best.pt
    ablation_alpha_020/best.pt
    ablation_no_curriculum/best.pt

Output CSV columns:
    run_name, severity, mean_makespan, cvar_10, worst_case, gap_to_ils

gap_to_ils = mean_makespan - ils_unshifted_mean  (positive = worse than ILS at φ=0)
ILS oracle is read from --ils_csv if provided, else gap_to_ils is left blank.

Usage:
    python -m evaluation.ablation_sweep \\
        --eval_dir  data/eval_dataset \\
        --ckpt_dir  experiments/results \\
        --out_csv   results/ablation_sweep.csv \\
        --severities 0.0 0.2 0.4 0.6 0.8 1.0 \\
        --n_seeds 5 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import os
from glob import glob

import numpy as np
import torch

from models.policy import HCARPPolicy
from training.configs.default import CFG
from training.configs.ablations import ABLATIONS
from evaluation.shift_evaluator import ShiftEvaluator

FIELDNAMES = ['run_name', 'severity', 'mean_makespan', 'cvar_10', 'worst_case', 'gap_to_ils']


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def load_policy(run_name: str, ckpt_path: str, device: str) -> HCARPPolicy:
    """Instantiate the correct architecture for this ablation variant."""
    overrides = ABLATIONS.get(run_name, {})
    cfg       = {**CFG, **overrides}

    d_shift           = cfg.get("d_shift", CFG["d_shift"])
    use_budget_signal = cfg.get("use_budget_signal", True)

    policy = HCARPPolicy(
        d_model           = cfg["d_model"],
        n_heads           = cfg["n_heads"],
        n_enc_layers      = cfg["n_enc_layers"],
        d_ff              = cfg["d_ff"],
        d_clss            = cfg["d_clss"],
        clip              = cfg["clip"],
        d_shift           = d_shift,
        use_budget_signal = use_budget_signal,
        device            = device,
    )
    state = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation severity sweep")
    parser.add_argument('--eval_dir',   required=True,
                        help='Path to HCARP eval .npz instances')
    parser.add_argument('--ckpt_dir',   required=True,
                        help='Root dir; ablation checkpoints at <ckpt_dir>/<run_name>/best.pt')
    parser.add_argument('--out_csv',    default='results/ablation_sweep.csv')
    parser.add_argument('--severities', nargs='+', type=float,
                        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--n_seeds',    type=int, default=5)
    parser.add_argument('--seed_start', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device',     default='cpu')
    parser.add_argument('--ils_csv',    default=None,
                        help='Optional: shifted_baselines.csv to compute gap_to_ils')
    args = parser.parse_args()

    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA not available — falling back to CPU')
        args.device = 'cpu'

    files = sorted(glob(os.path.join(args.eval_dir, '**', '*.npz'), recursive=True))
    assert files, f"No .npz files under {args.eval_dir}"
    print(f"Eval instances: {len(files)}")

    # Load ILS unshifted mean for gap computation (φ=0, ILS baseline)
    ils_mean_phi0: float | None = None
    if args.ils_csv and os.path.exists(args.ils_csv):
        import pandas as pd
        ils_df = pd.read_csv(args.ils_csv)
        mask   = (ils_df['baseline'] == 'ILS') & (np.isclose(ils_df['severity'], 0.0))
        if mask.any():
            ils_mean_phi0 = float(ils_df.loc[mask, 'makespan_T1'].mean())
            print(f"ILS unshifted oracle mean T1: {ils_mean_phi0:.3f}")

    # Discover checkpoints
    ckpt_map: dict[str, str] = {}
    for run_name in ABLATIONS:
        ckpt = os.path.join(args.ckpt_dir, run_name, 'best.pt')
        if os.path.exists(ckpt):
            ckpt_map[run_name] = ckpt
        else:
            print(f"  [skip] {run_name}: no checkpoint at {ckpt}")

    assert ckpt_map, "No ablation checkpoints found — train them first."
    print(f"Variants found: {list(ckpt_map)}\n")

    rows: list[dict] = []
    for run_name, ckpt_path in ckpt_map.items():
        print(f"=== {run_name} ===")
        policy = load_policy(run_name, ckpt_path, args.device)

        # Aggregate over n_seeds by running ShiftEvaluator once per seed
        seed_results: dict[float, list[dict]] = {s: [] for s in args.severities}

        for s_idx in range(args.n_seeds):
            seed      = args.seed_start + s_idx
            evaluator = ShiftEvaluator(files, batch_size=args.batch_size,
                                       alpha=0.1, seed=seed)
            sweep_out = evaluator.sweep(policy, severities=args.severities, greedy=True)
            for m in sweep_out:
                seed_results[m['severity']].append(m)

        for sev in args.severities:
            seed_ms = [m['mean']       for m in seed_results[sev]]
            seed_cv = [m['cvar']       for m in seed_results[sev]]
            seed_wc = [m['worst_case'] for m in seed_results[sev]]

            # ShiftEvaluator returns rewards (negative costs); convert to positive makespan
            mean_makespan = float(-np.mean(seed_ms))
            cvar_10       = float(-np.mean(seed_cv))   # CVaR of makespan
            worst_case    = float(-np.mean(seed_wc))

            gap = (mean_makespan - ils_mean_phi0) if ils_mean_phi0 else None

            row = {
                'run_name':     run_name,
                'severity':     sev,
                'mean_makespan': mean_makespan,
                'cvar_10':      cvar_10,
                'worst_case':   worst_case,
                'gap_to_ils':   round(gap, 4) if gap is not None else '',
            }
            rows.append(row)
            print(f"  φ={sev:.2f}  mean={mean_makespan:.3f}  "
                  f"CVaR={cvar_10:.3f}  worst={worst_case:.3f}"
                  + (f"  gap={gap:.3f}" if gap else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {args.out_csv}")


if __name__ == '__main__':
    main()
