"""
CVRP transfer evaluation — severity sweep for CVRP-50 and CVRP-100.

Evaluates ShARC (CVaR-RL trained on CVRP) vs RN-RL across shift severities
for two problem sizes. This is the primary generality experiment.

Output CSV columns:
    problem, n_nodes, method, severity, mean_makespan, cvar_10, worst_case, delta_cvar

delta_cvar = cvar_rn - cvar_sharc  (positive means ShARC is better on tail)

Usage:
    python -m evaluation.transfer_eval \\
        --eval_dir_50  data/cvrp50/eval \\
        --eval_dir_100 data/cvrp100/eval \\
        --cvar_ckpt_50  experiments/results/cvrp50_cvar_shift/best.pt \\
        --cvar_ckpt_100 experiments/results/cvrp100_cvar_shift/best.pt \\
        --rn_ckpt_50    experiments/results/cvrp50_rn/best.pt \\
        --rn_ckpt_100   experiments/results/cvrp100_rn/best.pt \\
        --out_csv results/transfer_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from glob import glob
from typing import Optional

import numpy as np
import torch

from env.cvrp_env import CVRPEnv
from env.shift import ShiftConfig, ShiftScheduler
from models.policy import HCARPPolicy
from training.configs.default import CFG

ALPHA      = 0.1
SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
FIELDNAMES = [
    'problem', 'n_nodes', 'method', 'severity',
    'mean_makespan', 'cvar_10', 'worst_case', 'delta_cvar',
]


def _bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() not in ('false', '0', 'no', 'off')


def load_policy(ckpt_path: str, d_shift: int, device: str) -> HCARPPolicy:
    policy = HCARPPolicy(
        d_model      = CFG['d_model'],
        n_heads      = CFG['n_heads'],
        n_enc_layers = CFG['n_enc_layers'],
        d_ff         = CFG['d_ff'],
        d_clss       = CFG['d_clss'],
        clip         = CFG['clip'],
        d_shift      = d_shift,
        device       = device,
    )
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()
    return policy


def _group_by_size(files: list[str], batch_size: int) -> list[list[str]]:
    """Group files by n_nodes then batch — CVRP instances must share N."""
    groups: dict[int, list[str]] = {}
    for f in files:
        try:
            n = int(np.load(f, allow_pickle=False)['demands'].shape[0])
        except Exception:
            n = 0
        groups.setdefault(n, []).append(f)
    batches = []
    for g in groups.values():
        for i in range(0, len(g), batch_size):
            batches.append(g[i: i + batch_size])
    return batches


@torch.no_grad()
def evaluate_at_severity(
    policy: HCARPPolicy,
    files: list[str],
    severity: float,
    batch_size: int,
    seed: int,
) -> dict:
    cfg = ShiftConfig(
        max_demand_shift  = 0.3 * severity,
        max_cost_shift    = 0.3 * severity,
        max_service_shift = 0.0,            # no service time in CVRP
        min_availability  = 1.0 - 0.3 * severity,
        mode              = 'adversarial',
    )
    scheduler = ShiftScheduler(cfg, seed=seed)
    costs: list[float] = []

    for batch in _group_by_size(files, batch_size):
        env = CVRPEnv(shift_scheduler=scheduler)
        env.load_files(batch)
        env.reset()
        _, _, _, info = policy.rollout(env, greedy=True)
        for entry in info.values():
            costs.append(float(entry['cost']))

    arr    = np.array(costs, dtype=float)
    n_tail = max(1, int(np.ceil(ALPHA * len(arr))))
    cvar   = float(np.sort(arr)[-n_tail:].mean())

    return {
        'mean_makespan': float(arr.mean()),
        'cvar_10':       cvar,
        'worst_case':    float(arr.max()),
    }


def run_for_size(
    label: str,
    n_nodes: int,
    eval_dir: str,
    cvar_ckpt: Optional[str],
    rn_ckpt: Optional[str],
    severities: list[float],
    n_seeds: int,
    seed_start: int,
    batch_size: int,
    device: str,
) -> list[dict]:
    """Evaluate ShARC and RN-RL on one problem size. Returns list of row dicts."""
    files = sorted(glob(os.path.join(eval_dir, '**', '*.npz'), recursive=True))
    if not files:
        print(f"  [warn] No .npz files under {eval_dir} — skipping {label}")
        return []

    print(f"\n--- {label} ({len(files)} instances) ---")
    rows: list[dict] = []

    policies: dict[str, tuple[Optional[str], int]] = {
        'sharc':   (cvar_ckpt, CFG['d_shift']),
        'rn_cvrp': (rn_ckpt,   0),
    }

    method_results: dict[str, dict[float, list[float]]] = {
        m: {s: [] for s in severities} for m in policies
    }

    for method, (ckpt_path, d_shift) in policies.items():
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"  [warn] {method} checkpoint missing ({ckpt_path}) — skipping")
            continue

        print(f"  Loading {method} from {ckpt_path}")
        policy = load_policy(ckpt_path, d_shift, device)

        for sev in severities:
            for s_idx in range(n_seeds):
                seed = seed_start + s_idx
                m    = evaluate_at_severity(policy, files, sev, batch_size, seed)
                method_results[method][sev].append(m['cvar_10'])

            print(
                f"    {method}  φ={sev:.1f}  "
                f"CVaR={np.mean(method_results[method][sev]):.3f}"
            )

    # Build rows with delta_cvar
    for sev in severities:
        sharc_cvars  = method_results.get('sharc',   {}).get(sev, [])
        rn_cvars     = method_results.get('rn_cvrp', {}).get(sev, [])
        sharc_mean   = float(np.mean(sharc_cvars))  if sharc_cvars  else None
        rn_mean      = float(np.mean(rn_cvars))     if rn_cvars     else None
        delta        = (rn_mean - sharc_mean) if (sharc_mean and rn_mean) else None

        for method in ['sharc', 'rn_cvrp']:
            cvars = method_results[method].get(sev, [])
            if not cvars:
                continue
            rows.append({
                'problem':      label,
                'n_nodes':      n_nodes,
                'method':       method,
                'severity':     sev,
                'mean_makespan': '',   # re-compute below if needed
                'cvar_10':      round(float(np.mean(cvars)), 4),
                'worst_case':   '',
                'delta_cvar':   round(delta, 4) if delta is not None else '',
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="CVRP transfer evaluation")
    parser.add_argument('--eval_dir_50',   required=True)
    parser.add_argument('--eval_dir_100',  required=True)
    parser.add_argument('--cvar_ckpt_50',  required=True)
    parser.add_argument('--cvar_ckpt_100', required=True)
    parser.add_argument('--rn_ckpt_50',    required=True)
    parser.add_argument('--rn_ckpt_100',   required=True)
    parser.add_argument('--out_csv',       default='results/transfer_eval.csv')
    parser.add_argument('--severities',    nargs='+', type=float, default=SEVERITIES)
    parser.add_argument('--n_seeds',       type=int,  default=5)
    parser.add_argument('--seed_start',    type=int,  default=42)
    parser.add_argument('--batch_size',    type=int,  default=32)
    parser.add_argument('--device',        default='cpu')
    args = parser.parse_args()

    # CUDA guard
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA not available — falling back to CPU')
        args.device = 'cpu'

    all_rows: list[dict] = []

    all_rows += run_for_size(
        label='cvrp50', n_nodes=50,
        eval_dir=args.eval_dir_50,
        cvar_ckpt=args.cvar_ckpt_50,
        rn_ckpt=args.rn_ckpt_50,
        severities=args.severities,
        n_seeds=args.n_seeds, seed_start=args.seed_start,
        batch_size=args.batch_size, device=args.device,
    )

    all_rows += run_for_size(
        label='cvrp100', n_nodes=100,
        eval_dir=args.eval_dir_100,
        cvar_ckpt=args.cvar_ckpt_100,
        rn_ckpt=args.rn_ckpt_100,
        severities=args.severities,
        n_seeds=args.n_seeds, seed_start=args.seed_start,
        batch_size=args.batch_size, device=args.device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nSaved: {args.out_csv}")


if __name__ == '__main__':
    main()
