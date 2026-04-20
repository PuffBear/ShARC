"""
Severity sweep for the paper's central figure.

Loads cvar_shift and rn_nominal checkpoints, evaluates both at
φ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}, records mean T_max and CVaR_0.1(T_max)
across the eval set.

Usage:
    python -m evaluation.severity_sweep \
        --eval_dir  data/eval_dataset \
        --cvar_ckpt experiments/results/cvar_shift/best.pt \
        --rn_ckpt   experiments/results/rn_nominal/best.pt \
        --out_csv   results/severity_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from glob import glob

import numpy as np
import torch

from env.hcarp_env import HCARPEnv
from env.shift import ShiftConfig, ShiftScheduler
from models.policy import HCARPPolicy
from training.configs.default import CFG
from training.train import make_batches


SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ALPHA = 0.1


def load_policy(ckpt_path: str, use_shift: bool, device: str) -> HCARPPolicy:
    d_shift = CFG["d_shift"] if use_shift else 0
    policy = HCARPPolicy(
        d_model      = CFG["d_model"],
        n_heads      = CFG["n_heads"],
        n_enc_layers = CFG["n_enc_layers"],
        d_ff         = CFG["d_ff"],
        d_clss       = CFG["d_clss"],
        clip         = CFG["clip"],
        d_shift      = d_shift,
        device       = device,
    )
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()
    return policy


@torch.no_grad()
def evaluate_at_severity(
    policy: HCARPPolicy,
    files: list[str],
    severity: float,
    batch_size: int,
    seed: int,
) -> dict:
    """
    Evaluate policy at a fixed shift severity φ.

    severity=0 → no shift; severity=1 → maximum shift (φ=1).
    Uses 'uniform' mode so magnitude is constant across the eval set.

    Returns mean T_max and CVaR_0.1(T_max) where T_max = T1 (worst vehicle time).
    """
    cfg = ShiftConfig(
        max_demand_shift = 0.3 * severity,
        max_cost_shift   = 0.3 * severity,
        min_availability = 1.0 - 0.3 * severity,
        mode             = "uniform",
    )
    scheduler = ShiftScheduler(cfg, seed=seed)

    T1_list: list[float] = []
    for batch in make_batches(files, batch_size, shuffle=False):
        env = HCARPEnv(shift_scheduler=scheduler)
        env.load_files(batch)
        env.reset()
        _, _, _, info = policy.rollout(env, greedy=True)
        for entry in info.values():
            T1_list.append(float(entry["T1"]))

    T1 = np.array(T1_list, dtype=float)
    n_tail = max(1, int(np.ceil(ALPHA * len(T1))))
    # CVaR of T_max: worst = highest T1 values (longest makespan)
    cvar_val = float(np.sort(T1)[-n_tail:].mean())

    return {
        "severity":   severity,
        "mean_T_max": float(np.mean(T1)),
        "cvar_T_max": cvar_val,
        "n":          len(T1),
    }


def main():
    parser = argparse.ArgumentParser(description="Severity sweep evaluation")
    parser.add_argument("--eval_dir",   required=True,       help="Directory with eval .npz instances")
    parser.add_argument("--cvar_ckpt",  required=True,       help="Path to cvar_shift best.pt checkpoint")
    parser.add_argument("--rn_ckpt",    required=True,       help="Path to rn_nominal best.pt checkpoint")
    parser.add_argument("--out_csv",    default="results/severity_sweep.csv")
    parser.add_argument("--severities", nargs="+", type=float, default=SEVERITIES,
                        help="List of φ values to sweep (default: 0 0.2 0.4 0.6 0.8 1.0)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.eval_dir, "**", "*.npz"), recursive=True))
    assert files, f"No .npz files found under {args.eval_dir}"
    print(f"Eval instances: {len(files)}")

    policies = {
        "cvar_shift": (args.cvar_ckpt, True),   # shift-conditioned model
        "rn_nominal": (args.rn_ckpt,   False),  # risk-neutral, no shift conditioning
    }

    rows = []
    for name, (ckpt, use_shift) in policies.items():
        print(f"\n=== {name} ({ckpt}) ===")
        policy = load_policy(ckpt, use_shift, args.device)
        for sev in args.severities:
            m = evaluate_at_severity(policy, files, sev, args.batch_size, args.seed)
            m["policy"] = name
            rows.append(m)
            print(
                f"  φ={sev:.1f}  mean_T_max={m['mean_T_max']:.4f}"
                f"  CVaR_0.1(T_max)={m['cvar_T_max']:.4f}  n={m['n']}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fieldnames = ["policy", "severity", "mean_T_max", "cvar_T_max", "n"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows({k: row[k] for k in fieldnames} for row in rows)

    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
