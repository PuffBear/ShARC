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


def _bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() not in ("false", "0", "no", "off")

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
    policy.load_state_dict(torch.load(ckpt_path, map_location=device), strict=use_shift)
    policy.eval()
    return policy


@torch.no_grad()
def evaluate_at_severity(
    policy: HCARPPolicy,
    files: list[str],
    severity: float,
    batch_size: int,
    seed: int,
    phi_demand_only: bool = False,
) -> dict:
    """
    Evaluate policy at a fixed shift severity φ.

    severity=0 → no shift; severity=1 → maximum shift (φ=1).
    Uses 'uniform' mode so magnitude is constant across the eval set.

    phi_demand_only=True: vary δ_demand and δ_cost only; fix p_availability=1.0 (no arc dropout).

    Returns mean T_max and CVaR_0.1(T_max) where T_max = T1 (worst vehicle time).
    """
    cfg = ShiftConfig(
        max_demand_shift  = 0.3 * severity,
        max_cost_shift    = 0.3 * severity,
        max_service_shift = 0.3 * severity,  # must scale with severity; default 0.3 would leak at φ=0
        min_availability  = 1.0 if phi_demand_only else 1.0 - 0.3 * severity,
        mode              = "adversarial",   # fixed positive shift (harder problems)
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
    parser.add_argument("--severities", "--phi_levels", nargs="+", type=float, default=SEVERITIES,
                        help="List of φ values to sweep (default: 0 0.2 0.4 0.6 0.8 1.0)")
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--n_seeds",         type=int,   default=5,
                        help="Number of random seeds to evaluate for robust confidence intervals")
    parser.add_argument("--seed_start",      type=int,   default=42)
    parser.add_argument("--device",          default="cpu")
    parser.add_argument("--rn_use_shift",    type=_bool, default=True,
                        help="True if rn checkpoint was trained with shift (d_shift=8).")
    parser.add_argument("--phi_demand_only", type=_bool, default=False,
                        help="Vary δ_demand and δ_cost only; fix p_availability=1.0 (no arc dropout).")
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.eval_dir, "**", "*.npz"), recursive=True))
    assert files, f"No .npz files found under {args.eval_dir}"
    print(f"Eval instances: {len(files)}, Sweeping {args.n_seeds} random seeds per severity.")

    def _label(path: str) -> str:
        return os.path.basename(os.path.dirname(path))

    policies = {
        _label(args.cvar_ckpt): (args.cvar_ckpt, True),
        _label(args.rn_ckpt):   (args.rn_ckpt,   args.rn_use_shift),
    }

    rows = []
    for name, (ckpt, use_shift) in policies.items():
        print(f"\n=== {name} ({ckpt}) ===")
        policy = load_policy(ckpt, use_shift, args.device)
        for sev in args.severities:
            sev_means = []
            sev_cvars = []
            
            for s_idx in range(args.n_seeds):
                seed = args.seed_start + s_idx
                metrics = evaluate_at_severity(policy, files, sev, args.batch_size, seed,
                                               phi_demand_only=args.phi_demand_only)
                sev_means.append(metrics["mean_T_max"])
                sev_cvars.append(metrics["cvar_T_max"])
            
            m = {
                "policy": name,
                "severity": sev,
                "mean_T_max": float(np.mean(sev_means)),
                "std_T_max":  float(np.std(sev_means)),
                "cvar_T_max": float(np.mean(sev_cvars)),
                "std_cvar_T_max": float(np.std(sev_cvars)),
                "n_instances": len(files),
                "n_seeds": args.n_seeds
            }
            rows.append(m)
            print(
                f"  φ={sev:.1f}  mean={m['mean_T_max']:.2f}±{m['std_T_max']:.2f}  "
                f"CVaR={m['cvar_T_max']:.2f}±{m['std_cvar_T_max']:.2f}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fieldnames = ["policy", "severity", "mean_T_max", "std_T_max", "cvar_T_max", "std_cvar_T_max", "n_instances", "n_seeds"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {args.out_csv}")

if __name__ == "__main__":
    main()
