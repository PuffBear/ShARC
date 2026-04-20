"""
REINFORCE and CVaR-REINFORCE training for HCARPPolicy.

Risk-neutral (default):
    Maximises E[R] via standard REINFORCE with greedy baseline.

CVaR mode (use_cvar=True):
    Maximises CVaR_α[R] — the expected reward in the worst α-fraction of episodes.
    Gradient weights come only from tail instances; baseline is the CVaR of the
    greedy rollout (Tamar et al., 2015; Bäuerle & Ott, 2011).

Usage:
    python -m training.train
    python -m training.train --use_cvar True --alpha 0.1
    python -m training.train --data_dir data/my_instances --n_epochs 500
"""

from __future__ import annotations

import argparse
import os
import random
import time
from glob import glob

import numpy as np
import torch

from env.hcarp_env import HCARPEnv
from models.policy import HCARPPolicy
from training.configs.default import CFG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scan_instances(data_dir: str) -> list[str]:
    files = sorted(glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    assert files, f"No .npz files found under {data_dir}"
    return files


def split_files(files: list[str], val_split: float, seed: int):
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    return shuffled[n_val:], shuffled[:n_val]


def make_batches(files: list[str], batch_size: int, shuffle: bool = True) -> list[list[str]]:
    if shuffle:
        files = files[:]
        random.shuffle(files)
        
    # HCARPEnv requires static batch np.stack, so group strictly by length
    from common.ops import import_instance
    groups = {}
    for f in files:
        try:
            sz = int(np.load(f)['req'].shape[0])
        except Exception:
            sz = 0
        if sz not in groups:
            groups[sz] = []
        groups[sz].append(f)
        
    batches = []
    for g_files in groups.values():
        for i in range(0, len(g_files), batch_size):
            batches.append(g_files[i : i + batch_size])
            
    if shuffle:
        random.shuffle(batches)
    return batches


def save_checkpoint(policy: HCARPPolicy, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(policy.state_dict(), path)


def load_checkpoint(policy: HCARPPolicy, path: str):
    policy.load_state_dict(torch.load(path, map_location=policy.device))


# ---------------------------------------------------------------------------
# CVaR loss
# ---------------------------------------------------------------------------

def cvar_loss(
    rewards: torch.Tensor,
    baseline: torch.Tensor,
    log_prob_sum: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    CVaR-REINFORCE loss (Tamar et al. 2015).

    Computes the policy gradient estimator that maximises CVaR_α[R]:
      ∇CVaR_α ≈ (1/α·B) · Σ_{i: R_i ≤ VaR_α} (R_i − CVaR_α[baseline]) · ∇log π_i

    Parameters
    ----------
    rewards      : [B]  stochastic rollout rewards
    baseline     : [B]  greedy rollout rewards (no grad)
    log_prob_sum : [B]  sum of log-probs over the stochastic trajectory
    alpha        : CVaR confidence level (fraction of worst episodes)

    Returns
    -------
    Scalar loss (negate to maximise CVaR).
    """
    B = rewards.size(0)
    n_tail = max(1, int(alpha * B))

    # VaR: alpha-quantile from the bottom (lowest reward = worst outcome)
    sorted_rew, indices = torch.sort(rewards)
    var_alpha = sorted_rew[n_tail - 1]

    # CVaR of the greedy baseline for a consistent advantage estimate
    sorted_base, _ = torch.sort(baseline)
    cvar_baseline = sorted_base[:n_tail].mean()

    # Exact masking: perfectly n_tail elements tracking exactly the lowest instances
    tail_mask = torch.zeros_like(rewards)
    tail_mask[indices[:n_tail]] = 1.0  # [B]

    # Advantage for tail instances only
    advantage = (rewards - cvar_baseline).detach()  # [B]

    # CVaR policy gradient weights: 1/(α·B) for tail, 0 otherwise
    cvar_weights = tail_mask / (alpha * float(B))

    return -(cvar_weights * advantage * log_prob_sum).sum()


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_batch(
    policy: HCARPPolicy,
    optimizer: torch.optim.Optimizer,
    files: list[str],
    max_grad_norm: float,
    device: str,
    shift_scheduler=None,
    use_cvar: bool = False,
    alpha: float = 0.1,
) -> dict:
    """
    Run one REINFORCE (or CVaR-REINFORCE) update on a batch of instances.
    """
    env = HCARPEnv(shift_scheduler=shift_scheduler)
    env.load_files(files)

    # --- Stochastic rollout (with gradient) ---
    env.reset()
    _, log_probs, rewards, _ = policy.rollout(env, greedy=False)

    # --- Greedy baseline (no gradient) ---
    env.reset()
    with torch.no_grad():
        _, _, baseline, _ = policy.rollout(env, greedy=True)

    log_prob_sum = log_probs.sum(dim=1)  # [B]

    # --- Loss ---
    if use_cvar:
        loss = cvar_loss(rewards, baseline, log_prob_sum, alpha)
    else:
        advantage = (rewards - baseline).detach()
        if advantage.std() > 1e-8:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        loss = -(advantage * log_prob_sum).mean()

    # --- Backward ---
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return dict(
        loss      = float(loss),
        reward    = float(rewards.mean()),
        baseline  = float(baseline.mean()),
        grad_norm = float(grad_norm),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    policy: HCARPPolicy,
    files: list[str],
    batch_size: int,
    shift_scheduler=None,
) -> dict:
    all_rewards, T1s, T2s, T3s = [], [], [], []

    for batch in make_batches(files, batch_size, shuffle=False):
        env = HCARPEnv(shift_scheduler=shift_scheduler)
        env.load_files(batch)
        env.reset()
        _, _, rewards, info = policy.rollout(env, greedy=True)

        all_rewards.extend(rewards.cpu().tolist())
        for i in info.values():
            T1s.append(i["T1"])
            T2s.append(i["T2"])
            T3s.append(i["T3"])

    return dict(
        val_reward = float(np.mean(all_rewards)),
        val_T1     = float(np.mean(T1s)),
        val_T2     = float(np.mean(T2s)),
        val_T3     = float(np.mean(T3s)),
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    set_seed(cfg["seed"])

    shift_scheduler = None
    if cfg.get("use_shift"):
        from env.shift import ShiftConfig, ShiftScheduler
        shift_cfg = ShiftConfig(
            max_demand_shift = cfg.get("max_demand_shift", 0.3),
            max_cost_shift   = cfg.get("max_cost_shift",   0.3),
            min_availability = cfg.get("min_availability", 0.7),
            warmup_steps     = cfg.get("shift_warmup",    1000),
            mode             = cfg.get("shift_mode",  "curriculum"),
        )
        shift_scheduler = ShiftScheduler(shift_cfg, seed=cfg["seed"])
        print(f"Shift scheduler: {shift_cfg}")

    all_files = scan_instances(cfg["data_dir"])
    train_files, val_files = split_files(all_files, cfg["val_split"], cfg["seed"])
    print(f"Instances — train: {len(train_files)}  val: {len(val_files)}")

    d_shift = cfg.get("d_shift", 8) if cfg.get("use_shift") else 0

    policy = HCARPPolicy(
        d_model      = cfg["d_model"],
        n_heads      = cfg["n_heads"],
        n_enc_layers = cfg["n_enc_layers"],
        d_ff         = cfg["d_ff"],
        d_clss       = cfg["d_clss"],
        clip         = cfg["clip"],
        d_shift      = d_shift,
        device       = cfg["device"],
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {n_params:,}  (d_shift={d_shift})")

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    ckpt_dir  = os.path.join(cfg["checkpoint_dir"], cfg["run_name"])
    best_path = os.path.join(ckpt_dir, "best.pt")
    best_val_reward = float("-inf")

    use_cvar = bool(cfg.get("use_cvar", False))
    alpha    = float(cfg.get("alpha",   0.1))
    print(f"Objective: {'CVaR-REINFORCE (α=' + str(alpha) + ')' if use_cvar else 'REINFORCE'}\n")

    for epoch in range(1, cfg["n_epochs"] + 1):
        policy.train()
        t0 = time.time()

        batches = make_batches(train_files, cfg["batch_size"])
        epoch_metrics: list[dict] = []

        for batch_files in batches:
            m = train_batch(
                policy, optimizer, batch_files,
                cfg["max_grad_norm"], cfg["device"],
                shift_scheduler=shift_scheduler,
                use_cvar=use_cvar,
                alpha=alpha,
            )
            epoch_metrics.append(m)
            if shift_scheduler is not None:
                shift_scheduler.advance()

        avg = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]}
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:4d}/{cfg['n_epochs']} | "
            f"loss {avg['loss']:8.2f} | "
            f"reward {avg['reward']:10.1f} | "
            f"baseline {avg['baseline']:10.1f} | "
            f"grad {avg['grad_norm']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if epoch % cfg["validate_every"] == 0:
            policy.eval()
            val = validate(policy, val_files, cfg["batch_size"], shift_scheduler)
            print(
                f"  [val] reward {val['val_reward']:10.1f} | "
                f"T1 {val['val_T1']:.3f}  T2 {val['val_T2']:.3f}  T3 {val['val_T3']:.3f}"
            )

            if val["val_reward"] > best_val_reward:
                best_val_reward = val["val_reward"]
                save_checkpoint(policy, best_path)
                print(f"  [val] *** new best — saved to {best_path}")

    final_path = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint(policy, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Best validation reward: {best_val_reward:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train HCARP policy")
    for key, val in CFG.items():
        t = type(val) if val is not None else str
        parser.add_argument(f"--{key}", type=t, default=val)
    return vars(parser.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
