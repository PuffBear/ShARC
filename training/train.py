"""
REINFORCE training for HCARPPolicy.

Algorithm (Kool et al. Attention Model, NeurIPS 2019):
  For each epoch:
    1. Sample a batch of instances.
    2. Stochastic rollout  → log_probs [B,T], reward [B]
    3. Greedy rollout      → baseline  [B]       (same instances, no grad)
    4. Advantage = reward - baseline
    5. Loss = -mean( advantage * log_probs.sum(dim=1) )
    6. Backward + gradient clip + Adam step.

Usage:
    python -m training.train
    python -m training.train --data_dir data/my_instances --n_epochs 500
"""

import argparse
import os
import random
import time
from copy import deepcopy
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
    """Return sorted list of all .npz paths under data_dir."""
    files = sorted(glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    assert files, f"No .npz files found under {data_dir}"
    return files


def split_files(files: list[str], val_split: float, seed: int):
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    return shuffled[n_val:], shuffled[:n_val]   # train, val


def make_batches(files: list[str], batch_size: int, shuffle: bool = True) -> list[list[str]]:
    """Split files into batches of fixed size."""
    if shuffle:
        files = files[:]
        random.shuffle(files)
    return [files[i : i + batch_size] for i in range(0, len(files), batch_size)]


def save_checkpoint(policy: HCARPPolicy, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(policy.state_dict(), path)


def load_checkpoint(policy: HCARPPolicy, path: str):
    policy.load_state_dict(torch.load(path, map_location=policy.device))


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_batch(
    policy: HCARPPolicy,
    optimizer: torch.optim.Optimizer,
    files: list[str],
    max_grad_norm: float,
    device: str,
) -> dict:
    """
    Run one REINFORCE update on a batch of instances.

    Returns a dict of scalar metrics.
    """
    env = HCARPEnv()
    env.load_files(files)

    # --- Stochastic rollout (with gradient) ---
    env.reset()
    _, log_probs, rewards, _ = policy.rollout(env, greedy=False)
    # log_probs: [B, T]   rewards: [B]

    # --- Greedy baseline (no gradient) ---
    env.reset()
    with torch.no_grad():
        _, _, baseline, _ = policy.rollout(env, greedy=True)
    # baseline: [B]

    # --- REINFORCE loss ---
    advantage = (rewards - baseline).detach()   # [B]
    # Normalise advantage for stability
    if advantage.std() > 1e-8:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    log_prob_sum = log_probs.sum(dim=1)          # [B]
    loss = -(advantage * log_prob_sum).mean()

    # --- Backward ---
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return dict(
        loss       = float(loss),
        reward     = float(rewards.mean()),
        baseline   = float(baseline.mean()),
        advantage  = float(advantage.mean()),
        grad_norm  = float(grad_norm),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    policy: HCARPPolicy,
    files: list[str],
    batch_size: int,
) -> dict:
    """
    Greedy rollout on the full validation set.
    Returns mean reward and mean T1/T2/T3.
    """
    all_rewards, T1s, T2s, T3s = [], [], [], []

    for batch in make_batches(files, batch_size, shuffle=False):
        env = HCARPEnv()
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

    # --- Data ---
    all_files = scan_instances(cfg["data_dir"])
    train_files, val_files = split_files(all_files, cfg["val_split"], cfg["seed"])
    print(f"Instances — train: {len(train_files)}  val: {len(val_files)}")

    # --- Model ---
    policy = HCARPPolicy(
        d_model      = cfg["d_model"],
        n_heads      = cfg["n_heads"],
        n_enc_layers = cfg["n_enc_layers"],
        d_ff         = cfg["d_ff"],
        d_clss       = cfg["d_clss"],
        clip         = cfg["clip"],
        device       = cfg["device"],
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {n_params:,}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    ckpt_dir  = os.path.join(cfg["checkpoint_dir"], cfg["run_name"])
    best_path = os.path.join(ckpt_dir, "best.pt")
    best_val_reward = float("-inf")

    print(f"\nStarting training — {cfg['n_epochs']} epochs, batch {cfg['batch_size']}\n")

    for epoch in range(1, cfg["n_epochs"] + 1):
        policy.train()
        t0 = time.time()

        batches = make_batches(train_files, cfg["batch_size"])
        epoch_metrics: list[dict] = []

        for batch_files in batches:
            m = train_batch(
                policy, optimizer, batch_files,
                cfg["max_grad_norm"], cfg["device"]
            )
            epoch_metrics.append(m)

        # Average metrics over all batches in this epoch
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

        # --- Validation ---
        if epoch % cfg["validate_every"] == 0:
            policy.eval()
            val = validate(policy, val_files, cfg["batch_size"])
            print(
                f"  [val] reward {val['val_reward']:10.1f} | "
                f"T1 {val['val_T1']:.3f}  T2 {val['val_T2']:.3f}  T3 {val['val_T3']:.3f}"
            )

            if val["val_reward"] > best_val_reward:
                best_val_reward = val["val_reward"]
                save_checkpoint(policy, best_path)
                print(f"  [val] *** new best — saved to {best_path}")

    # Save final checkpoint
    final_path = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint(policy, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Best validation reward: {best_val_reward:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train HCARP REINFORCE policy")
    for key, val in CFG.items():
        t = type(val) if val is not None else str
        parser.add_argument(f"--{key}", type=t, default=val)
    return vars(parser.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
