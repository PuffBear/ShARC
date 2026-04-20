"""
Evaluation metrics for HCARP policies.

compute_cvar    — CVaR at a given confidence level
compute_metrics — full metrics dict (mean, CVaR, worst-case, std)
gap_to_baseline — mean and CVaR gap between two policies
"""

import numpy as np


def compute_cvar(rewards: np.ndarray, alpha: float = 0.1) -> float:
    """
    CVaR_alpha: expected reward in the worst-alpha fraction of outcomes.

    For minimisation objectives expressed as negative costs, lower reward
    is worse. CVaR captures average performance in the worst-α tail.

    Parameters
    ----------
    rewards : 1-D array of episode rewards (negative costs)
    alpha   : tail fraction (e.g. 0.1 = worst 10%)

    Returns
    -------
    float — CVaR estimate (negative; closer to 0 is better)
    """
    rewards = np.asarray(rewards, dtype=float)
    n_tail = max(1, int(np.ceil(alpha * len(rewards))))
    sorted_r = np.sort(rewards)  # ascending: worst first
    return float(sorted_r[:n_tail].mean())


def compute_metrics(rewards: np.ndarray, alpha: float = 0.1) -> dict:
    """
    Compute a standard set of evaluation metrics.

    Parameters
    ----------
    rewards : 1-D array of episode rewards
    alpha   : CVaR confidence level

    Returns
    -------
    dict with keys: mean, std, cvar, worst_case, best_case, n_episodes, alpha
    """
    rewards = np.asarray(rewards, dtype=float)
    return {
        "mean":       float(np.mean(rewards)),
        "std":        float(np.std(rewards)),
        "cvar":       compute_cvar(rewards, alpha),
        "worst_case": float(np.min(rewards)),
        "best_case":  float(np.max(rewards)),
        "n_episodes": int(len(rewards)),
        "alpha":      alpha,
    }


def gap_to_baseline(
    policy_rewards: np.ndarray,
    baseline_rewards: np.ndarray,
    alpha: float = 0.1,
) -> dict:
    """
    Compute mean and CVaR gap between policy and a reference baseline.

    Positive gap means the policy outperforms the baseline.
    """
    policy_rewards   = np.asarray(policy_rewards,   dtype=float)
    baseline_rewards = np.asarray(baseline_rewards, dtype=float)

    mean_gap = float(np.mean(policy_rewards) - np.mean(baseline_rewards))
    cvar_gap = compute_cvar(policy_rewards, alpha) - compute_cvar(baseline_rewards, alpha)

    return {"mean_gap": mean_gap, "cvar_gap": float(cvar_gap)}
