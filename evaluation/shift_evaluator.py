"""
ShiftEvaluator — sweeps a policy across shift severity levels and records metrics.

The key paper figure: cost vs. shift severity for
  (a) CVaR policy
  (b) risk-neutral RL
  (c) ILS baseline
  (d) random agent

The CVaR policy should degrade more gracefully as severity increases.
"""

from __future__ import annotations

import numpy as np
import torch

from env.hcarp_env import HCARPEnv
from env.shift import ShiftConfig, ShiftScheduler
from evaluation.metrics import compute_metrics


class ShiftEvaluator:
    """
    Evaluate a policy across a range of shift severity levels.

    Parameters
    ----------
    files      : list of .npz instance paths to evaluate on
    batch_size : instances per batch
    alpha      : CVaR confidence level for metrics
    seed       : RNG seed for reproducible shift sampling
    """

    def __init__(
        self,
        files: list[str],
        batch_size: int = 32,
        alpha: float = 0.1,
        seed: int = 0,
    ):
        self.files = files
        self.batch_size = batch_size
        self.alpha = alpha
        self.seed = seed

    # ------------------------------------------------------------------

    def _make_scheduler(self, severity: float) -> ShiftScheduler:
        """
        Create a ShiftScheduler at a fixed severity level in [0, 1].

        severity=0 → no shift; severity=1 → maximum shift.
        Uses uniform mode so magnitude stays constant during the sweep.
        """
        cfg = ShiftConfig(
            max_demand_shift = 0.3 * severity,
            max_cost_shift   = 0.3 * severity,
            min_availability = 1.0 - 0.3 * severity,
            mode             = "uniform",
        )
        return ShiftScheduler(cfg, seed=self.seed)

    @torch.no_grad()
    def evaluate_policy(
        self,
        policy,
        shift_severity: float,
        greedy: bool = True,
    ) -> dict:
        """
        Evaluate a policy at a given shift severity.

        Parameters
        ----------
        policy         : HCARPPolicy (already trained)
        shift_severity : float in [0, 1]; 0 = no shift, 1 = max shift
        greedy         : use greedy decoding

        Returns
        -------
        metrics dict from compute_metrics()
        """
        scheduler = self._make_scheduler(shift_severity)
        all_rewards: list[float] = []

        for i in range(0, len(self.files), self.batch_size):
            batch = self.files[i : i + self.batch_size]
            env = HCARPEnv(shift_scheduler=scheduler)
            env.load_files(batch)
            env.reset()
            policy.eval()
            _, _, rewards, _ = policy.rollout(env, greedy=greedy)
            all_rewards.extend(rewards.cpu().tolist())

        return compute_metrics(np.array(all_rewards), self.alpha)

    def sweep(
        self,
        policy,
        severities: list[float] | None = None,
        greedy: bool = True,
    ) -> list[dict]:
        """
        Run evaluate_policy() across a list of severity levels.

        Parameters
        ----------
        policy      : HCARPPolicy
        severities  : list of values in [0, 1]; defaults to 11 evenly-spaced levels
        greedy      : use greedy decoding

        Returns
        -------
        List of dicts, each with 'severity' plus all keys from compute_metrics().
        """
        if severities is None:
            severities = [round(s, 2) for s in np.linspace(0.0, 1.0, 11)]

        results = []
        for sev in severities:
            m = self.evaluate_policy(policy, sev, greedy=greedy)
            m["severity"] = sev
            results.append(m)

        return results

    def compare(
        self,
        policies: dict[str, object],
        severities: list[float] | None = None,
        greedy: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Sweep multiple policies and return results keyed by name.

        Parameters
        ----------
        policies   : dict mapping name → HCARPPolicy
        severities : severity levels (see sweep())

        Returns
        -------
        dict mapping name → list of metric dicts (one per severity level)
        """
        return {
            name: self.sweep(policy, severities, greedy)
            for name, policy in policies.items()
        }
