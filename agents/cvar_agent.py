"""
CVaRAgent — wraps HCARPPolicy with CVaR-REINFORCE training and shift support.

This is the primary contribution agent: it optimises CVaR_α[R] under
continuous distribution shifts, making the policy robust to worst-case
performance degradation.
"""

from __future__ import annotations

import torch

from models.policy import HCARPPolicy
from env.shift import ShiftConfig, ShiftScheduler
from training.train import train_batch, validate, save_checkpoint, load_checkpoint


class CVaRAgent:
    """
    CVaR-optimising agent for HCARP under distribution shifts.

    Parameters
    ----------
    cfg : dict
        Configuration dict. Expected keys (with defaults):
          d_model, n_heads, n_enc_layers, d_ff, d_clss, clip  — model arch
          d_shift         — shift context projection dim (default 8)
          alpha           — CVaR confidence level (default 0.1)
          lr, max_grad_norm, batch_size                        — optimiser
          use_shift, shift_mode, shift_warmup,
          max_demand_shift, max_cost_shift, min_availability   — shift schedule
          device, seed
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cpu")

        self.policy = HCARPPolicy(
            d_model      = cfg.get("d_model",      128),
            n_heads      = cfg.get("n_heads",       8),
            n_enc_layers = cfg.get("n_enc_layers",  3),
            d_ff         = cfg.get("d_ff",          512),
            d_clss       = cfg.get("d_clss",        16),
            clip         = cfg.get("clip",          10.0),
            d_shift      = cfg.get("d_shift",       8),
            device       = self.device,
        )

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.get("lr", 1e-4)
        )

        shift_cfg = ShiftConfig(
            max_demand_shift = cfg.get("max_demand_shift", 0.3),
            max_cost_shift   = cfg.get("max_cost_shift",   0.3),
            min_availability = cfg.get("min_availability", 0.7),
            warmup_steps     = cfg.get("shift_warmup",    1000),
            mode             = cfg.get("shift_mode",  "curriculum"),
        )
        self.shift_scheduler = ShiftScheduler(shift_cfg, seed=cfg.get("seed", 42))
        self.alpha = cfg.get("alpha", 0.1)

    # ------------------------------------------------------------------

    def train_step(self, files: list[str]) -> dict:
        """Run one CVaR-REINFORCE update. Returns metrics dict."""
        self.policy.train()
        metrics = train_batch(
            policy          = self.policy,
            optimizer       = self.optimizer,
            files           = files,
            max_grad_norm   = self.cfg.get("max_grad_norm", 1.0),
            device          = self.device,
            shift_scheduler = self.shift_scheduler,
            use_cvar        = True,
            alpha           = self.alpha,
        )
        self.shift_scheduler.advance()
        return metrics

    def evaluate(self, files: list[str]) -> dict:
        """Greedy rollout on files under current shift schedule."""
        self.policy.eval()
        return validate(
            self.policy, files,
            batch_size      = self.cfg.get("batch_size", 32),
            shift_scheduler = self.shift_scheduler,
        )

    def save(self, path: str):
        save_checkpoint(self.policy, path)

    def load(self, path: str):
        load_checkpoint(self.policy, path)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())
