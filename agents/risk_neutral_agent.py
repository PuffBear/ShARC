"""
RiskNeutralAgent — wraps HCARPPolicy with standard REINFORCE (no shift, no CVaR).

Serves as the primary comparison baseline for CVaRAgent in ablation studies.
"""

from __future__ import annotations

import torch

from models.policy import HCARPPolicy
from training.train import train_batch, validate, save_checkpoint, load_checkpoint


class RiskNeutralAgent:
    """
    Standard REINFORCE agent (E[R] maximisation, no distribution shift).

    Parameters
    ----------
    cfg : dict
        Same schema as CVaRAgent; shift/CVaR keys are ignored.
        d_shift is forced to 0: no shift context conditioning.
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
            d_shift      = 0,   # no shift conditioning
            device       = self.device,
        )

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.get("lr", 1e-4)
        )

    # ------------------------------------------------------------------

    def train_step(self, files: list[str]) -> dict:
        """Run one standard REINFORCE update. Returns metrics dict."""
        self.policy.train()
        return train_batch(
            policy        = self.policy,
            optimizer     = self.optimizer,
            files         = files,
            max_grad_norm = self.cfg.get("max_grad_norm", 1.0),
            device        = self.device,
            use_cvar      = False,
        )

    def evaluate(self, files: list[str]) -> dict:
        """Greedy rollout on files (no shift)."""
        self.policy.eval()
        return validate(
            self.policy, files,
            batch_size = self.cfg.get("batch_size", 32),
        )

    def save(self, path: str):
        save_checkpoint(self.policy, path)

    def load(self, path: str):
        load_checkpoint(self.policy, path)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())
