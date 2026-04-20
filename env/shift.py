"""
Distribution shift utilities for HCARP training.

ShiftConfig  — dataclass holding shift hyperparameters.
ShiftScheduler — samples per-episode shift parameters along three axes:
    delta_demand    fractional perturbation to arc demands
    delta_cost      fractional perturbation to travel costs
    p_availability  probability that each required arc is available

Modes:
    "curriculum"  — shift magnitude grows linearly over warmup_steps
    "uniform"     — always sample from the full range
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ShiftConfig:
    max_demand_shift: float = 0.3       # max fractional demand change  e.g. 0.3 → ±30%
    max_cost_shift: float = 0.3         # max fractional travel-cost change
    min_availability: float = 0.7       # minimum arc-availability probability
    warmup_steps: int = 1000            # curriculum: steps to reach full magnitude
    mode: str = "curriculum"            # "curriculum" | "uniform"


class ShiftScheduler:
    """
    Produces per-episode shift parameters.

    Parameters
    ----------
    config : ShiftConfig
    seed   : optional RNG seed for reproducibility
    """

    CONTEXT_DIM = 3  # [delta_demand, delta_cost, p_availability]

    def __init__(self, config: ShiftConfig | None = None, seed: int | None = None):
        self.cfg = config or ShiftConfig()
        self._step = 0
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def advance(self):
        """Increment the internal step counter (call once per training batch)."""
        self._step += 1

    def _magnitude(self) -> float:
        if self.cfg.mode == "curriculum":
            return min(self._step / max(self.cfg.warmup_steps, 1), 1.0)
        return 1.0  # uniform: always full range

    def sample(self) -> dict:
        """
        Draw one set of shift parameters.

        Returns
        -------
        dict with keys: delta_demand, delta_cost, p_availability, context
            context : np.ndarray [3]  — raw shift values for policy conditioning
        """
        mag = self._magnitude()
        cfg = self.cfg

        delta_demand = mag * cfg.max_demand_shift * float(self._rng.uniform(-1.0, 1.0))
        delta_cost   = mag * cfg.max_cost_shift   * float(self._rng.uniform(-1.0, 1.0))
        delta_p      = mag * (1.0 - cfg.min_availability) * float(self._rng.uniform(0.0, 1.0))
        p_avail      = 1.0 - delta_p

        context = np.array([delta_demand, delta_cost, delta_p], dtype=np.float32)
        return {
            "delta_demand":   delta_demand,
            "delta_cost":     delta_cost,
            "p_availability": p_avail,
            "context":        context,
        }

    def sample_batch(self, B: int) -> list[dict]:
        """Sample B independent shift dicts (one per instance)."""
        return [self.sample() for _ in range(B)]

    @property
    def step(self) -> int:
        return self._step
