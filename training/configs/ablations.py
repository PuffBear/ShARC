"""
Ablation variant configs for ShARC.

Import ABLATIONS and merge with CFG before calling train():

    from training.configs.default import CFG
    from training.configs.ablations import ABLATIONS

    cfg = {**CFG, **ABLATIONS["no_shift_ctx"]}
    train(cfg)

Or pass the variant name to train.py via --run_name and --<key> overrides.

Seven variants:
  full_sharc       — no override; baseline for comparison
  no_shift_ctx     — d_shift=0; removes φ from decoder input
  no_budget_sig    — use_budget_signal=False; zeros β_t in decoder
  rn_alpha1        — use_cvar=False; standard REINFORCE (risk-neutral)
  alpha_005        — CVaR at α=0.05 (more conservative)
  alpha_020        — CVaR at α=0.20 (less conservative)
  no_curriculum    — shift_mode="uniform"; trains at full shift from step 0

Note on no_shift_ctx:
  d_shift=0 removes the shift-context projection entirely from the decoder.
  The policy checkpoint will have a different architecture; load with
  HCARPPolicy(..., d_shift=0).

Note on no_budget_sig:
  use_budget_signal=False zeroes β_t in the decoder forward pass but keeps
  the architecture identical — checkpoint is interchangeable with full_sharc.
"""

ABLATIONS: dict[str, dict] = {
    "full_sharc": {},   # no override — use CFG as-is

    "no_shift_ctx": {
        "d_shift":    0,        # removes φ projection from decoder
        "use_shift":  True,     # shift still applied to env (for fair eval)
        "run_name":   "ablation_no_shift_ctx",
    },

    "no_budget_sig": {
        "use_budget_signal": False,   # zeroes β_t slot in decoder context
        "run_name":          "ablation_no_budget_sig",
    },

    "rn_alpha1": {
        "use_cvar":  False,     # standard REINFORCE (risk-neutral, α effectively=1)
        "run_name":  "ablation_rn",
    },

    "alpha_005": {
        "alpha":    0.05,
        "use_cvar": True,
        "run_name": "ablation_alpha_005",
    },

    "alpha_020": {
        "alpha":    0.20,
        "use_cvar": True,
        "run_name": "ablation_alpha_020",
    },

    "alpha_030": {
        "alpha":    0.30,
        "use_cvar": True,
        "run_name": "ablation_alpha_030",
    },

    "no_curriculum": {
        "shift_mode": "uniform",    # trains at full shift from step 0 (no warmup)
        "run_name":   "ablation_no_curriculum",
    },
}
