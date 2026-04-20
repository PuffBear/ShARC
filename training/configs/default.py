"""
Default hyperparameters for HCARP policy training.
Import and override as needed.
"""

CFG = dict(
    # --- Data ---
    data_dir       = "data/test_dataset",
    val_split      = 0.1,
    seed           = 42,

    # --- Model ---
    d_model        = 128,
    n_heads        = 8,
    n_enc_layers   = 3,
    d_ff           = 512,
    d_clss         = 16,
    clip           = 10.0,
    d_shift        = 8,      # shift context projection dim (0 = no shift conditioning)

    # --- Training ---
    batch_size     = 32,
    n_epochs       = 200,
    lr             = 1e-4,
    max_grad_norm  = 1.0,

    # --- CVaR objective ---
    use_cvar       = False,  # True = CVaR-REINFORCE, False = standard REINFORCE
    alpha          = 0.1,    # CVaR confidence level: worst alpha-fraction

    # --- Distribution shift ---
    use_shift         = False,
    shift_mode        = "curriculum",
    shift_warmup      = 1000,
    max_demand_shift  = 0.3,
    max_cost_shift    = 0.3,
    min_availability  = 0.7,

    # --- Logging / checkpointing ---
    validate_every = 10,
    checkpoint_dir = "experiments/results",
    run_name       = "reinforce_default",

    # --- Device ---
    device         = "cpu",
)
