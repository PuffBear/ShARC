"""
Default hyperparameters for HCARP REINFORCE training.
Import and override as needed.
"""

CFG = dict(
    # --- Data ---
    data_dir       = "data/test_dataset",   # root dir with .npz files
    val_split      = 0.1,                   # fraction held out for validation
    seed           = 42,

    # --- Model ---
    d_model        = 128,
    n_heads        = 8,
    n_enc_layers   = 3,
    d_ff           = 512,
    d_clss         = 16,
    clip           = 10.0,

    # --- Training ---
    batch_size     = 32,
    n_epochs       = 200,
    lr             = 1e-4,
    max_grad_norm  = 1.0,

    # --- Logging / checkpointing ---
    validate_every = 10,          # epochs between validation runs
    checkpoint_dir = "experiments/results",
    run_name       = "reinforce_default",

    # --- Device ---
    device         = "cpu",       # change to "cuda" if available
)
