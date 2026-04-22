"""Quick inspection of all .pt checkpoints."""
import torch
import os
from collections import OrderedDict

CKPT_DIR = "experiments/experiments/results"

runs = ["rn_shift", "rn_nominal", "cvar_shift", "cvar_nominal"]

for run in runs:
    for tag in ["best", "final"]:
        path = os.path.join(CKPT_DIR, run, f"{tag}.pt")
        if not os.path.exists(path):
            continue

        ckpt = torch.load(path, map_location="cpu")

        print(f"\n{'='*60}")
        print(f"  {run}/{tag}.pt  ({os.path.getsize(path)/1e6:.1f} MB)")
        print(f"{'='*60}")

        if isinstance(ckpt, OrderedDict) or isinstance(ckpt, dict):
            # It's a state_dict
            n_params = sum(p.numel() for p in ckpt.values())
            print(f"  Type: state_dict ({len(ckpt)} keys, {n_params:,} params)")
            print(f"  Keys (first 10):")
            for i, (k, v) in enumerate(ckpt.items()):
                if i >= 10:
                    print(f"    ... and {len(ckpt)-10} more")
                    break
                print(f"    {k:40s} {str(tuple(v.shape)):20s} {'f32' if v.dtype==torch.float32 else str(v.dtype)}")

            # Check for NaN/Inf in weights
            nan_keys = [k for k, v in ckpt.items() if torch.isnan(v).any()]
            inf_keys = [k for k, v in ckpt.items() if torch.isinf(v).any()]
            if nan_keys:
                print(f"  ⚠️  NaN detected in: {nan_keys}")
            if inf_keys:
                print(f"  ⚠️  Inf detected in: {inf_keys}")
            if not nan_keys and not inf_keys:
                print(f"  ✅ No NaN/Inf in any parameters")

            # Weight statistics
            all_vals = torch.cat([v.float().flatten() for v in ckpt.values()])
            print(f"  Weight stats: mean={all_vals.mean():.6f}  std={all_vals.std():.4f}  "
                  f"min={all_vals.min():.4f}  max={all_vals.max():.4f}")
        else:
            print(f"  Type: {type(ckpt)}")
            print(f"  Content: {str(ckpt)[:200]}")
