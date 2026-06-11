"""
Run a single classical baseline (ILS, ACO, EA) on the eval dataset under a
specified shift severity. Called in a loop by jobs/shifted_baselines.sh over
baselines × severities.

Output CSV is append-mode and safe for concurrent writes via fcntl file locking.
Each existing (baseline, severity, instance_id, seed) row is skipped (idempotent).

Output columns:
    baseline, severity, instance_id, seed,
    makespan_T1, makespan_T2, makespan_T3,
    cvar_10, worst_case

cvar_10 and worst_case are computed over all instances for this (severity, seed)
and stored on every row for easy groupby downstream.

Usage:
    python baseline/run_baseline_at_shift.py \\
        --baseline ILS \\
        --eval_dir data/eval_dataset \\
        --severity 0.4 \\
        --n_seeds  5 \\
        --seed     42 \\
        --out_csv  results/shifted_baselines.csv
"""

from __future__ import annotations

import csv
import fcntl
import os
import sys
import tempfile
from glob import glob
from time import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from env.shift import ShiftConfig, ShiftScheduler

ALPHA = 0.1
FIELDNAMES = [
    'baseline', 'severity', 'instance_id', 'seed',
    'makespan_T1', 'makespan_T2', 'makespan_T3',
    'cvar_10', 'worst_case',
]


# ---------------------------------------------------------------------------
# Shift application at the npz level
# ---------------------------------------------------------------------------

def apply_shift(npz_path: str, shift: dict, tmp_dir: str, rng: np.random.Generator) -> str | None:
    """
    Load npz, perturb demands/costs/availability, save to tmp_dir.
    Returns new path, or None if availability dropout removed all required arcs.
    """
    es     = np.load(npz_path, allow_pickle=False)
    req    = es['req'].copy().astype(np.float64)     # cols: n1,n2,q,p,s,d
    nonreq = es['nonreq'].copy().astype(np.float64)

    dd = shift['delta_demand']
    dc = shift['delta_cost']
    ds = shift['delta_service']
    pa = shift['p_availability']

    req[:, 2] = np.clip(req[:, 2] * (1.0 + dd), 0.0, None)          # demand
    req[:, 4] = req[:, 4] * max(1.0 + ds, 0.0)                       # service time
    scale = max(1.0 + dc, 0.0)
    req[:, 5]    *= scale                                              # arc cost
    nonreq[:, 5] *= scale                                             # deadhead cost

    if pa < 1.0:
        keep = rng.random(len(req)) < pa
        req  = req[keep]

    if len(req) == 0:
        return None

    stem     = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = os.path.join(tmp_dir, f'{stem}_shifted.npz')
    kw = {k: es[k] for k in es.files}
    kw['req']    = req.astype(np.float32)
    kw['nonreq'] = nonreq.astype(np.float32)
    np.savez(out_path, **kw)
    return out_path


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def run_baseline_on_file(path: str, baseline: str, variant: str = 'P') -> np.ndarray | None:
    """Run one baseline on one shifted instance. Returns [T1, T2, T3] or None."""
    try:
        if baseline == 'ILS':
            from baseline.meta import InsertCheapestHCARP
            al = InsertCheapestHCARP()
            al.import_instance(path)
            result = al(variant=variant, num_sample=20)
        elif baseline == 'ACO':
            from baseline.meta import ACOHCARP
            al = ACOHCARP(n_ant=50)
            al.import_instance(path)
            result = al(n_epoch=100, variant=variant)
        elif baseline == 'EA':
            from baseline.meta import EAHCARP
            al = EAHCARP(n_population=200)
            al.import_instance(path)
            result = al(n_epoch=100, variant=variant)
        else:
            raise ValueError(f"Unknown baseline: {baseline!r}")
        return np.array(result[0], dtype=np.float32) if result is not None else None
    except Exception as e:
        print(f"  [warn] {baseline} failed on {os.path.basename(path)}: {e}")
        return None


# ---------------------------------------------------------------------------
# CSV helpers — append-mode with file locking
# ---------------------------------------------------------------------------

def _load_existing(csv_path: str) -> set[tuple]:
    """Return set of (baseline, severity_str, instance_id, seed_str) already done."""
    done: set[tuple] = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            done.add((
                row['baseline'],
                row['severity'],
                row['instance_id'],
                row['seed'],
            ))
    return done


def _append_rows(csv_path: str, rows: list[dict]):
    """Atomically append rows to csv_path using fcntl file locking."""
    if not rows:
        return
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    # Write to a temp file first, then append its contents under a lock
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, newline='') as tmp:
        w = csv.DictWriter(tmp, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerows(rows)
        tmp_path = tmp.name

    with open(csv_path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if write_header and os.path.getsize(csv_path) == 0:
                pass  # header was written into tmp; will be copied below
            with open(tmp_path, 'r') as src:
                f.write(src.read())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Baseline severity sweep under shift")
    parser.add_argument('--baseline',  required=True, choices=['ILS', 'ACO', 'EA'])
    parser.add_argument('--eval_dir',  required=True)
    parser.add_argument('--severity',  type=float, required=True,
                        help='Shift severity φ in [0, 1]')
    parser.add_argument('--n_seeds',   type=int, default=5)
    parser.add_argument('--seed',      type=int, default=42, help='Base seed')
    parser.add_argument('--variant',   type=str, default='P')
    parser.add_argument('--out_csv',   default='results/shifted_baselines.csv')
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.eval_dir, '**', '*.npz'), recursive=True))
    assert files, f"No .npz files under {args.eval_dir}"
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    # Load already-completed rows for idempotency
    done_keys = _load_existing(args.out_csv)
    sev_str   = str(args.severity)

    shift_cfg = ShiftConfig(
        max_demand_shift  = 0.3 * args.severity,
        max_cost_shift    = 0.3 * args.severity,
        max_service_shift = 0.3 * args.severity,
        min_availability  = 1.0 - 0.3 * args.severity,
        mode              = 'adversarial',
    )

    print(f"{args.baseline} | φ={args.severity} | {len(files)} instances × {args.n_seeds} seeds")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for seed_offset in range(args.n_seeds):
            seed     = args.seed + seed_offset
            seed_str = str(seed)
            scheduler = ShiftScheduler(shift_cfg, seed=seed)
            rng       = np.random.default_rng(seed)

            seed_T1s: list[float] = []   # for per-seed aggregate stats
            seed_rows: list[dict] = []

            for inst_path in files:
                instance_id = os.path.splitext(os.path.basename(inst_path))[0]

                if (args.baseline, sev_str, instance_id, seed_str) in done_keys:
                    continue  # already computed — idempotent

                shift        = scheduler.sample()
                shifted_path = apply_shift(inst_path, shift, tmp_dir, rng)
                if shifted_path is None:
                    continue  # all arcs dropped

                t0 = time()
                Ts = run_baseline_on_file(shifted_path, args.baseline, args.variant)
                elapsed = time() - t0

                if Ts is None:
                    continue

                seed_T1s.append(float(Ts[0]))
                seed_rows.append({
                    'baseline':    args.baseline,
                    'severity':    args.severity,
                    'instance_id': instance_id,
                    'seed':        seed,
                    'makespan_T1': float(Ts[0]),
                    'makespan_T2': float(Ts[1]),
                    'makespan_T3': float(Ts[2]),
                    'cvar_10':     None,   # filled below
                    'worst_case':  None,   # filled below
                })
                done_keys.add((args.baseline, sev_str, instance_id, seed_str))

            if not seed_T1s:
                print(f"  seed={seed}: no valid results")
                continue

            # Compute per-seed aggregates over all instances
            arr      = np.array(seed_T1s)
            n_tail   = max(1, int(np.ceil(ALPHA * len(arr))))
            cvar_10  = float(np.sort(arr)[-n_tail:].mean())
            worst    = float(arr.max())

            for r in seed_rows:
                r['cvar_10']    = cvar_10
                r['worst_case'] = worst

            _append_rows(args.out_csv, seed_rows)
            print(f"  seed={seed}  n={len(seed_T1s)}  "
                  f"mean_T1={arr.mean():.3f}  CVaR={cvar_10:.3f}  worst={worst:.3f}")

    print(f"\nDone → {args.out_csv}")


if __name__ == '__main__':
    main()
