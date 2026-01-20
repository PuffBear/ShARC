#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LINE_RE = re.compile(r"^(.*?)\s*:::\s*(.*?)\s*:::\s*([0-9.]+)\s*$")


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout, p.stderr


def parse_outputs(stdout: str) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping: file -> {"result": "...", "time": "..."}
    """
    out: Dict[str, Dict[str, str]] = {}
    for line in stdout.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        f, res, t = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        out[f] = {"result": res, "time": t}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Folder that contains subfolders with .npz (e.g., data_arc)")
    ap.add_argument("--out_csv", default="baseline_results.csv", help="CSV output path")
    ap.add_argument("--variant", default="P", help="Variant for EA/ILS/LP (usually P). ACO often uses U.")
    ap.add_argument("--aco_variant", default="U", help="Variant for ACO (often U).")
    ap.add_argument("--ils_num_sample", type=int, default=20)
    ap.add_argument("--ea_epochs", type=int, default=100)
    ap.add_argument("--ea_pop", type=int, default=200)
    ap.add_argument("--aco_epochs", type=int, default=100)
    ap.add_argument("--aco_ants", type=int, default=50)
    ap.add_argument("--seed", type=int, default=6868)
    ap.add_argument("--run_lp", action="store_true", help="Also run LP (needs Gurobi)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_path = Path(args.path).expanduser().resolve()

    # Commands (these scripts glob internally over args.path + '/*/*.npz')
    cmds = {
        "ILS": ["python3", "baseline/ils.py", "--path", str(data_path), "--variant", args.variant,
                "--num_sample", str(args.ils_num_sample), "--seed", str(args.seed)],
        "EA": ["python3", "baseline/ea.py", "--path", str(data_path), "--variant", args.variant,
               "--max_epoch", str(args.ea_epochs), "--n_population", str(args.ea_pop), "--seed", str(args.seed)],
        "ACO": ["python3", "baseline/aco.py", "--path", str(data_path), "--variant", args.aco_variant,
                "--max_epoch", str(args.aco_epochs), "--n_ant", str(args.aco_ants), "--seed", str(args.seed)],
    }
    if args.run_lp:
        cmds["LP"] = ["python3", "baseline/lp.py", "--path", str(data_path), "--variant", args.variant]

    # Run and collect
    collected: Dict[str, Dict[str, str]] = {}  # inst -> columns
    errors: List[str] = []

    for name, cmd in cmds.items():
        code, stdout, stderr = run_cmd(cmd, cwd=root)
        parsed = parse_outputs(stdout)

        if code != 0:
            errors.append(f"{name} exited with code {code}. stderr:\n{stderr}")

        # Merge per instance
        for f, vals in parsed.items():
            inst = Path(f).stem
            row = collected.setdefault(inst, {"instance": inst})
            row[f"{name}_result"] = vals["result"]
            row[f"{name}_time_s"] = vals["time"]

        # If a baseline produced no parseable lines, note it
        if not parsed:
            errors.append(f"{name} produced no parseable output lines. Check stdout/stderr.")

    # Write CSV
    # Create a stable column order
    cols = ["instance"]
    for name in cmds.keys():
        cols += [f"{name}_result", f"{name}_time_s"]

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for inst in sorted(collected.keys()):
            w.writerow({k: collected[inst].get(k, "") for k in cols})

    print(f"Wrote CSV: {out_csv}")
    if errors:
        print("\nWarnings/Errors:")
        for e in errors:
            print(" - " + e.replace("\n", "\n   "))


if __name__ == "__main__":
    main()
