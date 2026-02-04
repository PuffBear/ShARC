# converter: convert json files to compatible formats for Baseline solvers from Arc-DRL
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def convert_one(json_path: Path, out_path: Path, P: int = 3):
    obj = json.loads(json_path.read_text(encoding="utf-8"))

    cap = int(obj["capacity"])
    depot = int(obj["depot"])  # likely 1 in GDB
    tasks = obj["tasks"]

    # Vehicles: your GDB header had VEHICULOS, but your instance bundle currently doesn't store it.
    # We'll default to 1 if missing; you can override via CLI.
    M = int(obj.get("n_vehicles", 1))

    # Convert required edges to req rows: (n1, n2, q, p, s, d)
    # - shift nodes from 1..N -> 0..N-1
    # - q normalized by capacity (so feasibility threshold is 1)
    # - p priority class in [1..P]; if you have no priorities, put everything in class 1
    req_rows = []
    for e in tasks:
        u = int(e["u"]) - 1
        v = int(e["v"]) - 1
        cost = float(e["cost"])
        demand = float(e["demand"])
        q = demand / cap
        p = 1  # everything priority class 1
        s = cost  # service time/cost
        d = cost  # traversal distance/cost on the edge
        req_rows.append([u, v, q, p, s, d])

    req = np.array(req_rows, dtype=np.float32)

    # No non-required edges for GDB (ARISTAS_NOREQ=0). Use empty array with correct shape.
    nonreq = np.zeros((0, 6), dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        C=np.int32(cap),
        P=np.int32(P),
        M=np.int32(M),
        req=req,
        nonreq=nonreq,
        depot=np.int32(depot - 1),  # 0-based
        name=np.bytes_(str(obj.get("name", json_path.stem))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with instance JSON files")
    ap.add_argument("--out_dir", required=True, help="Where to write .npz files")
    ap.add_argument("--P", type=int, default=3, help="Number of priority classes (Arc-DRL expects 3)")
    ap.add_argument("--M", type=int, default=None, help="Override number of vehicles if your JSON doesn't store it")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No .json files found in {in_dir}")

    # Arc-DRL expects path like: <path>/*/*.npz
    # So we create a subfolder "gdb" inside out_dir.
    sub = out_dir / "gdb"
    sub.mkdir(parents=True, exist_ok=True)

    for jp in files:
        obj = json.loads(jp.read_text(encoding="utf-8"))
        if args.M is not None:
            obj["n_vehicles"] = args.M
            # write back temporarily in memory only
            tmp_path = jp  # not writing to disk
            # convert using in-memory object by dumping to str then loading would be messy
            # simplest: just compute M via override below
        out_path = sub / f"{jp.stem}.npz"

        if args.M is not None:
            # quick hack: convert_one reads from disk, so just pass M through object.get via injecting key
            # easiest: just call convert and then overwrite M in the npz by re-saving.
            convert_one(jp, out_path, P=args.P)
            z = np.load(out_path)
            np.savez_compressed(
                out_path,
                C=z["C"],
                P=z["P"],
                M=np.int32(args.M),
                req=z["req"],
                nonreq=z["nonreq"],
                depot=z["depot"],
                name=z["name"],
            )
        else:
            convert_one(jp, out_path, P=args.P)

    print(f"Converted {len(files)} JSON instances -> {sub}/*.npz")


if __name__ == "__main__":
    main()
