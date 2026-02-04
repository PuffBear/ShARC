#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

from parse_gdb import parse_gdb_dat


def floyd_warshall(n: int, edges: List[Tuple[int,int,int]]) -> List[List[float]]:
    # nodes are 1..n
    dist = [[math.inf]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dist[i][i] = 0.0
    for u,v,w in edges:
        dist[u][v] = min(dist[u][v], float(w))
        dist[v][u] = min(dist[v][u], float(w))  # undirected for GDB
    for k in range(1, n+1):
        dk = dist[k]
        for i in range(1, n+1):
            dik = dist[i][k]
            if dik == math.inf:
                continue
            di = dist[i]
            for j in range(1, n+1):
                alt = dik + dk[j]
                if alt < di[j]:
                    di[j] = alt
    return dist


def build_carp_instance(parsed: Dict[str, Any]) -> Dict[str, Any]:
    meta = parsed["meta"]
    tasks = parsed["required_edges"]

    n = int(meta["n_vertices"])
    depot = int(meta["depot"])
    capacity = int(meta["capacity"])

    # Use required edges as the underlying graph edges (since NOREQ = 0 here)
    graph_edges = [(t["u"], t["v"], t["cost"]) for t in tasks]

    dist = floyd_warshall(n, graph_edges)

    # Sanity: ensure connectedness from depot
    unreachable = [i for i in range(1, n+1) if dist[depot][i] == math.inf]
    if unreachable:
        raise ValueError(f"Graph has nodes unreachable from depot {depot}: {unreachable}")

    return {
        "name": meta["name"],
        "n_vertices": n,
        "depot": depot,
        "capacity": capacity,
        "n_tasks": len(tasks),
        "tasks": tasks,
        "dist": dist,  # (n+1)x(n+1) with 1-indexing
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to .dat file")
    ap.add_argument("--out", required=True, help="Output JSON path (instance bundle)")
    args = ap.parse_args()

    parsed = parse_gdb_dat(Path(args.inp))
    inst = build_carp_instance(parsed)
    Path(args.out).write_text(json.dumps(inst, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {inst['n_tasks']} tasks.")


if __name__ == "__main__":
    main()
