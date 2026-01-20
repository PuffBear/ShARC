#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

HEADER_RE = re.compile(r"^\s*([A-Z_ÁÉÍÓÚÜÑa-z0-9 ]+)\s*:\s*(.*?)\s*$")

# More forgiving edge parser:
# grabs (u,v) then grabs "coste <int>" and "demanda <int>" anywhere after
PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
COST_RE = re.compile(r"coste\s*(\d+)", re.IGNORECASE)
DEM_RE = re.compile(r"demanda\s*(\d+)", re.IGNORECASE)

COMMENT_TOKENS = ["//", "#", "%", ";"]


def strip_comment(line: str) -> str:
    s = line
    for tok in COMMENT_TOKENS:
        if tok in s:
            s = s.split(tok, 1)[0]
    return s.rstrip("\n")


def normalize_key(k: str) -> str:
    return k.strip().upper().replace(" ", "_")


def coerce_value(v: str) -> Any:
    v = v.strip()
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    return v


def parse_gdb_dat(path: Path, debug: bool = False) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    required_edges: List[Dict[str, Any]] = []

    in_req_list = False
    seen_req_header = False

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = strip_comment(raw).strip()
        if not line:
            continue

        # section start
        if line.upper().startswith("LISTA_ARISTAS_REQ"):
            in_req_list = True
            seen_req_header = True
            continue

        # inside req list
        if in_req_list:
            m_pair = PAIR_RE.search(line)
            m_cost = COST_RE.search(line)
            m_dem = DEM_RE.search(line)

            if m_pair and m_cost and m_dem:
                u = int(m_pair.group(1))
                v = int(m_pair.group(2))
                cost = int(m_cost.group(1))
                demand = int(m_dem.group(1))
                required_edges.append(
                    {"u": u, "v": v, "cost": cost, "demand": demand, "required": True}
                )
                continue

            # exit section if we hit a header like "DEPOSITO : 1"
            m_hdr_after = HEADER_RE.match(line)
            if m_hdr_after:
                in_req_list = False
            else:
                # ignore non-matching junk line inside section
                continue

        # header
        m = HEADER_RE.match(line)
        if m:
            key = normalize_key(m.group(1))
            val = coerce_value(m.group(2))
            meta[key] = val

    canonical = {
        "name": meta.get("NOMBRE"),
        "comment": meta.get("COMENTARIO"),
        "n_vertices": meta.get("VERTICES"),
        "n_req": meta.get("ARISTAS_REQ"),
        "n_noreq": meta.get("ARISTAS_NOREQ"),
        "n_vehicles": meta.get("VEHICULOS"),
        "capacity": meta.get("CAPACIDAD"),
        "req_total_cost": meta.get("COSTE_TOTAL_REQ"),
        "depot": meta.get("DEPOSITO"),
        "cost_type": meta.get("TIPO_COSTES_ARISTAS"),
    }

    warnings: List[str] = []
    if not meta:
        warnings.append("No headers parsed — file format may differ or file is empty/unreadable.")
    if not seen_req_header:
        warnings.append("Did not find LISTA_ARISTAS_REQ section header.")
    if canonical["n_req"] is not None and canonical["n_req"] != len(required_edges):
        warnings.append(f"ARISTAS_REQ says {canonical['n_req']} but parsed {len(required_edges)} edges")

    if debug:
        print(f"[debug] file={path}")
        print(f"[debug] headers={list(meta.keys())[:10]} (total {len(meta)})")
        print(f"[debug] parsed_required_edges={len(required_edges)}")

    return {"meta_raw": meta, "meta": canonical, "required_edges": required_edges, "warnings": warnings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to .dat file OR folder")
    ap.add_argument("--out", default=None, help="Output JSON file (if input is file) or folder (if input is folder)")
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    in_path = Path(args.inp).expanduser().resolve()

    if in_path.is_dir():
        files = sorted(in_path.rglob("*.dat"))
        if not files:
            raise SystemExit(f"No .dat files found in {in_path}")

        out_dir: Optional[Path] = None
        if args.out:
            out_dir = Path(args.out).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        for f in files:
            obj = parse_gdb_dat(f, debug=args.debug)

            if out_dir:
                (out_dir / f"{f.stem}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")

            summaries.append(
                {
                    "file": str(f),
                    "name": obj["meta"]["name"],
                    "n_vertices": obj["meta"]["n_vertices"],
                    "n_req_parsed": len(obj["required_edges"]),
                    "depot": obj["meta"]["depot"],
                    "capacity": obj["meta"]["capacity"],
                    "warnings": obj["warnings"],
                }
            )

        # ALWAYS print summary now
        print(json.dumps(summaries, indent=2))

    else:
        obj = parse_gdb_dat(in_path, debug=args.debug)

        if args.out:
            out_path = Path(args.out).expanduser().resolve()
            out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

        # ALWAYS print meta + quick stats
        print(json.dumps(obj["meta"], indent=2))
        print(f"required_edges_parsed = {len(obj['required_edges'])}")
        if obj["warnings"]:
            print("warnings:")
            for w in obj["warnings"]:
                print(f" - {w}")


if __name__ == "__main__":
    main()
