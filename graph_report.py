import argparse
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def build_graph(req, nonreq):
    all_arcs = np.vstack([req, nonreq])
    G = nx.MultiDiGraph()
    for row in all_arcs:
        u, v = int(row[0]), int(row[1])
        G.add_edge(
            u,
            v,
            demand=float(row[2]),
            priority=int(row[3]),
            service_time=float(row[4]),
            travel_time=float(row[5]),
            weight=float(row[5]),
        )
    return G, all_arcs


def validate_instance(G, req, nonreq, capacity, big_m=1e6, samples=100, rng_seed=455):
    all_arcs = np.vstack([req, nonreq])
    report = {"Status": "PASS", "Issues": []}

    if not nx.is_strongly_connected(G):
        report["Status"] = "FAIL"
        report["Issues"].append("Graph is not strongly connected.")

    if np.any(all_arcs[:, 5] <= 0):
        if report["Status"] != "FAIL":
            report["Status"] = "WARNING"
        report["Issues"].append("Zero or negative costs detected.")

    max_q = float(np.max(req[:, 2])) if req.size else 0.0
    if max_q > capacity:
        report["Status"] = "FAIL"
        report["Issues"].append(f"Infeasible: Max demand {max_q} exceeds capacity {capacity}")

    dist_matrix = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    nodes = list(G.nodes())
    if len(nodes) >= 3:
        rng = np.random.default_rng(rng_seed)
        for _ in range(samples):
            u, v, w = rng.choice(nodes, 3, replace=False)
            duw = dist_matrix[u].get(w, big_m)
            duv = dist_matrix[u].get(v, big_m)
            dvw = dist_matrix[v].get(w, big_m)
            if duw > duv + dvw + 1e-7:
                report["Status"] = "FAIL"
                report["Issues"].append(f"Triangle inequality violation at nodes {u, v, w}")
                break

    return report


def render_report_image(G, req, nonreq, report, out_path, title, seed=455):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1])
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    pos = nx.spring_layout(G, seed=seed, k=0.5)

    nx.draw_networkx_nodes(G, pos, node_size=400, node_color="lightskyblue", alpha=0.9, ax=ax_graph)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax_graph)

    req_edges = [(int(row[0]), int(row[1])) for row in req]
    nonreq_edges = [(int(row[0]), int(row[1])) for row in nonreq]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=req_edges,
        edge_color="crimson",
        width=2.5,
        arrowstyle="-|>",
        arrowsize=15,
        ax=ax_graph,
        label="Required Arcs (Tasks)",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=nonreq_edges,
        edge_color="gray",
        alpha=0.3,
        style="dashed",
        arrowstyle="-|>",
        arrowsize=10,
        ax=ax_graph,
        label="Traversal Arcs",
    )

    if 0 in G:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[0],
            node_color="gold",
            node_size=700,
            edgecolors="black",
            linewidths=2,
            ax=ax_graph,
        )

    ax_graph.set_title(title, fontsize=16)
    ax_graph.legend(loc="upper right", fontsize=11)
    ax_graph.axis("off")

    ax_text.axis("off")
    issues = report["Issues"] or ["No issues found."]
    lines = [f"Status: {report['Status']}", ""]
    lines += [f"- {issue}" for issue in issues]
    text = "\n".join(lines)
    ax_text.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#999999"),
    )

    fig.tight_layout()
    fig.savefig(out_path)


def normalize_name(raw):
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            raw = str(raw)
    return str(raw).strip()


def process_one(data_path, out_path, samples):
    data = np.load(data_path)
    req = data["req"]
    nonreq = data["nonreq"]
    capacity = float(data["C"])

    G, _ = build_graph(req, nonreq)
    report = validate_instance(G, req, nonreq, capacity, samples=samples)

    instance_name = normalize_name(data.get("name", data_path.stem))
    render_report_image(G, req, nonreq, report, out_path, title=f"ShARC Graph Report: {instance_name}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render graph + validation report for ShARC instances")
    ap.add_argument("path", help="Path to .npz instance file or a folder to scan recursively")
    ap.add_argument("--out", default=None, help="Output image path (file input only)")
    ap.add_argument("--out_dir", default=None, help="Output folder (dir input only)")
    ap.add_argument("--samples", type=int, default=100, help="Triangle inequality samples")
    args = ap.parse_args()

    data_path = Path(args.path).expanduser().resolve()

    if data_path.is_dir():
        if not args.out_dir:
            raise SystemExit("When input is a folder, pass --out_dir for reports output.")
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(data_path.rglob("*.npz"))
        if not files:
            raise SystemExit(f"No .npz files found under {data_path}")

        for f in files:
            rel = f.relative_to(data_path)
            out_path = out_dir / rel.parent / f"{f.stem}_report.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            written = process_one(f, out_path, args.samples)
            print(f"Wrote {written}")
    else:
        if args.out_dir:
            raise SystemExit("--out_dir is only valid when input is a folder.")
        out_path = Path(args.out) if args.out else data_path.with_name(f"{data_path.stem}_report.png")
        written = process_one(data_path, out_path, args.samples)
        print(f"Wrote {written}")


if __name__ == "__main__":
    main()
