import numpy as np
import networkx as nx

class ShARCGaurdrail:
    def __init__(self, data_path, big_m=1e6):
        self.data = np.load(data_path)
        self.Q = float(self.data['C'])
        self.M_vehicles = int(self.data['M'])
        self.big_m = big_m
        
    def validate_instance(self, req, nonreq):
        """
        Validates an instance (Static or Shifted).
        req: (N, 6) matrix of required arcs
        nonreq: (M, 6) matrix of traversal arcs
        """
        all_arcs = np.vstack([req, nonreq])
        G = nx.MultiDiGraph()
        
        # Build graph for connectivity and shortest path checks
        for row in all_arcs:
            G.add_edge(int(row[0]), int(row[1]), weight=row[5], demand=row[2])
            
        report = {"Status": "PASS", "Issues": []}

        # --- A1: Structural Invariants ---
        
        # 1. Graph Sanity: Strong Connectivity
        # Required for Directed CARP to ensure the agent never gets trapped
        if not nx.is_strongly_connected(G):
            report["Status"] = "FAIL"
            report["Issues"].append("Graph is not strongly connected.")

        # 2. Cost Sanity: c_e > 0
        if np.any(all_arcs[:, 5] <= 0):
            report["Status"] = "WARNING"
            report["Issues"].append("Zero or negative costs detected.")

        # 3. Demand Sanity & Feasibility Condition (q_e <= Q)
        max_q = np.max(req[:, 2])
        if max_q > self.Q:
            report["Status"] = "FAIL"
            report["Issues"].append(f"Infeasible: Max demand {max_q} exceeds capacity {self.Q}")

        # --- A2: Cross-check Costs w/ Shortest Paths ---
        
        # Compute all-pairs shortest paths using Floyd-Warshall or Dijkstra
        dist_matrix = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        
        # Verify Triangle Inequality: d(u,w) <= d(u,v) + d(v,w)
        # We sample random triples to save computation time on large graphs
        nodes = list(G.nodes())
        for _ in range(100): # Random sampling of 100 triples
            u, v, w = np.random.choice(nodes, 3, replace=False)
            duw = dist_matrix[u].get(w, self.big_m)
            duv = dist_matrix[u].get(v, self.big_m)
            dvw = dist_matrix[v].get(w, self.big_m)
            
            if duw > duv + dvw + 1e-7: # Use epsilon for float precision
                report["Status"] = "FAIL"
                report["Issues"].append(f"Triangle inequality violation at nodes {u,v,w}")
                break

        return report

# Usage for Static Data
guardrail = ShARCGaurdrail('instances/100/100_39_099.npz')
init_report = guardrail.validate_instance(guardrail.data['req'], guardrail.data['nonreq'])
print(f"Initial Check: {init_report}")