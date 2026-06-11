"""
CVRP RL environment — Capacitated Vehicle Routing Problem.

Implements the same interface as HCARPEnv so training/train.py works
without modification (pass --problem cvrp to select this env).

Instance format (.npz) — produced by data/gen_cvrp.py:
    coords    (N, 2)   customer xy coordinates in [0,1]^2
    depot     (2,)     depot xy coordinates
    demands   (N,)     normalised demands in (0, capacity]
    capacity  scalar   vehicle capacity (normalised; default 1.0)

Internally the env builds a (N+1, N+1) Euclidean distance matrix with
the depot at index 0 and customers at 1..N — same index convention as
HCARPEnv's arc indexing.

Observation dict (matches HCARPEnv exactly):
    adj            [B, N+1, N+1]  distance matrix (may be shift-scaled)
    service_time   [B, N+1]       all zeros (no arc service time in CVRP)
    demand         [B, N+1]       normalised demands; index 0 = depot = 0
    clss           [B, N+1]       0=depot, 1=customer (single priority)
    visited        [B, N+1]  bool
    cur_arc        [B, V]    int32  current node per vehicle (depot=0)
    remaining_cap  [B, V]    float32
    vehicle_time   [B, V]    float32  accumulated travel distance
    active_vehicle [B]       int32
    budget         [B]       float32  running max vehicle distance (β_t)
    shift_context  [B, 3]    float32
    action_mask    [B, N+1]  bool

Action:
    0   → end current vehicle's route (return to depot), advance to next vehicle
    i   → visit customer i (1-indexed)

Reward: −total_tour_distance when all customers visited, 0 at all other steps.

Info at done: {b: {"T1": cost, "T2": cost, "T3": cost, "cost": cost}}
    T1/T2/T3 aliases allow training/train.py's validate() to work unchanged.

Shift parameterisation (same φ as HCARP):
    δ_d: per-node demand multiplier  demands *= (1 + δ_d · ε_i),  ε_i ~ U(−1,1)
    δ_c: uniform edge-cost scale     dist_ij *= (1 + δ_c)
    δ_p: node availability dropout   node dropped with prob (1 − p_availability)
"""

from __future__ import annotations

import math
import numpy as np
from glob import glob


class CVRPEnv:
    """
    Batched CVRP environment matching the HCARPEnv interface.

    All instances in a batch must share the same N (customers) and V (vehicles).
    """

    def __init__(self, shift_scheduler=None):
        self._instances: list[dict] = []
        self.shift_scheduler = shift_scheduler

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_files(self, paths: list[str]) -> "CVRPEnv":
        self._instances = [self._parse(p) for p in paths]
        return self

    def load_batch(self, instances: list[dict]) -> "CVRPEnv":
        self._instances = instances
        return self

    def _parse(self, path: str) -> dict:
        es       = np.load(path, allow_pickle=False)
        depot    = es['depot'].astype(np.float32)    # (2,)
        coords   = es['coords'].astype(np.float32)   # (N, 2)
        demands  = es['demands'].astype(np.float32)  # (N,)
        capacity = float(es['capacity'])

        # Build (N+1, 2) node matrix: depot at 0, customers at 1..N
        all_coords = np.vstack([depot[None, :], coords])  # (N+1, 2)

        # Euclidean distance matrix
        diff = all_coords[:, None, :] - all_coords[None, :, :]
        adj  = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)  # (N+1, N+1)

        # Normalised demand vector: depot = 0, customers = demands / capacity
        norm_demands    = np.zeros(len(demands) + 1, dtype=np.float32)
        norm_demands[1:] = np.clip(demands / capacity, 0.0, 1.0)

        n_nodes = len(demands)
        service_time = np.zeros(n_nodes + 1, dtype=np.float32)

        clss    = np.ones(n_nodes + 1, dtype=np.int32)
        clss[0] = 0

        # Minimum vehicles needed (with 20% slack)
        total_demand = float(norm_demands.sum())
        n_vehicles   = max(2, math.ceil(total_demand * 1.2))

        return dict(
            adj          = adj,               # (N+1, N+1)
            raw_demands  = demands,           # (N,) unnormalised, for shift
            demands      = norm_demands,      # (N+1,) normalised
            capacity     = capacity,
            service_time = service_time,      # (N+1,) zeros
            clss         = clss,              # (N+1,)
            n_nodes      = n_nodes,
            n_vehicles   = n_vehicles,
            all_coords   = all_coords,        # (N+1, 2) for shift
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        assert self._instances, "call load_files() or load_batch() first"

        inst = self._instances
        B    = len(inst)
        N    = inst[0]["n_nodes"]
        V    = inst[0]["n_vehicles"]

        assert all(i["n_nodes"]    == N for i in inst), "batch must share n_nodes"
        assert all(i["n_vehicles"] == V for i in inst), "batch must share n_vehicles"

        self.B = B
        self.n = N      # HCARPEnv compat: self.n = n_arcs
        self.M = V

        # Static
        self.adj          = np.stack([i["adj"]          for i in inst])  # (B, N+1, N+1)
        self.service_time = np.stack([i["service_time"] for i in inst])  # (B, N+1)
        self.demand       = np.stack([i["demands"]      for i in inst])  # (B, N+1)
        self.clss         = np.stack([i["clss"]         for i in inst])  # (B, N+1)

        # Dynamic
        self.cur_arc       = np.zeros((B, V), dtype=np.int32)
        self.remaining_cap = np.ones((B, V),  dtype=np.float32)
        self.vehicle_time  = np.zeros((B, V), dtype=np.float32)
        self.budget        = np.zeros(B,       dtype=np.float32)

        self.visited       = np.zeros((B, N + 1), dtype=bool)
        self.visited[:, 0] = True   # depot is never a customer action

        self.active_vehicle = np.zeros(B, dtype=np.int32)
        self.done_mask      = np.zeros(B, dtype=bool)
        self._action_log: list[list[int]] = [[] for _ in range(B)]

        self.shift_context = np.zeros((B, 3), dtype=np.float32)

        if self.shift_scheduler is not None:
            shifts = self.shift_scheduler.sample_batch(B)
            rng    = np.random.default_rng()
            for b, sh in enumerate(shifts):
                dd = sh["delta_demand"]
                dc = sh["delta_cost"]
                pa = sh["p_availability"]

                # Per-node demand shift
                raw    = inst[b]["raw_demands"]
                cap    = inst[b]["capacity"]
                eps    = rng.uniform(-1.0, 1.0, size=len(raw)).astype(np.float32)
                perturbed        = np.clip(raw * (1.0 + dd * eps), 0.0, None)
                norm             = np.zeros(len(raw) + 1, dtype=np.float32)
                norm[1:]         = np.clip(perturbed / cap, 0.0, 1.0)
                self.demand[b]   = norm

                # Uniform edge-cost scale
                self.adj[b] = self.adj[b] * max(1.0 + dc, 0.0)

                # Node availability dropout
                for node_i in range(1, N + 1):
                    if rng.random() > pa:
                        self.visited[b, node_i] = True

                self.shift_context[b] = sh["context"]

        return self._get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, dict]:
        actions = np.asarray(actions, dtype=np.int32)
        assert actions.shape == (self.B,)

        reward = np.zeros(self.B, dtype=np.float32)
        info: dict = {}

        for b in range(self.B):
            if self.done_mask[b]:
                continue

            a = int(actions[b])
            v = int(self.active_vehicle[b])
            self._action_log[b].append(a)

            if a == 0:
                # Return current vehicle to depot, advance to next
                prev = int(self.cur_arc[b, v])
                if prev != 0:
                    self.vehicle_time[b, v] += float(self.adj[b, prev, 0])
                    self.cur_arc[b, v] = 0
                self.active_vehicle[b] = min(v + 1, self.M - 1)
            else:
                prev = int(self.cur_arc[b, v])
                self.vehicle_time[b, v] += float(self.adj[b, prev, a])
                self.remaining_cap[b, v] -= self.demand[b, a]
                self.cur_arc[b, v] = a
                self.visited[b, a] = True

            self.budget[b] = float(self.vehicle_time[b].max())

            if self.visited[b, 1:].all():
                self.done_mask[b] = True
                cost, r = self._compute_reward(b)
                reward[b] = r
                # T1/T2/T3 aliases for train.py validate() compatibility
                info[b] = {"cost": cost, "T1": cost, "T2": cost, "T3": cost}

        return self._get_obs(), reward, self.done_mask.copy(), info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        return {
            "adj":            self.adj,
            "service_time":   self.service_time,
            "demand":         self.demand,
            "clss":           self.clss,
            "visited":        self.visited.copy(),
            "cur_arc":        self.cur_arc.copy(),
            "remaining_cap":  self.remaining_cap.copy(),
            "vehicle_time":   self.vehicle_time.copy(),
            "active_vehicle": self.active_vehicle.copy(),
            "budget":         self.budget.copy(),
            "shift_context":  self.shift_context.copy(),
            "action_mask":    self._get_action_mask(),
        }

    def _get_action_mask(self) -> np.ndarray:
        B, N1 = self.B, self.n + 1
        mask  = np.zeros((B, N1), dtype=bool)

        for b in range(B):
            if self.done_mask[b]:
                mask[b, 0] = True
                continue

            v   = int(self.active_vehicle[b])
            cap = float(self.remaining_cap[b, v])

            unvisited    = ~self.visited[b]
            unvisited[0] = False
            fits         = self.demand[b] <= cap
            arc_feasible = unvisited & fits
            mask[b]      = arc_feasible

            no_arc_fits     = not arc_feasible.any()
            vehicle_has_arc = int(self.cur_arc[b, v]) != 0
            can_switch      = v < self.M - 1

            if can_switch and (vehicle_has_arc or no_arc_fits):
                mask[b, 0] = True

            if not mask[b].any():
                mask[b, 0] = True

        return mask

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, b: int) -> tuple[float, float]:
        """Add return-to-depot for last vehicle, return (total_cost, -total_cost)."""
        v    = int(self.active_vehicle[b])
        prev = int(self.cur_arc[b, v])
        if prev != 0:
            self.vehicle_time[b, v] += float(self.adj[b, prev, 0])
            self.cur_arc[b, v] = 0
        total_cost = float(self.vehicle_time[b].sum())
        return total_cost, -total_cost

    # ------------------------------------------------------------------
    # HCARPEnv interface
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self.n + 1

    def max_steps(self) -> int:
        return self.n * 2 + self.M

    @staticmethod
    def from_directory(path: str, limit: int = None) -> "CVRPEnv":
        files = sorted(glob(f"{path}/**/*.npz", recursive=True))
        if limit:
            files = files[:limit]
        assert files, f"no .npz files found under {path}"
        env = CVRPEnv()
        env.load_files(files)
        return env
