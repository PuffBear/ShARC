"""
HCARP RL Environment — Hierarchical Capacitated Arc Routing Problem.

Kept in pure NumPy so it meshes with the existing numba/numpy codebase.
The model layer handles tensor conversion.

Observation dict (all np.ndarray):
  Static — fixed for the entire episode:
    adj            [B, n+1, n+1]  deadhead distances (may be shift-scaled)
    service_time   [B, n+1]       service time per arc (0 for depot at index 0)
    demand         [B, n+1]       normalised demand per arc (may be shift-perturbed)
    clss           [B, n+1]       priority class: 0=depot, 1/2/3=tier

  Dynamic — updated each step:
    visited        [B, n+1]  bool   arc already serviced?
    cur_arc        [B, M]    int32  current arc index per vehicle
    remaining_cap  [B, M]    f32    remaining capacity ∈ [0,1]
    vehicle_time   [B, M]    f32    accumulated route time
    active_vehicle [B]       int32  which vehicle is being dispatched
    budget         [B]       f32    running max vehicle time (proxy for worst-case cost)

  Shift context (zero if no shift scheduler):
    shift_context  [B, 3]    f32    [delta_demand, delta_cost, p_availability]

  Mask:
    action_mask    [B, n+1]  bool  True = feasible action for active vehicle

Action:
  np.ndarray [B] of ints in {0 .. n}
    0   → end current vehicle's route, advance to next vehicle
    i   → service arc i (1-indexed into required arcs)

Reward:
  0.0 at intermediate steps.
  −(1e4·T1 + 1e2·T2 + 1e0·T3) per instance when done.
  Shape [B] float32.
"""

from __future__ import annotations

import numpy as np
from glob import glob
from common.ops import import_instance
from common.cal_reward import get_Ts


LEX_WEIGHTS = np.float32([1e4, 1e2, 1e0])


class HCARPEnv:
    """
    Batched RL environment for HCARP.

    All instances in a batch must share the same n_arcs and n_vehicles.
    Use HCARPEnv.from_directory() or load_files() / load_batch() to initialise.

    Parameters
    ----------
    shift_scheduler : ShiftScheduler | None
        If provided, per-instance distribution shifts are applied at reset().
    """

    def __init__(self, shift_scheduler=None):
        self._instances: list[dict] = []
        self.shift_scheduler = shift_scheduler

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_files(self, paths: list[str]) -> "HCARPEnv":
        self._instances = [self._parse(p) for p in paths]
        return self

    def load_batch(self, instances: list[dict]) -> "HCARPEnv":
        """Accept pre-parsed dicts (e.g. from the data generator)."""
        self._instances = instances
        return self

    def _parse(self, path: str) -> dict:
        dms, P, M, demands, clss, s, d, edge_indxs = import_instance(path)
        return dict(
            dms=dms.astype(np.float32),           # [n+1, n+1]
            demands=demands.astype(np.float32),    # [n+1], already /C
            clss=clss.astype(np.int32),            # [n+1]
            service_time=s.astype(np.float32),     # [n+1]
            n_arcs=int(len(demands) - 1),
            n_vehicles=int(len(M)),
            n_priorities=int(len(P)),
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """
        Initialise a fresh episode from the loaded instances.
        Applies per-instance distribution shifts if a shift_scheduler is set.
        Returns the first observation dict.
        """
        assert self._instances, "call load_files() or load_batch() first"

        inst = self._instances
        B = len(inst)
        n = inst[0]["n_arcs"]
        M = inst[0]["n_vehicles"]

        assert all(i["n_arcs"] == n for i in inst), "all instances must have same n_arcs"
        assert all(i["n_vehicles"] == M for i in inst), "all instances must have same n_vehicles"

        self.B = B
        self.n = n
        self.M = M

        # --- Static arrays (copies; shifts applied below) -----------------
        self.adj          = np.stack([i["dms"]          for i in inst])  # [B, n+1, n+1]
        self.service_time = np.stack([i["service_time"] for i in inst])  # [B, n+1]
        self.demand       = np.stack([i["demands"]      for i in inst])  # [B, n+1]
        self.clss         = np.stack([i["clss"]         for i in inst])  # [B, n+1]

        # --- Dynamic arrays -----------------------------------------------
        self.cur_arc       = np.zeros((B, M), dtype=np.int32)
        self.remaining_cap = np.ones((B, M),  dtype=np.float32)
        self.vehicle_time  = np.zeros((B, M), dtype=np.float32)
        self.budget        = np.zeros(B,       dtype=np.float32)

        self.visited = np.zeros((B, n + 1), dtype=bool)
        self.visited[:, 0] = True  # depot never an arc action

        self.active_vehicle = np.zeros(B, dtype=np.int32)
        self.done_mask      = np.zeros(B, dtype=bool)

        self._action_log: list[list[int]] = [[] for _ in range(B)]

        # --- Shift context (zero if no scheduler) -------------------------
        self.shift_context = np.zeros((B, 3), dtype=np.float32)

        if self.shift_scheduler is not None:
            shifts = self.shift_scheduler.sample_batch(B)
            for b, sh in enumerate(shifts):
                dd = sh["delta_demand"]
                dc = sh["delta_cost"]
                pa = sh["p_availability"]

                # Demand shift: clamp to [0, 1]; depot stays 0
                self.demand[b] = np.clip(self.demand[b] * (1.0 + dd), 0.0, 1.0)
                self.demand[b, 0] = 0.0

                # Cost shift: scale travel times (never go negative)
                self.adj[b] = self.adj[b] * max(1.0 + dc, 0.0)

                # Task availability: pre-mark unavailable arcs as visited
                for arc_i in range(1, n + 1):
                    if np.random.random() > pa:
                        self.visited[b, arc_i] = True

                self.shift_context[b] = sh["context"]

        return self._get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, dict]:
        """
        actions : int array [B] — one action per instance.

        Returns
        -------
        obs    : dict
        reward : float32 [B]  — non-zero only when instance finishes
        done   : bool    [B]
        info   : dict    — {b: {"T1": ..., "T2": ..., "T3": ...}} for done instances
        """
        actions = np.asarray(actions, dtype=np.int32)
        assert actions.shape == (self.B,)

        reward = np.zeros(self.B, dtype=np.float32)
        info: dict = {}

        for b in range(self.B):
            if self.done_mask[b]:
                continue

            a = int(actions[b])
            self._action_log[b].append(a)
            v = int(self.active_vehicle[b])

            if a == 0:
                self.active_vehicle[b] = min(v + 1, self.M - 1)
            else:
                prev = int(self.cur_arc[b, v])
                self.vehicle_time[b, v] += (float(self.adj[b, prev, a])
                                            + float(self.service_time[b, a]))
                self.remaining_cap[b, v] -= self.demand[b, a]
                self.cur_arc[b, v] = a
                self.visited[b, a] = True

            # Update running budget: current worst-case route time
            self.budget[b] = float(self.vehicle_time[b].max())

            if self.visited[b, 1:].all():
                self.done_mask[b] = True
                r, Ts = self._compute_reward(b)
                reward[b] = r
                info[b] = {"T1": float(Ts[0]), "T2": float(Ts[1]), "T3": float(Ts[2])}

        return self._get_obs(), reward, self.done_mask.copy(), info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        return {
            # Static
            "adj":            self.adj,
            "service_time":   self.service_time,
            "demand":         self.demand,
            "clss":           self.clss,
            # Dynamic
            "visited":        self.visited.copy(),
            "cur_arc":        self.cur_arc.copy(),
            "remaining_cap":  self.remaining_cap.copy(),
            "vehicle_time":   self.vehicle_time.copy(),
            "active_vehicle": self.active_vehicle.copy(),
            "budget":         self.budget.copy(),
            # Shift context
            "shift_context":  self.shift_context.copy(),
            # Feasibility
            "action_mask":    self._get_action_mask(),
        }

    def _get_action_mask(self) -> np.ndarray:
        """
        Returns bool array [B, n+1]; True = feasible.

        Arc i feasible for instance b iff:
          - not yet visited
          - demand[b,i] <= remaining_cap[b, active_vehicle[b]]

        Action 0 (separator) feasible iff:
          - active vehicle has serviced ≥1 arc  AND  a next vehicle exists
          - OR no arc fits (forced switch to avoid deadlock)
        """
        B, n1 = self.B, self.n + 1
        mask = np.zeros((B, n1), dtype=bool)

        for b in range(B):
            if self.done_mask[b]:
                mask[b, 0] = True
                continue

            v = int(self.active_vehicle[b])
            cap = float(self.remaining_cap[b, v])

            unvisited = ~self.visited[b]
            unvisited[0] = False
            fits = self.demand[b] <= cap
            arc_feasible = unvisited & fits
            mask[b] = arc_feasible

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

    def _compute_reward(self, b: int) -> tuple[float, np.ndarray]:
        """Use the (potentially shifted) adj and demand arrays for reward."""
        actions = np.array(self._action_log[b], dtype=np.int32)

        vars_np = {
            "adj":          self.adj[b],
            "service_time": self.service_time[b],
            "clss":         self.clss[b],
            "demand":       self.demand[b],
        }

        Ts = get_Ts(vars_np, actions=actions[None])  # [1, 3]
        Ts = Ts[0]                                   # [3]
        scalar = -float(Ts @ LEX_WEIGHTS)
        return scalar, Ts

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self.n + 1

    def max_steps(self) -> int:
        return self.n * 2 + self.M

    @staticmethod
    def from_directory(path: str, limit: int = None) -> "HCARPEnv":
        files = sorted(glob(f"{path}/**/*.npz", recursive=True))
        if limit:
            files = files[:limit]
        assert files, f"no .npz files found under {path}"
        env = HCARPEnv()
        env.load_files(files)
        return env
