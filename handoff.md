# Project Handoff: ShARC (ReMAR)

## Goals: what we're trying to build
We are building **ShARC** (Shift-robust Arc Routing under Capacity Constraints), an agent that solves the Hierarchical Capacitated Arc Routing Problem (H-CARP) under distribution shifts. The objective is to train a deep reinforcement learning (RL) policy using a risk-averse CVaR (Conditional Value at Risk) objective. This policy aims to outperform classical heuristics (ILS, EA, ACO) and risk-neutral RL policies when there are unexpected shifts in demand, routing costs, or task availability at test time.

## Current state: where the work stands right now
The codebase is currently in a functional, research-grade state ($\sim$2,400 lines of Python). The core algorithm (CVaR-REINFORCE with explicit shift conditioning) is implemented and validated. We have:
*   A vectorized, batched NumPy RL environment (`env/hcarp_env.py`).
*   A Transformer encoder-pointer decoder neural policy (`models/policy.py`).
*   Classical heuristic baselines and exact solvers for comparison.
*   A functional training loop and an evaluation pipeline capable of running severity sweeps.

## Files in flight: active files
Based on the identified future optimizations and technical debt, the following files will likely require active modification next:
*   `baseline/lp.py` and `run_all_baselines.py` - Needs a graceful fallback mechanism for when the Gurobi LP baseline times out.
*   `env/shift.py` - The `"adversarial"` shift mode is currently just an unimplemented stub.
*   `training/train.py` - Needs a fix to prevent CVaR weight degeneracy in small batches where $\lfloor\alpha B\rfloor$ could evaluate to 0.
*   `common/intra.py`, `common/inter.py`, `common/cal_reward.py` - Needs ahead-of-time (AOT) caching (`@nb.njit(cache=True)`) to eliminate the 10-30s Numba cold-start compilation overhead.
*   `requirements.txt` - Needs dependency version pinning for better reproducibility.

## Failed attempts: what didn't work and why
*   **Risk-Neutral RL:** Optimizing strictly for expected returns (Standard REINFORCE) resulted in a $\sim$20% degradation in CVaR metrics under maximum distribution shift ($s=1$) compared to the CVaR-RL agent. This showed that standard RL fails to prepare the policy for worst-case distribution shifts.
*   **Exact LP-HCARP Scaling:** The Gurobi mixed-integer linear programming (MILP) baseline is heavily prone to timeouts due to the computational complexity of solving H-CARP exactly. When it times out, it currently returns `None` without falling back to a heuristic.
*   **Python-Native Reward Loops:** Calculating routing rewards and running local search entirely in native Python was too slow to be viable for the RL training loop.

## Successful attempts: what worked and why, and the results
*   **CVaR-REINFORCE:** Focusing the policy gradient updates on only the worst 10% of trajectories effectively trained the model for tail robustness. This strategy preserved baseline optimality at zero-shift while massively outperforming risk-neutral and classical models under extreme distribution shifts.
*   **Shift Conditioning:** Explicitly feeding a shift-context embedding (severity of demand, cost, and availability shifts) into the decoder pointer network successfully enabled the agent to actively adapt its routing strategy during test time.
*   **Numba JIT Acceleration:** Integrating Numba JIT compilation for the local search loops (2-opt) and objective evaluations solved the performance bottleneck, yielding a 10x+ speedup.
*   **Pointer Network Decoder:** Using an attention-based pointer decoder with feasibility masking avoided invalid action sampling, naturally guiding the agent to adhere to vehicle capacity limits and lexicographic priorities.

## Next step: the single next thing to try
Implement the missing **adversarial shift mode** in `env/shift.py` and enforce **Numba AOT caching** across the `common/` module files to immediately eliminate the cold-start compilation delays hindering iterative experimentation.
