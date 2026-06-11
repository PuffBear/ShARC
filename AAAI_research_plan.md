# ShARC — AAAI Research Plan
**Target:** AAAI 2027 (submission window ~Aug 2026)  
**As of:** June 2026  
**Status going in:** Preliminary results on N=60, V=2, HCARP only. ILS/ACO pending. No transfer experiments. No draft.

---

## Is there an acceptable AAAI paper here?

**Short answer: yes, but only with the transfer experiments done.**

The intersection (arc routing × neural RL × CVaR under continuous compound shift) is genuinely unoccupied. Prof. Cao confirmed the empirical signal looks right. Prof. Aggarwal confirmed the theoretical gap is real and open. The architecture has three defensible novel components, not one.

**But Prof. Cao's concern is the correct one to take seriously:** AAAI reviewers will ask "why HCARP specifically?" and "is this a method or just an application?" Without CVRP/PDP transfer results, the paper reads as application novelty, which AAAI typically rejects. With them, the paper becomes a method paper that happens to be motivated by HCARP — that's publishable.

The other thing that would make this clearly strong rather than borderline: at least a clean negative theoretical result on CVaR constraint satisfaction under non-rectangular shifts, even if you can't prove a positive one. Prof. Aggarwal said "good problem, limited ideas" — that means the result isn't trivially ruled out. Even stating the gap precisely as a proposition with a failed proof attempt that reveals *why* it's hard is publishable in the related work / limitations section and makes the paper more rigorous than most empirical NCO papers.

**Realistic tier:** Strong AAAI submission if CVRP/PDP transfer works. Solid workshop paper (NeurIPS Optimization, ICLR Distrib. Robustness) if transfer doesn't pan out.

---

## Script Inventory

### What you have (and what it does)

| Script | Status | What it does |
|--------|--------|-------------|
| `env/hcarp_env.py` | ✅ Done | Vectorised HCARP environment; loads .npz instances |
| `env/shift.py` | ✅ Done | ShiftConfig + ShiftScheduler (curriculum / uniform / adversarial modes — all present) |
| `models/policy.py` | ✅ Done | Full HCARPPolicy with shift context + β_t budget signal |
| `models/encoder.py` | ✅ Done | Transformer encoder |
| `models/decoder.py` | ✅ Done | Pointer decoder with attention + feasibility masking |
| `training/train.py` | ✅ Done | CVaR-REINFORCE loop; CLI entry point; handles shift scheduler |
| `training/configs/default.py` | ✅ Done | Default hyperparameter config dict |
| `agents/cvar_agent.py` | ✅ Done | Thin wrapper; useful for interactive use |
| `agents/risk_neutral_agent.py` | ✅ Done | Risk-neutral REINFORCE baseline |
| `evaluation/shift_evaluator.py` | ✅ Done | ShiftEvaluator class; `sweep()` and `compare()` methods |
| `evaluation/metrics.py` | ✅ Done | `compute_cvar`, `compute_metrics`, `gap_to_baseline` |
| `evaluation/severity_sweep.py` | ✅ Done | CLI severity sweep — but only compares CVaR-RL vs RN-RL, not classical baselines |
| `baseline/ils.py` | ✅ Done | InsertCheapest ILS; CLI, parallelised via ProcessPoolExecutor |
| `baseline/aco.py` | ✅ Done | ACO baseline |
| `baseline/ea.py` | ✅ Done | Evolutionary algorithm baseline |
| `baseline/lp.py` | ⚠️ Partial | Gurobi MILP baseline; times out; no fallback → returns None |
| `baseline/meta.py` | ✅ Done | Shared baseline utilities |
| `run_all_baselines.py` | ⚠️ Partial | Orchestrates ILS/EA/ACO/LP via subprocess; produces CSV — but classical baselines run on unshifted instances only |
| `data/gen.py` | ✅ Done | OSM instance generation using osmnx; produces .npz files |
| `data/gen_eval.py` | ✅ Done | Eval split generation |
| `data/gen_eval_osm.py` | ✅ Done | OSM-specific eval generation |
| `plot_results.py` | ✅ Done | Severity curve plots |
| `visualize_routes.py` | ✅ Done | Route visualisation |
| `plot_all_gantts_and_stats.py` | ✅ Done | Gantt charts + stats |
| `graph_report.py` | ✅ Done | Graph statistics reporting |
| `inspect_checkpoints.py` | ✅ Done | Checkpoint inspection |
| `temp/build_instance_gdb.py` | ⚠️ Partial | GDB benchmark parser — exists but not integrated into main pipeline |

**One correction to handoff.md:** The adversarial shift mode in `env/shift.py` IS implemented (forces positive shifts with stochastic sampling). It is not a stub. It is a simplified adversarial mode (not a true inner-loop maximisation), which is fine for evaluation.

---

### What you need to add (with specs)

#### Script 1: `baseline/run_baseline_at_shift.py`
**Priority: High — needed for Phase 1**

The current classical baselines (`ils.py`, `aco.py`, `ea.py`) run on unshifted instances. For the paper comparison table, you need them evaluated under the same shifted conditions as the RL policy.

```
Purpose: Apply compound shift to instances before running a classical baseline,
         then collect mean T_max and CVaR_0.1(T_max).

Inputs:
  --baseline       str   one of: ILS | EA | ACO
  --eval_dir       str   path to .npz eval instances
  --severity       float shift severity φ ∈ [0, 1]
  --seed           int   RNG seed (default 42)
  --n_seeds        int   number of severity-seed trials (default 5)
  --out_csv        str   output CSV path

How it works:
  1. Load .npz instance
  2. Apply ShiftScheduler(severity=s, mode="adversarial") to perturb
     demands, costs, availability — produces a modified .npz in /tmp
  3. Run the chosen baseline on the modified instance
  4. Collect T1, T2, T3 from result; compute mean and CVaR_0.1(T_max)
  5. Repeat for n_seeds seeds; write CSV with severity, mean_T_max,
     std_T_max, cvar_T_max, std_cvar_T_max, n_instances, n_seeds

Output columns: baseline, severity, mean_T_max, std_T_max, cvar_T_max,
                std_cvar_T_max, n_instances, n_seeds

Note: The shift application step just means perturbing the .npz arrays
      (demands *= (1 + delta_demand), costs *= (1 + delta_cost),
       drop arcs with p > p_availability) before passing to the baseline solver.
```

---

#### Script 2: `evaluation/ablation_sweep.py`
**Priority: High — needed for Phase 3, the main novelty proof**

```
Purpose: Run the severity sweep for every ablation variant and produce
         a comparison table showing per-component contribution.

Inputs:
  --eval_dir     str   path to eval instances
  --ckpt_dir     str   directory containing per-ablation checkpoints:
                       ckpt_dir/no_shift_ctx/best.pt
                       ckpt_dir/no_budget_sig/best.pt
                       ckpt_dir/rn_alpha1/best.pt
                       ckpt_dir/full_sharc/best.pt
                       ckpt_dir/alpha_005/best.pt  (etc.)
  --out_csv      str   output CSV
  --severities   list  default [0.0, 0.25, 0.5, 0.75, 1.0]
  --n_seeds      int   default 5

What it runs:
  For each checkpoint variant:
    Load policy with correct d_shift config
    Run severity_sweep.evaluate_at_severity() at each φ level × n_seeds
    Aggregate: mean ± std for mean_T_max and cvar_T_max

Output: wide-format CSV suitable for a LaTeX ablation table
        columns: variant, severity, mean_T_max, cvar_T_max, delta_vs_full (%)

Note: "no_shift_ctx" means train.py with d_shift=0.
      "no_budget_sig" requires a models/decoder.py flag to zero out β_t.
      Coordinate with training/configs/ to save these variants consistently.
```

---

#### Script 3: `training/configs/ablations.py`
**Priority: High — needed before ablation training runs**

```
Purpose: Define config overrides for each ablation variant.
         Import and pass to train.py instead of default.py.

Contents (dict of dicts):

ABLATIONS = {
    "full_sharc": {},   # no override — use CFG as-is

    "no_shift_ctx": {
        "d_shift": 0,   # removes φ from decoder input
        "run_name": "ablation_no_shift_ctx",
    },

    "no_budget_sig": {
        "use_budget_signal": False,  # new flag — see decoder.py note below
        "run_name": "ablation_no_budget_sig",
    },

    "rn_alpha1": {
        "use_cvar": False,
        "run_name": "ablation_rn",
    },

    "alpha_005": {"alpha": 0.05, "run_name": "ablation_alpha_005"},
    "alpha_020": {"alpha": 0.20, "run_name": "ablation_alpha_020"},
    "alpha_030": {"alpha": 0.30, "run_name": "ablation_alpha_030"},

    "no_curriculum": {
        "shift_mode": "uniform",   # trains at full shift from step 0
        "run_name": "ablation_no_curriculum",
    },
}

Note: models/decoder.py needs a `use_budget_signal` flag added
      (default True) that zeroes β_t out of the context vector when False.
      One line change in the decoder forward pass.
```

---

#### Script 4: `env/cvrp_env.py`
**Priority: Critical — Phase 4, the AAAI make-or-break**

```
Purpose: CVRP environment with the same compound shift interface as HCARPEnv,
         so training/train.py works without modification.

Problem: capacitated VRP on a complete graph of N nodes.
Shift parameterisation (same φ = (δ_d, δ_c, δ_p)):
  δ_d: node demand multiplier (demand_i *= 1 + δ_d * ε_i, ε_i ~ Uniform(-1,1))
  δ_c: edge cost multiplier (cost_ij *= 1 + δ_c * ε_ij)
  δ_p: node availability dropout (node i removed with probability δ_p)

Interface (must match HCARPEnv):
  __init__(shift_scheduler=None)
  load_files(file_list: list[str])      # .npz with keys: coords, demands, capacity
  reset() -> obs
  rollout(policy, greedy) -> (actions, log_probs, rewards, info)

Reward: negative total tour length (sum of edge costs across all vehicles)
        — same sign convention as HCARP reward.

Instance format (.npz):
  coords:   [N, 2]   node (x, y) coordinates (normalised to [0,1])
  demands:  [N]      node demands (depot demand = 0)
  capacity: scalar   vehicle capacity

Data generation: use Kool et al. (2019) generation scheme —
  N nodes uniform in [0,1]^2, demands uniform integer in [1,9],
  capacity = 50 (N=50) or 80 (N=100).
  Generate with data/gen_cvrp.py (Script 6 below).

Feasibility masking: standard CVRP capacity mask —
  mask out nodes whose demand exceeds remaining vehicle capacity.

Note: CVaR objective, shift curriculum, and policy architecture
      are IDENTICAL to HCARP. Only the env changes.
```

---

#### Script 5: `data/gen_cvrp.py`
**Priority: Critical — needed before CVRP training**

```
Purpose: Generate CVRP instances in the .npz format expected by cvrp_env.py.

Inputs (CLI):
  --n_nodes      int   number of customer nodes (50 or 100)
  --n_instances  int   total instances to generate
  --out_dir      str   output directory
  --seed         int   RNG seed

Instance generation (Kool et al. 2019 scheme):
  coords   ~ Uniform([0,1]^2)  for N+1 nodes (index 0 = depot)
  demands  ~ Uniform({1,...,9}) for customer nodes; 0 for depot
  capacity = 50 if n_nodes == 50 else 80

Output: one .npz per instance, keys: coords, demands, capacity
Split: 80% train, 20% eval (by default; controlled by --val_split flag)
```

---

#### Script 6: `evaluation/transfer_eval.py`
**Priority: Critical — produces the CVRP transfer figure for the paper**

```
Purpose: Evaluate ShARC (trained on CVRP) vs baselines at each severity level;
         produce the cross-domain transfer figure alongside the HCARP figure.

Inputs:
  --eval_dir       str   CVRP eval instances dir
  --sharc_ckpt     str   ShARC checkpoint trained on CVRP
  --rn_ckpt        str   Risk-neutral checkpoint trained on CVRP
  --kool_ckpt      str   Kool et al. pretrained checkpoint (no shift training)
  --out_csv        str   output CSV
  --severities     list  default [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  --n_seeds        int   default 5

What it produces:
  CSV with columns: method, n_nodes, severity, mean_cost, std_cost,
                    cvar_cost, std_cvar_cost

  "kool_no_shift": Kool et al. greedy, evaluated under shift but never
                   trained with shift conditioning — shows what happens
                   to a standard NCO model under distribution shift.

Paper figure: 2×2 grid: rows = HCARP / CVRP-50, cols = mean cost / CVaR cost
              Each panel: severity on x-axis, one line per method.
```

---

#### Script 7: `evaluation/make_paper_figures.py`
**Priority: Medium — final step before submission**

```
Purpose: Read all result CSVs and produce publication-quality figures
         in the exact format needed for the paper.

Inputs:
  --hcarp_csv       results/hcarp_severity_sweep.csv
  --cvrp_csv        results/cvrp_transfer.csv
  --ablation_csv    results/ablation_sweep.csv
  --out_dir         figures/

Figures produced:
  fig1_severity_curves.pdf   — main result (HCARP CVaR + mean, all methods)
  fig2_transfer.pdf          — CVRP transfer (side-by-side with HCARP)
  fig3_ablation_table.pdf    — ablation bar chart or table
  fig4_tradeoff.pdf          — mean vs CVaR scatter at φ=1

Style: matplotlib with seaborn, consistent colour scheme,
       error bands = ±1 std over seeds.
```

---

#### Script 8: `data/gen_gdb.py`
**Priority: Low-Medium — gives a second non-OSM dataset for credibility**

```
Purpose: Convert GDB benchmark instances to the .npz format.
         (temp/build_instance_gdb.py exists but is not integrated.)

Action: Refactor temp/build_instance_gdb.py into a proper data/gen_gdb.py
        with the same output format as data/gen.py.

GDB files: standard benchmark available at
           https://www.uv.es/~rmarti/paper/carp.html
           Download the .dat files; parse with temp/parse_gdb.py (already exists).

Output: .npz with keys: req, nonreq, P, M, C (same as OSM instances).
```

---

## Phase 0 — Immediate engineering fixes (1 week)

| Task | File | Fix |
|------|------|-----|
| CVaR weight degeneracy | `training/train.py` line 122 | `n_tail = max(1, int(alpha * B))` — already has the guard; check it handles B<10 |
| Gurobi fallback | `baseline/lp.py` | On timeout or None result, return ILS result with a flag; never return None |
| Numba AOT caching | `common/intra.py`, `common/inter.py`, `common/cal_reward.py` | Add `@nb.njit(cache=True)` to all JIT'd functions |
| Pin dependencies | `requirements.txt` | Run `pip freeze > requirements.txt` in a clean venv |
| `use_budget_signal` flag | `models/decoder.py` | Add one-line flag to zero β_t (needed for ablation 2) |

---

## Phase 1 — Complete the HCARP baseline table (1–2 weeks, compute)

**Run:** `baseline/run_baseline_at_shift.py` (Script 1, write this first) for ILS, ACO, EA at φ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}, 5 seeds each.

**Also run:** `evaluation/severity_sweep.py` on the same eval set for CVaR-RL and RN-RL.

**Merge:** Combine CSVs into one table. Target:

| Method | φ=0 mean | φ=0 CVaR | φ=1 mean | φ=1 CVaR |
|--------|---------|---------|---------|---------|
| ILS | | | | |
| ACO | | | | |
| EA | | | | |
| RN-RL | | | | |
| ShARC (CVaR-RL) | | | | |

---

## Phase 2 — Scale up HCARP eval (1–2 weeks, compute)

- N=200 instances, V=2 and V=4 vehicles, 10 seeds per severity
- Add GDB instances via `data/gen_gdb.py` (Script 8)
- Same severity sweep using existing `evaluation/severity_sweep.py`

---

## Phase 3 — Ablation studies (1 week, compute)

1. Write `training/configs/ablations.py` (Script 3)
2. Add `use_budget_signal` flag to `models/decoder.py`
3. Train each of the 7 ablation variants: full_sharc, no_shift_ctx, no_budget_sig, rn_alpha1, alpha_005, alpha_020, no_curriculum
4. Run `evaluation/ablation_sweep.py` (Script 2) across all checkpoints
5. Produce ablation table showing CVaR₀.₁(T_max) at φ=1

---

## Phase 4 — CVRP/PDP transfer (2–3 weeks, critical)

1. Write `env/cvrp_env.py` (Script 4)
2. Write `data/gen_cvrp.py` (Script 5) — generate CVRP-50 and CVRP-100 instances
3. Train ShARC on CVRP-50 and CVRP-100 using existing `training/train.py` (no changes needed if env interface matches)
4. Write `evaluation/transfer_eval.py` (Script 6) to run the severity sweep
5. Compare against: RN-RL (CVRP), Kool et al. greedy (no shift training)

**Key claim to validate:** The CVaR-REINFORCE + shift curriculum method requires only an environment swap. Architecture and training code are unchanged.

---

## Phase 5 — Paper figures (1 week)

Write `evaluation/make_paper_figures.py` (Script 7). Produces all figures from CSVs in one reproducible script.

---

## Phase 6 — Theoretical framing (parallel, 1–2 weeks of writing)

**The gap (from Prof. Aggarwal's confirmation):**

Standard CMDP primal-dual → expectation-level constraint satisfaction  
CVaR-level → what ShARC needs, no existing result  
PCMDP (Bai et al. 2023) → almost-sure guarantee, no shift model  

**Recommended path (Path B):** Write this as a precisely-stated open problem in Section 7 with three paragraphs:
1. State the three-level hierarchy formally
2. Show where ShARC sits (empirically achieves CVaR-level, theoretically unsupported)
3. Reference PCMDP and state the gap: "whether peak-constraint structure survives non-rectangular shifts is open"

This is more honest and actually more impressive than a vague claim — it shows you understand the theory deeply enough to identify what's missing.

---

## Phase 7 — Paper draft

**Paper structure (8 pages AAAI):**

| Section | Content | Est. length |
|---------|---------|------------|
| 1. Introduction | Problem motivation, tail risk in routing, 3 bullet contributions | 0.75p |
| 2. Related Work | litcomparison.tex condensed, gap statement | 1p |
| 3. Problem Formulation | HCARP, non-rectangular shift model, CVaR objective | 1p |
| 4. ShARC | Encoder-decoder, shift injection, β_t, CVaR-REINFORCE | 1.5p |
| 5. Experiments | HCARP table, ablations, CVRP transfer, severity curves | 2p |
| 6. Analysis | Why tail gap widens, φ-observability scoping, β_t mechanism pseudocode | 0.75p |
| 7. Conclusion | Contributions, open theory gap, future work (online shift estimation) | 0.5p |

**Must-add items not currently in any document:**
- State clearly: "φ is provided as a context input; online φ estimation is future work"
- Add β_t pseudocode box in Section 4 (pre-empts Prof. Cao's question for reviewers)
- Ablation table in Section 5

**When to send to Prof. Cao:** After Phase 1 + Phase 4 results (week 5–6). Send the baseline table, the CVRP transfer figure, and a 2-page draft of Section 5.

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|------------|
| 1 | Phase 0: engineering | Clean reproducible codebase |
| 1 | Write Scripts 1, 3 | `run_baseline_at_shift.py`, `ablations.py` |
| 2–3 | Phase 1: HCARP baselines | Full baseline table CSV |
| 2–3 | Phase 2: scale-up training | N=200, V=4 results |
| 3 | Write Scripts 2, 4, 5 | `ablation_sweep.py`, `cvrp_env.py`, `gen_cvrp.py` |
| 4 | Phase 3: ablation training | 7 checkpoints trained |
| 4–5 | Phase 3: ablation eval | Ablation table |
| 5–7 | Phase 4: CVRP transfer | CVRP train + eval CSVs |
| 6 | Write Scripts 6, 7 | `transfer_eval.py`, `make_paper_figures.py` |
| 6–7 | Write Script 8 (if time) | GDB instance generator |
| 7 | Phase 6: theory write-up | Section 7 draft |
| 7–8 | Email Prof. Cao | Results + Section 5 draft |
| 8–10 | Phase 7: full draft | Complete paper |
| 10 | Submission | |

---

## Compute allocation

| Experiment | Hardware | Est. time | Priority |
|-----------|----------|----------|---------|
| HCARP baseline sweep (ILS/ACO/EA, shifted) | CPU | 4–8h | Immediate |
| HCARP scale-up training (N=200) | GPU | 6–12h per run | High |
| Ablation training (7 variants) | GPU | ~50h total | High |
| CVRP-50 training | GPU | 8–12h | Critical |
| CVRP-100 training | GPU | 12–18h | Critical |
| CVRP transfer eval | CPU/GPU | 2–4h | Critical |

---

## Novelty statement (for paper)

*"ShARC is the first neural combinatorial optimisation framework to address arc routing under continuous compound distribution shifts. Its three architectural contributions — shift-context injection into the decoder, within-episode budget signalling (β_t), and CVaR-REINFORCE with tail-normalised advantages — together close a research gap unaddressed by distributionally robust VRP methods (DRO-VRP, SHIELD) and risk-sensitive RL (Tamar, Chow). The compound shift model induces a non-rectangular uncertainty set over jointly-drifting demands, costs, and availability — a structure for which CVaR-level constraint guarantees remain theoretically open, as we discuss in Section 7."*
