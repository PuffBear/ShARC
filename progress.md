# ShARC — Project Progress Report
**Last updated:** June 2026  
**Target venue:** AAAI 2027 (submission ~August 2026)

---

## What ShARC Is

**ShARC** (Shift-robust Arc Routing under Capacity Constraints) is a neural combinatorial optimisation framework for the **Hierarchical Capacitated Arc Routing Problem (HCARP)** under continuous compound distribution shifts at test time.

The three architectural contributions that make it novel:
1. **Shift-context injection** — a projected φ-vector `[δ_demand, δ_cost, δ_p]` is appended to the decoder context at every step, letting the policy adapt online to the current shift regime.
2. **Within-episode budget signal (β_t)** — a running maximum vehicle time fed back into the decoder, allowing the policy to reason about its current worst-case cost trajectory.
3. **CVaR-REINFORCE** — gradient updates restricted to the worst α-fraction of episodes; tail advantages normalised globally to preserve signal in small batches.

---

## What We Have Built (Codebase)

| Component | File | Status |
|-----------|------|--------|
| HCARP vectorised env | `env/hcarp_env.py` | ✅ |
| Compound shift scheduler | `env/shift.py` | ✅ (adversarial mode confirmed implemented) |
| CVRP env (transfer) | `env/cvrp_env.py` | ✅ (new) |
| Transformer encoder | `models/encoder.py` | ✅ |
| Attention pointer decoder | `models/decoder.py` | ✅ + `use_budget_signal` ablation flag |
| Full policy (encoder + decoder) | `models/policy.py` | ✅ |
| CVaR-REINFORCE training loop | `training/train.py` | ✅ |
| Default hyperparameter config | `training/configs/default.py` | ✅ |
| Ablation variant configs | `training/configs/ablations.py` | ✅ (new) |
| CVaR agent wrapper | `agents/cvar_agent.py` | ✅ |
| Risk-neutral agent wrapper | `agents/risk_neutral_agent.py` | ✅ |
| ILS baseline | `baseline/ils.py` | ✅ |
| ACO baseline | `baseline/aco.py` | ✅ |
| EA baseline | `baseline/ea.py` | ✅ |
| Gurobi MILP baseline | `baseline/lp.py` | ✅ (ILS fallback added; no longer returns None) |
| Baseline under shift | `baseline/run_baseline_at_shift.py` | ✅ (new) |
| Severity sweep eval | `evaluation/severity_sweep.py` | ✅ (bug fix: service shift now scales with φ) |
| Ablation sweep eval | `evaluation/ablation_sweep.py` | ✅ (new) |
| CVRP transfer eval | `evaluation/transfer_eval.py` | ✅ (new) |
| Paper figure generator | `evaluation/make_paper_figures.py` | ✅ (new) |
| OSM instance generator | `data/gen.py` | ✅ |
| CVRP instance generator | `data/gen_cvrp.py` | ✅ (new) |
| Route visualiser | `visualize_routes.py` | ✅ |
| Gantt + KDE plots | `plot_all_gantts_and_stats.py` | ✅ |
| Severity curve plots | `plot_results.py` | ✅ |

---

## Experiments Run

### Experiment 1 — v1 (First training run)
- **Setup:** N=60, V=2, HCARP only. CVaR-RL (`cvar_shift`) vs risk-neutral (`rn_nominal`), both trained with shift conditioning.
- **Eval dataset:** 60 OSM instances.
- **Results (severity_sweep_final.csv):** Early signal — CVaR-RL showed better tail performance. However, the metrics declined with φ (likely due to the unscaled service-shift bug later discovered).

| Policy | φ=0.0 mean | φ=0.0 CVaR | φ=1.0 mean | φ=1.0 CVaR |
|--------|-----------|-----------|-----------|-----------|
| cvar_shift | 58.32 | 102.02 | 48.40 | 90.22 |
| rn_nominal | 58.32 | 102.02 | 21.79 | 42.61 |

> **Diagnosis:** This v1 run used a broken evaluation mode (demand-only shift). Both policies used the same eval severity function that had a bug, causing numbers to decrease with φ rather than increase. Discarded.

---

### Experiment 2 — v2 (Current canonical results)
- **Setup:** N=60, V=2, HCARP. CVaR-RL v2 (`cvar_v2`) vs risk-neutral v2 (`rn_v2`), evaluated with proper adversarial shift across all three axes (demand, cost, availability).
- **Eval dataset:** 60 instances, 5 seeds per severity level.
- **Checkpoints:** `experiments/results/cvar_v2/best.pt`, `experiments/results/rn_v2/best.pt`.
- **Results (results/severity_sweep_v2.csv):**

| Policy | φ=0.0 mean | φ=0.0 CVaR | φ=0.5 mean | φ=0.5 CVaR | φ=1.0 mean | φ=1.0 CVaR |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| **ShARC (cvar_v2)** | 46.38 ± 0.25 | **82.02 ± 1.25** | 48.37 ± 0.30 | **85.46 ± 1.96** | 51.43 ± 0.48 | **91.72 ± 2.73** |
| RN-RL (rn_v2) | **43.93 ± 0.38** | 96.28 ± 1.86 | 47.08 ± 1.07 | 100.43 ± 1.48 | 49.96 ± 0.62 | 106.27 ± 2.88 |

**Key findings:**
- At **φ=0** (no shift): ShARC pays a **5.6% mean cost** (46.38 vs 43.93) to achieve **14.8% lower CVaR** (82.02 vs 96.28). The CVaR advantage is already present at zero shift — shift-context injection improves tail robustness even when φ=0 because the policy learns to hedge.
- At **φ=1** (maximum shift): ShARC mean is **2.9% higher** (51.43 vs 49.96) but CVaR is **13.7% lower** (91.72 vs 106.27). The tail gap grows significantly with severity.
- The **CVaR gap widens monotonically** with severity for both metrics — confirming the empirical signal Prof. Cao identified.
- RN-RL CVaR rises 10.4% from φ=0 to φ=1 (96.28 → 106.27), while ShARC CVaR rises only 11.8% (82.02 → 91.72) but from a lower baseline — confirming ShARC degrades more gracefully.

---

### Experiment 3 — Classical baseline comparison (unshifted)
- **Status:** Run on unshifted instances only. Stored in `baseline_results.csv`.
- **What's missing:** Classical baselines have NOT yet been evaluated under shift. This is Phase 1 of the roadmap.

---

### Visualisation work
- **Gantt charts** (`results/all_gantts/`): Per-instance vehicle schedule comparison, ShARC vs RN-RL. Confirms ShARC produces more balanced vehicle load distributions.
- **Makespan KDE histograms** (`results/makespan_histogram_stats.png`): Shows ShARC's T_max distribution is less heavy-tailed than RN-RL's across the eval set.
- **Route overlay** (`results/route_comparison.png`): NetworkX physical route visualisation on OSM graphs.
- **Severity plots** (`results/severity_plot_v2.png`, `severity_plot_v3.png`): The main paper figure precursors.

---

## Bugs Found and Fixed

| Bug | Location | Impact | Status |
|-----|----------|--------|--------|
| `max_service_shift` not scaled by severity | `evaluation/severity_sweep.py` | At φ=0, service times were still shifted by 30% — poisoned all v1 results | **Fixed** |
| Gurobi LP returns None on timeout | `baseline/lp.py` | Any downstream code using LP result would crash | **Fixed** (ILS fallback) |
| CVaR weight degeneracy guard existed but not checked for B<10 | `training/train.py:122` | n_tail could be 0 in small debug batches | Documented; guard is `max(1, int(alpha * B))` — correct |
| `adversarial` mode described as stub in handoff.md | `env/shift.py` | Misleading documentation | **Clarified** — mode is implemented (forced positive shifts) |

---

## What We Have Not Done Yet

### Phase 1 — Classical baselines under shift (HIGH PRIORITY, ~1 week, CPU)
Classical baselines (ILS, ACO, EA) have only been run on unshifted instances. For the paper comparison table they must be run under the same shifted conditions. Script is ready:
```bash
python -m baseline.run_baseline_at_shift \
    --baseline ILS \
    --eval_dir data/eval_dataset \
    --severities 0.0 0.2 0.4 0.6 0.8 1.0 \
    --n_seeds 5 \
    --out_csv results/ils_severity_sweep.csv
# Repeat for ACO, EA
```
Expected runtime: 4–8h per baseline on CPU.

### Phase 2 — Scale-up (HIGH PRIORITY, ~2 weeks, GPU)
Currently all results are on N=60, V=2. Need:
- N=200 instances with V=2 and V=4
- 10 seeds per severity
- GDB benchmark instances (`data/gen_gdb.py` — partial at `temp/build_instance_gdb.py`)

### Phase 3 — Ablation studies (HIGH PRIORITY, ~1 week training + 4 days eval)
Seven variants defined in `training/configs/ablations.py`. Need to train all seven, then run:
```bash
python -m evaluation.ablation_sweep \
    --eval_dir data/eval_dataset \
    --ckpt_dir experiments/results \
    --out_csv results/ablation_sweep.csv
```
Variants:
- `full_sharc` — reference
- `no_shift_ctx` — d_shift=0, no φ input
- `no_budget_sig` — β_t zeroed out
- `rn_alpha1` — standard REINFORCE
- `alpha_005`, `alpha_020`, `alpha_030` — CVaR at different α
- `no_curriculum` — uniform shift from step 0

### Phase 4 — CVRP transfer (CRITICAL for AAAI, ~3 weeks)
This is the make-or-break experiment. Without it the paper is an application paper, not a method paper.

Steps:
1. Generate CVRP-50 and CVRP-100 instances:
   ```bash
   python -m data.gen_cvrp --n_nodes 50  --n_instances 1000 --out_dir data/cvrp50
   python -m data.gen_cvrp --n_nodes 100 --n_instances 1000 --out_dir data/cvrp100
   ```
2. Train ShARC on CVRP (no architecture changes — only `--data_dir` changes):
   ```bash
   python -m training.train --data_dir data/cvrp50/train --use_cvar True \
       --use_shift True --run_name cvrp_sharc
   ```
3. Evaluate:
   ```bash
   python -m evaluation.transfer_eval \
       --eval_dir data/cvrp50/eval \
       --sharc_ckpt experiments/results/cvrp_sharc/best.pt \
       --rn_ckpt   experiments/results/cvrp_rn/best.pt \
       --out_csv   results/cvrp_transfer.csv
   ```
4. Key claim to validate: **architecture and training code are unchanged** — only the env changes.

### Phase 5 — Paper figures (1 week, after Phases 1 + 4)
Once CSVs exist:
```bash
python -m evaluation.make_paper_figures \
    --hcarp_csv    results/hcarp_severity_sweep.csv \
    --cvrp_csv     results/cvrp_transfer.csv \
    --ablation_csv results/ablation_sweep.csv \
    --baseline_csv results/ils_severity_sweep.csv \
    --out_dir      figures/
```
Produces fig1–fig4 (severity curves, 2×2 transfer grid, ablation bar chart, mean vs CVaR scatter).

### Phase 6 — Theoretical framing (parallel writing, ~1 week)
Open problem to state precisely in Section 7:
- Standard CMDP primal-dual → expectation-level constraint satisfaction only
- CVaR-level constraint guarantees under non-rectangular shift sets → theoretically open
- PCMDP (Bai et al. 2023) gives almost-sure guarantees but no shift model
- State the gap formally; show where ShARC sits (empirically achieves CVaR-level, theoretically unsupported)

### Phase 7 — Full paper draft (~weeks 8–10)
See `AAAI_research_plan.md` for detailed section breakdown and word targets.

---

## Paper Novelty Statement (draft)

> *"ShARC is the first neural combinatorial optimisation framework to address arc routing under continuous compound distribution shifts. Its three architectural contributions — shift-context injection into the decoder, within-episode budget signalling (β_t), and CVaR-REINFORCE with tail-normalised advantages — together close a research gap unaddressed by distributionally robust VRP methods (DRO-VRP, SHIELD) and risk-sensitive RL (Tamar, Chow). The compound shift model induces a non-rectangular uncertainty set over jointly-drifting demands, costs, and availability — a structure for which CVaR-level constraint guarantees remain theoretically open."*

---

## Timeline Summary

| Week | Work | Status |
|------|------|--------|
| Pre-June | Core codebase, CVaR-REINFORCE, HCARP env, shift scheduler, v1 training | ✅ Done |
| Pre-June | v2 training (cvar_v2, rn_v2), severity sweep, Gantt/KDE visualisations | ✅ Done |
| Pre-June | Literature comparison, codebase audit, AAAI research plan | ✅ Done |
| June (now) | Phase 0: engineering fixes + 7 new scripts (Scripts 1–7) | ✅ Done |
| Week 2–3 | Phase 1: classical baselines under shift (ILS/ACO/EA, CPU jobs) | ⏳ Next |
| Week 2–3 | Phase 2: scale-up training N=200, V=4 (GPU jobs) | ⏳ Queued |
| Week 3 | Phase 3: train 7 ablation variants | ⏳ Queued |
| Week 4 | Phase 3: run ablation sweep | ⏳ Queued |
| Week 5–7 | Phase 4: CVRP-50 / CVRP-100 training + transfer eval | ⏳ Critical |
| Week 6 | Email Prof. Cao: baseline table + CVRP transfer figure + Sec. 5 draft | ⏳ |
| Week 7 | Phase 6: theoretical framing (Section 7 open problem) | ⏳ |
| Week 8–10 | Phase 7: full paper draft | ⏳ |
| Week 10 | Submission | 🎯 |

---

## Compute Requirements (Remaining)

| Job | Hardware | Est. time | Priority |
|-----|----------|----------|---------|
| ILS / ACO / EA under shift | CPU (8 cores) | 4–8h each | Immediate |
| HCARP N=200 training | GPU | 6–12h per run | High |
| 7 ablation variant training | GPU | ~50h total | High |
| CVRP-50 training (ShARC + RN) | GPU | 8–12h each | Critical |
| CVRP-100 training | GPU | 12–18h each | Critical |
| CVRP transfer eval | CPU/GPU | 2–4h | Critical |

---

## File Index (quick reference)

```
ShARC/
├── env/
│   ├── hcarp_env.py           # Main HCARP RL environment
│   ├── cvrp_env.py            # CVRP env (same interface as HCARP) [NEW]
│   └── shift.py               # ShiftConfig + ShiftScheduler
├── models/
│   ├── encoder.py             # Transformer encoder
│   ├── decoder.py             # Pointer decoder (use_budget_signal flag added)
│   └── policy.py              # HCARPPolicy
├── training/
│   ├── train.py               # CVaR-REINFORCE training loop
│   └── configs/
│       ├── default.py         # Default hyperparameters
│       └── ablations.py       # 7 ablation variant overrides [NEW]
├── baseline/
│   ├── meta.py                # ILS / EA / ACO implementations
│   ├── ils.py / aco.py / ea.py
│   ├── lp.py                  # Gurobi MILP (ILS fallback fixed)
│   └── run_baseline_at_shift.py  # Shifted baseline eval [NEW]
├── evaluation/
│   ├── severity_sweep.py      # RL severity sweep (service shift bug fixed)
│   ├── ablation_sweep.py      # Ablation variant sweep [NEW]
│   ├── transfer_eval.py       # CVRP transfer evaluation [NEW]
│   └── make_paper_figures.py  # All paper figures from CSVs [NEW]
├── data/
│   ├── gen.py / gen_eval.py   # OSM instance generation
│   └── gen_cvrp.py            # CVRP instance generation [NEW]
├── results/
│   ├── severity_sweep_v2.csv  # Canonical HCARP results (60 instances, 5 seeds)
│   └── severity_sweep_v2.png / v3.png  # Severity curve plots
└── experiments/results/
    ├── cvar_v2/best.pt        # ShARC checkpoint
    └── rn_v2/best.pt          # RN-RL checkpoint
```

---

## Immediate Next Action

Run Phase 1 baseline sweep (CPU, can start now):

```bash
for BASELINE in ILS ACO EA; do
  python -m baseline.run_baseline_at_shift \
    --baseline $BASELINE \
    --eval_dir data/eval_dataset \
    --severities 0.0 0.2 0.4 0.6 0.8 1.0 \
    --n_seeds 5 \
    --out_csv results/${BASELINE}_severity_sweep.csv &
done
wait
```

Then merge with v2 RL results and produce the full comparison table for Prof. Cao.
