"""
ShARC architecture diagram.
Requires: matplotlib
Run: python architecture_diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── colour palette ────────────────────────────────────────────────────────────
C = dict(
    data   = "#D6EAF8",   # light blue
    env    = "#D5F5E3",   # light green
    model  = "#FDEBD0",   # light orange
    train  = "#F9EBEA",   # light red
    eval_  = "#EAE6F5",   # light purple
    base   = "#FDFEFE",   # near white
    edge   = "#2C3E50",   # dark slate
    arrow  = "#566573",
    text   = "#1A252F",
    sub    = "#5D6D7E",
)

def box(ax, x, y, w, h, label, sublabels=(), color="#FFFFFF",
        fontsize=9, subsize=7.5, radius=0.04):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.01,rounding_size={radius}",
                          linewidth=1.2, edgecolor=C["edge"],
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    cy = y + h - 0.045
    ax.text(x + w/2, cy, label,
            ha="center", va="top", fontsize=fontsize,
            fontweight="bold", color=C["text"], zorder=4)
    for i, s in enumerate(sublabels):
        ax.text(x + w/2, cy - 0.055*(i+1), s,
                ha="center", va="top", fontsize=subsize,
                color=C["sub"], zorder=4)

def arrow(ax, x0, y0, x1, y1, label="", color=C["arrow"], lw=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10),
                zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.01, my, label, fontsize=6.5, color=C["sub"],
                ha="left", va="center", zorder=6,
                bbox=dict(fc="white", ec="none", pad=1))

def bracket(ax, x, y, w, h, label, color, lw=1.0):
    """Dashed group rectangle with label in top-left corner."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.01,rounding_size=0.03",
                          linewidth=lw, edgecolor=color,
                          linestyle="--", facecolor="none", zorder=2)
    ax.add_patch(rect)
    ax.text(x+0.01, y+h-0.01, label,
            fontsize=7, color=color, va="top", zorder=3,
            style="italic")

# ── canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

# ── group brackets ────────────────────────────────────────────────────────────
bracket(ax, 0.01, 0.55, 0.16, 0.40, "Data",        "#2E86C1")
bracket(ax, 0.19, 0.30, 0.27, 0.65, "RL Environment","#1E8449")
bracket(ax, 0.48, 0.30, 0.30, 0.65, "Neural Policy", "#CA6F1E")
bracket(ax, 0.80, 0.30, 0.18, 0.65, "Training",      "#922B21")
bracket(ax, 0.01, 0.01, 0.97, 0.26, "Baselines",     "#6C3483")

# ── DATA column ───────────────────────────────────────────────────────────────
box(ax, 0.02, 0.84, 0.14, 0.09, "Instance (.npz)",
    ("req / nonreq arcs", "C, M, P"),
    C["data"])

box(ax, 0.02, 0.72, 0.14, 0.09, "Data Generators",
    ("gen.py", "gen_eval_osm.py"),
    C["data"])

box(ax, 0.02, 0.60, 0.14, 0.09, "common/ops.py",
    ("import_instance()", "Floyd–Warshall"),
    C["data"])

arrow(ax, 0.09, 0.84, 0.09, 0.81)   # npz → generator (label omitted)
arrow(ax, 0.09, 0.72, 0.09, 0.69)
arrow(ax, 0.16, 0.645, 0.20, 0.645, "parsed\ngraph")  # ops → env

# ── ENVIRONMENT column ────────────────────────────────────────────────────────
box(ax, 0.20, 0.76, 0.25, 0.16, "HCARPEnv",
    ("reset() / step(action)",
     "obs: adj, visited, mask, ...",
     "reward: −(10⁴T₁+10²T₂+T₃)"),
    C["env"])

box(ax, 0.20, 0.56, 0.25, 0.16, "ShiftScheduler",
    ("modes: curriculum / uniform",
     "δ_demand, δ_cost, p_avail",
     "shift_context [B,3]"),
    C["env"])

box(ax, 0.20, 0.36, 0.25, 0.16, "cal_reward.py  +  local_search.py",
    ("reward_ins() — Numba JIT",
     "intra 2-opt  |  inter swap",
     "variants: Priority / Uniform"),
    C["env"])

arrow(ax, 0.325, 0.76, 0.325, 0.72)          # env ↔ reward
arrow(ax, 0.325, 0.56, 0.325, 0.52)          # shift → reward
arrow(ax, 0.325, 0.36, 0.325, 0.52, "shift\ncontext")
arrow(ax, 0.20, 0.84, 0.20, 0.92,            # shift injects into env
      color="#1E8449")
ax.annotate("", xy=(0.20, 0.84), xytext=(0.245, 0.56),
            arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                            lw=1.2, connectionstyle="arc3,rad=0.3"),
            zorder=5)

# ── POLICY column ─────────────────────────────────────────────────────────────
box(ax, 0.49, 0.76, 0.28, 0.16, "ArcEncoder",
    ("3× Transformer (pre-LN)",
     "d_model=128, 8 heads, d_ff=512",
     "arc embeddings H ∈ ℝ^{B×n×128}"),
    C["model"])

box(ax, 0.49, 0.56, 0.28, 0.16, "AttentionDecoder",
    ("context: h_cur ‖ cap ‖ time ‖ budget ‖ e_shift",
     "logits = clip(q·kᵀ/√d, ±10)",
     "mask infeasible → log_softmax"),
    C["model"])

box(ax, 0.49, 0.36, 0.28, 0.16, "HCARPPolicy",
    ("encode(obs) → arc_emb",
     "act(obs, arc_emb) → action, log_prob",
     "rollout(env) → τ, log π(τ), R"),
    C["model"])

arrow(ax, 0.63, 0.76, 0.63, 0.72)   # encoder → decoder
arrow(ax, 0.63, 0.56, 0.63, 0.52)   # decoder → policy

# env → encoder
arrow(ax, 0.45, 0.845, 0.49, 0.845, "obs")
# env → policy (action feedback)
arrow(ax, 0.45, 0.365, 0.49, 0.365, "action\nstep")
# policy → env
ax.annotate("", xy=(0.45, 0.44), xytext=(0.49, 0.44),
            arrowprops=dict(arrowstyle="<|-", color=C["arrow"],
                            lw=1.2),
            zorder=5)
ax.text(0.455, 0.455, "action", fontsize=6.5, color=C["sub"],
        ha="center", va="bottom")

# ── TRAINING column ───────────────────────────────────────────────────────────
box(ax, 0.81, 0.76, 0.16, 0.16, "REINFORCE",
    ("advantage A = R − R̂",
     "normalize (μ, σ)",
     "L = −E[A·log π]"),
    C["train"])

box(ax, 0.81, 0.56, 0.16, 0.16, "CVaR-REINFORCE",
    ("VaR_α = quantile_α{R}",
     "w_i = 1[R≤VaR] / αB",
     "L = −Σ wᵢ(Rᵢ−CVaR̂)·log π"),
    C["train"])

box(ax, 0.81, 0.36, 0.16, 0.16, "Train loop",
    ("Adam lr=1e-4",
     "grad clip=1.0",
     "save best checkpoint"),
    C["train"])

# policy → training
arrow(ax, 0.77, 0.845, 0.81, 0.845, "log π, R")
arrow(ax, 0.77, 0.645, 0.81, 0.645, "log π, R")
arrow(ax, 0.77, 0.445, 0.81, 0.445)

# training loop feeds back to policy
ax.annotate("", xy=(0.89, 0.76), xytext=(0.89, 0.52),
            arrowprops=dict(arrowstyle="-|>", color="#922B21",
                            lw=1.2, connectionstyle="arc3,rad=-0.35"),
            zorder=5)
ax.text(0.945, 0.64, "θ update", fontsize=6.5, color="#922B21",
        ha="center", rotation=90)

# ── BASELINES row ─────────────────────────────────────────────────────────────
bx_params = [
    (0.02, 0.06, 0.18, "ILS",
     ("cheapest insertion", "3× intra+inter 2-opt", "20 samples")),
    (0.22, 0.06, 0.18, "EA",
     ("pop=200, tournament=4", "crossover + 2-opt mutate", "50 epochs")),
    (0.42, 0.06, 0.18, "ACO",
     ("50 ants, α=β=1, ρ=0.5", "pheromone update", "500 epochs")),
    (0.62, 0.06, 0.18, "LP-HCARP",
     ("Gurobi MILP", "x_{m,a}, y_{m,a,k}, t_{m,k}", "600 s timeout")),
    (0.82, 0.06, 0.16, "ShiftEvaluator",
     ("severity ∈ [0,1]", "mean / CVaR₀.₁ / gap", "severity_sweep_v2.csv")),
]
for bx, by, bw, bl, bsl in bx_params:
    col = C["eval_"] if "Evaluator" in bl else C["base"]
    box(ax, bx, by, bw, 0.18, bl, bsl, col)

# arrows from shared data/env into baselines
for bx, by, bw, bl, _ in bx_params[:-1]:
    ax.annotate("", xy=(bx + bw/2, by+0.18),
                xytext=(0.09 if bx < 0.19 else 0.325, 0.36),
                arrowprops=dict(arrowstyle="-|>", color="#6C3483",
                                lw=0.9, linestyle="dashed",
                                connectionstyle="arc3,rad=0"),
                zorder=5)

# policy → evaluator
arrow(ax, 0.63, 0.36, 0.90, 0.24, "greedy\nrollout")

# ── title & legend ────────────────────────────────────────────────────────────
ax.text(0.5, 0.985, "ShARC — System Architecture",
        ha="center", va="top", fontsize=14, fontweight="bold",
        color=C["text"], transform=ax.transAxes)

legend_items = [
    mpatches.Patch(facecolor=C["data"],  edgecolor=C["edge"], label="Data / Graph ops"),
    mpatches.Patch(facecolor=C["env"],   edgecolor=C["edge"], label="RL Environment"),
    mpatches.Patch(facecolor=C["model"], edgecolor=C["edge"], label="Neural Policy"),
    mpatches.Patch(facecolor=C["train"], edgecolor=C["edge"], label="Training"),
    mpatches.Patch(facecolor=C["eval_"], edgecolor=C["edge"], label="Evaluation"),
    mpatches.Patch(facecolor=C["base"],  edgecolor=C["edge"], label="Baselines"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=7.5,
          framealpha=0.9, ncol=3, bbox_to_anchor=(1.0, 0.0))

plt.tight_layout(pad=0.3)
plt.savefig("architecture.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved architecture.png")
plt.show()
