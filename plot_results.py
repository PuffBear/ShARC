import os
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("results/severity_sweep_v2.csv")

# Separate policies
cvar_df = df[df['policy'] == 'cvar_v2']
rn_df   = df[df['policy'] == 'rn_v2']

# Try to load baselines
baseline_lines = {}
if os.path.exists("results/baseline_results.csv"):
    b_df = pd.read_csv("results/baseline_results.csv")
    if len(b_df) > 0 and 'ILS_result' in b_df.columns:
        baseline_lines['ILS (Heuristic)'] = b_df['ILS_result'].mean()
    if len(b_df) > 0 and 'ACO_result' in b_df.columns:
        baseline_lines['ACO (Heuristic)'] = b_df['ACO_result'].mean()

plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

fig.suptitle("HCARP Policy Robustness Under Distribution Shift", fontsize=16, fontweight='bold', y=1.02)
# The subtitle provides the context Prof Cao requested
subtitle = "Scale: $N=60$ graphs | Vehicles: $V=5$ | Evaluation: 5 seeds per severity ($T_{max}$ ± 1 std dev) | Shift: $\\phi \sim (\\delta_{demand}, \\delta_{cost}, \\delta_{service})$"
fig.text(0.5, 0.94, subtitle, ha='center', fontsize=11, fontstyle='italic', color='dimgrey')

# --- Plot 1: Mean T_max ---
ax1.plot(rn_df['severity'], rn_df['mean_T_max'], marker='o', label='Risk-Neutral (REINFORCE)', color='#d62728', linewidth=2)
ax1.fill_between(rn_df['severity'], rn_df['mean_T_max'] - rn_df['std_T_max'], rn_df['mean_T_max'] + rn_df['std_T_max'], color='#d62728', alpha=0.15)

ax1.plot(cvar_df['severity'], cvar_df['mean_T_max'], marker='s', label='Risk-Averse (CVaR=0.1)', color='#1f77b4', linewidth=2)
ax1.fill_between(cvar_df['severity'], cvar_df['mean_T_max'] - cvar_df['std_T_max'], cvar_df['mean_T_max'] + cvar_df['std_T_max'], color='#1f77b4', alpha=0.15)

for name, val in baseline_lines.items():
    ax1.axhline(val, linestyle='--', color='gray', alpha=0.8, label=f"Baseline: {name}")

ax1.set_title('Average Performance Under Shift', fontsize=13, fontweight='bold')
ax1.set_xlabel('Shift Severity ($\\phi$)', fontsize=12)
ax1.set_ylabel('Mean $T_{max}$ (Lower is Better)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.7)

# --- Plot 2: CVaR T_max ---
ax2.plot(rn_df['severity'], rn_df['cvar_T_max'], marker='o', label='Risk-Neutral (REINFORCE)', color='#d62728', linewidth=2)
ax2.fill_between(rn_df['severity'], rn_df['cvar_T_max'] - rn_df['std_cvar_T_max'], rn_df['cvar_T_max'] + rn_df['std_cvar_T_max'], color='#d62728', alpha=0.15)

ax2.plot(cvar_df['severity'], cvar_df['cvar_T_max'], marker='s', label='Risk-Averse (CVaR=0.1)', color='#1f77b4', linewidth=2)
ax2.fill_between(cvar_df['severity'], cvar_df['cvar_T_max'] - cvar_df['std_cvar_T_max'], cvar_df['cvar_T_max'] + cvar_df['std_cvar_T_max'], color='#1f77b4', alpha=0.15)

# Fill between to highlight the safety gap exactly like before
ax2.fill_between(rn_df['severity'], cvar_df['cvar_T_max'], rn_df['cvar_T_max'], color='green', alpha=0.1, label='CVaR Safety Margin')

for name, val in baseline_lines.items():
    ax2.axhline(val, linestyle='--', color='gray', alpha=0.8) # Keep legend in ax1 mostly

ax2.set_title('Worst-Case (Tail) Performance Under Shift', fontsize=13, fontweight='bold')
ax2.set_xlabel('Shift Severity ($\\phi$)', fontsize=12)
ax2.set_ylabel('$CVaR_{0.1}(T_{max})$ (Lower is Better)', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.88) # make room for titles
plt.savefig("results/severity_plot_v3.png", dpi=300, bbox_inches='tight')
print("Plot saved to results/severity_plot_v3.png")
