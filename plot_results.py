import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("results/severity_sweep_v2.csv")

# Separate policies
cvar_df = df[df['policy'] == 'cvar_v2']
rn_df = df[df['policy'] == 'rn_v2']

plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: Mean T_max ---
ax1.plot(rn_df['severity'], rn_df['mean_T_max'], marker='o', label='Risk-Neutral (REINFORCE)', color='#d62728', linewidth=2)
ax1.plot(cvar_df['severity'], cvar_df['mean_T_max'], marker='s', label='Risk-Averse (CVaR=0.1)', color='#1f77b4', linewidth=2)
ax1.set_title('Average Performance Under Shift', fontsize=14, fontweight='bold')
ax1.set_xlabel('Shift Severity ($\\phi$)', fontsize=12)
ax1.set_ylabel('Mean $T_{max}$ (Lower is Better)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.7)

# --- Plot 2: CVaR T_max ---
ax2.plot(rn_df['severity'], rn_df['cvar_T_max'], marker='o', label='Risk-Neutral (REINFORCE)', color='#d62728', linewidth=2)
ax2.plot(cvar_df['severity'], cvar_df['cvar_T_max'], marker='s', label='Risk-Averse (CVaR=0.1)', color='#1f77b4', linewidth=2)

# Fill between to highlight the safety gap
ax2.fill_between(rn_df['severity'], cvar_df['cvar_T_max'], rn_df['cvar_T_max'], color='green', alpha=0.1, label='CVaR Safety Margin')

ax2.set_title('Worst-Case (Tail) Performance Under Shift', fontsize=14, fontweight='bold')
ax2.set_xlabel('Shift Severity ($\\phi$)', fontsize=12)
ax2.set_ylabel('$CVaR_{0.1}(T_{max})$ (Lower is Better)', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("results/severity_plot_v2.png", dpi=300, bbox_inches='tight')
print("Plot saved to results/severity_plot_v2.png")
