"""
Generate all paper figures and LaTeX tables for the ShARC AAAI submission.

Figures (AAAI double-column width = 3.5 in), saved as PDF + PNG:
  Fig 1 — CVaR severity curves          (severity_sweep.csv)
  Fig 2 — Ablation bar at severity=0.8  (ablation_sweep.csv)
  Fig 3 — Worst-case vs severity        (shifted_baselines.csv + severity_sweep.csv)
  Fig 4 — delta_cvar grouped bar        (transfer_eval.csv)

LaTeX tables:
  Table 1 — Nominal performance         (baselines_unshifted.csv)
  Table 2 — Full ablation across severities (ablation_sweep.csv)
  Both use \\toprule/\\midrule/\\bottomrule; ShARC row is bolded.

Usage:
    python -m evaluation.make_paper_figures \\
        --severity_csv   results/severity_sweep_v2.csv \\
        --ablation_csv   results/ablation_sweep.csv \\
        --baselines_csv  results/shifted_baselines.csv \\
        --transfer_csv   results/transfer_eval.csv \\
        --nominal_csv    results/baselines_unshifted.csv \\
        --out_dir        paper/figures
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

# AAAI double-column figure width in inches
FIG_W = 3.5
FIG_H = 2.5

ABLATION_ORDER = [
    'full_sharc', 'no_shift_ctx', 'no_budget_sig',
    'rn_alpha1', 'alpha_005', 'alpha_020', 'alpha_030', 'no_curriculum',
]
ABLATION_LABELS = {
    'full_sharc':     'Full ShARC',
    'no_shift_ctx':   'No shift ctx',
    'no_budget_sig':  r'No $\beta_t$',
    'rn_alpha1':      'RN (\\alpha=1)',
    'alpha_005':      '\\alpha=0.05',
    'alpha_020':      '\\alpha=0.20',
    'alpha_030':      '\\alpha=0.30',
    'no_curriculum':  'No curriculum',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size':        8,
        'axes.linewidth':   0.6,
        'lines.linewidth':  1.2,
        'axes.spines.top':  False,
        'axes.spines.right': False,
        'figure.dpi':       150,
    })
    return plt


def _save(fig, out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'{stem}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{stem}.png'), bbox_inches='tight', dpi=300)
    print(f"  Saved {stem}.pdf / .png")


def _read(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  Loaded {path}: {len(df)} rows")
        return df
    print(f"  [warn] Missing: {path}")
    return pd.DataFrame()


def _method_col(df: pd.DataFrame) -> str | None:
    for c in ('policy', 'method', 'baseline', 'run_name'):
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Fig 1 — CVaR severity curves (severity_sweep.csv)
# ---------------------------------------------------------------------------

def fig1_cvar_severity(df: pd.DataFrame, out_dir: str):
    if df.empty:
        print("  [skip] Fig 1: no data"); return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    cvar_col = next((c for c in ('cvar_10', 'cvar', 'cvar_T_max') if c in df.columns), None)
    sev_col  = 'severity' if 'severity' in df.columns else 'shift_severity'
    mc       = _method_col(df)

    if mc and mc != 'run_name':
        for method, grp in df.groupby(mc):
            grp = grp.sort_values(sev_col)
            ax.plot(grp[sev_col], grp[cvar_col], label=method, marker='o', markersize=3)
    elif cvar_col:
        grp = df.sort_values(sev_col)
        ax.plot(grp[sev_col], grp[cvar_col], label='ShARC', marker='o', markersize=3)

    ax.set_xlabel('Shift Severity $\\varphi$')
    ax.set_ylabel('CVaR$_{10}$ Makespan')
    ax.set_title('CVaR$_{10}$ vs. Shift Severity')
    ax.legend(fontsize=6, framealpha=0.7)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, out_dir, 'fig1_cvar_severity')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — Ablation bar at severity=0.8 (ablation_sweep.csv)
# ---------------------------------------------------------------------------

def fig2_ablation_bar(df: pd.DataFrame, out_dir: str, severity: float = 0.8):
    if df.empty:
        print("  [skip] Fig 2: no data"); return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    sub = df[np.isclose(df['severity'], severity)].copy()
    if sub.empty:
        print(f"  [warn] Fig 2: no rows at severity={severity}"); plt.close(fig); return

    present = sub['run_name'].tolist()
    order   = [r for r in ABLATION_ORDER if r in present] + \
              [r for r in present if r not in ABLATION_ORDER]
    sub_ord = sub.set_index('run_name').reindex(order).dropna(subset=['cvar_10'])
    labels  = [ABLATION_LABELS.get(r, r) for r in sub_ord.index]
    colors  = ['#2166ac' if r == 'full_sharc' else '#92c5de' for r in sub_ord.index]

    x = np.arange(len(labels))
    ax.bar(x, sub_ord['cvar_10'], color=colors, edgecolor='k', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=6)
    ax.set_ylabel('CVaR$_{10}$ Makespan')
    ax.set_title(f'Ablation at $\\varphi$={severity}')
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, out_dir, 'fig2_ablation_bar')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 — Worst-case vs severity: all methods
# ---------------------------------------------------------------------------

def fig3_worst_case(baselines_df: pd.DataFrame, sweep_df: pd.DataFrame, out_dir: str):
    if baselines_df.empty and sweep_df.empty:
        print("  [skip] Fig 3: no data"); return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    sev_col = 'severity'

    # Classical baselines
    if not baselines_df.empty and 'worst_case' in baselines_df.columns:
        for bl, grp in baselines_df.groupby('baseline'):
            agg = grp.groupby(sev_col)['worst_case'].mean().reset_index().sort_values(sev_col)
            ax.plot(agg[sev_col], agg['worst_case'], label=bl,
                    marker='s', markersize=3, linestyle='--')

    # ShARC / neural policies
    if not sweep_df.empty and 'worst_case' in sweep_df.columns:
        mc = _method_col(sweep_df)
        if mc and mc in sweep_df.columns and mc not in ('run_name',):
            for method, grp in sweep_df.groupby(mc):
                grp = grp.sort_values(sev_col)
                ax.plot(grp[sev_col], grp['worst_case'], label=method,
                        marker='o', markersize=3)
        else:
            grp = sweep_df.sort_values(sev_col)
            ax.plot(grp[sev_col], grp['worst_case'], label='ShARC',
                    marker='o', markersize=3)

    ax.set_xlabel('Shift Severity $\\varphi$')
    ax.set_ylabel('Worst-case Makespan')
    ax.set_title('Worst-case Makespan vs. Shift Severity')
    ax.legend(fontsize=6, framealpha=0.7)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, out_dir, 'fig3_worst_case')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 — delta_cvar grouped bar: CVRP-50 / CVRP-100
# ---------------------------------------------------------------------------

def fig4_delta_cvar(df: pd.DataFrame, out_dir: str):
    if df.empty or 'delta_cvar' not in df.columns:
        print("  [skip] Fig 4: no data or missing delta_cvar"); return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    sub = df[df['method'] == 'sharc'].dropna(subset=['delta_cvar'])
    if sub.empty:
        sub = df.dropna(subset=['delta_cvar'])

    problems   = sorted(sub['problem'].unique()) if 'problem' in sub.columns else ['cvrp50']
    severities = sorted(sub['severity'].unique())
    width      = 0.35
    x          = np.arange(len(severities))

    for i, prob in enumerate(problems):
        vals = []
        for sev in severities:
            mask = (sub['problem'] == prob) & np.isclose(sub['severity'], sev)
            vals.append(float(sub.loc[mask, 'delta_cvar'].mean()) if mask.any() else 0.0)
        offset = (i - len(problems) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=prob, edgecolor='k', linewidth=0.4)

    ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s:.1f}' for s in severities], fontsize=6)
    ax.set_xlabel('Shift Severity $\\varphi$')
    ax.set_ylabel('$\\Delta$ CVaR$_{10}$  (RN $-$ ShARC)')
    ax.set_title('ShARC Tail Advantage (CVRP Transfer)')
    ax.legend(fontsize=6, framealpha=0.7)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, out_dir, 'fig4_delta_cvar')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table 1 — Nominal performance (baselines_unshifted.csv)
# ---------------------------------------------------------------------------

def table1_nominal(df: pd.DataFrame, out_dir: str):
    if df.empty:
        print("  [skip] Table 1: no data"); return

    mc       = _method_col(df)
    mean_col = next((c for c in ('mean_makespan', 'mean_T1', 'mean_T_max') if c in df.columns), None)
    cvar_col = next((c for c in ('cvar_10', 'cvar_T_max') if c in df.columns), None)
    wc_col   = next((c for c in ('worst_case',) if c in df.columns), None)

    if not mc or not mean_col:
        print("  [warn] Table 1: cannot identify method or mean column"); return

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Nominal performance ($\varphi=0$). Bold = ShARC.}',
        r'\label{tab:nominal}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'Method & Mean & CVaR$_{10}$ & Worst-case \\',
        r'\midrule',
    ]

    for _, row in df.iterrows():
        name = str(row[mc])
        bold = name.lower() in ('sharc', 'cvar_rl', 'full_sharc')

        def fmt(col):
            if col and col in df.columns:
                v = row[col]
                s = f'{v:.3f}'
                return r'\textbf{' + s + r'}' if bold else s
            return '---'

        name_tex = r'\textbf{' + name + r'}' if bold else name
        lines.append(f'{name_tex} & {fmt(mean_col)} & {fmt(cvar_col)} & {fmt(wc_col)} \\\\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, out_dir, 'table1_nominal')


# ---------------------------------------------------------------------------
# Table 2 — Full ablation across severities
# ---------------------------------------------------------------------------

def table2_ablation(df: pd.DataFrame, out_dir: str):
    if df.empty:
        print("  [skip] Table 2: no data"); return

    severities = sorted(df['severity'].unique())
    sev_hdrs   = ' & '.join([f'$\\varphi$={s:.1f}' for s in severities])

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Ablation: CVaR$_{10}$ makespan across shift severities. '
        r'Bold = full ShARC.}',
        r'\label{tab:ablation}',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{l' + 'c' * len(severities) + '}',
        r'\toprule',
        f'Run & {sev_hdrs} \\\\',
        r'\midrule',
    ]

    present = df['run_name'].unique().tolist()
    order   = [r for r in ABLATION_ORDER if r in present] + \
              [r for r in present if r not in ABLATION_ORDER]

    for run in order:
        bold      = run == 'full_sharc'
        label     = ABLATION_LABELS.get(run, run)
        label_tex = r'\textbf{' + label + r'}' if bold else label
        vals      = []
        for sev in severities:
            mask = (df['run_name'] == run) & np.isclose(df['severity'], sev)
            if mask.any():
                v = float(df.loc[mask, 'cvar_10'].mean())
                s = f'{v:.3f}'
                vals.append(r'\textbf{' + s + r'}' if bold else s)
            else:
                vals.append('---')
        lines.append(label_tex + ' & ' + ' & '.join(vals) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    _write_tex(lines, out_dir, 'table2_ablation')


def _write_tex(lines: list[str], out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{stem}.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved {stem}.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures and LaTeX tables")
    parser.add_argument('--severity_csv',  default='results/severity_sweep_v2.csv')
    parser.add_argument('--ablation_csv',  default='results/ablation_sweep.csv')
    parser.add_argument('--baselines_csv', default='results/shifted_baselines.csv')
    parser.add_argument('--transfer_csv',  default='results/transfer_eval.csv')
    parser.add_argument('--nominal_csv',   default='results/baselines_unshifted.csv')
    parser.add_argument('--out_dir',       default='paper/figures')
    args = parser.parse_args()

    severity_df  = _read(args.severity_csv)
    ablation_df  = _read(args.ablation_csv)
    baselines_df = _read(args.baselines_csv)
    transfer_df  = _read(args.transfer_csv)
    nominal_df   = _read(args.nominal_csv)

    print("\nFigures:")
    fig1_cvar_severity(severity_df,              args.out_dir)
    fig2_ablation_bar(ablation_df,               args.out_dir)
    fig3_worst_case(baselines_df, severity_df,   args.out_dir)
    fig4_delta_cvar(transfer_df,                 args.out_dir)

    print("\nLaTeX tables:")
    table1_nominal(nominal_df,   args.out_dir)
    table2_ablation(ablation_df, args.out_dir)

    print(f"\nAll outputs in: {args.out_dir}")


if __name__ == '__main__':
    main()
