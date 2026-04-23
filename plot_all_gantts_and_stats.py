import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm

from env.hcarp_env import HCARPEnv
from env.shift import ShiftConfig, ShiftScheduler
from models.policy import HCARPPolicy

def load_policy(ckpt_path, d_shift, device="cpu"):
    policy = HCARPPolicy(d_shift=d_shift, device=device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()
    return policy

def get_gantt_events(policy, env):
    env.reset()
    obs = env._get_obs()
    arc_emb = policy.encode(obs)
    events = {v: [] for v in range(env.M)}
    with torch.no_grad():
        for _ in range(env.max_steps() + env.M):
            v = int(obs["active_vehicle"][0])
            start_time = float(obs["vehicle_time"][0, v])
            actions, _ = policy.act(obs, arc_emb, greedy=True)
            a = int(actions[0].cpu().numpy())
            arc_class = int(obs["clss"][0, a]) if a != 0 else 0
            obs, _, done_np, info = env.step(np.array([a]))
            if a != 0:
                end_time = float(obs["vehicle_time"][0, v])
                events[v].append((start_time, end_time - start_time, a, arc_class))
            if done_np.all():
                break
    return events, info[0]['T1']

def plot_single_gantt(env, best_rn_events, best_rn_tmax, best_cvar_events, best_cvar_tmax, instance_name, out_path):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Hierarchical Routing Timeline Under Severe Shift (φ=1.0) | {instance_name}", fontsize=15, fontweight='bold', y=0.98)
    
    class_colors = {1: '#e74c3c', 2: '#f39c12', 3: '#3498db'}
    
    def plot_gantt(ax, events, title, t1_max):
        ax.set_title(f"{title}  |  Priority 1 Makespan ($T_1$): {t1_max:.2f}", fontsize=12, fontweight='bold', pad=10)
        ax.set_yticks(range(env.M))
        ax.set_yticklabels([f"Vehicle {v+1}" for v in range(env.M)])
        ax.set_ylabel("Dispatch Fleet", fontsize=11)
        
        labels_added = set()
        for v in range(env.M):
            for (start, duration, arc, cls) in events[v]:
                color = class_colors.get(cls, '#bdc3c7')
                label = f"Priority {cls}" if cls not in labels_added else ""
                ax.barh(v, duration, left=start, height=0.5, color=color, edgecolor='black', linewidth=0.5, alpha=0.9, label=label)
                if label: labels_added.add(cls)
                
        ax.axvline(x=t1_max, color='red', linestyle='--', linewidth=2, alpha=1.0, label='$T_1$ Bottleneck')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label: ax.legend(by_label.values(), by_label.keys(), loc='lower right')
        
    plot_gantt(ax1, best_rn_events, "Risk-Neutral (REINFORCE)", best_rn_tmax)
    plot_gantt(ax2, best_cvar_events, "Risk-Averse (CVaR=0.1)", best_cvar_tmax)
    
    ax2.set_xlabel("Elapsed Route Time (Simulation Hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_histogram(rn_tmaxs, cvar_tmaxs):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(rn_tmaxs, fill=True, color='#e74c3c', label=f'Risk-Neutral (Std: {np.std(rn_tmaxs):.2f})', alpha=0.5)
    sns.kdeplot(cvar_tmaxs, fill=True, color='#2ecc71', label=f'Risk-Averse (Std: {np.std(cvar_tmaxs):.2f})', alpha=0.5)
    
    # Mark Means
    plt.axvline(np.mean(rn_tmaxs), color='#c0392b', linestyle='--', label=f'RN Mean: {np.mean(rn_tmaxs):.2f}')
    plt.axvline(np.mean(cvar_tmaxs), color='#27ae60', linestyle='--', label=f'CVaR Mean: {np.mean(cvar_tmaxs):.2f}')
    
    plt.title("Distribution of $T_1$ Makespan Across 60 Shifted Instances", fontsize=14, fontweight='bold')
    plt.xlabel("Makespan $T_1$ (Lower is Better)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11)
    
    out_name = "results/makespan_histogram_stats.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Stats Histogram saved to {out_name}")

def main():
    files = sorted(glob("data/eval_dataset/**/*.npz", recursive=True)) or sorted(glob("data/eval_dataset/*.npz", recursive=True))
    
    os.makedirs("results/all_gantts", exist_ok=True)
    
    cfg = ShiftConfig(max_demand_shift=0.3, max_cost_shift=0.3, min_availability=1.0, mode="adversarial")
    scheduler = ShiftScheduler(cfg, seed=42)
    env = HCARPEnv(shift_scheduler=scheduler)
    
    cvar_policy = load_policy("experiments/results/cvar_v2/best.pt", d_shift=8)
    rn_policy   = load_policy("experiments/results/rn_v2/best.pt", d_shift=8)
    
    print("Generating 60 Gantt charts and collecting stats...")
    rn_tmaxs, cvar_tmaxs = [], []
    
    for f in tqdm(files):
        env.load_files([f])
        rn_events, rn_Tmax = get_gantt_events(rn_policy, env)
        cvar_events, cvar_Tmax = get_gantt_events(cvar_policy, env)
        
        rn_tmaxs.append(rn_Tmax)
        cvar_tmaxs.append(cvar_Tmax)
        
        inst_name = os.path.basename(f)
        out_path = f"results/all_gantts/gantt_{inst_name}.png"
        plot_single_gantt(env, rn_events, rn_Tmax, cvar_events, cvar_Tmax, inst_name, out_path)

    plot_histogram(rn_tmaxs, cvar_tmaxs)
    print("Done! Check results/all_gantts/ and results/makespan_histogram_stats.png")

if __name__ == "__main__":
    main()
