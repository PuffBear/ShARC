import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from env.hcarp_env import HCARPEnv
from env.shift import ShiftConfig, ShiftScheduler
from models.policy import HCARPPolicy

def load_policy(ckpt_path, d_shift, device="cpu"):
    policy = HCARPPolicy(d_shift=d_shift, device=device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()
    return policy

def get_gantt_events(policy, env):
    """Manually step through the environment and record start/end times for each vehicle."""
    env.reset()
    obs = env._get_obs()
    arc_emb = policy.encode(obs)
    
    events = {v: [] for v in range(env.M)}
    
    with torch.no_grad():
        for _ in range(env.max_steps() + env.M):
            v = int(obs["active_vehicle"][0])
            start_time = float(obs["vehicle_time"][0, v])
            
            actions, log_p = policy.act(obs, arc_emb, greedy=True)
            a = int(actions[0].cpu().numpy())
            
            # Fetch priority class before taking the step
            arc_class = int(obs["clss"][0, a]) if a != 0 else 0
            
            obs, _, done_np, info = env.step(np.array([a]))
            
            if a != 0:
                end_time = float(obs["vehicle_time"][0, v])
                events[v].append((start_time, end_time - start_time, a, arc_class))
                
            if done_np.all():
                break
                
    return events, info[0]['T1']

def main():
    files = sorted(glob("data/eval_dataset/**/*.npz", recursive=True))
    if not files:
        files = sorted(glob("data/eval_dataset/*.npz", recursive=True))
    
    # Setup policies and environment
    cfg = ShiftConfig(max_demand_shift=0.3, max_cost_shift=0.3, min_availability=1.0, mode="adversarial")
    scheduler = ShiftScheduler(cfg, seed=42)
    env = HCARPEnv(shift_scheduler=scheduler)
    
    cvar_policy = load_policy("experiments/results/cvar_v2/best.pt", d_shift=8)
    rn_policy   = load_policy("experiments/results/rn_v2/best.pt", d_shift=8)
    
    # Evaluate all files and store results
    print("Evaluating all 60 instances to find the best showcase examples...")
    results = []
    
    for f in files:
        env.load_files([f])
        rn_events, rn_Tmax = get_gantt_events(rn_policy, env)
        cvar_events, cvar_Tmax = get_gantt_events(cvar_policy, env)
        advantage = rn_Tmax - cvar_Tmax
        results.append({
            'file': f,
            'advantage': advantage,
            'rn_events': rn_events,
            'rn_Tmax': rn_Tmax,
            'cvar_events': cvar_events,
            'cvar_Tmax': cvar_Tmax
        })
        
    # Sort by how much CVaR beat RN
    results.sort(key=lambda x: x['advantage'], reverse=True)
    
    # Plot the top 3 instances
    top_k = 3
    print(f"Generating Gantt charts for the top {top_k} instances...")
    
    for i in range(top_k):
        res = results[i]
        best_file = res['file']
        best_rn_events, best_rn_tmax = res['rn_events'], res['rn_Tmax']
        best_cvar_events, best_cvar_tmax = res['cvar_events'], res['cvar_Tmax']
        
        print(f"[{i+1}] {best_file.split('/')[-1]} | RN T1: {best_rn_tmax:.2f} | CVaR T1: {best_cvar_tmax:.2f}")
        
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"Hierarchical Routing Timeline Under Severe Shift (φ=1.0) | {best_file.split('/')[-1]}", fontsize=15, fontweight='bold', y=0.98)
        
        class_colors = {
            1: '#e74c3c', # Red
            2: '#f39c12', # Orange
            3: '#3498db'  # Blue
        }
        
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
            ax.legend(by_label.values(), by_label.keys(), loc='lower right')
            
        plot_gantt(ax1, best_rn_events, "Risk-Neutral (REINFORCE)", best_rn_tmax)
        plot_gantt(ax2, best_cvar_events, "Risk-Averse (CVaR=0.1)", best_cvar_tmax)
        
        ax2.set_xlabel("Elapsed Route Time (Simulation Hours)", fontsize=12)
        
        plt.tight_layout()
        out_name = f"results/gantt_comparison_top{i+1}.png"
        plt.savefig(out_name, dpi=300, bbox_inches='tight')
        plt.close(fig) # close figure to free memory
        print(f"Saved to {out_name}")

if __name__ == "__main__":
    main()
