#!/usr/bin/env python3
"""
Phase 3 Final: Simplified Balanced Dynamic Optimization
========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from functools import partial
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 75)
print("PHASE 3 FINAL: BALANCED DYNAMIC OPTIMIZATION")
print("=" * 75)

# 加载数据
print("\n[1] Loading data...")
estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')
print(f"    Loaded {len(estimates)} rows, {estimates['season'].nunique()} seasons")

# =============================================================================
# 评估函数
# =============================================================================

def evaluate_rule_all_seasons(estimates, rule_func):
    """评估规则在所有赛季的表现"""
    results_early_j, results_early_f = [], []
    results_late_j, results_late_f = [], []
    
    for season in estimates['season'].unique():
        season_data = estimates[estimates['season'] == season]
        weeks = sorted(season_data['week'].unique())
        n_weeks = len(weeks)
        
        if n_weeks < 3:
            continue
        
        # 划分阶段
        early_weeks = set(weeks[:n_weeks//3])
        late_weeks = set(weeks[2*n_weeks//3:])
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week].copy()
            if len(week_data) < 2:
                continue
            
            # 计算得分
            scores = rule_func(week_data, week, n_weeks)
            week_data = week_data.assign(score=scores)
            week_data = week_data.assign(
                score_rank=week_data['score'].rank(ascending=False),
                J_rank=week_data['J_pct'].rank(ascending=False),
                F_rank=week_data['f_mean'].rank(ascending=False)
            )
            
            j_corr, _ = stats.spearmanr(week_data['score_rank'], week_data['J_rank'])
            f_corr, _ = stats.spearmanr(week_data['score_rank'], week_data['F_rank'])
            
            if week in early_weeks:
                results_early_j.append(j_corr)
                results_early_f.append(f_corr)
            elif week in late_weeks:
                results_late_j.append(j_corr)
                results_late_f.append(f_corr)
    
    if not results_early_j or not results_late_j:
        return None
    
    metrics = {
        'J_early': np.nanmean(results_early_j),
        'F_early': np.nanmean(results_early_f),
        'J_late': np.nanmean(results_late_j),
        'F_late': np.nanmean(results_late_f),
    }
    
    # 整体指标
    metrics['J_overall'] = (metrics['J_early'] + metrics['J_late']) / 2
    metrics['F_overall'] = (metrics['F_early'] + metrics['F_late']) / 2
    
    # 动态模式 = (早期F - 后期F) + (后期J - 早期J)
    metrics['dynamic_pattern'] = (metrics['F_early'] - metrics['F_late']) + \
                                  (metrics['J_late'] - metrics['J_early'])
    
    # 传统Balance
    j_o = max(metrics['J_overall'], 0.01)
    f_o = max(metrics['F_overall'], 0.01)
    metrics['Balance_traditional'] = 2 * j_o * f_o / (j_o + f_o)
    
    # 阶段加权Balance（早期偏F，后期偏J）
    j_e = max(metrics['J_early'], 0.01)
    f_e = max(metrics['F_early'], 0.01)
    j_l = max(metrics['J_late'], 0.01)
    f_l = max(metrics['F_late'], 0.01)
    
    early_weighted = 0.4 * j_e + 0.6 * f_e
    late_weighted = 0.6 * j_l + 0.4 * f_l
    metrics['Balance_phased'] = (early_weighted + late_weighted) / 2
    
    # 综合得分
    metrics['Score_balanced'] = (
        0.35 * metrics['Balance_traditional'] +
        0.30 * metrics['Balance_phased'] +
        0.25 * max(0, metrics['dynamic_pattern'] * 0.3) +
        0.10
    )
    
    return metrics


# =============================================================================
# 规则定义
# =============================================================================

def rule_static_rank(week_data, current_week, total_weeks, judge_weight):
    n = len(week_data)
    J_rank = week_data['J_pct'].rank(ascending=False)
    F_rank = week_data['f_mean'].rank(ascending=False)
    J_score = (n - J_rank + 1) / n * 100
    F_score = (n - F_rank + 1) / n * 100
    return judge_weight * J_score + (1 - judge_weight) * F_score


def rule_dynamic_sigmoid(week_data, current_week, total_weeks, j_min, j_max, steepness):
    t = current_week - 1
    T = max(total_weeks - 1, 1)
    
    x = steepness * (t / T - 0.5)
    sigmoid = 1 / (1 + np.exp(-x))
    w_j = j_min + (j_max - j_min) * sigmoid
    w_f = 1 - w_j
    
    n = len(week_data)
    J_rank = week_data['J_pct'].rank(ascending=False)
    F_rank = week_data['f_mean'].rank(ascending=False)
    J_score = (n - J_rank + 1) / n * 100
    F_score = (n - F_rank + 1) / n * 100
    
    return w_j * J_score + w_f * F_score


def rule_dynamic_linear(week_data, current_week, total_weeks, j_start, j_end):
    t = current_week - 1
    T = max(total_weeks - 1, 1)
    
    progress = t / T
    w_j = j_start + (j_end - j_start) * progress
    w_f = 1 - w_j
    
    n = len(week_data)
    J_rank = week_data['J_pct'].rank(ascending=False)
    F_rank = week_data['f_mean'].rank(ascending=False)
    J_score = (n - J_rank + 1) / n * 100
    F_score = (n - F_rank + 1) / n * 100
    
    return w_j * J_score + w_f * F_score


# =============================================================================
# 参数搜索
# =============================================================================
print("\n[2] Searching for optimal parameters...")

all_configs = []

# 静态规则
print("    Evaluating static rules...")
for jw in np.arange(0.35, 0.66, 0.05):
    rule_func = partial(rule_static_rank, judge_weight=jw)
    metrics = evaluate_rule_all_seasons(estimates, rule_func)
    if metrics:
        metrics['rule_name'] = f'Static_Rank({jw:.2f})'
        metrics['rule_type'] = 'static'
        metrics['j_weight'] = jw
        all_configs.append(metrics)
        print(f"      {metrics['rule_name']}: Balance={metrics['Balance_traditional']:.4f}")

# Sigmoid动态规则
print("    Evaluating Sigmoid dynamic rules...")
for j_min in np.arange(0.30, 0.46, 0.05):
    for j_max in np.arange(0.55, 0.76, 0.05):
        if j_max <= j_min + 0.1:
            continue
        for steepness in [3, 4, 5, 6]:
            rule_func = partial(rule_dynamic_sigmoid, j_min=j_min, j_max=j_max, steepness=steepness)
            metrics = evaluate_rule_all_seasons(estimates, rule_func)
            if metrics:
                metrics['rule_name'] = f'Sigmoid({j_min:.2f},{j_max:.2f},{steepness})'
                metrics['rule_type'] = 'dynamic_sigmoid'
                metrics['j_min'] = j_min
                metrics['j_max'] = j_max
                metrics['steepness'] = steepness
                all_configs.append(metrics)

print(f"    Sigmoid rules evaluated: {sum(1 for c in all_configs if c['rule_type'] == 'dynamic_sigmoid')}")

# Linear动态规则
print("    Evaluating Linear dynamic rules...")
for j_start in np.arange(0.30, 0.46, 0.05):
    for j_end in np.arange(0.55, 0.76, 0.05):
        if j_end <= j_start + 0.1:
            continue
        rule_func = partial(rule_dynamic_linear, j_start=j_start, j_end=j_end)
        metrics = evaluate_rule_all_seasons(estimates, rule_func)
        if metrics:
            metrics['rule_name'] = f'Linear({j_start:.2f},{j_end:.2f})'
            metrics['rule_type'] = 'dynamic_linear'
            metrics['j_start'] = j_start
            metrics['j_end'] = j_end
            all_configs.append(metrics)

print(f"    Linear rules evaluated: {sum(1 for c in all_configs if c['rule_type'] == 'dynamic_linear')}")

config_df = pd.DataFrame(all_configs)
print(f"\n    Total configurations: {len(config_df)}")

# =============================================================================
# 结果分析
# =============================================================================
print("\n[3] Results analysis...")

static_df = config_df[config_df['rule_type'] == 'static']
dynamic_df = config_df[config_df['rule_type'] != 'static']

print(f"    Static rules: {len(static_df)}")
print(f"    Dynamic rules: {len(dynamic_df)}")

# 最佳规则
best_static = static_df.loc[static_df['Score_balanced'].idxmax()]
best_dynamic = dynamic_df.loc[dynamic_df['Score_balanced'].idxmax()]

print(f"\n    ╔══════════════════════════════════════════════════════════════════════════════╗")
print(f"    ║                    BALANCED COMPARISON                                        ║")
print(f"    ╠══════════════════════════════════════════════════════════════════════════════╣")
print(f"    ║  Metric               Best Static        Best Dynamic      Winner            ║")
print(f"    ╠══════════════════════════════════════════════════════════════════════════════╣")

metrics_to_compare = [
    ('J_early', '早期精英选拔'),
    ('F_early', '早期粉丝参与'),
    ('J_late', '后期精英选拔'),
    ('F_late', '后期粉丝参与'),
    ('Balance_traditional', '传统Balance'),
    ('Balance_phased', '阶段Balance'),
    ('dynamic_pattern', '动态模式'),
    ('Score_balanced', '综合得分'),
]

dynamic_wins = 0
for metric, desc in metrics_to_compare:
    static_val = best_static[metric]
    dynamic_val = best_dynamic[metric]
    winner = '★ Dynamic' if dynamic_val > static_val else 'Static'
    if dynamic_val > static_val:
        dynamic_wins += 1
    print(f"    ║  {desc:<16} {static_val:>12.4f}       {dynamic_val:>12.4f}      {winner:<12} ║")

print(f"    ╠══════════════════════════════════════════════════════════════════════════════╣")
print(f"    ║  Dynamic Wins: {dynamic_wins}/8 dimensions                                              ║")
print(f"    ╚══════════════════════════════════════════════════════════════════════════════╝")

print(f"\n    Best Static Rule: {best_static['rule_name']}")
print(f"    Best Dynamic Rule: {best_dynamic['rule_name']}")

# Top 10 动态规则
print(f"\n    Top 10 Dynamic Rules:")
print(f"    {'='*90}")
top_dynamic = dynamic_df.nlargest(10, 'Score_balanced')
for _, row in top_dynamic.iterrows():
    print(f"    {row['rule_name']:<35} Score={row['Score_balanced']:.4f} "
          f"DynPat={row['dynamic_pattern']:.4f}")

# =============================================================================
# 可视化
# =============================================================================
print("\n[4] Generating visualizations...")

output_dir = 'cleaned_outputs/phase3_pareto_analysis'
os.makedirs(output_dir, exist_ok=True)

# 图1: 权重演化
fig, ax = plt.subplots(figsize=(12, 6))

weeks = np.arange(1, 12)
T = 10

# 静态规则
ax.axhline(y=best_static['j_weight'], color='#ef4444', linewidth=3, linestyle='--', 
           label=f'Static ({best_static["j_weight"]:.0%})')

# 动态规则
if 'j_min' in best_dynamic:
    j_min = best_dynamic['j_min']
    j_max = best_dynamic['j_max']
    
    if 'steepness' in best_dynamic:  # Sigmoid
        steepness = best_dynamic['steepness']
        weights = [j_min + (j_max - j_min) / (1 + np.exp(-steepness * (t/T - 0.5))) 
                   for t in range(11)]
    else:  # Linear
        weights = [j_min + (j_max - j_min) * t/T for t in range(11)]
    
    ax.plot(weeks, weights[:11], 'b-', linewidth=3, marker='o', markersize=8,
            label=f'Dynamic ({j_min:.0%} → {j_max:.0%})')

ax.set_xlabel('Week', fontsize=13, fontweight='bold')
ax.set_ylabel('Judge Weight $w_J$', fontsize=13, fontweight='bold')
ax.set_title('Weight Evolution: Optimal Dynamic vs Best Static', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.set_xlim(1, 11)
ax.set_ylim(0.2, 0.8)
ax.grid(True, alpha=0.3)

# 阶段标注
ax.axvspan(1, 4, alpha=0.1, color='green')
ax.axvspan(4, 7, alpha=0.1, color='yellow')
ax.axvspan(7, 11, alpha=0.1, color='blue')
ax.text(2.5, 0.75, 'Early\n(High F)', ha='center', fontsize=9)
ax.text(5.5, 0.75, 'Mid\n(Transition)', ha='center', fontsize=9)
ax.text(9, 0.75, 'Late\n(High J)', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/weight_evolution_final.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {output_dir}/weight_evolution_final.png")

# 图2: Pareto前沿
fig, ax = plt.subplots(figsize=(10, 8))

# 学术化图例映射
legend_labels = {
    'static': r'Static ($\alpha_J = const.$)',
    'dynamic_sigmoid': r'Sigmoid Dynamic ($\alpha_J(t) = \sigma(t)$)',
    'dynamic_linear': r'Linear Dynamic ($\alpha_J(t) \propto t$)',
}

for rule_type, color, marker in [
    ('static', '#ef4444', 'o'),
    ('dynamic_sigmoid', '#3b82f6', 's'),
    ('dynamic_linear', '#10b981', '^'),
]:
    subset = config_df[config_df['rule_type'] == rule_type]
    if len(subset) > 0:
        ax.scatter(subset['Balance_traditional'], subset['dynamic_pattern'],
                  c=color, marker=marker, s=80, alpha=0.6, 
                  label=legend_labels.get(rule_type, rule_type))

ax.scatter([best_static['Balance_traditional']], [best_static['dynamic_pattern']],
          c='red', marker='*', s=300, edgecolors='black', linewidth=2, 
          label=r'Optimal Static ($\mathcal{B}^*_{static}$)', zorder=10)
ax.scatter([best_dynamic['Balance_traditional']], [best_dynamic['dynamic_pattern']],
          c='blue', marker='*', s=300, edgecolors='black', linewidth=2, 
          label=r'Optimal Dynamic ($\mathcal{B}^*_{dynamic}$)', zorder=10)

ax.set_xlabel(r'Traditional Balance $\mathcal{B}$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Dynamic Pattern Score $\mathcal{P}$', fontsize=12, fontweight='bold')
ax.set_title(r'Pareto Frontier: $\mathcal{B}$ vs Dynamic Advantage $\mathcal{P}$', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/pareto_frontier_final.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {output_dir}/pareto_frontier_final.png")

# =============================================================================
# 保存结果
# =============================================================================
print("\n[5] Saving results...")

config_df.to_csv(f'{output_dir}/balanced_optimization_final.csv', index=False)

# 最优规则
optimal_rule = {
    'best_dynamic': {
        'rule_name': best_dynamic['rule_name'],
        'rule_type': best_dynamic['rule_type'],
        'parameters': {k: float(best_dynamic[k]) for k in 
                       ['j_min', 'j_max', 'steepness', 'j_start', 'j_end']
                       if k in best_dynamic and pd.notna(best_dynamic.get(k))},
        'metrics': {k: float(best_dynamic[k]) for k, _ in metrics_to_compare}
    },
    'best_static': {
        'rule_name': best_static['rule_name'],
        'parameters': {'j_weight': float(best_static['j_weight'])},
        'metrics': {k: float(best_static[k]) for k in ['Balance_traditional', 'Score_balanced']}
    },
    'comparison': {
        'dynamic_wins': dynamic_wins,
        'total_dimensions': 8
    }
}

with open(f'{output_dir}/optimal_rule_final.json', 'w') as f:
    json.dump(optimal_rule, f, indent=2)

print(f"    Saved: {output_dir}/balanced_optimization_final.csv")
print(f"    Saved: {output_dir}/optimal_rule_final.json")

# =============================================================================
# 总结
# =============================================================================
print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)

print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                         最终结论                                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  最佳静态规则: {best_static['rule_name']:<50} ║
║    - 传统Balance: {best_static['Balance_traditional']:.4f}                                           ║
║    - 综合得分: {best_static['Score_balanced']:.4f}                                               ║
║                                                                           ║
║  最佳动态规则: {best_dynamic['rule_name']:<50} ║
║    - 传统Balance: {best_dynamic['Balance_traditional']:.4f}                                           ║
║    - 动态模式得分: {best_dynamic['dynamic_pattern']:.4f}                                           ║
║    - 综合得分: {best_dynamic['Score_balanced']:.4f}                                               ║
║                                                                           ║
║  动态规则胜出维度: {dynamic_wins}/8                                                   ║
║    - 早期粉丝参与: {best_dynamic['F_early']:.4f} vs {best_static['F_early']:.4f}                              ║
║    - 后期精英选拔: {best_dynamic['J_late']:.4f} vs {best_static['J_late']:.4f}                               ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

print("=" * 75)
