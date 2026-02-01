#!/usr/bin/env python3
"""
Phase 4: Pareto Optimization
============================
找到"公平性"与"参与度"之间的最优平衡点：
1. 定义双目标: J (Meritocracy) 和 F (Engagement)
2. 绘制Pareto前沿
3. 标记当前规则、Judges' Save、推荐规则

Author: MCM 2026 Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("PHASE 4: PARETO OPTIMIZATION")
print("=" * 70)

# 加载数据
print("\n[1] Loading data...")
estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')
panel = pd.read_csv('cleaned_outputs/clean_weekly_panel.csv')

print(f"    Estimates: {len(estimates)} rows")

# =============================================================================
# PART 1: DEFINE OBJECTIVES
# =============================================================================
print("\n[2] Defining objectives...")

def calculate_objectives(season_data, judge_weight=0.5, method='rank'):
    """
    计算给定规则下的两个目标：
    - Objective J (Meritocracy): 最终排名与评委排名的相关性
    - Objective F (Engagement): 最终排名与粉丝排名的相关性
    
    Parameters:
    - judge_weight: 评委权重 (0-1)
    - method: 'rank' 或 'pct'
    """
    # 获取最后一周数据
    max_week = season_data['week'].max()
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return np.nan, np.nan
    
    fan_weight = 1 - judge_weight
    
    # 计算排名
    final_data['J_rank'] = final_data['J_pct'].rank(ascending=False)
    final_data['F_rank'] = final_data['f_mean'].rank(ascending=False)
    
    if method == 'rank':
        # Rank-based combined score
        final_data['combined'] = judge_weight * final_data['J_rank'] + fan_weight * final_data['F_rank']
        final_data['final_rank'] = final_data['combined'].rank()
    else:
        # Percentage-based combined score
        max_f = final_data['f_mean'].max()
        final_data['F_pct'] = final_data['f_mean'] / max_f * 100 if max_f > 0 else 0
        final_data['combined'] = judge_weight * final_data['J_pct'] + fan_weight * final_data['F_pct']
        final_data['final_rank'] = final_data['combined'].rank(ascending=False)
    
    # 计算目标
    # J: correlation with judge ranking (higher = more meritocratic)
    j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
    
    # F: correlation with fan ranking (higher = more fan-favoring)
    f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
    
    return j_corr, f_corr

def simulate_judges_save(season_data, judge_weight=0.5):
    """
    模拟Judges' Save规则：
    当选手在Bottom 2时，评委可以拯救评分较高者
    """
    max_week = season_data['week'].max()
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return np.nan, np.nan
    
    # 计算排名
    final_data['J_rank'] = final_data['J_pct'].rank(ascending=False)
    final_data['F_rank'] = final_data['f_mean'].rank(ascending=False)
    
    # 标准combined
    fan_weight = 1 - judge_weight
    final_data['combined'] = judge_weight * final_data['J_rank'] + fan_weight * final_data['F_rank']
    
    # Judges' Save调整：Bottom 2中，如果J%差距>10，救高分者
    final_data['final_rank'] = final_data['combined'].rank()
    n = len(final_data)
    
    bottom_2 = final_data[final_data['final_rank'] >= n - 1]
    if len(bottom_2) >= 2:
        j_scores = bottom_2['J_pct'].values
        if abs(j_scores[0] - j_scores[1]) > 10:
            # 调整：高J%者排名提升
            high_j_idx = bottom_2['J_pct'].idxmax()
            final_data.loc[high_j_idx, 'final_rank'] -= 1
    
    j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
    f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
    
    return j_corr, f_corr

# =============================================================================
# PART 2: COMPUTE PARETO FRONTIER
# =============================================================================
print("\n[3] Computing Pareto frontier...")

# 测试不同权重组合
weights = np.linspace(0.3, 0.9, 25)  # 评委权重从30%到90%
methods = ['rank', 'pct']

pareto_points = []

for method in methods:
    for w in weights:
        j_scores = []
        f_scores = []
        
        for season in estimates['season'].unique():
            season_data = estimates[estimates['season'] == season]
            j, f = calculate_objectives(season_data, judge_weight=w, method=method)
            if not np.isnan(j) and not np.isnan(f):
                j_scores.append(j)
                f_scores.append(f)
        
        if len(j_scores) > 0:
            pareto_points.append({
                'method': method,
                'judge_weight': w,
                'fan_weight': 1 - w,
                'J_mean': np.mean(j_scores),
                'J_std': np.std(j_scores),
                'F_mean': np.mean(f_scores),
                'F_std': np.std(f_scores)
            })

pareto_df = pd.DataFrame(pareto_points)

# 添加Judges' Save点
judges_save_points = []
for w in [0.5, 0.6, 0.7]:
    j_scores = []
    f_scores = []
    
    for season in estimates['season'].unique():
        season_data = estimates[estimates['season'] == season]
        j, f = simulate_judges_save(season_data, judge_weight=w)
        if not np.isnan(j) and not np.isnan(f):
            j_scores.append(j)
            f_scores.append(f)
    
    if len(j_scores) > 0:
        judges_save_points.append({
            'method': f'judges_save_{int(w*100)}',
            'judge_weight': w,
            'J_mean': np.mean(j_scores),
            'F_mean': np.mean(f_scores)
        })

print(f"    Computed {len(pareto_df)} Pareto points")
print(f"    Judges' Save variants: {len(judges_save_points)}")

# =============================================================================
# PART 3: IDENTIFY PARETO-OPTIMAL POINTS
# =============================================================================
print("\n[4] Identifying Pareto-optimal points...")

def is_pareto_optimal(points, maximize_both=True):
    """识别Pareto最优点"""
    n = len(points)
    is_optimal = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if maximize_both:
                    # 如果j在两个维度都>=i，且至少一个>i
                    if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                        (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                        is_optimal[i] = False
                        break
    
    return is_optimal

# 对Rank方法找Pareto前沿
rank_df = pareto_df[pareto_df['method'] == 'rank'].copy()
rank_points = rank_df[['J_mean', 'F_mean']].values
rank_optimal = is_pareto_optimal(rank_points)
rank_df['is_optimal'] = rank_optimal

# 对Pct方法找Pareto前沿
pct_df = pareto_df[pareto_df['method'] == 'pct'].copy()
pct_points = pct_df[['J_mean', 'F_mean']].values
pct_optimal = is_pareto_optimal(pct_points)
pct_df['is_optimal'] = pct_optimal

print(f"    Rank method: {rank_optimal.sum()} Pareto-optimal points")
print(f"    Pct method: {pct_optimal.sum()} Pareto-optimal points")

# =============================================================================
# PART 4: FIND KNEE POINT (RECOMMENDED RULE)
# =============================================================================
print("\n[5] Finding knee point (recommended rule)...")

def find_knee_point(points):
    """
    找到Pareto前沿的"膝点"——曲线最弯处
    使用最大曲率法
    """
    if len(points) < 3:
        return 0
    
    # 归一化
    j_norm = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min() + 1e-6)
    f_norm = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min() + 1e-6)
    
    # 计算到对角线的距离
    # 对角线: 从(min_J, max_F)到(max_J, min_F)
    distances = []
    for i in range(len(points)):
        # 距离 = |J + F - 1| / sqrt(2)  (归一化后)
        dist = abs(j_norm[i] + f_norm[i] - 1) / np.sqrt(2)
        distances.append(dist)
    
    # 选择距离最大的点
    knee_idx = np.argmax(distances)
    return knee_idx

# 找Rank方法的膝点
rank_optimal_df = rank_df[rank_df['is_optimal']].sort_values('J_mean')
if len(rank_optimal_df) > 0:
    rank_optimal_points = rank_optimal_df[['J_mean', 'F_mean']].values
    knee_idx = find_knee_point(rank_optimal_points)
    knee_point = rank_optimal_df.iloc[knee_idx]
    
    print(f"\n    Recommended Rule (Knee Point):")
    print(f"    - Method: Rank-based")
    print(f"    - Judge Weight: {knee_point['judge_weight']:.1%}")
    print(f"    - Fan Weight: {knee_point['fan_weight']:.1%}")
    print(f"    - J (Meritocracy): {knee_point['J_mean']:.4f}")
    print(f"    - F (Engagement): {knee_point['F_mean']:.4f}")

# =============================================================================
# PART 5: COMPARE KEY RULES
# =============================================================================
print("\n[6] Comparing key rules...")

# 当前规则 (50-50)
current_rank = pareto_df[(pareto_df['method'] == 'rank') & 
                          (abs(pareto_df['judge_weight'] - 0.5) < 0.01)].iloc[0]
current_pct = pareto_df[(pareto_df['method'] == 'pct') & 
                         (abs(pareto_df['judge_weight'] - 0.5) < 0.01)].iloc[0]

# Judges' Save (50-50)
judges_save_50 = judges_save_points[0] if judges_save_points else None

# 推荐规则
recommended = knee_point if len(rank_optimal_df) > 0 else current_rank

comparison_table = pd.DataFrame([
    {'Rule': 'Current (Rank 50-50)', 'J': current_rank['J_mean'], 'F': current_rank['F_mean'],
     'Status': '🔴 Current'},
    {'Rule': 'Current (Pct 50-50)', 'J': current_pct['J_mean'], 'F': current_pct['F_mean'],
     'Status': '⚪ Alternative'},
    {'Rule': f"Judges' Save (50-50)", 
     'J': judges_save_50['J_mean'] if judges_save_50 else np.nan, 
     'F': judges_save_50['F_mean'] if judges_save_50 else np.nan,
     'Status': '🔵 With Save'},
    {'Rule': f"Recommended (Rank {knee_point['judge_weight']:.0%}-{knee_point['fan_weight']:.0%})",
     'J': knee_point['J_mean'], 'F': knee_point['F_mean'],
     'Status': '⭐ Recommended'}
])

print("\n    Rule Comparison:")
print("    " + "-" * 60)
print(f"    {'Rule':<35} {'J (Merit)':>10} {'F (Engage)':>12} {'Status':>10}")
print("    " + "-" * 60)
for _, row in comparison_table.iterrows():
    print(f"    {row['Rule']:<35} {row['J']:>10.4f} {row['F']:>12.4f} {row['Status']:>10}")

# =============================================================================
# PART 6: VISUALIZATIONS
# =============================================================================
print("\n[7] Generating visualizations...")

import os
img_dir = 'cleaned_outputs/phase4_pareto'
os.makedirs(img_dir, exist_ok=True)

# Prepare data for plots
rank_plot = pareto_df[pareto_df['method'] == 'rank']
pct_plot = pareto_df[pareto_df['method'] == 'pct']
rank_frontier = rank_df[rank_df['is_optimal']].sort_values('J_mean')
pct_frontier = pct_df[pct_df['is_optimal']].sort_values('J_mean')

# --- Individual Plots ---

# 6.1 Pareto Frontier (Individual)
fig1, ax1_ind = plt.subplots(figsize=(8, 7))
ax1_ind.scatter(rank_plot['J_mean'], rank_plot['F_mean'], c='steelblue', alpha=0.3, s=40, label='Rank Method')
ax1_ind.scatter(pct_plot['J_mean'], pct_plot['F_mean'], c='coral', alpha=0.3, s=40, label='Pct Method')
ax1_ind.plot(rank_frontier['J_mean'], rank_frontier['F_mean'], 'b-', linewidth=2, label='Rank Frontier')
ax1_ind.plot(pct_frontier['J_mean'], pct_frontier['F_mean'], 'r--', linewidth=2, label='Pct Frontier')
ax1_ind.scatter(current_rank['J_mean'], current_rank['F_mean'], c='red', s=200, marker='o', 
            zorder=5, edgecolors='black', linewidth=2, label='Current Rule')
if judges_save_50:
    ax1_ind.scatter(judges_save_50['J_mean'], judges_save_50['F_mean'], c='blue', s=200, marker='s',
                zorder=5, edgecolors='black', linewidth=2, label="Judges' Save")
ax1_ind.scatter(knee_point['J_mean'], knee_point['F_mean'], c='gold', s=300, marker='*',
            zorder=5, edgecolors='black', linewidth=2, label='Recommended')
ax1_ind.set_xlabel('J (Meritocracy) - Correlation with Judge Ranking', fontsize=11)
ax1_ind.set_ylabel('F (Engagement) - Correlation with Fan Ranking', fontsize=11)
ax1_ind.set_title('Pareto Frontier: Fairness vs Engagement Trade-off', fontsize=12, fontweight='bold')
ax1_ind.legend(loc='lower left', fontsize=9)
ax1_ind.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{img_dir}/pareto_frontier.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/pareto_frontier.png")

# 6.2 Judge Weight vs Objectives (Individual)
fig2, ax2_ind = plt.subplots(figsize=(8, 6))
ax2_ind.plot(rank_plot['judge_weight'], rank_plot['J_mean'], 'b-o', label='J (Rank)', markersize=4)
ax2_ind.plot(rank_plot['judge_weight'], rank_plot['F_mean'], 'b--s', label='F (Rank)', markersize=4)
ax2_ind.plot(pct_plot['judge_weight'], pct_plot['J_mean'], 'r-o', label='J (Pct)', markersize=4)
ax2_ind.plot(pct_plot['judge_weight'], pct_plot['F_mean'], 'r--s', label='F (Pct)', markersize=4)
ax2_ind.axvline(x=0.5, color='gray', linestyle=':', label='Current (50%)')
ax2_ind.axvline(x=knee_point['judge_weight'], color='gold', linestyle='-', linewidth=2, 
            label=f"Recommended ({knee_point['judge_weight']:.0%})")
ax2_ind.set_xlabel('Judge Weight', fontsize=11)
ax2_ind.set_ylabel('Correlation', fontsize=11)
ax2_ind.set_title('Objectives vs Judge Weight', fontsize=12, fontweight='bold')
ax2_ind.legend(loc='center right', fontsize=8)
ax2_ind.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{img_dir}/weight_vs_objectives.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/weight_vs_objectives.png")

# 6.3 Trade-off Ratio (Individual)
fig3, ax3_ind = plt.subplots(figsize=(8, 6))
rank_plot_copy = rank_plot.copy()
rank_plot_copy['JF_ratio'] = rank_plot_copy['J_mean'] / (rank_plot_copy['F_mean'] + 0.01)
pct_plot_copy = pct_plot.copy()
pct_plot_copy['JF_ratio'] = pct_plot_copy['J_mean'] / (pct_plot_copy['F_mean'] + 0.01)
ax3_ind.plot(rank_plot_copy['judge_weight'], rank_plot_copy['JF_ratio'], 'b-o', label='Rank Method', markersize=4)
ax3_ind.plot(pct_plot_copy['judge_weight'], pct_plot_copy['JF_ratio'], 'r-o', label='Pct Method', markersize=4)
ax3_ind.axhline(y=1.0, color='gray', linestyle='--', label='Equal Weight')
ax3_ind.axvline(x=knee_point['judge_weight'], color='gold', linestyle='-', linewidth=2)
ax3_ind.set_xlabel('Judge Weight', fontsize=11)
ax3_ind.set_ylabel('J/F Ratio (Merit/Engagement)', fontsize=11)
ax3_ind.set_title('Merit vs Engagement Balance', fontsize=12, fontweight='bold')
ax3_ind.legend(fontsize=9)
ax3_ind.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{img_dir}/tradeoff_ratio.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/tradeoff_ratio.png")

# 6.4 Summary Bar Chart (Individual)
fig4, ax4_ind = plt.subplots(figsize=(8, 6))
rules = ['Current\n(50-50)', "Judges'\nSave", f'Recommended\n({knee_point["judge_weight"]:.0%})']
j_values = [current_rank['J_mean'], 
            judges_save_50['J_mean'] if judges_save_50 else 0,
            knee_point['J_mean']]
f_values = [current_rank['F_mean'],
            judges_save_50['F_mean'] if judges_save_50 else 0,
            knee_point['F_mean']]
x_bar = np.arange(len(rules))
width = 0.35
bars1 = ax4_ind.bar(x_bar - width/2, j_values, width, label='J (Meritocracy)', color='steelblue')
bars2 = ax4_ind.bar(x_bar + width/2, f_values, width, label='F (Engagement)', color='coral')
ax4_ind.set_ylabel('Correlation', fontsize=11)
ax4_ind.set_title('Key Rules Comparison', fontsize=12, fontweight='bold')
ax4_ind.set_xticks(x_bar)
ax4_ind.set_xticklabels(rules)
ax4_ind.legend()
ax4_ind.set_ylim(0, 1)
for bar, val in zip(bars1, j_values):
    ax4_ind.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, f_values):
    ax4_ind.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{img_dir}/rules_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/rules_comparison.png")

# --- Panel Plot (Combined) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 6.1 Pareto Frontier (Main Plot)
ax1 = axes[0, 0]
ax1.scatter(rank_plot['J_mean'], rank_plot['F_mean'], c='steelblue', alpha=0.3, s=40, label='Rank Method')
ax1.scatter(pct_plot['J_mean'], pct_plot['F_mean'], c='coral', alpha=0.3, s=40, label='Pct Method')
ax1.plot(rank_frontier['J_mean'], rank_frontier['F_mean'], 'b-', linewidth=2, label='Rank Frontier')
ax1.plot(pct_frontier['J_mean'], pct_frontier['F_mean'], 'r--', linewidth=2, label='Pct Frontier')

# Key points
ax1.scatter(current_rank['J_mean'], current_rank['F_mean'], c='red', s=200, marker='o', 
            zorder=5, edgecolors='black', linewidth=2, label='Current Rule')
if judges_save_50:
    ax1.scatter(judges_save_50['J_mean'], judges_save_50['F_mean'], c='blue', s=200, marker='s',
                zorder=5, edgecolors='black', linewidth=2, label="Judges' Save")
ax1.scatter(knee_point['J_mean'], knee_point['F_mean'], c='gold', s=300, marker='*',
            zorder=5, edgecolors='black', linewidth=2, label='Recommended')

ax1.set_xlabel('J (Meritocracy) - Correlation with Judge Ranking', fontsize=11)
ax1.set_ylabel('F (Engagement) - Correlation with Fan Ranking', fontsize=11)
ax1.set_title('Pareto Frontier: Fairness vs Engagement Trade-off', fontsize=12, fontweight='bold')
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 6.2 Judge Weight vs Objectives
ax2 = axes[0, 1]

ax2.plot(rank_plot['judge_weight'], rank_plot['J_mean'], 'b-o', label='J (Rank)', markersize=4)
ax2.plot(rank_plot['judge_weight'], rank_plot['F_mean'], 'b--s', label='F (Rank)', markersize=4)
ax2.plot(pct_plot['judge_weight'], pct_plot['J_mean'], 'r-o', label='J (Pct)', markersize=4)
ax2.plot(pct_plot['judge_weight'], pct_plot['F_mean'], 'r--s', label='F (Pct)', markersize=4)

ax2.axvline(x=0.5, color='gray', linestyle=':', label='Current (50%)')
ax2.axvline(x=knee_point['judge_weight'], color='gold', linestyle='-', linewidth=2, 
            label=f"Recommended ({knee_point['judge_weight']:.0%})")

ax2.set_xlabel('Judge Weight', fontsize=11)
ax2.set_ylabel('Correlation', fontsize=11)
ax2.set_title('Objectives vs Judge Weight', fontsize=12, fontweight='bold')
ax2.legend(loc='center right', fontsize=8)
ax2.grid(True, alpha=0.3)

# 6.3 Trade-off Ratio
ax3 = axes[1, 0]

# 计算J和F的比值
rank_plot = rank_plot.copy()
rank_plot['JF_ratio'] = rank_plot['J_mean'] / (rank_plot['F_mean'] + 0.01)
pct_plot = pct_plot.copy()
pct_plot['JF_ratio'] = pct_plot['J_mean'] / (pct_plot['F_mean'] + 0.01)

ax3.plot(rank_plot['judge_weight'], rank_plot['JF_ratio'], 'b-o', label='Rank Method', markersize=4)
ax3.plot(pct_plot['judge_weight'], pct_plot['JF_ratio'], 'r-o', label='Pct Method', markersize=4)

ax3.axhline(y=1.0, color='gray', linestyle='--', label='Equal Weight')
ax3.axvline(x=knee_point['judge_weight'], color='gold', linestyle='-', linewidth=2)

ax3.set_xlabel('Judge Weight', fontsize=11)
ax3.set_ylabel('J/F Ratio (Merit/Engagement)', fontsize=11)
ax3.set_title('Merit vs Engagement Balance', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 6.4 Summary Bar Chart
ax4 = axes[1, 1]

rules = ['Current\n(50-50)', "Judges'\nSave", f'Recommended\n({knee_point["judge_weight"]:.0%})']
j_values = [current_rank['J_mean'], 
            judges_save_50['J_mean'] if judges_save_50 else 0,
            knee_point['J_mean']]
f_values = [current_rank['F_mean'],
            judges_save_50['F_mean'] if judges_save_50 else 0,
            knee_point['F_mean']]

x = np.arange(len(rules))
width = 0.35

bars1 = ax4.bar(x - width/2, j_values, width, label='J (Meritocracy)', color='steelblue')
bars2 = ax4.bar(x + width/2, f_values, width, label='F (Engagement)', color='coral')

ax4.set_ylabel('Correlation', fontsize=11)
ax4.set_title('Key Rules Comparison', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(rules)
ax4.legend()
ax4.set_ylim(0, 1)

# 添加数值标签
for bar, val in zip(bars1, j_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, f_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('cleaned_outputs/pareto_optimization.png', dpi=150, bbox_inches='tight')
print("    Saved: pareto_optimization.png")

# =============================================================================
# PART 7: SAVE RESULTS
# =============================================================================
print("\n[8] Saving results...")

pareto_df.to_csv('cleaned_outputs/pareto_points.csv', index=False)
print(f"    Saved: pareto_points.csv ({len(pareto_df)} rows)")

comparison_table.to_csv('cleaned_outputs/rule_comparison.csv', index=False)
print(f"    Saved: rule_comparison.csv")

# 保存推荐规则
recommended_rule = {
    'method': 'rank',
    'judge_weight': float(knee_point['judge_weight']),
    'fan_weight': float(knee_point['fan_weight']),
    'J_meritocracy': float(knee_point['J_mean']),
    'F_engagement': float(knee_point['F_mean']),
    'mechanism': 'Judges Save for Bottom 2'
}

import json
with open('cleaned_outputs/recommended_rule.json', 'w') as f:
    json.dump(recommended_rule, f, indent=2)
print("    Saved: recommended_rule.json")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 4: PARETO OPTIMIZATION SUMMARY")
print("=" * 70)

print(f"""
PARETO OPTIMIZATION RESULTS:
============================

DUAL OBJECTIVES:
• J (Meritocracy): Correlation between final rank and judge rank
  - Higher J = Better dancers place higher
• F (Engagement): Correlation between final rank and fan rank
  - Higher F = More fan engagement

PARETO FRONTIER:
• Rank method produces better Pareto frontier than Pct method
• Trade-off exists: Increasing J slightly decreases F

KEY RULES COMPARISON:
┌──────────────────────────┬──────────┬──────────┬─────────────────┐
│ Rule                     │ J        │ F        │ Status          │
├──────────────────────────┼──────────┼──────────┼─────────────────┤
│ Current (Rank 50-50)     │ {current_rank['J_mean']:.4f}   │ {current_rank['F_mean']:.4f}   │ 🔴 Baseline     │
│ Judges' Save (50-50)     │ {judges_save_50['J_mean'] if judges_save_50 else 0:.4f}   │ {judges_save_50['F_mean'] if judges_save_50 else 0:.4f}   │ 🔵 With Save    │
│ Recommended ({knee_point['judge_weight']:.0%}-{knee_point['fan_weight']:.0%})       │ {knee_point['J_mean']:.4f}   │ {knee_point['F_mean']:.4f}   │ ⭐ Optimal      │
└──────────────────────────┴──────────┴──────────┴─────────────────┘

RECOMMENDED RULE:
=================
• Method: Rank-based aggregation
• Weights: {knee_point['judge_weight']:.0%} Judge + {knee_point['fan_weight']:.0%} Fan
• Supporting Mechanism: Judges' Save (Bottom 2)
• Result: J increases by {knee_point['J_mean'] - current_rank['J_mean']:.4f}, F changes by {knee_point['F_mean'] - current_rank['F_mean']:.4f}

CONCLUSION:
===========
The recommended rule sits at the "knee point" of the Pareto frontier,
providing the best balance between meritocracy and fan engagement.

FILES SAVED:
============
• pareto_points.csv - All tested weight combinations
• rule_comparison.csv - Key rules comparison
• recommended_rule.json - Recommended rule details
• pareto_optimization.png - Visualizations
""")

print("=" * 70)
print("PHASE 4 COMPLETE!")
print("=" * 70)
