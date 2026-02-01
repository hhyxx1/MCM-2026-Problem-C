#!/usr/bin/env python3
"""
单独生成 pareto_integrated.png 图
====================================
从 phase4_pareto_v2.py 提取，独立生成帕累托优化综合图

Author: MCM 2026 Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("生成 pareto_integrated.png")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1] Loading data...")
estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')
print(f"    Estimates: {len(estimates)} rows, {estimates['season'].nunique()} seasons")

# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def calculate_objectives_static(season_data, judge_weight=0.5, method='rank'):
    """静态规则：固定权重"""
    max_week = season_data['week'].max()
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return np.nan, np.nan
    
    fan_weight = 1 - judge_weight
    
    final_data['J_rank'] = final_data['J_pct'].rank(ascending=False)
    final_data['F_rank'] = final_data['f_mean'].rank(ascending=False)
    
    if method == 'rank':
        final_data['combined'] = judge_weight * final_data['J_rank'] + fan_weight * final_data['F_rank']
        final_data['final_rank'] = final_data['combined'].rank()
    else:
        max_f = final_data['f_mean'].max()
        final_data['F_pct'] = final_data['f_mean'] / max_f * 100 if max_f > 0 else 0
        final_data['combined'] = judge_weight * final_data['J_pct'] + fan_weight * final_data['F_pct']
        final_data['final_rank'] = final_data['combined'].rank(ascending=False)
    
    j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
    f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
    
    return j_corr, f_corr


def calculate_objectives_dynamic(season_data, base_weight=0.5, delta=0.03, log_strength=0.5):
    """动态对数加权规则"""
    weeks = sorted(season_data['week'].unique())
    max_week = max(weeks)
    t = len(weeks) - 1
    
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return np.nan, np.nan
    
    w_j = min(base_weight + delta * t, 0.8)
    w_f = 1 - w_j
    
    final_data['J_rank'] = final_data['J_pct'].rank(ascending=False)
    final_data['F_rank'] = final_data['f_mean'].rank(ascending=False)
    
    max_f = final_data['f_mean'].max()
    if max_f > 0:
        f_linear = final_data['f_mean'] / max_f * 100
        if log_strength > 0:
            f_log = np.log1p(final_data['f_mean'] * 100)
            max_f_log = f_log.max()
            f_log_norm = f_log / max_f_log * 100 if max_f_log > 0 else 0
            f_transformed = log_strength * f_log_norm + (1 - log_strength) * f_linear
        else:
            f_transformed = f_linear
    else:
        f_transformed = 0
    
    final_data['combined'] = w_j * final_data['J_pct'] + w_f * f_transformed
    final_data['final_rank'] = final_data['combined'].rank(ascending=False)
    
    j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
    f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
    
    return j_corr, f_corr


def calculate_objectives_dynamic_with_save(season_data, base_weight=0.5, delta=0.03, log_strength=0.5):
    """动态对数加权 + Judges' Save"""
    weeks = sorted(season_data['week'].unique())
    max_week = max(weeks)
    t = len(weeks) - 1
    
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return np.nan, np.nan
    
    w_j = min(base_weight + delta * t, 0.8)
    w_f = 1 - w_j
    
    final_data['J_rank'] = final_data['J_pct'].rank(ascending=False)
    final_data['F_rank'] = final_data['f_mean'].rank(ascending=False)
    
    max_f = final_data['f_mean'].max()
    if max_f > 0:
        f_linear = final_data['f_mean'] / max_f * 100
        if log_strength > 0:
            f_log = np.log1p(final_data['f_mean'] * 100)
            max_f_log = f_log.max()
            f_log_norm = f_log / max_f_log * 100 if max_f_log > 0 else 0
            f_transformed = log_strength * f_log_norm + (1 - log_strength) * f_linear
        else:
            f_transformed = f_linear
    else:
        f_transformed = 0
    
    final_data['combined'] = w_j * final_data['J_pct'] + w_f * f_transformed
    final_data['final_rank'] = final_data['combined'].rank(ascending=False)
    
    # Judges' Save
    n = len(final_data)
    bottom_2 = final_data[final_data['final_rank'] >= n - 1]
    if len(bottom_2) >= 2:
        j_scores = bottom_2['J_pct'].values
        if abs(j_scores[0] - j_scores[1]) > 10:
            high_j_idx = bottom_2['J_pct'].idxmax()
            final_data.loc[high_j_idx, 'final_rank'] -= 1
    
    j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
    f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
    
    return j_corr, f_corr

# =============================================================================
# COMPUTE ALL PARETO POINTS
# =============================================================================
print("\n[2] Computing Pareto points...")

all_points = []

# 静态 Rank 规则
print("    Static Rank rules...")
for w in np.linspace(0.3, 0.9, 25):
    j_list, f_list = [], []
    for season in estimates['season'].unique():
        j, f = calculate_objectives_static(estimates[estimates['season'] == season], w, 'rank')
        if not np.isnan(j) and not np.isnan(f):
            j_list.append(j)
            f_list.append(f)
    if j_list:
        all_points.append({
            'rule': 'Static Rank',
            'params': f'{w:.0%}',
            'judge_weight': w,
            'J': np.mean(j_list),
            'F': np.mean(f_list)
        })

# 静态 Pct 规则
print("    Static Pct rules...")
for w in np.linspace(0.3, 0.9, 25):
    j_list, f_list = [], []
    for season in estimates['season'].unique():
        j, f = calculate_objectives_static(estimates[estimates['season'] == season], w, 'pct')
        if not np.isnan(j) and not np.isnan(f):
            j_list.append(j)
            f_list.append(f)
    if j_list:
        all_points.append({
            'rule': 'Static Pct',
            'params': f'{w:.0%}',
            'judge_weight': w,
            'J': np.mean(j_list),
            'F': np.mean(f_list)
        })

# 动态规则（无对数）
print("    Dynamic (no log) rules...")
for base in [0.45, 0.5, 0.55]:
    for delta in [0.01, 0.015, 0.02, 0.025]:
        j_list, f_list = [], []
        for season in estimates['season'].unique():
            j, f = calculate_objectives_dynamic(estimates[estimates['season'] == season], 
                                                 base, delta, log_strength=0)
            if not np.isnan(j) and not np.isnan(f):
                j_list.append(j)
                f_list.append(f)
        if j_list:
            all_points.append({
                'rule': 'Dynamic',
                'params': f'b={base:.0%},δ={delta}',
                'base_weight': base,
                'delta': delta,
                'log_strength': 0,
                'J': np.mean(j_list),
                'F': np.mean(f_list)
            })

# 动态对数规则
print("    Dynamic+Log rules...")
for base in [0.45, 0.5, 0.55]:
    for delta in [0.01, 0.015, 0.02, 0.025]:
        for log_s in [0.1, 0.15, 0.2, 0.25, 0.3]:
            j_list, f_list = [], []
            for season in estimates['season'].unique():
                j, f = calculate_objectives_dynamic(estimates[estimates['season'] == season],
                                                     base, delta, log_s)
                if not np.isnan(j) and not np.isnan(f):
                    j_list.append(j)
                    f_list.append(f)
            if j_list:
                all_points.append({
                    'rule': 'Dynamic+Log',
                    'params': f'b={base:.0%},δ={delta},α={log_s}',
                    'base_weight': base,
                    'delta': delta,
                    'log_strength': log_s,
                    'J': np.mean(j_list),
                    'F': np.mean(f_list)
                })

# 推荐规则 (动态对数 + Judges' Save)
print("    Recommended (Dynamic+Log+Save) rules...")
for base in [0.45, 0.5, 0.55]:
    for delta in [0.01, 0.015, 0.02]:
        for log_s in [0.1, 0.15, 0.2]:
            j_list, f_list = [], []
            for season in estimates['season'].unique():
                j, f = calculate_objectives_dynamic_with_save(estimates[estimates['season'] == season],
                                                               base, delta, log_s)
                if not np.isnan(j) and not np.isnan(f):
                    j_list.append(j)
                    f_list.append(f)
            if j_list:
                all_points.append({
                    'rule': 'Recommended',
                    'params': f'b={base:.0%},δ={delta},α={log_s}+Save',
                    'base_weight': base,
                    'delta': delta,
                    'log_strength': log_s,
                    'J': np.mean(j_list),
                    'F': np.mean(f_list)
                })

df = pd.DataFrame(all_points)
df['Balance'] = 2 * df['J'] * df['F'] / (df['J'] + df['F'] + 1e-6)

print(f"\n    Total points: {len(df)}")

# =============================================================================
# FIND PARETO FRONTIER AND KEY POINTS
# =============================================================================
print("\n[3] Finding Pareto frontier and key points...")

def find_pareto_frontier(points):
    """找帕累托前沿"""
    vals = points[['J', 'F']].values
    n = len(vals)
    is_optimal = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if (vals[j, 0] >= vals[i, 0] and vals[j, 1] >= vals[i, 1] and
                    (vals[j, 0] > vals[i, 0] or vals[j, 1] > vals[i, 1])):
                    is_optimal[i] = False
                    break
    
    return points[is_optimal].copy()

# 各规则分组
static_rank = df[df['rule'] == 'Static Rank'].copy()
static_pct = df[df['rule'] == 'Static Pct'].copy()
dynamic = df[df['rule'] == 'Dynamic'].copy()
dynamic_log = df[df['rule'] == 'Dynamic+Log'].copy()
recommended = df[df['rule'] == 'Recommended'].copy()

# 帕累托前沿
rank_frontier = find_pareto_frontier(static_rank).sort_values('J')
pct_frontier = find_pareto_frontier(static_pct).sort_values('J')

# 当前规则
current_rule = static_pct[abs(static_pct['judge_weight'] - 0.5) < 0.01].iloc[0]

# 找推荐规则中的最优
if len(recommended) > 0:
    rec_frontier = find_pareto_frontier(recommended)
    if len(rec_frontier) > 0:
        good_points = rec_frontier[(rec_frontier['J'] > current_rule['J']) & 
                                    (rec_frontier['F'] > 0.55)]
        if len(good_points) > 0:
            best_recommended = good_points.loc[good_points['Balance'].idxmax()]
        else:
            best_recommended = rec_frontier.loc[rec_frontier['Balance'].idxmax()]
    else:
        best_recommended = recommended.loc[recommended['Balance'].idxmax()]
else:
    best_recommended = None

# 找动态对数规则的最优
if len(dynamic_log) > 0:
    dyn_frontier = find_pareto_frontier(dynamic_log)
    if len(dyn_frontier) > 0:
        best_dynamic = dyn_frontier.loc[dyn_frontier['Balance'].idxmax()]
    else:
        best_dynamic = dynamic_log.loc[dynamic_log['Balance'].idxmax()]
else:
    best_dynamic = None

# Rank 50-50
rank_50 = static_rank[abs(static_rank['judge_weight'] - 0.5) < 0.01].iloc[0]

print(f"\n    Current (Pct 50-50): J={current_rule['J']:.4f}, F={current_rule['F']:.4f}, Balance={current_rule['Balance']:.4f}")
print(f"    Rank 50-50: J={rank_50['J']:.4f}, F={rank_50['F']:.4f}, Balance={rank_50['Balance']:.4f}")
if best_dynamic is not None:
    print(f"    Best Dynamic+Log: J={best_dynamic['J']:.4f}, F={best_dynamic['F']:.4f}, Balance={best_dynamic['Balance']:.4f}")
    print(f"      Params: {best_dynamic['params']}")
if best_recommended is not None:
    print(f"    Best Recommended: J={best_recommended['J']:.4f}, F={best_recommended['F']:.4f}, Balance={best_recommended['Balance']:.4f}")
    print(f"      Params: {best_recommended['params']}")

# =============================================================================
# VISUALIZATION - 生成 pareto_integrated.png
# =============================================================================
print("\n[4] Generating pareto_integrated.png...")

img_dir = 'cleaned_outputs/phase4_pareto'
os.makedirs(img_dir, exist_ok=True)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# =============================================================================
# SUBPLOT 1 (子图a): 主帕累托图 - Pareto Frontier
# =============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#fafafa')

# 绘制帕累托前沿
ax1.plot(rank_frontier['J'], rank_frontier['F'], 
         'o-', color='#3b82f6', linewidth=2.5, markersize=6,
         label='Rank Frontier', alpha=0.8)
ax1.plot(pct_frontier['J'], pct_frontier['F'],
         's--', color='#ef4444', linewidth=2.5, markersize=6,
         label='Pct Frontier', alpha=0.8)

# 绘制动态规则点（背景）
ax1.scatter(dynamic['J'], dynamic['F'], c='#f59e0b', s=40, alpha=0.3, marker='^', label='Dynamic')
ax1.scatter(dynamic_log['J'], dynamic_log['F'], c='#a855f7', s=50, alpha=0.4, marker='D', label='Dynamic+Log')
ax1.scatter(recommended['J'], recommended['F'], c='#10b981', s=80, alpha=0.5, marker='*', label='Recommended')

# 当前规则标记
ax1.scatter(current_rule['J'], current_rule['F'],
            c='#f97316', s=400, marker='h', zorder=10,
            edgecolors='#c2410c', linewidth=3)
ax1.annotate('Current\n(Pct 50-50)', 
             xy=(current_rule['J'], current_rule['F']),
             xytext=(current_rule['J']-0.1, current_rule['F']+0.08),
             fontsize=10, ha='right', fontweight='bold', color='#c2410c',
             arrowprops=dict(arrowstyle='->', color='#c2410c'))

# Rank 50-50 (膝点)
ax1.scatter(rank_50['J'], rank_50['F'],
            c='#3b82f6', s=300, marker='p', zorder=10,
            edgecolors='#1e40af', linewidth=3)
ax1.annotate('Rank 50-50\n(Knee Point)', 
             xy=(rank_50['J'], rank_50['F']),
             xytext=(rank_50['J']+0.08, rank_50['F']+0.05),
             fontsize=10, ha='left', color='#1e40af',
             arrowprops=dict(arrowstyle='->', color='#1e40af'))

# 最优推荐规则
if best_recommended is not None:
    ax1.scatter(best_recommended['J'], best_recommended['F'],
                c='#10b981', s=500, marker='*', zorder=10,
                edgecolors='#047857', linewidth=3)
    ax1.annotate(f"★ Recommended\n({best_recommended['params']})", 
                 xy=(best_recommended['J'], best_recommended['F']),
                 xytext=(best_recommended['J']+0.05, best_recommended['F']-0.1),
                 fontsize=10, ha='left', fontweight='bold', color='#047857',
                 arrowprops=dict(arrowstyle='->', color='#047857', lw=2))

# 改进箭头
if best_recommended is not None:
    ax1.annotate('', 
                 xy=(best_recommended['J'], best_recommended['F']),
                 xytext=(current_rule['J'], current_rule['F']),
                 arrowprops=dict(arrowstyle='->', color='#059669', lw=2.5,
                                connectionstyle='arc3,rad=0.2'))

ax1.set_xlabel('J (Meritocracy)', fontsize=13, fontweight='bold')
ax1.set_ylabel('F (Engagement)', fontsize=13, fontweight='bold')
ax1.set_title('(a) Pareto Frontier: Static vs Dynamic Log-Weighting Rules', fontsize=16, fontweight='bold')
ax1.legend(loc='lower left', fontsize=9, ncol=3, framealpha=0.95)
ax1.grid(True, alpha=0.4, linestyle='--')

# =============================================================================
# SUBPLOT 2 (子图b): 动态权重演化
# =============================================================================
ax2 = fig.add_subplot(gs[1, 0])

base_w = best_recommended['base_weight'] if best_recommended is not None else 0.5
delta_val = best_recommended['delta'] if best_recommended is not None else 0.03

weeks = np.arange(0, 12)
judge_weights = np.minimum(base_w + delta_val * weeks, 0.8)
fan_weights = 1 - judge_weights

ax2.fill_between(weeks, 0, judge_weights, alpha=0.6, color='#3b82f6', label='Judge Weight $w_J$')
ax2.fill_between(weeks, judge_weights, 1, alpha=0.6, color='#f97316', label='Fan Weight $w_F$')
ax2.plot(weeks, judge_weights, 'b-', linewidth=2.5)

formula = f'$w_J(t) = {base_w} + {delta_val}t$\n$w_F(t) = 1 - w_J(t)$'
ax2.text(0.98, 0.95, formula, transform=ax2.transAxes, fontsize=11,
         va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

ax2.set_xlabel('Week (t)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Weight', fontsize=12, fontweight='bold')
ax2.set_title('(b) Dynamic Weight Evolution', fontsize=14, fontweight='bold')
ax2.legend(loc='center right', fontsize=10)
ax2.set_xlim(0, 11)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

# =============================================================================
# SUBPLOT 3 (子图c): 规则对比柱状图
# =============================================================================
ax3 = fig.add_subplot(gs[1, 1])

labels = ['Current\n(Pct 50-50)', 'Rank 50-50']
j_vals = [current_rule['J'], rank_50['J']]
f_vals = [current_rule['F'], rank_50['F']]
balance_vals = [current_rule['Balance'], rank_50['Balance']]

if best_dynamic is not None:
    labels.append('Dynamic+Log')
    j_vals.append(best_dynamic['J'])
    f_vals.append(best_dynamic['F'])
    balance_vals.append(best_dynamic['Balance'])

if best_recommended is not None:
    labels.append('Recommended\n(+Save)')
    j_vals.append(best_recommended['J'])
    f_vals.append(best_recommended['F'])
    balance_vals.append(best_recommended['Balance'])

x = np.arange(len(labels))
width = 0.25

bars1 = ax3.bar(x - width, j_vals, width, label='J (Meritocracy)', color='steelblue', alpha=0.8)
bars2 = ax3.bar(x, f_vals, width, label='F (Engagement)', color='coral', alpha=0.8)
bars3 = ax3.bar(x + width, balance_vals, width, label='Balance', color='forestgreen', alpha=0.8)

ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('(c) Key Rules Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=9)
ax3.legend(fontsize=10)
ax3.set_ylim(0, 1.1)
ax3.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8)

fig.suptitle('Phase 4: Pareto Optimization with Dynamic Log-Weighting Rule', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{img_dir}/pareto_integrated.png', dpi=200, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {img_dir}/pareto_integrated.png")

# 同时显示图像
plt.show()

print("\n" + "=" * 70)
print("图表说明：")
print("=" * 70)
print("""
【子图 (a) - Pareto Frontier: 帕累托前沿图】
  - X轴 J (Meritocracy): 表示规则对"实力导向"的尊重程度
    (与评委评分排名的Spearman相关系数，越高表示越尊重专业评分)
  - Y轴 F (Engagement): 表示规则对"观众参与度"的尊重程度
    (与粉丝投票排名的Spearman相关系数，越高表示越尊重观众意愿)
  - 蓝色线 (Rank Frontier): 基于排名的静态规则帕累托前沿
  - 红色线 (Pct Frontier): 基于百分比的静态规则帕累托前沿
  - 橙色六边形 (Current): 当前采用的规则 (Pct 50-50)
  - 蓝色五边形 (Knee Point): Rank 50-50 膝点（权衡点）
  - 绿色星形 (Recommended): 推荐的动态对数规则+评委挽救机制

【子图 (b) - Dynamic Weight Evolution: 动态权重演化图】
  - X轴: 比赛周数 (t)
  - Y轴: 权重值 (0-1)
  - 蓝色区域 (w_J): 评委权重，随时间线性增加
  - 橙色区域 (w_F): 粉丝权重，随时间线性减少
  - 公式: w_J(t) = base + δ*t，意味着越接近决赛，专业评分越重要

【子图 (c) - Key Rules Comparison: 关键规则对比柱状图】
  - X轴: 四种规则
    1. Current (Pct 50-50): 当前规则 (百分比加权，50-50)
    2. Rank 50-50: 排名加权规则 (膝点)
    3. Dynamic+Log: 动态对数规则
    4. Recommended (+Save): 推荐规则 (动态对数 + 评委挽救)
  - Y轴: 得分 (0-1)
  - 蓝色柱 (J - Meritocracy): 实力导向得分
  - 橙色柱 (F - Engagement): 观众参与度得分
  - 绿色柱 (Balance): 平衡指标 = 2*J*F/(J+F)，调和平均数
    表示在J和F之间的综合权衡能力，越高说明两者兼顾越好
""")
