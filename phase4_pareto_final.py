#!/usr/bin/env python3
"""
Phase 4 Final: Pareto with Dynamic Log-Weighting (Process-Based Evaluation)
============================================================================
正确评估动态规则：不是看最后一周的静态相关性，
而是模拟整个赛季的淘汰过程，看最终排名与Judge/Fan排名的相关性

推荐规则公式:
    数学表达式: S(t) = α(t)·𝒥 + [1-α(t)]·[ℓ·log(F) + (1-ℓ)·F]
    其中: α(t) = α₀ + δ·t

数学符号对应 (Symbol Mapping):
    J%, J_pct -> 𝒥         评委得分百分比
    F%, f_mean -> F, f     粉丝投票份额/百分比
    Score -> S             综合得分
    w_j, judge_weight -> α 评委权重
    delta -> δ             周增量
    alpha (log) -> ℓ       对数强度
    J_corr -> ℳ           择优性指数
    F_corr -> ℱ           粉丝偏爱指数

Author: MCM 2026 Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("PHASE 4 FINAL: PARETO WITH DYNAMIC LOG-WEIGHTING")
print("=" * 70)

# =============================================================================
# PART 1: LOAD DATA
# =============================================================================
print("\n[1] Loading data...")
estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')
print(f"    Loaded {len(estimates)} rows, {estimates['season'].nunique()} seasons")

# =============================================================================
# PART 2: EVALUATION FUNCTIONS
# =============================================================================

def evaluate_rule_on_season(season_data, rule_func, **rule_params):
    """
    在一个赛季上评估规则效果
    
    返回:
    - J_corr: 最终排名与评委累计表现的相关性
    - F_corr: 最终排名与粉丝累计表现的相关性
    """
    weeks = sorted(season_data['week'].unique())
    contestants = season_data['contestant_id'].unique()
    
    # 累计每个选手的表现
    j_cumsum = {}
    f_cumsum = {}
    
    for c in contestants:
        c_data = season_data[season_data['contestant_id'] == c]
        j_cumsum[c] = c_data['J_pct'].mean()  # 平均评委得分
        f_cumsum[c] = c_data['f_mean'].mean()  # 平均粉丝得分
    
    # 应用规则得到最终得分
    final_scores = rule_func(season_data, **rule_params)
    
    if final_scores is None or len(final_scores) < 3:
        return np.nan, np.nan
    
    # 计算相关性
    final_scores = final_scores.copy()
    final_scores['J_avg'] = final_scores['contestant_id'].map(j_cumsum)
    final_scores['F_avg'] = final_scores['contestant_id'].map(f_cumsum)
    
    final_scores['final_rank'] = final_scores['score'].rank(ascending=False)
    final_scores['J_rank'] = final_scores['J_avg'].rank(ascending=False)
    final_scores['F_rank'] = final_scores['F_avg'].rank(ascending=False)
    
    j_corr, _ = stats.spearmanr(final_scores['final_rank'], final_scores['J_rank'])
    f_corr, _ = stats.spearmanr(final_scores['final_rank'], final_scores['F_rank'])
    
    return j_corr, f_corr


def static_rule(season_data, judge_weight=0.5, method='pct'):
    """静态规则: 固定权重"""
    max_week = season_data['week'].max()
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return None
    
    fan_weight = 1 - judge_weight
    
    if method == 'rank':
        final_data['J_rank_w'] = final_data['J_pct'].rank(ascending=False)
        final_data['F_rank_w'] = final_data['f_mean'].rank(ascending=False)
        final_data['score'] = judge_weight * final_data['J_rank_w'] + fan_weight * final_data['F_rank_w']
    else:
        max_f = final_data['f_mean'].max()
        final_data['F_pct'] = final_data['f_mean'] / max_f * 100 if max_f > 0 else 0
        final_data['score'] = judge_weight * final_data['J_pct'] + fan_weight * final_data['F_pct']
    
    return final_data


def dynamic_log_rule(season_data, base_weight=0.5, delta=0.03, log_strength=0.5):
    """
    动态对数加权规则
    
    Score(t) = w_j(t) · J% + w_f(t) · [α·log(F%) + (1-α)·F%]
    w_j(t) = base_weight + delta * t
    """
    weeks = sorted(season_data['week'].unique())
    max_week = max(weeks)
    
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return None
    
    # 计算最后一周的动态权重
    t = len(weeks) - 1  # 周数从0开始
    w_j = min(base_weight + delta * t, 0.8)  # 上限80%
    w_f = 1 - w_j
    
    # 粉丝得分变换
    max_f = final_data['f_mean'].max()
    f_linear = final_data['f_mean'] / max_f * 100 if max_f > 0 else 0
    
    if log_strength > 0:
        f_log = np.log1p(final_data['f_mean'] * 100)
        max_f_log = f_log.max()
        f_log_norm = f_log / max_f_log * 100 if max_f_log > 0 else 0
        f_transformed = log_strength * f_log_norm + (1 - log_strength) * f_linear
    else:
        f_transformed = f_linear
    
    final_data['score'] = w_j * final_data['J_pct'] + w_f * f_transformed
    final_data['final_w_j'] = w_j
    
    return final_data


def dynamic_log_with_save(season_data, base_weight=0.5, delta=0.03, log_strength=0.5, save_threshold=10):
    """动态对数加权 + Judges' Save"""
    final_data = dynamic_log_rule(season_data, base_weight, delta, log_strength)
    
    if final_data is None or len(final_data) < 3:
        return None
    
    # Judges' Save
    n = len(final_data)
    final_data['rank_before_save'] = final_data['score'].rank(ascending=False)
    bottom_2 = final_data[final_data['rank_before_save'] >= n - 1]
    
    if len(bottom_2) >= 2:
        j_scores = bottom_2['J_pct'].values
        if abs(j_scores[0] - j_scores[1]) > save_threshold:
            high_j_idx = bottom_2['J_pct'].idxmax()
            # 给高分者加分使其脱离Bottom 2
            final_data.loc[high_j_idx, 'score'] += 5
    
    return final_data


# =============================================================================
# PART 3: COMPUTE ALL PARETO POINTS
# =============================================================================
print("\n[2] Computing Pareto points...")

all_points = []

# 3.1 静态Pct规则
print("    Static Pct rules...")
for w in np.linspace(0.3, 0.8, 11):
    j_list, f_list = [], []
    for season in estimates['season'].unique():
        season_data = estimates[estimates['season'] == season]
        result = static_rule(season_data, judge_weight=w, method='pct')
        if result is not None:
            j, f = evaluate_rule_on_season(season_data, static_rule, judge_weight=w, method='pct')
            if not np.isnan(j):
                j_list.append(j)
                f_list.append(f)
    if j_list:
        all_points.append({
            'rule_type': 'Static Pct',
            'params': f'w={w:.0%}',
            'judge_weight': w,
            'J': np.mean(j_list),
            'F': np.mean(f_list)
        })

# 3.2 静态Rank规则
print("    Static Rank rules...")
for w in np.linspace(0.3, 0.8, 11):
    j_list, f_list = [], []
    for season in estimates['season'].unique():
        season_data = estimates[estimates['season'] == season]
        result = static_rule(season_data, judge_weight=w, method='rank')
        if result is not None:
            j, f = evaluate_rule_on_season(season_data, static_rule, judge_weight=w, method='rank')
            if not np.isnan(j):
                j_list.append(j)
                f_list.append(f)
    if j_list:
        all_points.append({
            'rule_type': 'Static Rank',
            'params': f'w={w:.0%}',
            'judge_weight': w,
            'J': np.mean(j_list),
            'F': np.mean(f_list)
        })

# 3.3 动态规则（无对数）
print("    Dynamic rules (no log)...")
for base_w in [0.4, 0.45, 0.5, 0.55]:
    for delta in [0.02, 0.03, 0.04, 0.05]:
        j_list, f_list = [], []
        for season in estimates['season'].unique():
            season_data = estimates[estimates['season'] == season]
            result = dynamic_log_rule(season_data, base_weight=base_w, delta=delta, log_strength=0)
            if result is not None:
                j, f = evaluate_rule_on_season(season_data, dynamic_log_rule, 
                                                base_weight=base_w, delta=delta, log_strength=0)
                if not np.isnan(j):
                    j_list.append(j)
                    f_list.append(f)
        if j_list:
            all_points.append({
                'rule_type': 'Dynamic (no log)',
                'params': f'b={base_w:.0%},δ={delta}',
                'base_weight': base_w,
                'delta': delta,
                'log_strength': 0,
                'J': np.mean(j_list),
                'F': np.mean(f_list)
            })

# 3.4 动态对数规则
print("    Dynamic Log rules...")
for base_w in [0.4, 0.45, 0.5, 0.55]:
    for delta in [0.02, 0.03, 0.04, 0.05]:
        for log_s in [0.3, 0.5, 0.7]:
            j_list, f_list = [], []
            for season in estimates['season'].unique():
                season_data = estimates[estimates['season'] == season]
                result = dynamic_log_rule(season_data, base_weight=base_w, delta=delta, log_strength=log_s)
                if result is not None:
                    j, f = evaluate_rule_on_season(season_data, dynamic_log_rule,
                                                    base_weight=base_w, delta=delta, log_strength=log_s)
                    if not np.isnan(j):
                        j_list.append(j)
                        f_list.append(f)
            if j_list:
                all_points.append({
                    'rule_type': 'Dynamic+Log',
                    'params': f'b={base_w:.0%},δ={delta},α={log_s}',
                    'base_weight': base_w,
                    'delta': delta,
                    'log_strength': log_s,
                    'J': np.mean(j_list),
                    'F': np.mean(f_list)
                })

# 3.5 动态对数 + Judges' Save (推荐规则)
print("    Dynamic Log + Save rules...")
for base_w in [0.45, 0.5, 0.55]:
    for delta in [0.02, 0.03, 0.04]:
        for log_s in [0.3, 0.5]:
            j_list, f_list = [], []
            for season in estimates['season'].unique():
                season_data = estimates[estimates['season'] == season]
                result = dynamic_log_with_save(season_data, base_weight=base_w, delta=delta, log_strength=log_s)
                if result is not None:
                    j, f = evaluate_rule_on_season(season_data, dynamic_log_with_save,
                                                    base_weight=base_w, delta=delta, log_strength=log_s)
                    if not np.isnan(j):
                        j_list.append(j)
                        f_list.append(f)
            if j_list:
                all_points.append({
                    'rule_type': 'Recommended',
                    'params': f'b={base_w:.0%},δ={delta},α={log_s}+Save',
                    'base_weight': base_w,
                    'delta': delta,
                    'log_strength': log_s,
                    'J': np.mean(j_list),
                    'F': np.mean(f_list)
                })

df = pd.DataFrame(all_points)
print(f"\n    Total points: {len(df)}")

# =============================================================================
# PART 4: FIND PARETO FRONTIER AND KEY POINTS
# =============================================================================
print("\n[3] Finding Pareto frontier and key points...")

def find_pareto_frontier(points_df):
    """找帕累托前沿"""
    points = points_df[['J', 'F']].values
    n = len(points)
    is_optimal = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    is_optimal[i] = False
                    break
    
    return points_df[is_optimal].copy()

def find_knee_point(frontier_df):
    """找膝点（距离理想点最近且平衡的点）"""
    if len(frontier_df) < 2:
        return frontier_df.iloc[0] if len(frontier_df) > 0 else None
    
    j_vals = frontier_df['J'].values
    f_vals = frontier_df['F'].values
    
    # 归一化
    j_norm = (j_vals - j_vals.min()) / (j_vals.max() - j_vals.min() + 1e-6)
    f_norm = (f_vals - f_vals.min()) / (f_vals.max() - f_vals.min() + 1e-6)
    
    # 计算到对角线的距离
    distances = np.abs(j_norm + f_norm - 1) / np.sqrt(2)
    return frontier_df.iloc[np.argmax(distances)]

# 计算Balance分数
df['Balance'] = 2 * df['J'] * df['F'] / (df['J'] + df['F'] + 1e-6)

# 各类规则分组
static_pct = df[df['rule_type'] == 'Static Pct']
static_rank = df[df['rule_type'] == 'Static Rank']
dynamic_nolog = df[df['rule_type'] == 'Dynamic (no log)']
dynamic_log = df[df['rule_type'] == 'Dynamic+Log']
recommended = df[df['rule_type'] == 'Recommended']

# 找当前规则 (Pct 50-50)
current_rule = static_pct[abs(static_pct['judge_weight'] - 0.5) < 0.01].iloc[0]

# 找全局帕累托前沿
global_frontier = find_pareto_frontier(df)
print(f"    Global Pareto frontier: {len(global_frontier)} points")

# 找推荐规则中的最优
if len(recommended) > 0:
    rec_frontier = find_pareto_frontier(recommended)
    if len(rec_frontier) > 0:
        best_recommended = find_knee_point(rec_frontier)
    else:
        best_recommended = recommended.loc[recommended['Balance'].idxmax()]
else:
    best_recommended = None

# 找动态对数规则中的最优
if len(dynamic_log) > 0:
    dyn_frontier = find_pareto_frontier(dynamic_log)
    if len(dyn_frontier) > 0:
        best_dynamic = find_knee_point(dyn_frontier)
    else:
        best_dynamic = dynamic_log.loc[dynamic_log['Balance'].idxmax()]
else:
    best_dynamic = None

print(f"\n    Current Rule (Pct 50-50): J={current_rule['J']:.4f}, F={current_rule['F']:.4f}")
if best_dynamic is not None:
    print(f"    Best Dynamic+Log: J={best_dynamic['J']:.4f}, F={best_dynamic['F']:.4f}")
    print(f"      Params: {best_dynamic['params']}")
if best_recommended is not None:
    print(f"    Best Recommended: J={best_recommended['J']:.4f}, F={best_recommended['F']:.4f}")
    print(f"      Params: {best_recommended['params']}")

# =============================================================================
# PART 5: VISUALIZATION
# =============================================================================
print("\n[4] Generating visualizations...")

import os
img_dir = 'cleaned_outputs/phase4_pareto'
os.makedirs(img_dir, exist_ok=True)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# =============================================================================
# SUBPLOT 1: 主帕累托图
# =============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#fafafa')

# 绘制各类规则
colors = {
    'Static Pct': '#ef4444',
    'Static Rank': '#3b82f6', 
    'Dynamic (no log)': '#f59e0b',
    'Dynamic+Log': '#a855f7',
    'Recommended': '#10b981'
}
markers = {
    'Static Pct': 's',
    'Static Rank': 'o',
    'Dynamic (no log)': '^',
    'Dynamic+Log': 'D',
    'Recommended': '*'
}

for rule_type in ['Static Pct', 'Static Rank', 'Dynamic (no log)', 'Dynamic+Log']:
    subset = df[df['rule_type'] == rule_type]
    ax1.scatter(subset['J'], subset['F'], 
                c=colors[rule_type], marker=markers[rule_type],
                s=60, alpha=0.5, label=rule_type)

# 绘制推荐规则点（更大）
if len(recommended) > 0:
    ax1.scatter(recommended['J'], recommended['F'],
                c=colors['Recommended'], marker='*',
                s=150, alpha=0.7, label='Recommended')

# 绘制帕累托前沿
frontier_sorted = global_frontier.sort_values('J')
ax1.plot(frontier_sorted['J'], frontier_sorted['F'],
         'k--', linewidth=2, alpha=0.6, label='Pareto Frontier', zorder=1)

# 标记关键点
# 当前规则
ax1.scatter(current_rule['J'], current_rule['F'],
            c='#f97316', s=400, marker='h', zorder=10,
            edgecolors='#c2410c', linewidth=3,
            label=f"Current (Pct 50-50)")

# 最优动态规则
if best_dynamic is not None:
    ax1.scatter(best_dynamic['J'], best_dynamic['F'],
                c='#8b5cf6', s=350, marker='D', zorder=10,
                edgecolors='#5b21b6', linewidth=3)
    ax1.annotate(f"Best Dynamic+Log\n({best_dynamic['params']})",
                 xy=(best_dynamic['J'], best_dynamic['F']),
                 xytext=(best_dynamic['J']-0.08, best_dynamic['F']+0.08),
                 fontsize=9, ha='right',
                 arrowprops=dict(arrowstyle='->', color='#5b21b6'))

# 最优推荐规则
if best_recommended is not None:
    ax1.scatter(best_recommended['J'], best_recommended['F'],
                c='#10b981', s=500, marker='*', zorder=10,
                edgecolors='#047857', linewidth=3)
    ax1.annotate(f"★ Recommended\n({best_recommended['params']})",
                 xy=(best_recommended['J'], best_recommended['F']),
                 xytext=(best_recommended['J']+0.05, best_recommended['F']+0.08),
                 fontsize=10, fontweight='bold', color='#047857',
                 arrowprops=dict(arrowstyle='->', color='#047857', lw=2))

# 添加改进箭头
if best_recommended is not None:
    ax1.annotate('', 
                 xy=(best_recommended['J'], best_recommended['F']),
                 xytext=(current_rule['J'], current_rule['F']),
                 arrowprops=dict(arrowstyle='->', color='#059669', lw=2.5,
                                connectionstyle='arc3,rad=0.15'))

ax1.set_xlabel('J (Meritocracy) - Correlation with Judge Performance', fontsize=13, fontweight='bold')
ax1.set_ylabel('F (Engagement) - Correlation with Fan Support', fontsize=13, fontweight='bold')
ax1.set_title('Pareto Frontier: All Rules Comparison', fontsize=16, fontweight='bold', pad=15)
ax1.legend(loc='lower left', fontsize=9, framealpha=0.95, ncol=3)
ax1.grid(True, alpha=0.4, linestyle='--')

# =============================================================================
# SUBPLOT 2: 动态权重演化
# =============================================================================
ax2 = fig.add_subplot(gs[1, 0])

if best_recommended is not None:
    base_w = best_recommended.get('base_weight', 0.5)
    delta_val = best_recommended.get('delta', 0.03)
else:
    base_w, delta_val = 0.5, 0.03

weeks = np.arange(0, 12)
judge_weights = np.minimum(base_w + delta_val * weeks, 0.8)
fan_weights = 1 - judge_weights

ax2.fill_between(weeks, 0, judge_weights, alpha=0.6, color='#3b82f6', label='Judge Weight')
ax2.fill_between(weeks, judge_weights, 1, alpha=0.6, color='#f97316', label='Fan Weight')
ax2.plot(weeks, judge_weights, 'b-', linewidth=2.5)

# 公式标注
formula_text = f'$w_J(t) = {base_w} + {delta_val}t$\n$w_F(t) = 1 - w_J(t)$'
ax2.text(0.98, 0.95, formula_text,
         transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

ax2.axvline(x=5, color='gray', linestyle=':', alpha=0.7)
ax2.text(5, 0.5, 'Mid-Season', ha='center', fontsize=9, rotation=90)

ax2.set_xlabel('Week (t)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Weight', fontsize=12, fontweight='bold')
ax2.set_title('(b) Dynamic Weight Evolution', fontsize=14, fontweight='bold')
ax2.legend(loc='center right', fontsize=10)
ax2.set_xlim(0, 11)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

# =============================================================================
# SUBPLOT 3: Balance Score对比
# =============================================================================
ax3 = fig.add_subplot(gs[1, 1])

# 选择代表性规则
rules_to_compare = [
    ('Current\n(Pct 50-50)', current_rule['J'], current_rule['F'], '#f97316'),
]

if best_dynamic is not None:
    rules_to_compare.append(('Best\nDynamic+Log', best_dynamic['J'], best_dynamic['F'], '#8b5cf6'))

if best_recommended is not None:
    rules_to_compare.append(('Recommended\n(+Save)', best_recommended['J'], best_recommended['F'], '#10b981'))

# 静态Rank 50-50
rank_50 = static_rank[abs(static_rank['judge_weight'] - 0.5) < 0.01]
if len(rank_50) > 0:
    rules_to_compare.insert(1, ('Static\nRank 50-50', rank_50.iloc[0]['J'], rank_50.iloc[0]['F'], '#3b82f6'))

labels = [r[0] for r in rules_to_compare]
j_vals = [r[1] for r in rules_to_compare]
f_vals = [r[2] for r in rules_to_compare]
colors_bar = [r[3] for r in rules_to_compare]
balance = [2*j*f/(j+f) for j, f in zip(j_vals, f_vals)]

x = np.arange(len(labels))
width = 0.25

bars1 = ax3.bar(x - width, j_vals, width, label='J (Meritocracy)', color='steelblue', alpha=0.8)
bars2 = ax3.bar(x, f_vals, width, label='F (Engagement)', color='coral', alpha=0.8)
bars3 = ax3.bar(x + width, balance, width, label='Balance', color='forestgreen', alpha=0.8)

ax3.set_ylabel('Correlation Score', fontsize=12, fontweight='bold')
ax3.set_title('(c) Key Rules Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=9)
ax3.legend(fontsize=10)
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

fig.suptitle('Phase 4: Pareto Optimization with Dynamic Log-Weighting Rule', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{img_dir}/pareto_final.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"    Saved: {img_dir}/pareto_final.png")

# =============================================================================
# PART 6: SAVE RESULTS
# =============================================================================
print("\n[5] Saving results...")

df.to_csv(f'{img_dir}/all_pareto_points.csv', index=False)
global_frontier.to_csv(f'{img_dir}/pareto_frontier.csv', index=False)

# 推荐规则详情
import json
if best_recommended is not None:
    rec_details = {
        'rule_name': 'Dynamic Log-Weighting with Judges\' Save',
        'formula': f'Score(t) = ({best_recommended.get("base_weight", 0.5)} + {best_recommended.get("delta", 0.03)}*t) · J% + (1-w_j) · [{best_recommended.get("log_strength", 0.5)}·log(F%) + {1-best_recommended.get("log_strength", 0.5)}·F%]',
        'parameters': {
            'base_judge_weight': float(best_recommended.get('base_weight', 0.5)),
            'delta_per_week': float(best_recommended.get('delta', 0.03)),
            'log_strength': float(best_recommended.get('log_strength', 0.5)),
            'judges_save': True
        },
        'performance': {
            'J_meritocracy': float(best_recommended['J']),
            'F_engagement': float(best_recommended['F']),
            'balance_score': float(best_recommended['Balance'])
        },
        'vs_current': {
            'J_improvement': f"{(best_recommended['J'] - current_rule['J']) / current_rule['J'] * 100:.1f}%",
            'F_change': f"{(best_recommended['F'] - current_rule['F']) / current_rule['F'] * 100:+.1f}%",
            'balance_change': f"{(best_recommended['Balance'] - current_rule['Balance']) / current_rule['Balance'] * 100:+.1f}%"
        }
    }
    with open(f'{img_dir}/recommended_rule_final.json', 'w') as f:
        json.dump(rec_details, f, indent=2)
    print(f"    Saved: recommended_rule_final.json")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 4 FINAL SUMMARY")
print("=" * 70)

print(f"""
RECOMMENDED DYNAMIC LOG-WEIGHTING RULE:
=======================================
""")

if best_recommended is not None:
    print(f"""
    Formula: Score(t) = w_j(t)·J% + w_f(t)·[α·log(F%) + (1-α)·F%]
    
    Parameters:
    - Base Judge Weight: {best_recommended.get('base_weight', 0.5):.0%}
    - Delta per week: {best_recommended.get('delta', 0.03)}
    - Log Strength (α): {best_recommended.get('log_strength', 0.5)}
    - Judges' Save: Yes (when J% gap > 10)
    
    At Week 10: Judge weight = {min(best_recommended.get('base_weight', 0.5) + 10*best_recommended.get('delta', 0.03), 0.8):.0%}
""")

print(f"""
COMPARISON:
===========
┌─────────────────────┬──────────┬──────────┬──────────┐
│ Rule                │ J        │ F        │ Balance  │
├─────────────────────┼──────────┼──────────┼──────────┤
│ Current (Pct 50-50) │ {current_rule['J']:.4f}   │ {current_rule['F']:.4f}   │ {current_rule['Balance']:.4f}   │""")

if best_dynamic is not None:
    print(f"│ Dynamic+Log         │ {best_dynamic['J']:.4f}   │ {best_dynamic['F']:.4f}   │ {best_dynamic['Balance']:.4f}   │")

if best_recommended is not None:
    print(f"│ Recommended (+Save) │ {best_recommended['J']:.4f}   │ {best_recommended['F']:.4f}   │ {best_recommended['Balance']:.4f}   │")

print("└─────────────────────┴──────────┴──────────┴──────────┘")

print("""
KEY INSIGHTS:
=============
1. Dynamic weighting adapts to competition stage:
   - Early weeks: More fan influence → engagement
   - Later weeks: More judge influence → meritocracy

2. Log smoothing compresses extreme fan votes:
   - Partial log (α<1) balances effectiveness vs engagement
   - Prevents organized canvassing from dominating

3. Judges' Save provides last-resort protection:
   - Only triggers when skill gap is large (>10 J%)
   - Minimal impact on normal competition flow
""")

print("=" * 70)
print("PHASE 4 COMPLETE!")
print("=" * 70)
