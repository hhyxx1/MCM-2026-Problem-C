#!/usr/bin/env python3
"""
单独生成子图(c) - Key Rules Comparison 柱状图
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

print("生成 Key Rules Comparison 柱状图...")

# =============================================================================
# LOAD DATA & COMPUTE
# =============================================================================
estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')

def calculate_objectives_static(season_data, judge_weight=0.5, method='rank'):
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

# 计算各规则得分
all_points = []

# Static Rank
for w in np.linspace(0.3, 0.9, 25):
    j_list, f_list = [], []
    for season in estimates['season'].unique():
        j, f = calculate_objectives_static(estimates[estimates['season'] == season], w, 'rank')
        if not np.isnan(j) and not np.isnan(f):
            j_list.append(j)
            f_list.append(f)
    if j_list:
        all_points.append({'rule': 'Static Rank', 'judge_weight': w, 'J': np.mean(j_list), 'F': np.mean(f_list)})

# Static Pct
for w in np.linspace(0.3, 0.9, 25):
    j_list, f_list = [], []
    for season in estimates['season'].unique():
        j, f = calculate_objectives_static(estimates[estimates['season'] == season], w, 'pct')
        if not np.isnan(j) and not np.isnan(f):
            j_list.append(j)
            f_list.append(f)
    if j_list:
        all_points.append({'rule': 'Static Pct', 'judge_weight': w, 'J': np.mean(j_list), 'F': np.mean(f_list)})

# Dynamic+Log
for base in [0.45, 0.5, 0.55]:
    for delta in [0.01, 0.015, 0.02, 0.025]:
        for log_s in [0.1, 0.15, 0.2, 0.25, 0.3]:
            j_list, f_list = [], []
            for season in estimates['season'].unique():
                j, f = calculate_objectives_dynamic(estimates[estimates['season'] == season], base, delta, log_s)
                if not np.isnan(j) and not np.isnan(f):
                    j_list.append(j)
                    f_list.append(f)
            if j_list:
                all_points.append({'rule': 'Dynamic+Log', 'base_weight': base, 'delta': delta, 
                                   'log_strength': log_s, 'J': np.mean(j_list), 'F': np.mean(f_list)})

# Recommended
for base in [0.45, 0.5, 0.55]:
    for delta in [0.01, 0.015, 0.02]:
        for log_s in [0.1, 0.15, 0.2]:
            j_list, f_list = [], []
            for season in estimates['season'].unique():
                j, f = calculate_objectives_dynamic_with_save(estimates[estimates['season'] == season], base, delta, log_s)
                if not np.isnan(j) and not np.isnan(f):
                    j_list.append(j)
                    f_list.append(f)
            if j_list:
                all_points.append({'rule': 'Recommended', 'base_weight': base, 'delta': delta,
                                   'log_strength': log_s, 'J': np.mean(j_list), 'F': np.mean(f_list)})

df = pd.DataFrame(all_points)
df['Balance'] = 2 * df['J'] * df['F'] / (df['J'] + df['F'] + 1e-6)

# 找关键点
static_rank = df[df['rule'] == 'Static Rank']
static_pct = df[df['rule'] == 'Static Pct']
dynamic_log = df[df['rule'] == 'Dynamic+Log']
recommended = df[df['rule'] == 'Recommended']

current_rule = static_pct[abs(static_pct['judge_weight'] - 0.5) < 0.01].iloc[0]
rank_50 = static_rank[abs(static_rank['judge_weight'] - 0.5) < 0.01].iloc[0]
best_dynamic = dynamic_log.loc[dynamic_log['Balance'].idxmax()]
best_recommended = recommended.loc[recommended['Balance'].idxmax()]

# =============================================================================
# 生成单独的柱状图
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

labels = ['Current\n(Pct 50-50)', 'Rank 50-50', 'Dynamic+Log', 'Recommended\n(+Save)']
j_vals = [current_rule['J'], rank_50['J'], best_dynamic['J'], best_recommended['J']]
f_vals = [current_rule['F'], rank_50['F'], best_dynamic['F'], best_recommended['F']]
balance_vals = [current_rule['Balance'], rank_50['Balance'], best_dynamic['Balance'], best_recommended['Balance']]

x = np.arange(len(labels))
width = 0.25

bars1 = ax.bar(x - width, j_vals, width, label='J (Meritocracy)', color='steelblue', alpha=0.8)
bars2 = ax.bar(x, f_vals, width, label='F (Engagement)', color='coral', alpha=0.8)
bars3 = ax.bar(x + width, balance_vals, width, label='Balance', color='forestgreen', alpha=0.8)

ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Key Rules Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=12, loc='upper right')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                f'{h:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()

img_dir = 'cleaned_outputs/phase4_pareto'
os.makedirs(img_dir, exist_ok=True)
plt.savefig(f'{img_dir}/key_rules_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {img_dir}/key_rules_comparison.png")

plt.show()

print("""
【柱状图说明】

X轴 - 四种评分规则:
  1. Current (Pct 50-50): 当前规则，评委和粉丝各占50%，按百分比加权
  2. Rank 50-50: 排名加权规则，按排名而非百分比加权
  3. Dynamic+Log: 动态对数规则，权重随时间变化 + 对数变换压缩极端票数
  4. Recommended (+Save): 推荐规则，动态对数 + 评委挽救机制

Y轴 - 三种指标（每个规则对应3根柱子）:
  蓝色柱 J (Meritocracy): 实力导向得分
    - 与评委评分排名的Spearman相关系数
    - 越高表示规则越尊重专业评分

  橙色柱 F (Engagement): 观众参与度得分
    - 与粉丝投票排名的Spearman相关系数
    - 越高表示规则越尊重观众投票

  绿色柱 Balance: 平衡指标
    - 公式: 2*J*F/(J+F)，即调和平均数
    - 越高说明在J和F之间权衡得越好
""")
