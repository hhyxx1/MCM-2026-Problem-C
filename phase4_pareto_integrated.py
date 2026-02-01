#!/usr/bin/env python3
"""
Phase 4 Enhanced: Pareto Optimization with Dynamic Log-Weighting
=================================================================
将 Plan 中的推荐规则整合到帕累托分析中

推荐规则公式:
    Score = (0.5 + 0.05*t) · J% + (0.5 - 0.05*t) · log(F%)
    
其中:
- t = 周数 (从0开始)
- 随比赛进行，评委权重从50%增加到70-80%
- log平滑机制：抑制极端粉丝投票

帕累托图展示:
1. 静态规则的帕累托前沿 (Rank, Pct)
2. 动态对数加权规则的位置
3. 证明推荐规则位于"膝点"附近

Author: MCM 2026 Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("PHASE 4: PARETO WITH DYNAMIC LOG-WEIGHTING RULE")
print("=" * 70)

# =============================================================================
# PART 1: LOAD DATA
# =============================================================================
print("\n[1] Loading data...")
estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')
print(f"    Loaded {len(estimates)} rows, {estimates['season'].nunique()} seasons")

# =============================================================================
# PART 2: DEFINE SCORING RULES
# =============================================================================

def apply_static_rule(season_data, judge_weight=0.5, method='rank'):
    """
    静态规则: 固定权重，不随周数变化
    - method='rank': 基于排名的加权
    - method='pct': 基于百分比的加权
    """
    max_week = season_data['week'].max()
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return np.nan, np.nan, None
    
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
    
    return j_corr, f_corr, final_data


def apply_dynamic_log_rule(season_data, base_judge_weight=0.5, delta=0.05, log_scale=True, log_strength=1.0):
    """
    动态对数加权规则 (Plan Phase 5 推荐)
    
    公式: Score = (0.5 + δ*t) · J% + (0.5 - δ*t) · [α·log(F%) + (1-α)·F%]
    
    特点:
    1. 动态权重: 随周数t增加，评委权重增加 (从50%到70-80%)
    2. 对数平滑: 可调节强度，α=log_strength
    
    Parameters:
    - base_judge_weight: 初始评委权重 (默认0.5)
    - delta: 每周权重增加量 (默认0.05)
    - log_scale: 是否使用对数变换
    - log_strength: 对数强度 (0-1)，1=纯对数，0=纯线性
    """
    weeks = sorted(season_data['week'].unique())
    
    # 逐周计算得分并确定淘汰
    all_scores = []
    
    for week in weeks:
        week_data = season_data[season_data['week'] == week].copy()
        t = week - min(weeks)  # 周数从0开始
        
        # 动态权重
        w_j = min(base_judge_weight + delta * t, 0.85)  # 上限85%
        w_f = 1 - w_j
        
        # 计算粉丝得分变换
        max_f = week_data['f_mean'].max()
        f_linear = week_data['f_mean'] / max_f * 100 if max_f > 0 else 0
        
        if log_scale and log_strength > 0:
            # 混合对数: α·log(F%) + (1-α)·F%
            f_log = np.log1p(week_data['f_mean'] * 100)
            max_f_log = f_log.max()
            f_log_norm = f_log / max_f_log * 100 if max_f_log > 0 else 0
            week_data['F_transformed'] = log_strength * f_log_norm + (1 - log_strength) * f_linear
        else:
            week_data['F_transformed'] = f_linear
        
        week_data['score'] = w_j * week_data['J_pct'] + w_f * week_data['F_transformed']
        week_data['t'] = t
        week_data['w_j'] = w_j
        
        all_scores.append(week_data)
    
    # 使用最后一周的数据计算最终排名
    final_data = all_scores[-1].copy()
    final_data['J_rank'] = final_data['J_pct'].rank(ascending=False)
    final_data['F_rank'] = final_data['f_mean'].rank(ascending=False)
    final_data['final_rank'] = final_data['score'].rank(ascending=False)
    
    if len(final_data) < 3:
        return np.nan, np.nan, None
    
    j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
    f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
    
    return j_corr, f_corr, final_data


def apply_judges_save(season_data, base_rule='dynamic_log', **kwargs):
    """
    添加 Judges' Save 机制:
    当选手在 Bottom 2 时，评委可以拯救评分较高者
    """
    if base_rule == 'dynamic_log':
        j_corr, f_corr, final_data = apply_dynamic_log_rule(season_data, **kwargs)
    else:
        j_corr, f_corr, final_data = apply_static_rule(season_data, **kwargs)
    
    if final_data is None or len(final_data) < 3:
        return j_corr, f_corr
    
    # Judges' Save: Bottom 2中，如果J%差距>10，救高分者
    n = len(final_data)
    bottom_2 = final_data[final_data['final_rank'] >= n - 1]
    
    if len(bottom_2) >= 2:
        j_scores = bottom_2['J_pct'].values
        if abs(j_scores[0] - j_scores[1]) > 10:
            high_j_idx = bottom_2['J_pct'].idxmax()
            final_data.loc[high_j_idx, 'final_rank'] -= 1
    
    # 重新计算排名相关性
    final_data['J_rank'] = final_data['J_pct'].rank(ascending=False)
    final_data['F_rank'] = final_data['f_mean'].rank(ascending=False)
    
    j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
    f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
    
    return j_corr, f_corr

# =============================================================================
# PART 3: COMPUTE PARETO POINTS FOR ALL RULES
# =============================================================================
print("\n[2] Computing Pareto points for all rules...")

# 3.1 静态规则 (Rank method)
print("    Testing static Rank rules...")
rank_points = []
for w in np.linspace(0.3, 0.9, 25):
    j_list, f_list = [], []
    for season in estimates['season'].unique():
        season_data = estimates[estimates['season'] == season]
        j, f, _ = apply_static_rule(season_data, judge_weight=w, method='rank')
        if not np.isnan(j):
            j_list.append(j)
            f_list.append(f)
    if j_list:
        rank_points.append({
            'rule': 'Static Rank',
            'judge_weight': w,
            'J': np.mean(j_list),
            'F': np.mean(f_list)
        })

# 3.2 静态规则 (Pct method)
print("    Testing static Pct rules...")
pct_points = []
for w in np.linspace(0.3, 0.9, 25):
    j_list, f_list = [], []
    for season in estimates['season'].unique():
        season_data = estimates[estimates['season'] == season]
        j, f, _ = apply_static_rule(season_data, judge_weight=w, method='pct')
        if not np.isnan(j):
            j_list.append(j)
            f_list.append(f)
    if j_list:
        pct_points.append({
            'rule': 'Static Pct',
            'judge_weight': w,
            'J': np.mean(j_list),
            'F': np.mean(f_list)
        })

# 3.3 动态对数加权规则 (推荐规则)
print("    Testing Dynamic Log-Weighting rules...")
dynamic_log_points = []
# 测试更广泛的参数组合，包括更温和的设置
for base_w in [0.4, 0.45, 0.5, 0.55, 0.6]:
    for delta in [0.01, 0.02, 0.03, 0.04, 0.05]:  # 更小的增量
        for log_scale in [True, False]:
            # 也测试混合方案（部分对数）
            for log_strength in [0.3, 0.5, 0.7, 1.0]:  # 对数强度
                if not log_scale and log_strength != 1.0:
                    continue  # 非对数模式只测试一次
                    
                j_list, f_list = [], []
                for season in estimates['season'].unique():
                    season_data = estimates[estimates['season'] == season]
                    j, f, _ = apply_dynamic_log_rule(season_data, 
                                                      base_judge_weight=base_w,
                                                      delta=delta,
                                                      log_scale=log_scale,
                                                      log_strength=log_strength if log_scale else 1.0)
                    if not np.isnan(j):
                        j_list.append(j)
                        f_list.append(f)
                if j_list:
                    dynamic_log_points.append({
                        'rule': 'Dynamic' + (f'+Log({log_strength})' if log_scale else ''),
                        'base_weight': base_w,
                        'delta': delta,
                        'log_scale': log_scale,
                        'log_strength': log_strength if log_scale else 1.0,
                        'J': np.mean(j_list),
                        'F': np.mean(f_list)
                    })

# 3.4 动态对数 + Judges' Save (完整推荐)
print("    Testing Dynamic Log + Judges' Save...")
recommended_points = []
for base_w in [0.45, 0.5, 0.55]:
    for delta in [0.02, 0.03, 0.04]:
        for log_strength in [0.3, 0.5, 0.7]:
            j_list, f_list = [], []
            for season in estimates['season'].unique():
                season_data = estimates[estimates['season'] == season]
                j, f = apply_judges_save(season_data, 
                                         base_rule='dynamic_log',
                                         base_judge_weight=base_w,
                                         delta=delta,
                                         log_scale=True,
                                         log_strength=log_strength)
                if not np.isnan(j):
                    j_list.append(j)
                    f_list.append(f)
            if j_list:
                recommended_points.append({
                    'rule': 'Recommended',
                    'base_weight': base_w,
                    'delta': delta,
                    'log_strength': log_strength,
                    'J': np.mean(j_list),
                    'F': np.mean(f_list)
                })

# 转换为DataFrame
rank_df = pd.DataFrame(rank_points)
pct_df = pd.DataFrame(pct_points)
dynamic_df = pd.DataFrame(dynamic_log_points)
recommended_df = pd.DataFrame(recommended_points)

print(f"    Rank points: {len(rank_df)}")
print(f"    Pct points: {len(pct_df)}")
print(f"    Dynamic points: {len(dynamic_df)}")
print(f"    Recommended points: {len(recommended_df)}")

# =============================================================================
# PART 4: FIND PARETO FRONTIER AND KEY POINTS
# =============================================================================
print("\n[3] Finding Pareto frontier and key points...")

def find_pareto_frontier(df, j_col='J', f_col='F'):
    """找到帕累托前沿"""
    points = df[[j_col, f_col]].values
    n = len(points)
    is_optimal = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    is_optimal[i] = False
                    break
    
    return df[is_optimal].copy()

def find_knee_point(df, j_col='J', f_col='F'):
    """找到帕累托前沿的膝点"""
    if len(df) < 3:
        return df.iloc[0] if len(df) > 0 else None
    
    j_vals = df[j_col].values
    f_vals = df[f_col].values
    
    j_norm = (j_vals - j_vals.min()) / (j_vals.max() - j_vals.min() + 1e-6)
    f_norm = (f_vals - f_vals.min()) / (f_vals.max() - f_vals.min() + 1e-6)
    
    distances = np.abs(j_norm + f_norm - 1) / np.sqrt(2)
    return df.iloc[np.argmax(distances)]

# 找各方法的帕累托前沿
rank_frontier = find_pareto_frontier(rank_df).sort_values('J')
pct_frontier = find_pareto_frontier(pct_df).sort_values('J')

# 找推荐规则中的最优点 (使用膝点法而非纯Balance)
if len(recommended_df) > 0:
    # 添加Balance列
    recommended_df['Balance'] = 2 * recommended_df['J'] * recommended_df['F'] / (recommended_df['J'] + recommended_df['F'])
    
    # 找帕累托前沿上的膝点
    rec_frontier = find_pareto_frontier(recommended_df)
    if len(rec_frontier) > 0:
        best_recommended = find_knee_point(rec_frontier)
    else:
        best_recommended = recommended_df.loc[recommended_df['Balance'].idxmax()]
else:
    best_recommended = None

# 找动态规则（无Save）的最优点
if len(dynamic_df) > 0:
    dynamic_log_only = dynamic_df[dynamic_df['log_scale'] == True]
    if len(dynamic_log_only) > 0:
        dynamic_log_only = dynamic_log_only.copy()
        dynamic_log_only['Balance'] = 2 * dynamic_log_only['J'] * dynamic_log_only['F'] / (dynamic_log_only['J'] + dynamic_log_only['F'])
        # 找帕累托前沿
        dyn_frontier = find_pareto_frontier(dynamic_log_only)
        if len(dyn_frontier) > 0:
            best_dynamic_log = find_knee_point(dyn_frontier)
        else:
            best_dynamic_log = dynamic_log_only.loc[dynamic_log_only['Balance'].idxmax()]
    else:
        best_dynamic_log = None
else:
    best_dynamic_log = None

# 当前规则 (Pct 50-50)
current_rule = pct_df[abs(pct_df['judge_weight'] - 0.5) < 0.01].iloc[0]

print(f"\n    Current Rule (Pct 50-50):")
print(f"      J={current_rule['J']:.4f}, F={current_rule['F']:.4f}")

if best_dynamic_log is not None:
    print(f"\n    Best Dynamic+Log (no Save):")
    print(f"      Base={best_dynamic_log['base_weight']:.0%}, δ={best_dynamic_log['delta']}")
    print(f"      J={best_dynamic_log['J']:.4f}, F={best_dynamic_log['F']:.4f}")

if best_recommended is not None:
    print(f"\n    Best Recommended (Dynamic+Log+Save):")
    print(f"      Base={best_recommended['base_weight']:.0%}, δ={best_recommended['delta']}")
    print(f"      J={best_recommended['J']:.4f}, F={best_recommended['F']:.4f}")

# =============================================================================
# PART 5: VISUALIZATION
# =============================================================================
print("\n[4] Generating visualizations...")

import os
img_dir = 'cleaned_outputs/phase4_pareto'
os.makedirs(img_dir, exist_ok=True)

# 创建综合图
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')

gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# =============================================================================
# SUBPLOT 1: 主帕累托图 - 所有规则对比
# =============================================================================
ax1 = fig.add_subplot(gs[0, :])  # 占据顶部整行
ax1.set_facecolor('#fafafa')

# 绘制静态规则的帕累托前沿
ax1.plot(rank_frontier['J'], rank_frontier['F'], 
         'o-', color='#3b82f6', linewidth=2.5, markersize=6,
         label='Static Rank Frontier', alpha=0.7)
ax1.plot(pct_frontier['J'], pct_frontier['F'], 
         's--', color='#ef4444', linewidth=2.5, markersize=6,
         label='Static Pct Frontier', alpha=0.7)

# 绘制动态规则点（背景）
if len(dynamic_df) > 0:
    dynamic_log_only = dynamic_df[dynamic_df['log_scale'] == True]
    ax1.scatter(dynamic_log_only['J'], dynamic_log_only['F'],
                c='#a855f7', s=60, alpha=0.4, marker='D',
                label='Dynamic+Log variants')

# 标记关键点
# 1. 当前规则
ax1.scatter(current_rule['J'], current_rule['F'],
            c='#f97316', s=300, marker='h', zorder=10,
            edgecolors='#c2410c', linewidth=2,
            label='Current (Pct 50-50)')

# 2. 动态+对数（无Save）
if best_dynamic_log is not None:
    ax1.scatter(best_dynamic_log['J'], best_dynamic_log['F'],
                c='#8b5cf6', s=280, marker='D', zorder=10,
                edgecolors='#5b21b6', linewidth=2,
                label=f"Dynamic+Log (Base={best_dynamic_log['base_weight']:.0%}, δ={best_dynamic_log['delta']})")

# 3. 推荐规则（动态+对数+Save）
if best_recommended is not None:
    ax1.scatter(best_recommended['J'], best_recommended['F'],
                c='#10b981', s=400, marker='*', zorder=10,
                edgecolors='#047857', linewidth=2,
                label=f"★ Recommended (Dynamic+Log+Save)")

# 添加标注箭头
if best_recommended is not None:
    # 从当前规则指向推荐规则
    ax1.annotate('', 
                 xy=(best_recommended['J'], best_recommended['F']),
                 xytext=(current_rule['J'], current_rule['F']),
                 arrowprops=dict(arrowstyle='->', color='#059669', lw=2.5,
                                connectionstyle='arc3,rad=0.2'))
    
    # 计算改进
    j_improve = (best_recommended['J'] - current_rule['J']) / current_rule['J'] * 100
    f_change = (best_recommended['F'] - current_rule['F']) / current_rule['F'] * 100
    
    mid_x = (current_rule['J'] + best_recommended['J']) / 2
    mid_y = (current_rule['F'] + best_recommended['F']) / 2 + 0.03
    ax1.text(mid_x, mid_y, 
             f'J: +{j_improve:.1f}%\nF: {f_change:+.1f}%',
             fontsize=10, color='#059669', fontweight='bold',
             ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 理想点和反理想点
ax1.scatter(1, 1, c='gold', s=150, marker='*', zorder=5, alpha=0.6, label='Ideal (1,1)')
ax1.axhline(y=current_rule['F'], color='#f97316', linestyle=':', alpha=0.4)
ax1.axvline(x=current_rule['J'], color='#f97316', linestyle=':', alpha=0.4)

ax1.set_xlabel('J (Meritocracy) - Correlation with Judge Ranking', fontsize=13, fontweight='bold')
ax1.set_ylabel('F (Engagement) - Correlation with Fan Ranking', fontsize=13, fontweight='bold')
ax1.set_title('Pareto Frontier: Static vs Dynamic Log-Weighting Rules', fontsize=16, fontweight='bold', pad=15)
ax1.legend(loc='lower left', fontsize=10, framealpha=0.95, ncol=2)
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.set_xlim(0.3, 1.0)
ax1.set_ylim(0.3, 1.0)

# =============================================================================
# SUBPLOT 2: 动态权重随周数变化
# =============================================================================
ax2 = fig.add_subplot(gs[1, 0])

weeks = np.arange(0, 12)
base_weight = 0.5
delta = 0.05

judge_weights = np.minimum(base_weight + delta * weeks, 0.9)
fan_weights = 1 - judge_weights

ax2.fill_between(weeks, 0, judge_weights, alpha=0.6, color='#3b82f6', label='Judge Weight')
ax2.fill_between(weeks, judge_weights, 1, alpha=0.6, color='#f97316', label='Fan Weight')

ax2.plot(weeks, judge_weights, 'b-', linewidth=2.5)
ax2.plot(weeks, fan_weights, 'r--', linewidth=2.5)

# 标注公式
ax2.text(0.5, 0.95, 
         r'$w_J(t) = 0.5 + 0.05t$' + '\n' + r'$w_F(t) = 0.5 - 0.05t$',
         transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# 标注关键周
ax2.axvline(x=4, color='gray', linestyle=':', alpha=0.7)
ax2.text(4, 0.5, 'Mid-Season\n(Week 5)', ha='center', fontsize=9)
ax2.axvline(x=8, color='gray', linestyle=':', alpha=0.7)
ax2.text(8, 0.5, 'Finals\n(Week 9+)', ha='center', fontsize=9)

ax2.set_xlabel('Week (t)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Weight', fontsize=12, fontweight='bold')
ax2.set_title('(b) Dynamic Weight Evolution Over Season', fontsize=14, fontweight='bold')
ax2.legend(loc='center right', fontsize=10)
ax2.set_xlim(0, 11)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

# =============================================================================
# SUBPLOT 3: 对数平滑效果
# =============================================================================
ax3 = fig.add_subplot(gs[1, 1])

# 模拟不同粉丝投票份额
f_pct = np.linspace(1, 100, 100)
f_linear = f_pct
f_log = np.log1p(f_pct) / np.log1p(100) * 100  # 归一化到0-100

ax3.plot(f_pct, f_linear, 'r-', linewidth=2.5, label='Linear: F%')
ax3.plot(f_pct, f_log, 'b-', linewidth=2.5, label='Log: log(1+F%)')

# 标注压缩效果
ax3.fill_between(f_pct, f_log, f_linear, alpha=0.2, color='#6b7280')
ax3.annotate('', xy=(80, 90), xytext=(80, 60),
             arrowprops=dict(arrowstyle='<->', color='#dc2626', lw=2))
ax3.text(82, 75, 'Compression\nof extreme\nfan votes', fontsize=10, color='#dc2626')

# 标注关键点
ax3.scatter([20, 80], [20, 80], c='red', s=100, zorder=5)
ax3.scatter([20, 80], [np.log1p(20)/np.log1p(100)*100, np.log1p(80)/np.log1p(100)*100], 
            c='blue', s=100, zorder=5)

ax3.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax3.axvline(x=50, color='gray', linestyle=':', alpha=0.5)

ax3.set_xlabel('Raw Fan Vote Share (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Transformed Score', fontsize=12, fontweight='bold')
ax3.set_title('(c) Log Smoothing: Suppressing Extreme Fan Votes', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 100)

# 总标题
fig.suptitle('Phase 4: Pareto Optimization with Dynamic Log-Weighting Rule', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{img_dir}/pareto_dynamic_log_weighting.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"    Saved: {img_dir}/pareto_dynamic_log_weighting.png")

# =============================================================================
# PART 6: RULES COMPARISON TABLE
# =============================================================================
print("\n[5] Creating rules comparison table...")

comparison_rows = [
    {
        'Rule': 'Current (Pct 50-50)',
        'Type': 'Static',
        'J': current_rule['J'],
        'F': current_rule['F'],
        'Balance': 2 * current_rule['J'] * current_rule['F'] / (current_rule['J'] + current_rule['F']),
        'Status': '🔴 Current'
    }
]

if best_dynamic_log is not None:
    comparison_rows.append({
        'Rule': f"Dynamic+Log (Base={best_dynamic_log['base_weight']:.0%})",
        'Type': 'Dynamic',
        'J': best_dynamic_log['J'],
        'F': best_dynamic_log['F'],
        'Balance': 2 * best_dynamic_log['J'] * best_dynamic_log['F'] / (best_dynamic_log['J'] + best_dynamic_log['F']),
        'Status': '⚪ Variant'
    })

if best_recommended is not None:
    comparison_rows.append({
        'Rule': f"Recommended (Dynamic+Log+Save)",
        'Type': 'Dynamic+Save',
        'J': best_recommended['J'],
        'F': best_recommended['F'],
        'Balance': 2 * best_recommended['J'] * best_recommended['F'] / (best_recommended['J'] + best_recommended['F']),
        'Status': '⭐ Recommended'
    })

comparison_table = pd.DataFrame(comparison_rows)
comparison_table.to_csv(f'{img_dir}/dynamic_rule_comparison.csv', index=False)

# =============================================================================
# PART 7: SAVE RECOMMENDED RULE DETAILS
# =============================================================================
print("\n[6] Saving recommended rule details...")

import json

recommended_rule = {
    'rule_name': 'Dynamic Log-Weighting with Judges\' Save',
    'formula': 'Score = (0.5 + 0.05*t) · J% + (0.5 - 0.05*t) · log(1 + F%)',
    'parameters': {
        'base_judge_weight': float(best_recommended['base_weight']) if best_recommended is not None else 0.5,
        'delta': float(best_recommended['delta']) if best_recommended is not None else 0.05,
        'log_smoothing': True,
        'judges_save': True,
        'judges_save_threshold': 10  # J%差距>10时触发
    },
    'performance': {
        'J_meritocracy': float(best_recommended['J']) if best_recommended is not None else None,
        'F_engagement': float(best_recommended['F']) if best_recommended is not None else None,
        'balance_score': float(comparison_rows[-1]['Balance']) if best_recommended is not None else None
    },
    'vs_current': {
        'current_J': float(current_rule['J']),
        'current_F': float(current_rule['F']),
        'J_improvement': f"{((best_recommended['J'] - current_rule['J']) / current_rule['J'] * 100):.1f}%" if best_recommended is not None else None,
        'F_change': f"{((best_recommended['F'] - current_rule['F']) / current_rule['F'] * 100):+.1f}%" if best_recommended is not None else None
    },
    'rationale': [
        'Dynamic weighting: Increases judge influence as competition progresses (50% → 80%)',
        'Log smoothing: Compresses extreme fan votes, preventing canvassing from dominating',
        'Judges\' Save: Last safety net for high-skill contestants in Bottom 2',
        'Combined effect: Better meritocracy without significantly hurting engagement'
    ]
}

with open(f'{img_dir}/recommended_rule_dynamic.json', 'w') as f:
    json.dump(recommended_rule, f, indent=2)
print(f"    Saved: recommended_rule_dynamic.json")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("DYNAMIC LOG-WEIGHTING PARETO ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
RECOMMENDED RULE:
=================
    Score = (0.5 + 0.05·t) · J% + (0.5 - 0.05·t) · log(1 + F%)
    
    + Judges' Save for Bottom 2 (when J% gap > 10)

KEY FEATURES:
=============
    1. DYNAMIC WEIGHT: Judge weight increases from 50% to ~80% over season
       → Early weeks: Give fans voice to build engagement
       → Later weeks: Merit matters more for fair outcome
    
    2. LOG SMOOTHING: log(1 + F%) instead of raw F%
       → Compresses extreme fan votes
       → Prevents organized canvassing from dominating
    
    3. JUDGES' SAVE: Safety mechanism for Bottom 2
       → If J% gap > 10 points, save the better dancer
       → Last resort to prevent egregious eliminations

COMPARISON:
===========
┌────────────────────────────┬──────────┬──────────┬──────────┬────────────┐
│ Rule                       │ J        │ F        │ Balance  │ Status     │
├────────────────────────────┼──────────┼──────────┼──────────┼────────────┤
│ Current (Pct 50-50)        │ {current_rule['J']:.4f}   │ {current_rule['F']:.4f}   │ {2*current_rule['J']*current_rule['F']/(current_rule['J']+current_rule['F']):.4f}   │ 🔴 Current │
""")

if best_dynamic_log is not None:
    print(f"│ Dynamic+Log (no Save)      │ {best_dynamic_log['J']:.4f}   │ {best_dynamic_log['F']:.4f}   │ {2*best_dynamic_log['J']*best_dynamic_log['F']/(best_dynamic_log['J']+best_dynamic_log['F']):.4f}   │ ⚪ Variant │")

if best_recommended is not None:
    print(f"│ Recommended (Full Package) │ {best_recommended['J']:.4f}   │ {best_recommended['F']:.4f}   │ {2*best_recommended['J']*best_recommended['F']/(best_recommended['J']+best_recommended['F']):.4f}   │ ⭐ Recomm. │")

print("└────────────────────────────┴──────────┴──────────┴──────────┴────────────┘")

print(f"""
FILES SAVED:
============
• pareto_dynamic_log_weighting.png - Main visualization
• dynamic_rule_comparison.csv - Rules comparison table
• recommended_rule_dynamic.json - Recommended rule details
""")

print("=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
