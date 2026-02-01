#!/usr/bin/env python3
"""
Phase 3: Pareto Optimization with Dynamic Log-Weighting Analysis
================================================================
基于已有数据重新分析帕累托优化与动态加权+对数平滑规则模型

核心任务:
1. 双目标定义: J(精英选拔) vs F(粉丝参与)
2. 规则空间搜索: 静态规则 + 动态加权 + 对数平滑
3. Pareto前沿分析与膝点识别
4. 敏感性分析与参数优化

推荐规则公式:
    Score(t) = w_j(t)·J% + w_f(t)·[α·log(F%) + (1-α)·F%]
    其中 w_j(t) = base + δ·t

Author: MCM 2026 Team
Date: 2026-02-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import json
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

print("=" * 75)
print("PHASE 3: PARETO OPTIMIZATION WITH DYNAMIC LOG-WEIGHTING ANALYSIS")
print("=" * 75)

# =============================================================================
# 1. 加载数据
# =============================================================================
print("\n[1] Loading and preprocessing data...")

estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')
print(f"    Loaded {len(estimates)} observations from {estimates['season'].nunique()} seasons")
print(f"    Columns: {list(estimates.columns)}")

# 数据质量检查
seasons = sorted(estimates['season'].unique())
print(f"    Seasons: {min(seasons)} - {max(seasons)}")

# 统计每赛季的周数
season_stats = estimates.groupby('season').agg({
    'week': 'max',
    'celebrity_name': 'nunique',
    'f_mean': 'mean'
}).round(4)
season_stats.columns = ['max_week', 'n_contestants', 'mean_fan_share']
print(f"\n    Season Statistics (sample):")
print(season_stats.head(10).to_string())

# =============================================================================
# 2. 目标函数定义
# =============================================================================
print("\n[2] Defining objective functions...")

def calculate_objectives_static(season_data, judge_weight=0.5, method='rank'):
    """
    静态规则目标函数计算
    
    Parameters:
    -----------
    season_data : DataFrame - 单赛季数据
    judge_weight : float - 评委权重 (0-1)
    method : str - 'rank' (基于排名) 或 'pct' (基于百分比)
    
    Returns:
    --------
    j_corr, f_corr : Spearman相关系数
    """
    max_week = season_data['week'].max()
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return np.nan, np.nan
    
    fan_weight = 1 - judge_weight
    
    # 计算排名
    final_data['J_rank'] = final_data['J_pct'].rank(ascending=False)
    final_data['F_rank'] = final_data['f_mean'].rank(ascending=False)
    
    if method == 'rank':
        # Rank制: 加权排名
        final_data['combined'] = judge_weight * final_data['J_rank'] + fan_weight * final_data['F_rank']
        final_data['final_rank'] = final_data['combined'].rank()
    else:
        # Pct制: 加权百分比
        max_f = final_data['f_mean'].max()
        final_data['F_pct'] = final_data['f_mean'] / max_f * 100 if max_f > 0 else 0
        final_data['combined'] = judge_weight * final_data['J_pct'] + fan_weight * final_data['F_pct']
        final_data['final_rank'] = final_data['combined'].rank(ascending=False)
    
    # Spearman相关
    j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
    f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
    
    return j_corr, f_corr


def calculate_objectives_dynamic(season_data, base_weight=0.5, delta=0.02, log_strength=0.2):
    """
    动态对数加权规则
    
    Score(t) = w_j(t)·J% + w_f(t)·[α·log(F%) + (1-α)·F%]
    
    其中 w_j(t) = base + delta * t (每周增加δ)
    
    Parameters:
    -----------
    base_weight : float - 第1周评委权重
    delta : float - 每周权重增量
    log_strength : float (α) - 对数平滑强度
    """
    weeks = sorted(season_data['week'].unique())
    max_week = max(weeks)
    t = len(weeks) - 1  # 周数索引
    
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return np.nan, np.nan
    
    # 动态权重（在最后一周的权重）
    w_j = min(base_weight + delta * t, 0.85)  # 上限85%
    w_f = 1 - w_j
    
    # 排名计算
    final_data['J_rank'] = final_data['J_pct'].rank(ascending=False)
    final_data['F_rank'] = final_data['f_mean'].rank(ascending=False)
    
    # 粉丝得分变换 (对数平滑)
    max_f = final_data['f_mean'].max()
    if max_f > 0:
        # 线性部分
        f_linear = final_data['f_mean'] / max_f * 100
        
        # 对数平滑部分
        if log_strength > 0:
            # log1p 平滑极端值
            f_log = np.log1p(final_data['f_mean'] * 100)
            max_f_log = f_log.max()
            f_log_norm = f_log / max_f_log * 100 if max_f_log > 0 else 0
            
            # 混合: α·log + (1-α)·linear
            f_transformed = log_strength * f_log_norm + (1 - log_strength) * f_linear
        else:
            f_transformed = f_linear
    else:
        f_transformed = 0
    
    # 综合得分
    final_data['combined'] = w_j * final_data['J_pct'] + w_f * f_transformed
    final_data['final_rank'] = final_data['combined'].rank(ascending=False)
    
    j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
    f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
    
    return j_corr, f_corr


def calculate_objectives_with_save(season_data, base_weight=0.5, delta=0.02, log_strength=0.2, 
                                    save_threshold=10):
    """
    动态对数加权 + Judges' Save 机制
    
    Judges' Save触发条件:
    - Bottom 2选手的评委分差距 > save_threshold
    - 高分者被挽救，低分者被淘汰
    """
    weeks = sorted(season_data['week'].unique())
    max_week = max(weeks)
    t = len(weeks) - 1
    
    final_data = season_data[season_data['week'] == max_week].copy()
    
    if len(final_data) < 3:
        return np.nan, np.nan
    
    # 动态权重
    w_j = min(base_weight + delta * t, 0.85)
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
    
    # Judges' Save 机制
    n = len(final_data)
    if n >= 2:
        bottom_2 = final_data[final_data['final_rank'] >= n - 1]
        if len(bottom_2) >= 2:
            j_scores = bottom_2['J_pct'].values
            if abs(j_scores[0] - j_scores[1]) > save_threshold:
                # 挽救评委分数高的选手
                high_j_idx = bottom_2['J_pct'].idxmax()
                final_data.loc[high_j_idx, 'final_rank'] -= 1
    
    j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
    f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
    
    return j_corr, f_corr


# =============================================================================
# 3. 规则空间搜索
# =============================================================================
print("\n[3] Searching rule space...")

all_points = []
seasons_list = estimates['season'].unique()

# 3.1 静态 Rank 规则
print("    [3.1] Static Rank rules (30%-90%)...")
for w in np.linspace(0.30, 0.90, 25):
    j_list, f_list = [], []
    for season in seasons_list:
        j, f = calculate_objectives_static(estimates[estimates['season'] == season], w, 'rank')
        if not np.isnan(j) and not np.isnan(f):
            j_list.append(j)
            f_list.append(f)
    if j_list:
        all_points.append({
            'rule': 'Static Rank',
            'params': f'w={w:.0%}',
            'judge_weight': w,
            'J': np.mean(j_list),
            'F': np.mean(f_list),
            'J_std': np.std(j_list),
            'F_std': np.std(f_list),
            'n_seasons': len(j_list)
        })

# 3.2 静态 Pct 规则
print("    [3.2] Static Pct rules (30%-90%)...")
for w in np.linspace(0.30, 0.90, 25):
    j_list, f_list = [], []
    for season in seasons_list:
        j, f = calculate_objectives_static(estimates[estimates['season'] == season], w, 'pct')
        if not np.isnan(j) and not np.isnan(f):
            j_list.append(j)
            f_list.append(f)
    if j_list:
        all_points.append({
            'rule': 'Static Pct',
            'params': f'w={w:.0%}',
            'judge_weight': w,
            'J': np.mean(j_list),
            'F': np.mean(f_list),
            'J_std': np.std(j_list),
            'F_std': np.std(f_list),
            'n_seasons': len(j_list)
        })

# 3.3 动态规则（无对数平滑）
print("    [3.3] Dynamic rules (no log smoothing)...")
for base in np.linspace(0.40, 0.55, 7):
    for delta in [0.01, 0.015, 0.02, 0.025, 0.03]:
        j_list, f_list = [], []
        for season in seasons_list:
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
                'F': np.mean(f_list),
                'J_std': np.std(j_list),
                'F_std': np.std(f_list),
                'n_seasons': len(j_list)
            })

# 3.4 动态对数加权规则
print("    [3.4] Dynamic + Log smoothing rules...")
for base in np.linspace(0.40, 0.55, 7):
    for delta in [0.01, 0.015, 0.02, 0.025]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
            j_list, f_list = [], []
            for season in seasons_list:
                j, f = calculate_objectives_dynamic(estimates[estimates['season'] == season],
                                                     base, delta, alpha)
                if not np.isnan(j) and not np.isnan(f):
                    j_list.append(j)
                    f_list.append(f)
            if j_list:
                all_points.append({
                    'rule': 'Dynamic+Log',
                    'params': f'b={base:.0%},δ={delta},α={alpha}',
                    'base_weight': base,
                    'delta': delta,
                    'log_strength': alpha,
                    'J': np.mean(j_list),
                    'F': np.mean(f_list),
                    'J_std': np.std(j_list),
                    'F_std': np.std(f_list),
                    'n_seasons': len(j_list)
                })

# 3.5 动态对数 + Judges' Save（推荐规则）
print("    [3.5] Recommended rules (Dynamic + Log + Judges' Save)...")
for base in np.linspace(0.40, 0.55, 7):
    for delta in [0.01, 0.015, 0.02, 0.025]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
            j_list, f_list = [], []
            for season in seasons_list:
                j, f = calculate_objectives_with_save(estimates[estimates['season'] == season],
                                                       base, delta, alpha)
                if not np.isnan(j) and not np.isnan(f):
                    j_list.append(j)
                    f_list.append(f)
            if j_list:
                all_points.append({
                    'rule': 'Recommended',
                    'params': f'b={base:.0%},δ={delta},α={alpha}+Save',
                    'base_weight': base,
                    'delta': delta,
                    'log_strength': alpha,
                    'J': np.mean(j_list),
                    'F': np.mean(f_list),
                    'J_std': np.std(j_list),
                    'F_std': np.std(f_list),
                    'n_seasons': len(j_list)
                })

# 转换为DataFrame
df = pd.DataFrame(all_points)

# 计算调和平均Balance
df['Balance'] = 2 * df['J'] * df['F'] / (df['J'] + df['F'] + 1e-9)

print(f"\n    Total rule configurations tested: {len(df)}")
print(f"    By rule type:")
for rule in df['rule'].unique():
    count = len(df[df['rule'] == rule])
    print(f"      - {rule}: {count}")

# =============================================================================
# 4. Pareto前沿分析
# =============================================================================
print("\n[4] Pareto frontier analysis...")

def find_pareto_frontier(points):
    """
    找Pareto前沿点（双目标最大化）
    """
    vals = points[['J', 'F']].values
    n = len(vals)
    is_optimal = np.ones(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # 如果j在两个目标上都不差于i，且至少一个严格更好
                if (vals[j, 0] >= vals[i, 0] and vals[j, 1] >= vals[i, 1] and
                    (vals[j, 0] > vals[i, 0] or vals[j, 1] > vals[i, 1])):
                    is_optimal[i] = False
                    break
    
    return points[is_optimal].copy()


def find_knee_point(frontier, method='distance'):
    """
    找Pareto前沿的膝点（最佳权衡点）
    
    方法:
    - 'distance': 到理想点和反理想点连线的最大距离
    - 'curvature': 最大曲率点
    """
    if len(frontier) < 3:
        return frontier.iloc[0] if len(frontier) > 0 else None
    
    j = frontier['J'].values
    f = frontier['F'].values
    
    # 归一化
    j_norm = (j - j.min()) / (j.max() - j.min() + 1e-9)
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-9)
    
    if method == 'distance':
        # 到 (0,1)-(1,0) 对角线的距离
        distances = np.abs(j_norm + f_norm - 1) / np.sqrt(2)
        knee_idx = np.argmax(distances)
    else:
        # 曲率方法
        dx = np.gradient(j_norm)
        dy = np.gradient(f_norm)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-9)**1.5
        knee_idx = np.argmax(curvature)
    
    return frontier.iloc[knee_idx]


# 按规则类型分组
static_rank = df[df['rule'] == 'Static Rank'].copy()
static_pct = df[df['rule'] == 'Static Pct'].copy()
dynamic = df[df['rule'] == 'Dynamic'].copy()
dynamic_log = df[df['rule'] == 'Dynamic+Log'].copy()
recommended = df[df['rule'] == 'Recommended'].copy()

# 计算各规则的Pareto前沿
rank_frontier = find_pareto_frontier(static_rank).sort_values('J')
pct_frontier = find_pareto_frontier(static_pct).sort_values('J')
dynamic_frontier = find_pareto_frontier(dynamic).sort_values('J')
dynamic_log_frontier = find_pareto_frontier(dynamic_log).sort_values('J')
recommended_frontier = find_pareto_frontier(recommended).sort_values('J')

# 全局Pareto前沿
global_frontier = find_pareto_frontier(df).sort_values('J')

print(f"\n    Pareto frontier sizes:")
print(f"      - Static Rank: {len(rank_frontier)} points")
print(f"      - Static Pct: {len(pct_frontier)} points")
print(f"      - Dynamic: {len(dynamic_frontier)} points")
print(f"      - Dynamic+Log: {len(dynamic_log_frontier)} points")
print(f"      - Recommended: {len(recommended_frontier)} points")
print(f"      - Global: {len(global_frontier)} points")

# =============================================================================
# 5. 关键点识别
# =============================================================================
print("\n[5] Identifying key points...")

# 当前规则 (Pct 50-50)
current_rule = static_pct[abs(static_pct['judge_weight'] - 0.5) < 0.02].iloc[0]

# Rank 50-50
rank_50 = static_rank[abs(static_rank['judge_weight'] - 0.5) < 0.02].iloc[0]

# Rank膝点
rank_knee = find_knee_point(rank_frontier)

# 推荐规则中的最优点
# 策略: 选择Balance最高且J > 当前规则的点
if len(recommended_frontier) > 0:
    good_points = recommended_frontier[
        (recommended_frontier['J'] > current_rule['J']) & 
        (recommended_frontier['F'] > 0.5)
    ]
    if len(good_points) > 0:
        best_recommended = good_points.loc[good_points['Balance'].idxmax()]
    else:
        best_recommended = recommended_frontier.loc[recommended_frontier['Balance'].idxmax()]
else:
    best_recommended = recommended.loc[recommended['Balance'].idxmax()]

# Dynamic+Log最优点
if len(dynamic_log_frontier) > 0:
    best_dynamic_log = dynamic_log_frontier.loc[dynamic_log_frontier['Balance'].idxmax()]
else:
    best_dynamic_log = dynamic_log.loc[dynamic_log['Balance'].idxmax()]

print(f"\n    Key Points Summary:")
print(f"    {'='*65}")
print(f"    {'Rule':<25} {'J':>8} {'F':>8} {'Balance':>10}")
print(f"    {'-'*65}")
print(f"    Current (Pct 50-50)    {current_rule['J']:>8.4f} {current_rule['F']:>8.4f} {current_rule['Balance']:>10.4f}")
print(f"    Rank 50-50             {rank_50['J']:>8.4f} {rank_50['F']:>8.4f} {rank_50['Balance']:>10.4f}")
print(f"    Rank Knee Point        {rank_knee['J']:>8.4f} {rank_knee['F']:>8.4f} {rank_knee['Balance']:>10.4f}")
print(f"    Best Dynamic+Log       {best_dynamic_log['J']:>8.4f} {best_dynamic_log['F']:>8.4f} {best_dynamic_log['Balance']:>10.4f}")
print(f"    Best Recommended       {best_recommended['J']:>8.4f} {best_recommended['F']:>8.4f} {best_recommended['Balance']:>10.4f}")
print(f"    {'='*65}")

# =============================================================================
# 6. 改进量计算
# =============================================================================
print("\n[6] Computing improvements vs current rule...")

def compute_improvement(new_rule, baseline):
    """计算相对改进"""
    j_change = (new_rule['J'] - baseline['J']) / abs(baseline['J']) * 100 if baseline['J'] != 0 else 0
    f_change = (new_rule['F'] - baseline['F']) / abs(baseline['F']) * 100 if baseline['F'] != 0 else 0
    bal_change = (new_rule['Balance'] - baseline['Balance']) / abs(baseline['Balance']) * 100 if baseline['Balance'] != 0 else 0
    return j_change, f_change, bal_change

improvements = {}
for name, rule in [('Rank 50-50', rank_50), 
                   ('Rank Knee', rank_knee),
                   ('Best Dynamic+Log', best_dynamic_log),
                   ('Best Recommended', best_recommended)]:
    j_c, f_c, b_c = compute_improvement(rule, current_rule)
    improvements[name] = {'J_change': j_c, 'F_change': f_c, 'Balance_change': b_c}
    print(f"    {name}: J {j_c:+.1f}%, F {f_c:+.1f}%, Balance {b_c:+.1f}%")

# =============================================================================
# 7. 敏感性分析
# =============================================================================
print("\n[7] Sensitivity analysis...")

# 分析参数对目标的敏感性
print("\n    [7.1] Parameter sensitivity for Dynamic+Log rules:")

# 固定其他参数，分析base_weight的影响
if 'base_weight' in dynamic_log.columns:
    base_sensitivity = dynamic_log.groupby('base_weight').agg({
        'J': ['mean', 'std'],
        'F': ['mean', 'std'],
        'Balance': 'mean'
    }).round(4)
    print("\n    Base Weight Sensitivity:")
    print(base_sensitivity.to_string())

# 分析delta的影响
if 'delta' in dynamic_log.columns:
    delta_sensitivity = dynamic_log.groupby('delta').agg({
        'J': ['mean', 'std'],
        'F': ['mean', 'std'],
        'Balance': 'mean'
    }).round(4)
    print("\n    Delta Sensitivity:")
    print(delta_sensitivity.to_string())

# 分析log_strength的影响
if 'log_strength' in dynamic_log.columns:
    alpha_sensitivity = dynamic_log.groupby('log_strength').agg({
        'J': ['mean', 'std'],
        'F': ['mean', 'std'],
        'Balance': 'mean'
    }).round(4)
    print("\n    Log Strength (α) Sensitivity:")
    print(alpha_sensitivity.to_string())

# =============================================================================
# 8. 可视化
# =============================================================================
print("\n[8] Generating visualizations...")

import os
output_dir = 'cleaned_outputs/phase3_pareto_analysis'
os.makedirs(output_dir, exist_ok=True)

# 图1: 主Pareto前沿图
fig1, ax1 = plt.subplots(figsize=(14, 10))
ax1.set_facecolor('#f8f9fa')

# 绘制各规则的点（背景）
ax1.scatter(static_pct['J'], static_pct['F'], c='#ef4444', s=30, alpha=0.3, marker='s', label='Static Pct')
ax1.scatter(static_rank['J'], static_rank['F'], c='#3b82f6', s=30, alpha=0.3, marker='o', label='Static Rank')
ax1.scatter(dynamic['J'], dynamic['F'], c='#f59e0b', s=40, alpha=0.4, marker='^', label='Dynamic')
ax1.scatter(dynamic_log['J'], dynamic_log['F'], c='#a855f7', s=50, alpha=0.5, marker='D', label='Dynamic+Log')
ax1.scatter(recommended['J'], recommended['F'], c='#10b981', s=60, alpha=0.5, marker='*', label='Recommended')

# 绘制Pareto前沿线
ax1.plot(rank_frontier['J'], rank_frontier['F'], 'o-', color='#1e40af', 
         linewidth=2.5, markersize=8, label='Rank Frontier', alpha=0.9)
ax1.plot(pct_frontier['J'], pct_frontier['F'], 's--', color='#b91c1c', 
         linewidth=2.5, markersize=8, label='Pct Frontier', alpha=0.9)

# 标记关键点
# 当前规则
ax1.scatter(current_rule['J'], current_rule['F'], c='#f97316', s=500, marker='h', 
            zorder=10, edgecolors='#c2410c', linewidth=3, label='Current (Pct 50-50)')
ax1.annotate('Current\n(Pct 50-50)', 
             xy=(current_rule['J'], current_rule['F']),
             xytext=(current_rule['J']-0.08, current_rule['F']+0.1),
             fontsize=11, ha='right', fontweight='bold', color='#c2410c',
             arrowprops=dict(arrowstyle='->', color='#c2410c', lw=2))

# Rank 50-50
ax1.scatter(rank_50['J'], rank_50['F'], c='#3b82f6', s=400, marker='p', 
            zorder=10, edgecolors='#1e40af', linewidth=3)
ax1.annotate('Rank 50-50', 
             xy=(rank_50['J'], rank_50['F']),
             xytext=(rank_50['J']+0.05, rank_50['F']+0.08),
             fontsize=11, ha='left', color='#1e40af',
             arrowprops=dict(arrowstyle='->', color='#1e40af', lw=2))

# 推荐规则
ax1.scatter(best_recommended['J'], best_recommended['F'], c='#10b981', s=600, marker='*', 
            zorder=10, edgecolors='#047857', linewidth=3)
ax1.annotate(f"★ Recommended\n{best_recommended['params']}", 
             xy=(best_recommended['J'], best_recommended['F']),
             xytext=(best_recommended['J']+0.05, best_recommended['F']-0.12),
             fontsize=10, ha='left', fontweight='bold', color='#047857',
             arrowprops=dict(arrowstyle='->', color='#047857', lw=2))

# 改进箭头
ax1.annotate('', 
             xy=(best_recommended['J'], best_recommended['F']),
             xytext=(current_rule['J'], current_rule['F']),
             arrowprops=dict(arrowstyle='-|>', color='#059669', lw=3,
                            connectionstyle='arc3,rad=0.15'))

# 理想区域标注
ax1.fill_between([0.6, 1.0], 0.6, 1.0, alpha=0.1, color='green', label='Ideal Region')

ax1.set_xlabel('J (Meritocracy - Judge Alignment)', fontsize=14, fontweight='bold')
ax1.set_ylabel('F (Engagement - Fan Alignment)', fontsize=14, fontweight='bold')
ax1.set_title('Pareto Frontier: Fairness-Engagement Trade-off\n(Phase 3 Analysis)', 
              fontsize=16, fontweight='bold')
ax1.legend(loc='lower left', fontsize=9, ncol=2, framealpha=0.95)
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig(f'{output_dir}/pareto_frontier_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {output_dir}/pareto_frontier_analysis.png")

# 图2: 动态权重演化
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图: 权重随周次变化
ax2a = axes[0]
base_w = best_recommended['base_weight'] if 'base_weight' in best_recommended else 0.45
delta_val = best_recommended['delta'] if 'delta' in best_recommended else 0.01

weeks = np.arange(1, 13)
judge_weights = np.minimum(base_w + delta_val * (weeks - 1), 0.85)
fan_weights = 1 - judge_weights

ax2a.fill_between(weeks, 0, judge_weights, alpha=0.7, color='#3b82f6', label='Judge Weight $w_J(t)$')
ax2a.fill_between(weeks, judge_weights, 1, alpha=0.7, color='#f97316', label='Fan Weight $w_F(t)$')
ax2a.plot(weeks, judge_weights, 'b-', linewidth=3)
ax2a.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% line')

formula = f'$w_J(t) = {base_w:.2f} + {delta_val}(t-1)$\n$w_F(t) = 1 - w_J(t)$'
ax2a.text(0.98, 0.95, formula, transform=ax2a.transAxes, fontsize=12,
         va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

ax2a.set_xlabel('Week (t)', fontsize=13, fontweight='bold')
ax2a.set_ylabel('Weight', fontsize=13, fontweight='bold')
ax2a.set_title('(a) Dynamic Weight Evolution', fontsize=14, fontweight='bold')
ax2a.legend(loc='center right', fontsize=10)
ax2a.set_xlim(1, 12)
ax2a.set_ylim(0, 1)
ax2a.set_xticks(range(1, 13))
ax2a.grid(True, alpha=0.3)

# 右图: 对数平滑效果
ax2b = axes[1]
f_raw = np.linspace(0.01, 1, 100)
f_linear = f_raw * 100

# 不同α值的对数平滑
for alpha in [0.1, 0.2, 0.3, 0.5]:
    f_log = np.log1p(f_raw * 100)
    f_log_norm = f_log / f_log.max() * 100
    f_transformed = alpha * f_log_norm + (1 - alpha) * f_linear
    ax2b.plot(f_linear, f_transformed, linewidth=2, label=f'α = {alpha}')

ax2b.plot(f_linear, f_linear, 'k--', linewidth=2, alpha=0.5, label='No smoothing (α=0)')

ax2b.set_xlabel('Original Fan Score F%', fontsize=13, fontweight='bold')
ax2b.set_ylabel('Transformed Fan Score', fontsize=13, fontweight='bold')
ax2b.set_title('(b) Log Smoothing Effect on Extreme Values', fontsize=14, fontweight='bold')
ax2b.legend(fontsize=10)
ax2b.grid(True, alpha=0.3)
ax2b.set_xlim(0, 100)
ax2b.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f'{output_dir}/dynamic_weighting_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {output_dir}/dynamic_weighting_analysis.png")

# 图3: 规则对比柱状图
fig3, ax3 = plt.subplots(figsize=(12, 6))

rules = ['Current\n(Pct 50-50)', 'Rank 50-50', 'Best\nDynamic+Log', 'Best\nRecommended']
j_vals = [current_rule['J'], rank_50['J'], best_dynamic_log['J'], best_recommended['J']]
f_vals = [current_rule['F'], rank_50['F'], best_dynamic_log['F'], best_recommended['F']]
balance_vals = [current_rule['Balance'], rank_50['Balance'], 
                best_dynamic_log['Balance'], best_recommended['Balance']]

x = np.arange(len(rules))
width = 0.25

bars1 = ax3.bar(x - width, j_vals, width, label='J (Meritocracy)', color='#3b82f6', alpha=0.8)
bars2 = ax3.bar(x, f_vals, width, label='F (Engagement)', color='#f97316', alpha=0.8)
bars3 = ax3.bar(x + width, balance_vals, width, label='Balance', color='#10b981', alpha=0.8)

ax3.set_xlabel('Voting Rule', fontsize=13, fontweight='bold')
ax3.set_ylabel('Spearman Correlation', fontsize=13, fontweight='bold')
ax3.set_title('Rule Comparison: J, F, and Balance Metrics', fontsize=15, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(rules, fontsize=11)
ax3.legend(fontsize=11)
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/rule_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {output_dir}/rule_comparison.png")

# 图4: 敏感性分析热力图
if 'base_weight' in dynamic_log.columns and 'delta' in dynamic_log.columns:
    fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 创建透视表
    for idx, (metric, title) in enumerate([('J', 'Meritocracy (J)'), 
                                            ('F', 'Engagement (F)'), 
                                            ('Balance', 'Balance')]):
        pivot = dynamic_log.pivot_table(values=metric, 
                                         index='base_weight', 
                                         columns='delta', 
                                         aggfunc='mean')
        im = axes[idx].imshow(pivot.values, cmap='RdYlGn', aspect='auto')
        axes[idx].set_xticks(range(len(pivot.columns)))
        axes[idx].set_xticklabels([f'{d}' for d in pivot.columns], rotation=45)
        axes[idx].set_yticks(range(len(pivot.index)))
        axes[idx].set_yticklabels([f'{b:.0%}' for b in pivot.index])
        axes[idx].set_xlabel('δ (weekly increment)', fontsize=11)
        axes[idx].set_ylabel('Base Weight', fontsize=11)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sensitivity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_dir}/sensitivity_heatmap.png")

# =============================================================================
# 9. 保存结果
# =============================================================================
print("\n[9] Saving results...")

# 保存所有点
df.to_csv(f'{output_dir}/all_pareto_points.csv', index=False)
print(f"    Saved: {output_dir}/all_pareto_points.csv")

# 保存Pareto前沿
global_frontier.to_csv(f'{output_dir}/global_pareto_frontier.csv', index=False)
print(f"    Saved: {output_dir}/global_pareto_frontier.csv")

# 保存推荐规则
recommended_rule = {
    'rule_name': 'Dynamic Log-Weighting with Judges\' Save',
    'formula': f'Score(t) = w_j(t)·J% + w_f(t)·[α·log(F%) + (1-α)·F%]',
    'formula_details': {
        'w_j(t)': f'{base_w:.2f} + {delta_val}·(t-1)',
        'w_f(t)': '1 - w_j(t)',
        'alpha': best_recommended.get('log_strength', 0.2),
        'judges_save_threshold': 10
    },
    'parameters': {
        'base_judge_weight': float(best_recommended.get('base_weight', base_w)),
        'delta_per_week': float(best_recommended.get('delta', delta_val)),
        'log_strength': float(best_recommended.get('log_strength', 0.2)),
        'judges_save': True
    },
    'performance': {
        'J': float(best_recommended['J']),
        'F': float(best_recommended['F']),
        'Balance': float(best_recommended['Balance'])
    },
    'vs_current_rule': {
        'current_rule': 'Pct 50-50',
        'J_change_pct': float(improvements['Best Recommended']['J_change']),
        'F_change_pct': float(improvements['Best Recommended']['F_change']),
        'Balance_change_pct': float(improvements['Best Recommended']['Balance_change'])
    },
    'interpretation': {
        'J': 'Spearman correlation between final ranking and judge ranking (higher = more meritocratic)',
        'F': 'Spearman correlation between final ranking and fan ranking (higher = more engagement)',
        'Balance': 'Harmonic mean of J and F (higher = better trade-off)'
    }
}

with open(f'{output_dir}/recommended_rule.json', 'w') as f:
    json.dump(recommended_rule, f, indent=2)
print(f"    Saved: {output_dir}/recommended_rule.json")

# 保存对比表
comparison_df = pd.DataFrame([
    {
        'Rule': 'Current (Pct 50-50)',
        'J': current_rule['J'],
        'F': current_rule['F'],
        'Balance': current_rule['Balance'],
        'J_change': 0,
        'F_change': 0,
        'Balance_change': 0
    },
    {
        'Rule': 'Rank 50-50',
        'J': rank_50['J'],
        'F': rank_50['F'],
        'Balance': rank_50['Balance'],
        **improvements['Rank 50-50']
    },
    {
        'Rule': 'Best Dynamic+Log',
        'J': best_dynamic_log['J'],
        'F': best_dynamic_log['F'],
        'Balance': best_dynamic_log['Balance'],
        **improvements['Best Dynamic+Log']
    },
    {
        'Rule': 'Best Recommended (with Save)',
        'J': best_recommended['J'],
        'F': best_recommended['F'],
        'Balance': best_recommended['Balance'],
        **improvements['Best Recommended']
    }
])
comparison_df.to_csv(f'{output_dir}/rule_comparison.csv', index=False)
print(f"    Saved: {output_dir}/rule_comparison.csv")

# =============================================================================
# 10. 总结报告
# =============================================================================
print("\n" + "=" * 75)
print("PHASE 3 ANALYSIS SUMMARY")
print("=" * 75)

print(f"""
1. OBJECTIVE FUNCTIONS DEFINED:
   - J (Meritocracy): Spearman correlation with judge ranking
   - F (Engagement): Spearman correlation with fan ranking  
   - Balance: Harmonic mean = 2JF/(J+F)

2. RULE SPACE SEARCHED:
   - Static Rank: {len(static_rank)} configurations
   - Static Pct: {len(static_pct)} configurations
   - Dynamic (no log): {len(dynamic)} configurations
   - Dynamic + Log: {len(dynamic_log)} configurations
   - Recommended (+ Save): {len(recommended)} configurations
   - Total: {len(df)} configurations

3. KEY FINDINGS:
   
   Current Rule (Pct 50-50):
   - J = {current_rule['J']:.4f}, F = {current_rule['F']:.4f}, Balance = {current_rule['Balance']:.4f}
   
   Best Recommended Rule:
   - Parameters: {best_recommended['params']}
   - J = {best_recommended['J']:.4f} ({improvements['Best Recommended']['J_change']:+.1f}% vs current)
   - F = {best_recommended['F']:.4f} ({improvements['Best Recommended']['F_change']:+.1f}% vs current)
   - Balance = {best_recommended['Balance']:.4f} ({improvements['Best Recommended']['Balance_change']:+.1f}% vs current)

4. RECOMMENDED RULE FORMULA:
   
   Score(t) = w_j(t)·J% + w_f(t)·[α·log(F%) + (1-α)·F%]
   
   Where:
   - w_j(t) = {base_w:.2f} + {delta_val}·(t-1)  (Judge weight increases each week)
   - w_f(t) = 1 - w_j(t)  (Fan weight decreases each week)
   - α = {best_recommended.get('log_strength', 0.2)}  (Log smoothing strength)
   - Judges' Save: Activates when Bottom-2 judge score gap > 10%

5. DESIGN RATIONALE:
   - Dynamic Weighting: Early rounds favor fans (engagement), later rounds favor judges (fairness)
   - Log Smoothing: Compresses extreme fan votes, reducing manipulation impact
   - Judges' Save: Protects high-skilled contestants from fan-driven eliminations

6. OUTPUT FILES:
   - {output_dir}/pareto_frontier_analysis.png
   - {output_dir}/dynamic_weighting_analysis.png
   - {output_dir}/rule_comparison.png
   - {output_dir}/sensitivity_heatmap.png
   - {output_dir}/all_pareto_points.csv
   - {output_dir}/global_pareto_frontier.csv
   - {output_dir}/recommended_rule.json
   - {output_dir}/rule_comparison.csv
""")

print("=" * 75)
print("Phase 3 Pareto Analysis Complete!")
print("=" * 75)
