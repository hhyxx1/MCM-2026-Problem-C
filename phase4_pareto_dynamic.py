#!/usr/bin/env python3
"""
Phase 4 Enhanced: Pareto Optimization with Dynamic Season Weighting
====================================================================
在帕累托分析中引入动态加权，反映不同赛季的重要性差异

动态加权公式:
    w_s = exp(-λ(S_max - s)) × (n_s / max(n))
    
其中:
- s = 赛季编号
- λ = 时序衰减系数 (0.05-0.1)
- n_s = 该赛季的样本量

可视化设计:
1. 加权vs非加权帕累托前沿对比
2. 赛季权重分布图
3. 分时期帕累托轨迹
4. 权重敏感性分析

Author: MCM 2026 Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("PHASE 4 ENHANCED: PARETO WITH DYNAMIC WEIGHTING")
print("=" * 70)

# =============================================================================
# PART 1: LOAD DATA AND COMPUTE SEASON WEIGHTS
# =============================================================================
print("\n[1] Loading data and computing season weights...")

estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')

# 计算每个赛季的统计信息
season_stats = estimates.groupby('season').agg({
    'contestant_id': 'count',  # 样本量
    'week': 'max',  # 最大周数
    'J_pct': 'std',  # J分数标准差（数据质量指标）
    'f_mean': 'std'  # F分数标准差
}).reset_index()
season_stats.columns = ['season', 'n_samples', 'max_week', 'J_std', 'F_std']

# 动态加权参数
LAMBDA = 0.08  # 时序衰减系数
S_max = season_stats['season'].max()
n_max = season_stats['n_samples'].max()

# 计算综合权重
season_stats['time_weight'] = np.exp(-LAMBDA * (S_max - season_stats['season']))
season_stats['sample_weight'] = season_stats['n_samples'] / n_max
season_stats['combined_weight'] = season_stats['time_weight'] * season_stats['sample_weight']

# 归一化权重
season_stats['normalized_weight'] = season_stats['combined_weight'] / season_stats['combined_weight'].sum()

print(f"    Total seasons: {len(season_stats)}")
print(f"    Season range: {season_stats['season'].min()} - {S_max}")
print(f"    Decay parameter λ: {LAMBDA}")

# 创建权重字典
season_weights = dict(zip(season_stats['season'], season_stats['normalized_weight']))

# =============================================================================
# PART 2: OBJECTIVE FUNCTIONS WITH WEIGHTING
# =============================================================================

def calculate_objectives(season_data, judge_weight=0.5, method='rank'):
    """计算单个赛季的J和F目标值"""
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

def calculate_weighted_objectives(estimates_df, judge_weight, method, season_weights_dict):
    """计算加权后的整体目标值"""
    j_values = []
    f_values = []
    weights = []
    seasons = []
    
    for season in estimates_df['season'].unique():
        season_data = estimates_df[estimates_df['season'] == season]
        j, f = calculate_objectives(season_data, judge_weight=judge_weight, method=method)
        
        if not np.isnan(j) and not np.isnan(f):
            w = season_weights_dict.get(season, 1.0 / len(estimates_df['season'].unique()))
            j_values.append(j)
            f_values.append(f)
            weights.append(w)
            seasons.append(season)
    
    if len(j_values) == 0:
        return np.nan, np.nan, [], [], []
    
    # 归一化权重
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # 加权平均
    J_weighted = np.average(j_values, weights=weights)
    F_weighted = np.average(f_values, weights=weights)
    
    return J_weighted, F_weighted, j_values, f_values, seasons

def calculate_unweighted_objectives(estimates_df, judge_weight, method):
    """计算非加权（简单平均）的目标值"""
    j_scores = []
    f_scores = []
    
    for season in estimates_df['season'].unique():
        season_data = estimates_df[estimates_df['season'] == season]
        j, f = calculate_objectives(season_data, judge_weight=judge_weight, method=method)
        if not np.isnan(j) and not np.isnan(f):
            j_scores.append(j)
            f_scores.append(f)
    
    if len(j_scores) == 0:
        return np.nan, np.nan
    
    return np.mean(j_scores), np.mean(f_scores)

# =============================================================================
# PART 3: COMPUTE PARETO FRONTIERS (WEIGHTED VS UNWEIGHTED)
# =============================================================================
print("\n[2] Computing Pareto frontiers...")

weights_range = np.linspace(0.3, 0.9, 25)

# 存储结果
weighted_points = []
unweighted_points = []

for w in weights_range:
    # Rank method - Weighted
    J_w, F_w, _, _, _ = calculate_weighted_objectives(estimates, w, 'rank', season_weights)
    if not np.isnan(J_w):
        weighted_points.append({
            'method': 'rank', 'judge_weight': w, 'J': J_w, 'F': F_w, 'weighted': True
        })
    
    # Rank method - Unweighted
    J_u, F_u = calculate_unweighted_objectives(estimates, w, 'rank')
    if not np.isnan(J_u):
        unweighted_points.append({
            'method': 'rank', 'judge_weight': w, 'J': J_u, 'F': F_u, 'weighted': False
        })

weighted_df = pd.DataFrame(weighted_points)
unweighted_df = pd.DataFrame(unweighted_points)

print(f"    Weighted points: {len(weighted_df)}")
print(f"    Unweighted points: {len(unweighted_df)}")

# =============================================================================
# PART 4: PER-SEASON ANALYSIS FOR TRAJECTORY PLOT
# =============================================================================
print("\n[3] Computing per-season trajectories...")

# 将赛季分为三个时期
early_seasons = list(range(1, 12))      # S1-S11 (早期)
mid_seasons = list(range(12, 23))       # S12-S22 (中期)  
recent_seasons = list(range(23, 35))    # S23-S34 (近期)

period_names = ['Early (S1-11)', 'Middle (S12-22)', 'Recent (S23-34)']
period_seasons = [early_seasons, mid_seasons, recent_seasons]
period_colors = ['#94a3b8', '#60a5fa', '#f97316']  # 灰蓝、蓝、橙

period_frontiers = {}

for period_name, seasons, color in zip(period_names, period_seasons, period_colors):
    period_data = estimates[estimates['season'].isin(seasons)]
    
    if len(period_data) == 0:
        continue
    
    points = []
    for w in weights_range:
        j_scores = []
        f_scores = []
        
        for season in period_data['season'].unique():
            season_data = period_data[period_data['season'] == season]
            j, f = calculate_objectives(season_data, judge_weight=w, method='rank')
            if not np.isnan(j) and not np.isnan(f):
                j_scores.append(j)
                f_scores.append(f)
        
        if len(j_scores) > 0:
            points.append({
                'judge_weight': w,
                'J': np.mean(j_scores),
                'F': np.mean(f_scores)
            })
    
    if len(points) > 0:
        period_frontiers[period_name] = pd.DataFrame(points)

print(f"    Computed frontiers for {len(period_frontiers)} periods")

# =============================================================================
# PART 5: FIND KNEE POINTS
# =============================================================================
print("\n[4] Finding knee points...")

def find_knee_point(df):
    """找到帕累托前沿的膝点"""
    if len(df) < 3:
        return df.iloc[0] if len(df) > 0 else None
    
    j_vals = df['J'].values
    f_vals = df['F'].values
    
    j_norm = (j_vals - j_vals.min()) / (j_vals.max() - j_vals.min() + 1e-6)
    f_norm = (f_vals - f_vals.min()) / (f_vals.max() - f_vals.min() + 1e-6)
    
    distances = np.abs(j_norm + f_norm - 1) / np.sqrt(2)
    knee_idx = np.argmax(distances)
    
    return df.iloc[knee_idx]

knee_weighted = find_knee_point(weighted_df)
knee_unweighted = find_knee_point(unweighted_df)

print(f"\n    Weighted Knee Point:")
print(f"      Judge Weight: {knee_weighted['judge_weight']:.1%}")
print(f"      J: {knee_weighted['J']:.4f}, F: {knee_weighted['F']:.4f}")

print(f"\n    Unweighted Knee Point:")
print(f"      Judge Weight: {knee_unweighted['judge_weight']:.1%}")
print(f"      J: {knee_unweighted['J']:.4f}, F: {knee_unweighted['F']:.4f}")

# =============================================================================
# PART 6: VISUALIZATIONS
# =============================================================================
print("\n[5] Generating visualizations...")

import os
img_dir = 'cleaned_outputs/phase4_pareto'
os.makedirs(img_dir, exist_ok=True)

# 创建4合1综合图
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('white')

# 调整子图布局
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)

# =============================================================================
# SUBPLOT 1: 加权vs非加权帕累托前沿对比
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#fafafa')

# 绘制两条前沿
ax1.plot(unweighted_df['J'], unweighted_df['F'], 
         'o-', color='#94a3b8', linewidth=2, markersize=5,
         label='Unweighted (Simple Average)', alpha=0.7)
ax1.plot(weighted_df['J'], weighted_df['F'], 
         's-', color='#3b82f6', linewidth=2.5, markersize=6,
         label='Weighted (Dynamic)', alpha=0.9)

# 标记膝点
ax1.scatter(knee_unweighted['J'], knee_unweighted['F'],
            c='#94a3b8', s=200, marker='*', zorder=10,
            edgecolors='#475569', linewidth=1.5,
            label=f"Unweighted Knee ({knee_unweighted['judge_weight']:.0%})")
ax1.scatter(knee_weighted['J'], knee_weighted['F'],
            c='#f97316', s=250, marker='*', zorder=10,
            edgecolors='#c2410c', linewidth=2,
            label=f"Weighted Knee ({knee_weighted['judge_weight']:.0%})")

# 添加箭头显示加权效应
mid_idx = len(weighted_df) // 2
if mid_idx < len(unweighted_df):
    ax1.annotate('', 
                 xy=(weighted_df.iloc[mid_idx]['J'], weighted_df.iloc[mid_idx]['F']),
                 xytext=(unweighted_df.iloc[mid_idx]['J'], unweighted_df.iloc[mid_idx]['F']),
                 arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2))
    ax1.text((weighted_df.iloc[mid_idx]['J'] + unweighted_df.iloc[mid_idx]['J'])/2 + 0.02,
             (weighted_df.iloc[mid_idx]['F'] + unweighted_df.iloc[mid_idx]['F'])/2,
             'Weighting\nEffect', fontsize=9, color='#dc2626', fontweight='bold',
             ha='left', va='center')

ax1.set_xlabel('J (Meritocracy)', fontsize=12, fontweight='bold')
ax1.set_ylabel('F (Engagement)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Weighted vs Unweighted Pareto Frontier', fontsize=14, fontweight='bold', pad=10)
ax1.legend(loc='lower left', fontsize=9, framealpha=0.95)
ax1.grid(True, alpha=0.4, linestyle='--')

# =============================================================================
# SUBPLOT 2: 赛季权重分布（双轴柱状图）
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# 创建颜色渐变
n_seasons = len(season_stats)
colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_seasons))

# 绘制权重柱状图
bars = ax2.bar(season_stats['season'], season_stats['normalized_weight'] * 100, 
               color=colors, edgecolor='#1e40af', linewidth=0.5, alpha=0.8)

# 标注三个时期
for period_name, seasons, color in zip(period_names, period_seasons, period_colors):
    valid_seasons = [s for s in seasons if s <= S_max]
    if valid_seasons:
        ax2.axvspan(min(valid_seasons)-0.5, max(valid_seasons)+0.5, 
                    alpha=0.15, color=color, zorder=0)
        mid_s = (min(valid_seasons) + max(valid_seasons)) / 2
        ax2.text(mid_s, ax2.get_ylim()[1] * 0.92, period_name.split()[0], 
                 ha='center', fontsize=10, color=color, fontweight='bold')

# 在右轴显示累积权重
ax2_twin = ax2.twinx()
cumulative = np.cumsum(season_stats['normalized_weight'].values) * 100
ax2_twin.plot(season_stats['season'], cumulative, 
              'r-o', linewidth=2, markersize=4, label='Cumulative %')
ax2_twin.set_ylabel('Cumulative Weight (%)', fontsize=11, color='#dc2626')
ax2_twin.tick_params(axis='y', labelcolor='#dc2626')
ax2_twin.set_ylim(0, 105)

ax2.set_xlabel('Season', fontsize=12, fontweight='bold')
ax2.set_ylabel('Season Weight (%)', fontsize=11, color='#1e40af')
ax2.tick_params(axis='y', labelcolor='#1e40af')
ax2.set_title('(b) Dynamic Season Weights Distribution', fontsize=14, fontweight='bold', pad=10)

# 添加公式标注
formula_box = ax2.text(0.98, 0.02, 
                       r'$w_s = e^{-\lambda(S_{max}-s)} \times \frac{n_s}{\max(n)}$' + f'\n$\\lambda={LAMBDA}$',
                       transform=ax2.transAxes, fontsize=10,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# =============================================================================
# SUBPLOT 3: 分时期帕累托轨迹
# =============================================================================
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#fafafa')

# 绘制各时期的前沿
line_styles = ['-', '--', '-']
markers = ['o', 's', '^']

for idx, (period_name, color, ls, mk) in enumerate(zip(period_names, period_colors, line_styles, markers)):
    if period_name in period_frontiers:
        df = period_frontiers[period_name]
        ax3.plot(df['J'], df['F'], 
                 linestyle=ls, color=color, linewidth=2.5, 
                 marker=mk, markersize=6, alpha=0.85,
                 label=period_name)

# 添加时期演化箭头
if len(period_frontiers) >= 2:
    periods_list = list(period_frontiers.keys())
    for i in range(len(periods_list) - 1):
        df1 = period_frontiers[periods_list[i]]
        df2 = period_frontiers[periods_list[i+1]]
        
        # 在50%权重点之间画箭头
        mid1 = df1[df1['judge_weight'].between(0.48, 0.52)].iloc[0] if len(df1[df1['judge_weight'].between(0.48, 0.52)]) > 0 else df1.iloc[len(df1)//2]
        mid2 = df2[df2['judge_weight'].between(0.48, 0.52)].iloc[0] if len(df2[df2['judge_weight'].between(0.48, 0.52)]) > 0 else df2.iloc[len(df2)//2]
        
        ax3.annotate('', xy=(mid2['J'], mid2['F']), xytext=(mid1['J'], mid1['F']),
                     arrowprops=dict(arrowstyle='->', color='#6b7280', 
                                     lw=1.5, ls='--', mutation_scale=15))

# 添加"Evolution"标注
ax3.text(0.5, 0.95, '← Evolution →', transform=ax3.transAxes,
         fontsize=11, ha='center', color='#6b7280', fontstyle='italic')

ax3.set_xlabel('J (Meritocracy)', fontsize=12, fontweight='bold')
ax3.set_ylabel('F (Engagement)', fontsize=12, fontweight='bold')
ax3.set_title('(c) Pareto Frontier Evolution by Period', fontsize=14, fontweight='bold', pad=10)
ax3.legend(loc='lower left', fontsize=10, framealpha=0.95)
ax3.grid(True, alpha=0.4, linestyle='--')

# =============================================================================
# SUBPLOT 4: 权重敏感性热力图
# =============================================================================
ax4 = fig.add_subplot(gs[1, 1])

# 测试不同λ值的影响
lambda_values = [0.02, 0.05, 0.08, 0.12, 0.15]
judge_weights_test = [0.4, 0.5, 0.6, 0.7, 0.8]

heatmap_data_J = np.zeros((len(lambda_values), len(judge_weights_test)))
heatmap_data_F = np.zeros((len(lambda_values), len(judge_weights_test)))

for i, lam in enumerate(lambda_values):
    # 重新计算权重
    temp_weights = {}
    for _, row in season_stats.iterrows():
        s = row['season']
        time_w = np.exp(-lam * (S_max - s))
        sample_w = row['n_samples'] / n_max
        temp_weights[s] = time_w * sample_w
    
    # 归一化
    total_w = sum(temp_weights.values())
    temp_weights = {k: v/total_w for k, v in temp_weights.items()}
    
    for j, jw in enumerate(judge_weights_test):
        J_w, F_w, _, _, _ = calculate_weighted_objectives(estimates, jw, 'rank', temp_weights)
        heatmap_data_J[i, j] = J_w if not np.isnan(J_w) else 0
        heatmap_data_F[i, j] = F_w if not np.isnan(F_w) else 0

# 计算J-F平衡指标 (J和F的调和平均)
balance_data = 2 * heatmap_data_J * heatmap_data_F / (heatmap_data_J + heatmap_data_F + 1e-6)

# 绘制热力图
im = ax4.imshow(balance_data, cmap='RdYlGn', aspect='auto', 
                vmin=balance_data.min(), vmax=balance_data.max())

# 设置刻度
ax4.set_xticks(range(len(judge_weights_test)))
ax4.set_xticklabels([f'{w:.0%}' for w in judge_weights_test])
ax4.set_yticks(range(len(lambda_values)))
ax4.set_yticklabels([f'λ={l}' for l in lambda_values])

# 添加数值标注
for i in range(len(lambda_values)):
    for j in range(len(judge_weights_test)):
        text = ax4.text(j, i, f'{balance_data[i, j]:.3f}',
                       ha='center', va='center', color='black', fontsize=9)

# 标记最优点
max_idx = np.unravel_index(np.argmax(balance_data), balance_data.shape)
ax4.add_patch(plt.Rectangle((max_idx[1]-0.5, max_idx[0]-0.5), 1, 1, 
                              fill=False, edgecolor='#dc2626', linewidth=3))
ax4.text(max_idx[1], max_idx[0]-0.35, '★ Optimal', 
         ha='center', va='bottom', color='#dc2626', fontsize=9, fontweight='bold')

ax4.set_xlabel('Judge Weight', fontsize=12, fontweight='bold')
ax4.set_ylabel('Decay Parameter', fontsize=12, fontweight='bold')
ax4.set_title('(d) Sensitivity: Balance Score (Harmonic Mean of J & F)', 
              fontsize=14, fontweight='bold', pad=10)

# 添加colorbar
cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
cbar.set_label('Balance Score', fontsize=10)

# 总标题
fig.suptitle('Pareto Optimization with Dynamic Season Weighting', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{img_dir}/pareto_dynamic_weighting.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"    Saved: {img_dir}/pareto_dynamic_weighting.png")

# =============================================================================
# PART 7: SAVE ENHANCED RESULTS
# =============================================================================
print("\n[6] Saving enhanced results...")

# 保存权重信息
season_stats.to_csv(f'{img_dir}/season_weights.csv', index=False)
print(f"    Saved: season_weights.csv")

# 保存加权帕累托点
weighted_df.to_csv(f'{img_dir}/weighted_pareto_points.csv', index=False)
print(f"    Saved: weighted_pareto_points.csv")

# 保存对比结果
comparison = pd.DataFrame({
    'Type': ['Unweighted', 'Weighted'],
    'Knee_Judge_Weight': [knee_unweighted['judge_weight'], knee_weighted['judge_weight']],
    'Knee_J': [knee_unweighted['J'], knee_weighted['J']],
    'Knee_F': [knee_unweighted['F'], knee_weighted['F']],
    'Balance_Score': [2*knee_unweighted['J']*knee_unweighted['F']/(knee_unweighted['J']+knee_unweighted['F']),
                      2*knee_weighted['J']*knee_weighted['F']/(knee_weighted['J']+knee_weighted['F'])]
})
comparison.to_csv(f'{img_dir}/weighting_comparison.csv', index=False)
print(f"    Saved: weighting_comparison.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("DYNAMIC WEIGHTING ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
DYNAMIC WEIGHTING FORMULA:
==========================
    w_s = exp(-λ(S_max - s)) × (n_s / max(n))
    
    Parameters:
    - λ (decay): {LAMBDA}
    - S_max: {S_max}
    - Weight range: {season_stats['normalized_weight'].min():.4f} - {season_stats['normalized_weight'].max():.4f}

WEIGHTING EFFECT:
=================
    Recent seasons (S{S_max-5}-S{S_max}) account for {season_stats[season_stats['season'] >= S_max-5]['normalized_weight'].sum()*100:.1f}% of total weight
    Early seasons (S1-S10) account for {season_stats[season_stats['season'] <= 10]['normalized_weight'].sum()*100:.1f}% of total weight

KNEE POINT COMPARISON:
======================
┌──────────────────┬────────────────┬──────────┬──────────┬──────────────┐
│ Method           │ Judge Weight   │ J        │ F        │ Balance      │
├──────────────────┼────────────────┼──────────┼──────────┼──────────────┤
│ Unweighted       │ {knee_unweighted['judge_weight']:.0%}           │ {knee_unweighted['J']:.4f}   │ {knee_unweighted['F']:.4f}   │ {2*knee_unweighted['J']*knee_unweighted['F']/(knee_unweighted['J']+knee_unweighted['F']):.4f}       │
│ Weighted         │ {knee_weighted['judge_weight']:.0%}           │ {knee_weighted['J']:.4f}   │ {knee_weighted['F']:.4f}   │ {2*knee_weighted['J']*knee_weighted['F']/(knee_weighted['J']+knee_weighted['F']):.4f}       │
└──────────────────┴────────────────┴──────────┴──────────┴──────────────┘

PERIOD EVOLUTION:
=================
""")

for period_name in period_frontiers:
    df = period_frontiers[period_name]
    mid_point = df[df['judge_weight'].between(0.48, 0.52)].iloc[0] if len(df[df['judge_weight'].between(0.48, 0.52)]) > 0 else df.iloc[len(df)//2]
    print(f"    {period_name}: J={mid_point['J']:.4f}, F={mid_point['F']:.4f}")

print(f"""
FILES SAVED:
============
• pareto_dynamic_weighting.png - 4-panel visualization
• season_weights.csv - Per-season weight breakdown
• weighted_pareto_points.csv - Weighted Pareto frontier
• weighting_comparison.csv - Weighted vs Unweighted comparison
""")

print("=" * 70)
print("DYNAMIC WEIGHTING ANALYSIS COMPLETE!")
print("=" * 70)
