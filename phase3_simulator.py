#!/usr/bin/env python3
"""
Phase 3: DWTS Simulator
=======================
比较Rank-based和Percentage-based两种聚合规则：
1. 模拟每个赛季在两种规则下的淘汰路径
2. 量化差异: 周差异、最终排名差异
3. 评估哪种方法更"偏向粉丝"

Author: MCM 2026 Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("PHASE 3: DWTS SIMULATOR")
print("=" * 70)

# 加载数据
print("\n[1] Loading data...")
estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')
panel = pd.read_csv('cleaned_outputs/clean_weekly_panel.csv')

print(f"    Fan vote estimates: {len(estimates)} rows")
print(f"    Seasons: {estimates['season'].min()}-{estimates['season'].max()}")

# =============================================================================
# CORE SIMULATION FUNCTIONS
# =============================================================================

def simulate_rank_method(season_data):
    """
    Rank-based method: Combined = 0.5 * J_rank + 0.5 * F_rank
    每周淘汰combined_rank最高（最差）的k人
    """
    results = {}
    remaining = season_data['celebrity_name'].unique().tolist()
    
    for week in sorted(season_data['week'].unique()):
        week_data = season_data[(season_data['week'] == week) & 
                                 (season_data['celebrity_name'].isin(remaining))]
        
        if len(week_data) == 0:
            continue
        
        # 实际淘汰人数
        n_eliminated = week_data['was_eliminated'].sum()
        
        if n_eliminated == 0:
            results[week] = {
                'eliminated': [],
                'method': 'rank',
                'n_contestants': len(remaining)
            }
            continue
        
        # 计算排名
        df = week_data.copy()
        df['J_rank'] = df['J_pct'].rank(ascending=False, method='average')
        df['F_rank'] = df['f_mean'].rank(ascending=False, method='average')
        df['combined_rank'] = 0.5 * df['J_rank'] + 0.5 * df['F_rank']
        
        # 淘汰combined_rank最高的k人
        df_sorted = df.sort_values('combined_rank', ascending=False)
        eliminated = df_sorted.head(n_eliminated)['celebrity_name'].tolist()
        
        results[week] = {
            'eliminated': eliminated,
            'method': 'rank',
            'n_contestants': len(remaining),
            'details': df[['celebrity_name', 'J_pct', 'f_mean', 'J_rank', 'F_rank', 'combined_rank']].to_dict('records')
        }
        
        # 更新剩余选手
        remaining = [c for c in remaining if c not in eliminated]
    
    return results, remaining

def simulate_pct_method(season_data):
    """
    Percentage-based method: Combined = 0.5 * J_pct + 0.5 * F_pct
    其中 F_pct = f(i,w) * 100 / max(f(i,w))  # 归一化到0-100
    每周淘汰combined_pct最低（最差）的k人
    """
    results = {}
    remaining = season_data['celebrity_name'].unique().tolist()
    
    for week in sorted(season_data['week'].unique()):
        week_data = season_data[(season_data['week'] == week) & 
                                 (season_data['celebrity_name'].isin(remaining))]
        
        if len(week_data) == 0:
            continue
        
        # 实际淘汰人数
        n_eliminated = week_data['was_eliminated'].sum()
        
        if n_eliminated == 0:
            results[week] = {
                'eliminated': [],
                'method': 'pct',
                'n_contestants': len(remaining)
            }
            continue
        
        # 计算百分比得分
        df = week_data.copy()
        # F_pct: 将f(i,w)归一化到0-100，与J_pct相同量纲
        max_f = df['f_mean'].max()
        if max_f > 0:
            df['F_pct'] = df['f_mean'] / max_f * 100
        else:
            df['F_pct'] = 100 / len(df)
        
        df['combined_pct'] = 0.5 * df['J_pct'] + 0.5 * df['F_pct']
        
        # 淘汰combined_pct最低的k人
        df_sorted = df.sort_values('combined_pct', ascending=True)
        eliminated = df_sorted.head(n_eliminated)['celebrity_name'].tolist()
        
        results[week] = {
            'eliminated': eliminated,
            'method': 'pct',
            'n_contestants': len(remaining),
            'details': df[['celebrity_name', 'J_pct', 'f_mean', 'F_pct', 'combined_pct']].to_dict('records')
        }
        
        # 更新剩余选手
        remaining = [c for c in remaining if c not in eliminated]
    
    return results, remaining

def compare_methods(rank_results, pct_results, rank_final, pct_final, actual_eliminated):
    """比较两种方法的结果"""
    comparison = {
        'weekly_diff': 0,
        'weeks_different': [],
        'final_diff': rank_final != pct_final,
        'rank_final': rank_final,
        'pct_final': pct_final
    }
    
    all_weeks = set(rank_results.keys()) | set(pct_results.keys())
    
    for week in sorted(all_weeks):
        if week not in rank_results or week not in pct_results:
            continue
        
        rank_elim = set(rank_results[week]['eliminated'])
        pct_elim = set(pct_results[week]['eliminated'])
        
        if rank_elim != pct_elim:
            comparison['weekly_diff'] += 1
            comparison['weeks_different'].append({
                'week': week,
                'rank_eliminated': list(rank_elim),
                'pct_eliminated': list(pct_elim),
                'actual_eliminated': [c for c in actual_eliminated if actual_eliminated.get(c) == week]
            })
    
    return comparison

# =============================================================================
# RUN SIMULATION FOR ALL SEASONS
# =============================================================================
print("\n[2] Running simulation for all seasons...")

all_comparisons = []

for season in sorted(estimates['season'].unique()):
    season_data = estimates[estimates['season'] == season].copy()
    
    # 获取实际淘汰信息
    actual_eliminated = {}
    for _, row in season_data[season_data['was_eliminated'] == True].iterrows():
        actual_eliminated[row['celebrity_name']] = row['week']
    
    # 模拟两种方法
    rank_results, rank_final = simulate_rank_method(season_data)
    pct_results, pct_final = simulate_pct_method(season_data)
    
    # 比较
    comparison = compare_methods(rank_results, pct_results, rank_final, pct_final, actual_eliminated)
    comparison['season'] = season
    comparison['n_weeks'] = len(rank_results)
    
    all_comparisons.append(comparison)
    
    # 打印每季结果
    if comparison['weekly_diff'] > 0:
        print(f"    Season {season}: {comparison['weekly_diff']} weeks differ "
              f"(Final: {comparison['final_diff']})")
    else:
        print(f"    Season {season}: Methods agree (No difference)")

# 转换为DataFrame
comparison_df = pd.DataFrame(all_comparisons)

# =============================================================================
# ANALYZE DIFFERENCES
# =============================================================================
print("\n[3] Analyzing differences...")

# 总体统计
total_seasons = len(comparison_df)
seasons_with_diff = (comparison_df['weekly_diff'] > 0).sum()
total_weekly_diff = comparison_df['weekly_diff'].sum()
total_weeks = comparison_df['n_weeks'].sum()
final_diff = comparison_df['final_diff'].sum()

print(f"\n    Total seasons: {total_seasons}")
print(f"    Seasons with differences: {seasons_with_diff} ({seasons_with_diff/total_seasons:.1%})")
print(f"    Total weekly differences: {total_weekly_diff}/{total_weeks} ({total_weekly_diff/total_weeks:.1%})")
print(f"    Finals changed: {final_diff}")

# 详细差异分析
print("\n    Detailed differences by season:")
for _, row in comparison_df[comparison_df['weekly_diff'] > 0].iterrows():
    print(f"\n    Season {row['season']}:")
    for diff in row['weeks_different']:
        print(f"      Week {diff['week']}:")
        print(f"        Rank method: {diff['rank_eliminated']}")
        print(f"        Pct method:  {diff['pct_eliminated']}")
        print(f"        Actual:      {diff['actual_eliminated']}")

# =============================================================================
# FAN FAVOR INDEX (FFI) CALCULATION
# =============================================================================
print("\n[4] Calculating Fan Favor Index (FFI)...")

def calculate_ffi(season_data, method='rank'):
    """
    FFI = Spearman correlation between final ranking and fan ranking
    Higher FFI = More fan-favoring
    """
    # 获取最后一周的数据来确定最终排名
    max_week = season_data['week'].max()
    final_week = season_data[season_data['week'] == max_week].copy()
    
    if len(final_week) < 2:
        return np.nan
    
    # Fan ranking (基于f_mean)
    final_week['fan_rank'] = final_week['f_mean'].rank(ascending=False)
    
    if method == 'rank':
        # Rank method: 最终排名基于combined_rank
        final_week['J_rank'] = final_week['J_pct'].rank(ascending=False)
        final_week['F_rank'] = final_week['f_mean'].rank(ascending=False)
        final_week['combined_rank'] = 0.5 * final_week['J_rank'] + 0.5 * final_week['F_rank']
        final_week['final_rank'] = final_week['combined_rank'].rank()
    else:
        # Pct method: 最终排名基于combined_pct
        max_f = final_week['f_mean'].max()
        final_week['F_pct'] = final_week['f_mean'] / max_f * 100 if max_f > 0 else 0
        final_week['combined_pct'] = 0.5 * final_week['J_pct'] + 0.5 * final_week['F_pct']
        final_week['final_rank'] = final_week['combined_pct'].rank(ascending=False)
    
    # Spearman correlation
    corr, _ = stats.spearmanr(final_week['final_rank'], final_week['fan_rank'])
    return corr

# 计算每季的FFI
ffi_results = []

for season in sorted(estimates['season'].unique()):
    season_data = estimates[estimates['season'] == season]
    
    ffi_rank = calculate_ffi(season_data, method='rank')
    ffi_pct = calculate_ffi(season_data, method='pct')
    
    ffi_results.append({
        'season': season,
        'FFI_rank': ffi_rank,
        'FFI_pct': ffi_pct,
        'FFI_diff': ffi_pct - ffi_rank if not (np.isnan(ffi_rank) or np.isnan(ffi_pct)) else np.nan
    })

ffi_df = pd.DataFrame(ffi_results)

print("\n    Fan Favor Index (FFI) by Season:")
print(f"    Average FFI (Rank method): {ffi_df['FFI_rank'].mean():.4f}")
print(f"    Average FFI (Pct method):  {ffi_df['FFI_pct'].mean():.4f}")
print(f"    Average difference (Pct - Rank): {ffi_df['FFI_diff'].mean():.4f}")

# 哪种方法更偏向粉丝
if ffi_df['FFI_diff'].mean() > 0:
    print(f"\n    Conclusion: Percentage method is more fan-favoring (FFI +{ffi_df['FFI_diff'].mean():.4f})")
else:
    print(f"\n    Conclusion: Rank method is more fan-favoring (FFI {ffi_df['FFI_diff'].mean():.4f})")

# =============================================================================
# JUDGE FAVOR INDEX (JFI) CALCULATION
# =============================================================================
print("\n[5] Calculating Judge Favor Index (JFI)...")

def calculate_jfi(season_data, method='rank'):
    """
    JFI = Spearman correlation between final ranking and judge ranking
    Higher JFI = More judge-favoring (meritocratic)
    """
    max_week = season_data['week'].max()
    final_week = season_data[season_data['week'] == max_week].copy()
    
    if len(final_week) < 2:
        return np.nan
    
    # Judge ranking (基于J_pct)
    final_week['judge_rank'] = final_week['J_pct'].rank(ascending=False)
    
    if method == 'rank':
        final_week['J_rank'] = final_week['J_pct'].rank(ascending=False)
        final_week['F_rank'] = final_week['f_mean'].rank(ascending=False)
        final_week['combined_rank'] = 0.5 * final_week['J_rank'] + 0.5 * final_week['F_rank']
        final_week['final_rank'] = final_week['combined_rank'].rank()
    else:
        max_f = final_week['f_mean'].max()
        final_week['F_pct'] = final_week['f_mean'] / max_f * 100 if max_f > 0 else 0
        final_week['combined_pct'] = 0.5 * final_week['J_pct'] + 0.5 * final_week['F_pct']
        final_week['final_rank'] = final_week['combined_pct'].rank(ascending=False)
    
    corr, _ = stats.spearmanr(final_week['final_rank'], final_week['judge_rank'])
    return corr

# 计算JFI
for idx, row in ffi_df.iterrows():
    season_data = estimates[estimates['season'] == row['season']]
    ffi_df.loc[idx, 'JFI_rank'] = calculate_jfi(season_data, 'rank')
    ffi_df.loc[idx, 'JFI_pct'] = calculate_jfi(season_data, 'pct')

ffi_df['JFI_diff'] = ffi_df['JFI_pct'] - ffi_df['JFI_rank']

print(f"\n    Average JFI (Rank method): {ffi_df['JFI_rank'].mean():.4f}")
print(f"    Average JFI (Pct method):  {ffi_df['JFI_pct'].mean():.4f}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n[6] Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 6.1 Weekly Differences by Season
ax1 = axes[0, 0]
seasons = comparison_df['season']
diffs = comparison_df['weekly_diff']
colors = ['red' if d > 0 else 'green' for d in diffs]
ax1.bar(seasons, diffs, color=colors, alpha=0.7)
ax1.set_xlabel('Season')
ax1.set_ylabel('Number of Different Weeks')
ax1.set_title('Weekly Elimination Differences\n(Rank vs Percentage)')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 6.2 FFI Comparison
ax2 = axes[0, 1]
x = ffi_df['season']
width = 0.35
ax2.bar(x - width/2, ffi_df['FFI_rank'], width, label='Rank Method', color='steelblue', alpha=0.7)
ax2.bar(x + width/2, ffi_df['FFI_pct'], width, label='Pct Method', color='coral', alpha=0.7)
ax2.set_xlabel('Season')
ax2.set_ylabel('Fan Favor Index (FFI)')
ax2.set_title('FFI Comparison by Season')
ax2.legend()
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

# 6.3 JFI Comparison
ax3 = axes[0, 2]
ax3.bar(x - width/2, ffi_df['JFI_rank'], width, label='Rank Method', color='steelblue', alpha=0.7)
ax3.bar(x + width/2, ffi_df['JFI_pct'], width, label='Pct Method', color='coral', alpha=0.7)
ax3.set_xlabel('Season')
ax3.set_ylabel('Judge Favor Index (JFI)')
ax3.set_title('JFI Comparison by Season')
ax3.legend()
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

# 6.4 FFI vs JFI Trade-off
ax4 = axes[1, 0]
# Rank method points
ax4.scatter(ffi_df['JFI_rank'], ffi_df['FFI_rank'], c='steelblue', 
            label='Rank Method', alpha=0.6, s=60)
# Pct method points
ax4.scatter(ffi_df['JFI_pct'], ffi_df['FFI_pct'], c='coral',
            label='Pct Method', alpha=0.6, s=60, marker='s')
# Means
ax4.scatter(ffi_df['JFI_rank'].mean(), ffi_df['FFI_rank'].mean(), 
            c='darkblue', s=200, marker='*', label='Rank Mean', zorder=5)
ax4.scatter(ffi_df['JFI_pct'].mean(), ffi_df['FFI_pct'].mean(),
            c='darkred', s=200, marker='*', label='Pct Mean', zorder=5)
ax4.set_xlabel('Judge Favor Index (JFI)')
ax4.set_ylabel('Fan Favor Index (FFI)')
ax4.set_title('Trade-off: JFI vs FFI')
ax4.legend(loc='lower left')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# 6.5 Difference Distribution
ax5 = axes[1, 1]
ax5.hist(ffi_df['FFI_diff'].dropna(), bins=15, color='mediumpurple', alpha=0.7, 
         edgecolor='black', label='FFI Diff (Pct - Rank)')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax5.axvline(x=ffi_df['FFI_diff'].mean(), color='green', linestyle='--', linewidth=2,
            label=f'Mean = {ffi_df["FFI_diff"].mean():.3f}')
ax5.set_xlabel('FFI Difference (Pct - Rank)')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of FFI Difference')
ax5.legend()

# 6.6 Era Analysis
ax6 = axes[1, 2]
ffi_df['era'] = ffi_df['season'].apply(
    lambda s: 'Early\n(S1-10)' if s <= 10 else ('Middle\n(S11-20)' if s <= 20 else 
              ('Late\n(S21-27)' if s <= 27 else 'TikTok\n(S28+)'))
)

era_ffi = ffi_df.groupby('era').agg({
    'FFI_rank': 'mean',
    'FFI_pct': 'mean',
    'JFI_rank': 'mean',
    'JFI_pct': 'mean'
}).round(4)

era_order = ['Early\n(S1-10)', 'Middle\n(S11-20)', 'Late\n(S21-27)', 'TikTok\n(S28+)']
era_ffi = era_ffi.reindex(era_order)

x = np.arange(len(era_order))
ax6.bar(x - 0.2, era_ffi['FFI_rank'], 0.2, label='FFI Rank', color='steelblue')
ax6.bar(x, era_ffi['FFI_pct'], 0.2, label='FFI Pct', color='coral')
ax6.bar(x + 0.2, era_ffi['JFI_rank'], 0.2, label='JFI Rank', color='lightblue', hatch='//')
ax6.bar(x + 0.4, era_ffi['JFI_pct'], 0.2, label='JFI Pct', color='lightsalmon', hatch='//')
ax6.set_xticks(x + 0.1)
ax6.set_xticklabels(era_order)
ax6.set_ylabel('Index Value')
ax6.set_title('FFI & JFI by Era')
ax6.legend(loc='lower right')

plt.tight_layout()
plt.savefig('cleaned_outputs/simulator_comparison.png', dpi=150, bbox_inches='tight')
print("    Saved: simulator_comparison.png")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n[7] Saving results...")

# 保存比较结果
comparison_df.to_csv('cleaned_outputs/method_comparison.csv', index=False)
print(f"    Saved: method_comparison.csv ({len(comparison_df)} rows)")

# 保存FFI/JFI结果
ffi_df.to_csv('cleaned_outputs/favor_indices.csv', index=False)
print(f"    Saved: favor_indices.csv ({len(ffi_df)} rows)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 3 SIMULATOR SUMMARY")
print("=" * 70)

print(f"""
METHOD COMPARISON RESULTS:
==========================
• Total seasons analyzed: {total_seasons}
• Seasons with different eliminations: {seasons_with_diff} ({seasons_with_diff/total_seasons:.1%})
• Total weekly differences: {total_weekly_diff}/{total_weeks} ({total_weekly_diff/total_weeks:.1%})
• Finals affected: {final_diff}

FAN FAVOR INDEX (FFI):
======================
• FFI measures how closely final ranking aligns with fan preferences
• Higher FFI = More fan-favoring

  Method          Mean FFI    Std
  ----------------------------------
  Rank Method     {ffi_df['FFI_rank'].mean():.4f}      {ffi_df['FFI_rank'].std():.4f}
  Pct Method      {ffi_df['FFI_pct'].mean():.4f}      {ffi_df['FFI_pct'].std():.4f}

JUDGE FAVOR INDEX (JFI):
========================
• JFI measures how closely final ranking aligns with judge scores
• Higher JFI = More meritocratic

  Method          Mean JFI    Std
  ----------------------------------
  Rank Method     {ffi_df['JFI_rank'].mean():.4f}      {ffi_df['JFI_rank'].std():.4f}
  Pct Method      {ffi_df['JFI_pct'].mean():.4f}      {ffi_df['JFI_pct'].std():.4f}

KEY FINDINGS:
=============
1. The two methods produce different eliminations in {seasons_with_diff} out of {total_seasons} seasons
2. Percentage method has {"higher" if ffi_df['FFI_pct'].mean() > ffi_df['FFI_rank'].mean() else "lower"} FFI → {"More" if ffi_df['FFI_pct'].mean() > ffi_df['FFI_rank'].mean() else "Less"} fan-favoring
3. Rank method has {"higher" if ffi_df['JFI_rank'].mean() > ffi_df['JFI_pct'].mean() else "lower"} JFI → {"More" if ffi_df['JFI_rank'].mean() > ffi_df['JFI_pct'].mean() else "Less"} meritocratic

CONCLUSION:
===========
{'RANK METHOD is recommended: It is more meritocratic (higher JFI) while maintaining reasonable fan engagement.' if ffi_df['JFI_rank'].mean() > ffi_df['JFI_pct'].mean() else 'PERCENTAGE METHOD is recommended: Better balance between merit and engagement.'}

FILES SAVED:
============
• method_comparison.csv - Season-level comparison
• favor_indices.csv - FFI and JFI by season
• simulator_comparison.png - Visualizations
""")

print("=" * 70)
print("PHASE 3 SIMULATOR COMPLETE!")
print("=" * 70)
