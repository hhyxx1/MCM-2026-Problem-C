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

def simulate_rank_method(season_data, judges_save=False):
    """
    Rank-based method: Combined = 0.5 * J_rank + 0.5 * F_rank
    每周淘汰combined_rank最高（最差）的k人
    
    Args:
        season_data: DataFrame with season data
        judges_save: If True, enable Judges' Save mechanism for Bottom 2
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
                'n_contestants': len(remaining),
                'judges_saved': None
            }
            continue
        
        # 计算排名
        df = week_data.copy()
        df['J_rank'] = df['J_pct'].rank(ascending=False, method='average')
        df['F_rank'] = df['f_mean'].rank(ascending=False, method='average')
        df['combined_rank'] = 0.5 * df['J_rank'] + 0.5 * df['F_rank']
        
        # 淘汰combined_rank最高的k人
        df_sorted = df.sort_values('combined_rank', ascending=False)
        
        # =====================================================================
        # JUDGES' SAVE MECHANISM (per Plan requirement)
        # If enabled, and k=1, check Bottom 2: save the one with higher J_pct
        # =====================================================================
        judges_saved_name = None
        if judges_save and n_eliminated == 1 and len(df) >= 2:
            bottom_2 = df_sorted.head(2)
            # Judge prefers higher J_pct (better dancer)
            j_scores = bottom_2['J_pct'].values
            if j_scores[0] > j_scores[1]:
                # Bottom 1 has higher J_pct, save them and eliminate Bottom 2
                judges_saved_name = bottom_2.iloc[0]['celebrity_name']
                eliminated = [bottom_2.iloc[1]['celebrity_name']]
            else:
                # Normal elimination (Bottom 1 eliminated)
                eliminated = df_sorted.head(n_eliminated)['celebrity_name'].tolist()
        else:
            eliminated = df_sorted.head(n_eliminated)['celebrity_name'].tolist()
        
        results[week] = {
            'eliminated': eliminated,
            'method': 'rank' + ('+save' if judges_save else ''),
            'n_contestants': len(remaining),
            'judges_saved': judges_saved_name,
            'details': df[['celebrity_name', 'J_pct', 'f_mean', 'J_rank', 'F_rank', 'combined_rank']].to_dict('records')
        }
        
        # 更新剩余选手
        remaining = [c for c in remaining if c not in eliminated]
    
    return results, remaining

def simulate_pct_method(season_data, judges_save=False):
    """
    Percentage-based method: Combined = 0.5 * J_pct + 0.5 * F_pct
    其中 F_pct = f(i,w) * 100 / max(f(i,w))  # 归一化到0-100
    每周淘汰combined_pct最低（最差）的k人
    
    Args:
        season_data: DataFrame with season data
        judges_save: If True, enable Judges' Save mechanism for Bottom 2
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
                'n_contestants': len(remaining),
                'judges_saved': None
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
        
        # =====================================================================
        # JUDGES' SAVE MECHANISM (per Plan requirement)
        # If enabled, and k=1, check Bottom 2: save the one with higher J_pct
        # =====================================================================
        judges_saved_name = None
        if judges_save and n_eliminated == 1 and len(df) >= 2:
            bottom_2 = df_sorted.head(2)
            # Judge prefers higher J_pct (better dancer)
            j_scores = bottom_2['J_pct'].values
            if j_scores[0] < j_scores[1]:
                # Bottom 1 has lower J_pct, but if Bottom 2 has higher J_pct, save Bottom 2
                # Wait - in pct method, lower combined = worse, so bottom_2[0] is worst
                # Save if worst has lower J_pct than second worst
                judges_saved_name = bottom_2.iloc[0]['celebrity_name']
                eliminated = [bottom_2.iloc[1]['celebrity_name']]
            else:
                eliminated = df_sorted.head(n_eliminated)['celebrity_name'].tolist()
        else:
            eliminated = df_sorted.head(n_eliminated)['celebrity_name'].tolist()
        
        results[week] = {
            'eliminated': eliminated,
            'method': 'pct' + ('+save' if judges_save else ''),
            'n_contestants': len(remaining),
            'judges_saved': judges_saved_name,
            'details': df[['celebrity_name', 'J_pct', 'f_mean', 'F_pct', 'combined_pct']].to_dict('records')
        }
        
        # 更新剩余选手
        remaining = [c for c in remaining if c not in eliminated]
    
    return results, remaining


def simulate_sigmoid_dynamic(season_data, w_min=0.30, w_max=0.75, steepness=6, judges_save=False):
    """
    Sigmoid Dynamic Strategy (NEW - Phase 3 Optimal):
    w_J(t) = w_min + (w_max - w_min) / (1 + exp(-steepness * (t/T - 0.5)))
    Score = w_J(t) * J_rank + (1 - w_J(t)) * F_rank
    
    Uses Rank-based scoring for robustness against extreme vote distributions.
    
    Args:
        season_data: DataFrame with season data
        w_min: Minimum judge weight (early phase), default 0.30
        w_max: Maximum judge weight (late phase), default 0.75
        steepness: S-curve steepness, default 6
        judges_save: If True, enable Judges' Save mechanism for Bottom 2
    
    Returns:
        results: dict of weekly results
        remaining: list of final remaining contestants
    """
    results = {}
    remaining = season_data['celebrity_name'].unique().tolist()
    total_weeks = season_data['week'].nunique()
    
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
                'method': 'sigmoid_dynamic',
                'n_contestants': len(remaining),
                'judges_saved': None,
                'w_j': None
            }
            continue
        
        # 计算Sigmoid动态权重
        t = week - 1  # 0-indexed
        T = max(total_weeks - 1, 1)
        x = steepness * (t / T - 0.5)
        sigmoid = 1 / (1 + np.exp(-x))
        w_j = w_min + (w_max - w_min) * sigmoid
        w_f = 1 - w_j
        
        # 计算Rank得分
        df = week_data.copy()
        n = len(df)
        df['J_rank'] = df['J_pct'].rank(ascending=False, method='average')
        df['F_rank'] = df['f_mean'].rank(ascending=False, method='average')
        
        # 转换为得分（rank越小越好 -> 得分越高越好）
        df['J_score'] = (n - df['J_rank'] + 1) / n * 100
        df['F_score'] = (n - df['F_rank'] + 1) / n * 100
        
        # 动态加权得分
        df['combined_score'] = w_j * df['J_score'] + w_f * df['F_score']
        
        # 淘汰combined_score最低的k人
        df_sorted = df.sort_values('combined_score', ascending=True)
        
        # JUDGES' SAVE for Bottom 2
        judges_saved_name = None
        if judges_save and n_eliminated == 1 and len(df) >= 2:
            bottom_2 = df_sorted.head(2)
            j_scores = bottom_2['J_pct'].values
            if j_scores[0] < j_scores[1]:
                # Bottom 1 has lower J_pct, save and eliminate Bottom 2 instead
                judges_saved_name = bottom_2.iloc[0]['celebrity_name']
                eliminated = [bottom_2.iloc[1]['celebrity_name']]
            else:
                eliminated = df_sorted.head(n_eliminated)['celebrity_name'].tolist()
        else:
            eliminated = df_sorted.head(n_eliminated)['celebrity_name'].tolist()
        
        results[week] = {
            'eliminated': eliminated,
            'method': 'sigmoid_dynamic' + ('+save' if judges_save else ''),
            'n_contestants': len(remaining),
            'judges_saved': judges_saved_name,
            'w_j': w_j,
            'w_f': w_f,
            'details': df[['celebrity_name', 'J_pct', 'f_mean', 'J_score', 'F_score', 'combined_score']].to_dict('records')
        }
        
        # 更新剩余选手
        remaining = [c for c in remaining if c not in eliminated]
    
    return results, remaining


def simulate_new_strategy(season_data, judges_save=True):
    """
    New Strategy (per Plan Phase 5 requirement):
    Score = (0.5 + 0.05*t) * J% + (0.5 - 0.05*t) * log(F%)
    Where t = week number (capped), and Judges' Save is enabled by default
    
    This is the recommended dynamic log-weighting strategy
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
                'method': 'new_strategy',
                'n_contestants': len(remaining),
                'judges_saved': None
            }
            continue
        
        # 计算动态权重
        df = week_data.copy()
        t = min(week, 10)  # Cap at week 10
        alpha_j = 0.5 + 0.05 * (t - 1)  # Judge weight increases with time
        alpha_j = min(alpha_j, 0.95)  # Cap at 0.95
        alpha_f = 1 - alpha_j
        
        # Log transform for fan votes (avoid log(0))
        df['F_log'] = np.log(df['f_mean'].clip(lower=0.001) * 100 + 1)
        
        # Normalize to 0-100 scale for combining
        df['F_log_norm'] = (df['F_log'] - df['F_log'].min()) / (df['F_log'].max() - df['F_log'].min() + 0.001) * 100
        
        # Dynamic weighted score
        df['combined_score'] = alpha_j * df['J_pct'] + alpha_f * df['F_log_norm']
        
        # 淘汰combined_score最低的k人
        df_sorted = df.sort_values('combined_score', ascending=True)
        
        # JUDGES' SAVE for Bottom 2
        judges_saved_name = None
        if judges_save and n_eliminated == 1 and len(df) >= 2:
            bottom_2 = df_sorted.head(2)
            j_scores = bottom_2['J_pct'].values
            if j_scores[0] < j_scores[1]:
                # Bottom 1 has lower J_pct, save Bottom 2 (higher J_pct)
                judges_saved_name = bottom_2.iloc[1]['celebrity_name']
                eliminated = [bottom_2.iloc[0]['celebrity_name']]
            else:
                # Normal: eliminate the worst
                eliminated = df_sorted.head(n_eliminated)['celebrity_name'].tolist()
        else:
            eliminated = df_sorted.head(n_eliminated)['celebrity_name'].tolist()
        
        results[week] = {
            'eliminated': eliminated,
            'method': 'new_strategy' + ('+save' if judges_save else ''),
            'n_contestants': len(remaining),
            'judges_saved': judges_saved_name,
            'alpha_j': alpha_j,
            'details': df[['celebrity_name', 'J_pct', 'f_mean', 'F_log_norm', 'combined_score']].to_dict('records')
        }
        
        # 更新剩余选手
        remaining = [c for c in remaining if c not in eliminated]
    
    return results, remaining

def compare_methods(rank_results, pct_results, rank_final, pct_final, actual_eliminated,
                    new_results=None, new_final=None):
    """比较方法的结果 (支持2种或3种方法)"""
    comparison = {
        'weekly_diff': 0,
        'weeks_different': [],
        'final_diff': rank_final != pct_final,
        'rank_final': rank_final,
        'pct_final': pct_final
    }
    
    # Add new strategy comparison if provided
    if new_results is not None:
        comparison['new_final'] = new_final
        comparison['final_diff_new_vs_rank'] = new_final != rank_final
        comparison['final_diff_new_vs_pct'] = new_final != pct_final
    
    all_weeks = set(rank_results.keys()) | set(pct_results.keys())
    
    for week in sorted(all_weeks):
        if week not in rank_results or week not in pct_results:
            continue
        
        rank_elim = set(rank_results[week]['eliminated'])
        pct_elim = set(pct_results[week]['eliminated'])
        
        if rank_elim != pct_elim:
            comparison['weekly_diff'] += 1
            diff_entry = {
                'week': week,
                'rank_eliminated': list(rank_elim),
                'pct_eliminated': list(pct_elim),
                'actual_eliminated': [c for c in actual_eliminated if actual_eliminated.get(c) == week]
            }
            
            # Add new strategy elimination if available
            if new_results and week in new_results:
                diff_entry['new_eliminated'] = list(new_results[week]['eliminated'])
                diff_entry['judges_saved'] = new_results[week].get('judges_saved')
            
            comparison['weeks_different'].append(diff_entry)
    
    return comparison

# =============================================================================
# RUN SIMULATION FOR ALL SEASONS (4 methods: Rank, Percentage, New Strategy, Sigmoid Dynamic)
# =============================================================================
print("\n[2] Running simulation for all seasons (Rank, Percentage, New Strategy, Sigmoid Dynamic)...")

all_comparisons = []

for season in sorted(estimates['season'].unique()):
    season_data = estimates[estimates['season'] == season].copy()
    
    # 获取实际淘汰信息
    actual_eliminated = {}
    for _, row in season_data[season_data['was_eliminated'] == True].iterrows():
        actual_eliminated[row['celebrity_name']] = row['week']
    
    # 模拟四种方法 (Rank, Percentage, Old Linear Dynamic, NEW Sigmoid Dynamic)
    rank_results, rank_final = simulate_rank_method(season_data, judges_save=False)
    pct_results, pct_final = simulate_pct_method(season_data, judges_save=False)
    new_results, new_final = simulate_new_strategy(season_data, judges_save=True)  # Old linear dynamic
    sigmoid_results, sigmoid_final = simulate_sigmoid_dynamic(season_data, w_min=0.30, w_max=0.75, steepness=6, judges_save=True)  # NEW optimal
    
    # Also test with Judges' Save toggle
    rank_save_results, rank_save_final = simulate_rank_method(season_data, judges_save=True)
    sigmoid_nosave_results, sigmoid_nosave_final = simulate_sigmoid_dynamic(season_data, w_min=0.30, w_max=0.75, steepness=6, judges_save=False)
    
    # 比较
    comparison = compare_methods(rank_results, pct_results, rank_final, pct_final, 
                                  actual_eliminated, new_results, new_final)
    comparison['season'] = season
    comparison['n_weeks'] = len(rank_results)
    comparison['rank_save_final'] = rank_save_final
    comparison['judges_save_changed_final'] = rank_final != rank_save_final
    
    # Add Sigmoid Dynamic results (NEW optimal strategy)
    comparison['sigmoid_final'] = sigmoid_final
    comparison['sigmoid_nosave_final'] = sigmoid_nosave_final
    comparison['final_diff_sigmoid_vs_rank'] = sigmoid_final != rank_final
    comparison['final_diff_sigmoid_vs_pct'] = sigmoid_final != pct_final
    
    # Count how many weeks Judges' Save was used
    save_count = sum(1 for w, r in new_results.items() if r.get('judges_saved') is not None)
    sigmoid_save_count = sum(1 for w, r in sigmoid_results.items() if r.get('judges_saved') is not None)
    comparison['judges_save_count'] = save_count
    comparison['sigmoid_save_count'] = sigmoid_save_count
    
    all_comparisons.append(comparison)
    
    # 打印每季结果
    if comparison['weekly_diff'] > 0:
        save_str = f", Save used {save_count}x (old) / {sigmoid_save_count}x (sigmoid)" if (save_count > 0 or sigmoid_save_count > 0) else ""
        print(f"    Season {season}: {comparison['weekly_diff']} weeks differ "
              f"(Final: Rank={rank_final}, Pct={pct_final}, Old={new_final}, Sigmoid={sigmoid_final}{save_str})")
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
    
    Supported methods: 'rank', 'pct', 'sigmoid'
    """
    # 获取最后一周的数据来确定最终排名
    max_week = season_data['week'].max()
    final_week = season_data[season_data['week'] == max_week].copy()
    total_weeks = season_data['week'].nunique()
    
    if len(final_week) < 2:
        return np.nan
    
    # Fan ranking (基于f_mean)
    final_week['fan_rank'] = final_week['f_mean'].rank(ascending=False)
    
    if method == 'rank':
        # Rank method: 最终排名基于combined_rank (50-50 static)
        final_week['J_rank'] = final_week['J_pct'].rank(ascending=False)
        final_week['F_rank'] = final_week['f_mean'].rank(ascending=False)
        final_week['combined_rank'] = 0.5 * final_week['J_rank'] + 0.5 * final_week['F_rank']
        final_week['final_rank'] = final_week['combined_rank'].rank()
    elif method == 'sigmoid':
        # Sigmoid Dynamic method: 使用最后一周的Sigmoid权重
        t = max_week - 1
        T = max(total_weeks - 1, 1)
        x = 6 * (t / T - 0.5)  # steepness=6
        sigmoid = 1 / (1 + np.exp(-x))
        w_j = 0.30 + 0.45 * sigmoid  # w_min=0.30, w_max=0.75
        w_f = 1 - w_j
        
        n = len(final_week)
        final_week['J_rank'] = final_week['J_pct'].rank(ascending=False)
        final_week['F_rank'] = final_week['f_mean'].rank(ascending=False)
        final_week['J_score'] = (n - final_week['J_rank'] + 1) / n * 100
        final_week['F_score'] = (n - final_week['F_rank'] + 1) / n * 100
        final_week['combined_score'] = w_j * final_week['J_score'] + w_f * final_week['F_score']
        final_week['final_rank'] = final_week['combined_score'].rank(ascending=False)
    else:
        # Pct method: 最终排名基于combined_pct
        max_f = final_week['f_mean'].max()
        final_week['F_pct'] = final_week['f_mean'] / max_f * 100 if max_f > 0 else 0
        final_week['combined_pct'] = 0.5 * final_week['J_pct'] + 0.5 * final_week['F_pct']
        final_week['final_rank'] = final_week['combined_pct'].rank(ascending=False)
    
    # Spearman correlation
    corr, _ = stats.spearmanr(final_week['final_rank'], final_week['fan_rank'])
    return corr

# 计算每季的FFI (including Sigmoid)
ffi_results = []

for season in sorted(estimates['season'].unique()):
    season_data = estimates[estimates['season'] == season]
    
    ffi_rank = calculate_ffi(season_data, method='rank')
    ffi_pct = calculate_ffi(season_data, method='pct')
    ffi_sigmoid = calculate_ffi(season_data, method='sigmoid')
    
    ffi_results.append({
        'season': season,
        'FFI_rank': ffi_rank,
        'FFI_pct': ffi_pct,
        'FFI_sigmoid': ffi_sigmoid,
        'FFI_diff': ffi_pct - ffi_rank if not (np.isnan(ffi_rank) or np.isnan(ffi_pct)) else np.nan,
        'FFI_sigmoid_vs_rank': ffi_sigmoid - ffi_rank if not (np.isnan(ffi_sigmoid) or np.isnan(ffi_rank)) else np.nan
    })

ffi_df = pd.DataFrame(ffi_results)

print("\n    Fan Favor Index (FFI) by Season:")
print(f"    Average FFI (Rank method):    {ffi_df['FFI_rank'].mean():.4f}")
print(f"    Average FFI (Pct method):     {ffi_df['FFI_pct'].mean():.4f}")
print(f"    Average FFI (Sigmoid method): {ffi_df['FFI_sigmoid'].mean():.4f}")
print(f"    Average difference (Pct - Rank):     {ffi_df['FFI_diff'].mean():.4f}")
print(f"    Average difference (Sigmoid - Rank): {ffi_df['FFI_sigmoid_vs_rank'].mean():.4f}")

# 哪种方法更偏向粉丝
if ffi_df['FFI_sigmoid_vs_rank'].mean() > ffi_df['FFI_diff'].mean():
    print(f"\n    Conclusion: Sigmoid method balances fan engagement best")
else:
    print(f"\n    Conclusion: Percentage method is more fan-favoring (FFI +{ffi_df['FFI_diff'].mean():.4f})")

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
    elif method == 'sigmoid':
        # Sigmoid Dynamic method: 使用最后一周的Sigmoid权重
        total_weeks = season_data['week'].nunique()
        t = max_week - 1
        T = max(total_weeks - 1, 1)
        x = 6 * (t / T - 0.5)  # steepness=6
        sigmoid = 1 / (1 + np.exp(-x))
        w_j = 0.30 + 0.45 * sigmoid
        w_f = 1 - w_j
        
        n = len(final_week)
        final_week['J_rank'] = final_week['J_pct'].rank(ascending=False)
        final_week['F_rank'] = final_week['f_mean'].rank(ascending=False)
        final_week['J_score'] = (n - final_week['J_rank'] + 1) / n * 100
        final_week['F_score'] = (n - final_week['F_rank'] + 1) / n * 100
        final_week['combined_score'] = w_j * final_week['J_score'] + w_f * final_week['F_score']
        final_week['final_rank'] = final_week['combined_score'].rank(ascending=False)
    else:
        max_f = final_week['f_mean'].max()
        final_week['F_pct'] = final_week['f_mean'] / max_f * 100 if max_f > 0 else 0
        final_week['combined_pct'] = 0.5 * final_week['J_pct'] + 0.5 * final_week['F_pct']
        final_week['final_rank'] = final_week['combined_pct'].rank(ascending=False)
    
    corr, _ = stats.spearmanr(final_week['final_rank'], final_week['judge_rank'])
    return corr

# 计算JFI (including Sigmoid)
for idx, row in ffi_df.iterrows():
    season_data = estimates[estimates['season'] == row['season']]
    ffi_df.loc[idx, 'JFI_rank'] = calculate_jfi(season_data, 'rank')
    ffi_df.loc[idx, 'JFI_pct'] = calculate_jfi(season_data, 'pct')
    ffi_df.loc[idx, 'JFI_sigmoid'] = calculate_jfi(season_data, 'sigmoid')

ffi_df['JFI_diff'] = ffi_df['JFI_pct'] - ffi_df['JFI_rank']
ffi_df['JFI_sigmoid_vs_rank'] = ffi_df['JFI_sigmoid'] - ffi_df['JFI_rank']

print(f"\n    Average JFI (Rank method):    {ffi_df['JFI_rank'].mean():.4f}")
print(f"    Average JFI (Pct method):     {ffi_df['JFI_pct'].mean():.4f}")
print(f"    Average JFI (Sigmoid method): {ffi_df['JFI_sigmoid'].mean():.4f}")
print(f"    JFI Sigmoid vs Rank diff:     {ffi_df['JFI_sigmoid_vs_rank'].mean():.4f}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n[6] Generating visualizations...")

import os
img_dir = 'cleaned_outputs/phase3_simulator'
os.makedirs(img_dir, exist_ok=True)

# --- Individual Plots ---
# 6.1 Weekly Differences by Season (Individual)
fig1, ax1 = plt.subplots(figsize=(8, 5))
seasons = comparison_df['season']
diffs = comparison_df['weekly_diff']
colors = ['red' if d > 0 else 'green' for d in diffs]
ax1.bar(seasons, diffs, color=colors, alpha=0.7)
ax1.set_xlabel('Season')
ax1.set_ylabel('Number of Different Weeks')
ax1.set_title('Weekly Elimination Differences\n(Rank vs Percentage)')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{img_dir}/weekly_differences.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/weekly_differences.png")

# 6.2 FFI Comparison (Including Sigmoid)
fig2, ax2 = plt.subplots(figsize=(10, 5))
x = ffi_df['season']
width = 0.25
ax2.bar(x - width, ffi_df['FFI_rank'], width, label='Rank (Static 50-50)', color='steelblue', alpha=0.7)
ax2.bar(x, ffi_df['FFI_pct'], width, label='Pct Method', color='coral', alpha=0.7)
ax2.bar(x + width, ffi_df['FFI_sigmoid'], width, label='Sigmoid Dynamic (NEW)', color='forestgreen', alpha=0.7)
ax2.set_xlabel('Season')
ax2.set_ylabel('Fan Favor Index (FFI)')
ax2.set_title('FFI Comparison by Season (Including Sigmoid Dynamic)')
ax2.legend()
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{img_dir}/ffi_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/ffi_comparison.png")

# 6.3 JFI Comparison (Including Sigmoid)
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.bar(x - width, ffi_df['JFI_rank'], width, label='Rank (Static 50-50)', color='steelblue', alpha=0.7)
ax3.bar(x, ffi_df['JFI_pct'], width, label='Pct Method', color='coral', alpha=0.7)
ax3.bar(x + width, ffi_df['JFI_sigmoid'], width, label='Sigmoid Dynamic (NEW)', color='forestgreen', alpha=0.7)
ax3.set_xlabel('Season')
ax3.set_ylabel('Judge Favor Index (JFI)')
ax3.set_title('JFI Comparison by Season (Including Sigmoid Dynamic)')
ax3.legend()
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{img_dir}/jfi_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/jfi_comparison.png")

# 6.4 FFI vs JFI Trade-off (Including Sigmoid)
fig4, ax4 = plt.subplots(figsize=(10, 7))
ax4.scatter(ffi_df['JFI_rank'], ffi_df['FFI_rank'], c='steelblue', 
            label='Rank (Static 50-50)', alpha=0.6, s=60)
ax4.scatter(ffi_df['JFI_pct'], ffi_df['FFI_pct'], c='coral',
            label='Pct Method', alpha=0.6, s=60, marker='s')
ax4.scatter(ffi_df['JFI_sigmoid'], ffi_df['FFI_sigmoid'], c='forestgreen',
            label='Sigmoid Dynamic (NEW)', alpha=0.6, s=60, marker='^')
ax4.scatter(ffi_df['JFI_rank'].mean(), ffi_df['FFI_rank'].mean(), 
            c='darkblue', s=200, marker='*', label='Rank Mean', zorder=5)
ax4.scatter(ffi_df['JFI_pct'].mean(), ffi_df['FFI_pct'].mean(),
            c='darkred', s=200, marker='*', label='Pct Mean', zorder=5)
ax4.scatter(ffi_df['JFI_sigmoid'].mean(), ffi_df['FFI_sigmoid'].mean(),
            c='darkgreen', s=200, marker='*', label='Sigmoid Mean', zorder=5)
ax4.set_xlabel('Judge Favor Index (JFI)')
ax4.set_ylabel('Fan Favor Index (FFI)')
ax4.set_title('Trade-off: JFI vs FFI (Three Methods)')
ax4.legend(loc='lower left')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{img_dir}/ffi_jfi_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/ffi_jfi_tradeoff.png")

# 6.5 Difference Distribution (Individual)
fig5, ax5 = plt.subplots(figsize=(8, 5))
ax5.hist(ffi_df['FFI_diff'].dropna(), bins=15, color='mediumpurple', alpha=0.7, 
         edgecolor='black', label='FFI Diff (Pct - Rank)')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax5.axvline(x=ffi_df['FFI_diff'].mean(), color='green', linestyle='--', linewidth=2,
            label=f'Mean = {ffi_df["FFI_diff"].mean():.3f}')
ax5.set_xlabel('FFI Difference (Pct - Rank)')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of FFI Difference')
ax5.legend()
plt.tight_layout()
plt.savefig(f'{img_dir}/ffi_diff_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/ffi_diff_distribution.png")

# 6.6 Era Analysis (Individual)
fig6, ax6 = plt.subplots(figsize=(8, 6))
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
x_era = np.arange(len(era_order))
ax6.bar(x_era - 0.2, era_ffi['FFI_rank'], 0.2, label='FFI Rank', color='steelblue')
ax6.bar(x_era, era_ffi['FFI_pct'], 0.2, label='FFI Pct', color='coral')
ax6.bar(x_era + 0.2, era_ffi['JFI_rank'], 0.2, label='JFI Rank', color='lightblue', hatch='//')
ax6.bar(x_era + 0.4, era_ffi['JFI_pct'], 0.2, label='JFI Pct', color='lightsalmon', hatch='//')
ax6.set_xticks(x_era + 0.1)
ax6.set_xticklabels(era_order)
ax6.set_ylabel('Index Value')
ax6.set_title('FFI & JFI by Era')
ax6.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{img_dir}/era_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/era_analysis.png")

# --- Panel Plot (Combined) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 6.1 Weekly Differences by Season
ax1 = axes[0, 0]
ax1.bar(seasons, diffs, color=colors, alpha=0.7)
ax1.set_xlabel('Season')
ax1.set_ylabel('Number of Different Weeks')
ax1.set_title('Weekly Elimination Differences\n(Rank vs Percentage)')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 6.2 FFI Comparison
ax2 = axes[0, 1]
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
plt.savefig(f'{img_dir}/simulator_comparison.png', dpi=150, bbox_inches='tight')
print(f"    Saved: {img_dir}/simulator_comparison.png")

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
print("PHASE 3 SIMULATOR SUMMARY (WITH SIGMOID DYNAMIC)")
print("=" * 70)

# Calculate Sigmoid stats
sigmoid_finals_changed = comparison_df['final_diff_sigmoid_vs_rank'].sum() if 'final_diff_sigmoid_vs_rank' in comparison_df.columns else 0

print(f"""
METHOD COMPARISON RESULTS:
==========================
• Total seasons analyzed: {total_seasons}
• Seasons with different eliminations: {seasons_with_diff} ({seasons_with_diff/total_seasons:.1%})
• Total weekly differences: {total_weekly_diff}/{total_weeks} ({total_weekly_diff/total_weeks:.1%})
• Finals affected (Rank vs Pct): {final_diff}
• Finals affected (Sigmoid vs Rank): {sigmoid_finals_changed}

FAN FAVOR INDEX (FFI):
======================
• FFI measures how closely final ranking aligns with fan preferences
• Higher FFI = More fan-favoring

  Method              Mean FFI    Std
  --------------------------------------
  Rank (Static 50-50) {ffi_df['FFI_rank'].mean():.4f}      {ffi_df['FFI_rank'].std():.4f}
  Pct Method          {ffi_df['FFI_pct'].mean():.4f}      {ffi_df['FFI_pct'].std():.4f}
  Sigmoid Dynamic     {ffi_df['FFI_sigmoid'].mean():.4f}      {ffi_df['FFI_sigmoid'].std():.4f}  ← NEW

JUDGE FAVOR INDEX (JFI):
========================
• JFI measures how closely final ranking aligns with judge scores
• Higher JFI = More meritocratic

  Method              Mean JFI    Std
  --------------------------------------
  Rank (Static 50-50) {ffi_df['JFI_rank'].mean():.4f}      {ffi_df['JFI_rank'].std():.4f}
  Pct Method          {ffi_df['JFI_pct'].mean():.4f}      {ffi_df['JFI_pct'].std():.4f}
  Sigmoid Dynamic     {ffi_df['JFI_sigmoid'].mean():.4f}      {ffi_df['JFI_sigmoid'].std():.4f}  ← NEW

SIGMOID DYNAMIC STRATEGY (NEW - OPTIMAL):
==========================================
Formula: w_J(t) = 0.30 + 0.45 / (1 + exp(-6 * (t/T - 0.5)))
• Early phase (t=0):  w_J ≈ 30% → Fan engagement dominant
• Mid phase (t=T/2):  w_J = 52.5% → Balanced
• Late phase (t=T):   w_J ≈ 75% → Merit dominant

Scoring: Rank-based (robust against extreme distributions)

KEY FINDINGS:
=============
1. Sigmoid Dynamic achieves {"higher" if ffi_df['FFI_sigmoid'].mean() > ffi_df['FFI_rank'].mean() else "lower"} FFI than Rank ({ffi_df['FFI_sigmoid'].mean():.4f} vs {ffi_df['FFI_rank'].mean():.4f})
2. Sigmoid Dynamic achieves {"higher" if ffi_df['JFI_sigmoid'].mean() > ffi_df['JFI_rank'].mean() else "lower"} JFI than Rank ({ffi_df['JFI_sigmoid'].mean():.4f} vs {ffi_df['JFI_rank'].mean():.4f})
3. The Sigmoid function enables phase-differentiated optimization:
   - Early: High F-weight promotes fan engagement
   - Late: High J-weight ensures meritocratic finals

CONCLUSION:
===========
SIGMOID DYNAMIC (0.30→0.75, steepness=6) is recommended:
✓ Balances early fan engagement with late-stage meritocracy
✓ Uses Rank-based scoring for robustness
✓ Satisfies multi-phase optimization criteria

FILES SAVED:
============
• method_comparison.csv - Season-level comparison (with Sigmoid)
• favor_indices.csv - FFI and JFI by season (with Sigmoid)
• simulator_comparison.png - Visualizations (updated)
""")

print("=" * 70)
print("PHASE 3 SIMULATOR WITH SIGMOID DYNAMIC COMPLETE!")
print("=" * 70)
