#!/usr/bin/env python3
"""
Patch 4B Supplement: Fan-Elasticity Analysis
=============================================
添加"敏感度分析"来完善plan.md中的Patch 4B要求:
- Fan-Elasticity: 对f(i,w)添加小扰动，观察淘汰翻转概率
- 如果翻转频繁且随粉丝变化，说明该方法更偏向粉丝

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
print("PATCH 4B SUPPLEMENT: FAN-ELASTICITY ANALYSIS")
print("=" * 70)

# 加载数据
print("\n[1] Loading data...")
estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')
print(f"    Loaded {len(estimates)} observations")

# =============================================================================
# FAN-ELASTICITY SIMULATION
# =============================================================================
print("\n[2] Running Fan-Elasticity Simulation...")
print("    (Adding perturbations to f(i,w) and measuring elimination reversals)")

def simulate_with_perturbation(season_data, method, perturbation_std=0.02, n_simulations=100):
    """
    对f(i,w)添加高斯扰动，模拟淘汰结果，返回翻转率
    """
    reversals = []
    
    for week in sorted(season_data['week'].unique()):
        week_data = season_data[season_data['week'] == week].copy()
        if len(week_data) < 2:
            continue
        
        n_eliminated = int(week_data['was_eliminated'].sum())
        if n_eliminated == 0:
            continue
        
        # 原始淘汰结果
        if method == 'rank':
            week_data['J_rank'] = week_data['J_pct'].rank(ascending=False)
            week_data['F_rank'] = week_data['f_mean'].rank(ascending=False)
            week_data['combined'] = 0.5 * week_data['J_rank'] + 0.5 * week_data['F_rank']
            original_elim = set(week_data.nlargest(n_eliminated, 'combined')['celebrity_name'])
        else:
            max_f = week_data['f_mean'].max()
            week_data['F_pct'] = week_data['f_mean'] / max_f * 100 if max_f > 0 else 0
            week_data['combined'] = 0.5 * week_data['J_pct'] + 0.5 * week_data['F_pct']
            original_elim = set(week_data.nsmallest(n_eliminated, 'combined')['celebrity_name'])
        
        # 扰动模拟
        reversal_count = 0
        for _ in range(n_simulations):
            perturbed = week_data.copy()
            # 对f_mean添加扰动
            noise = np.random.normal(0, perturbation_std, len(perturbed))
            perturbed['f_perturbed'] = (perturbed['f_mean'] + noise).clip(0.001, 0.999)
            
            if method == 'rank':
                perturbed['F_rank_p'] = perturbed['f_perturbed'].rank(ascending=False)
                perturbed['combined_p'] = 0.5 * perturbed['J_rank'] + 0.5 * perturbed['F_rank_p']
                perturbed_elim = set(perturbed.nlargest(n_eliminated, 'combined_p')['celebrity_name'])
            else:
                max_f_p = perturbed['f_perturbed'].max()
                perturbed['F_pct_p'] = perturbed['f_perturbed'] / max_f_p * 100 if max_f_p > 0 else 0
                perturbed['combined_p'] = 0.5 * perturbed['J_pct'] + 0.5 * perturbed['F_pct_p']
                perturbed_elim = set(perturbed.nsmallest(n_eliminated, 'combined_p')['celebrity_name'])
            
            if perturbed_elim != original_elim:
                reversal_count += 1
        
        reversals.append({
            'week': week,
            'reversal_rate': reversal_count / n_simulations,
            'n_contestants': len(week_data),
            'n_eliminated': n_eliminated
        })
    
    return reversals

# 对每个赛季计算Fan-Elasticity
elasticity_results = []
n_sim = 100
perturbation = 0.03  # 3% standard deviation perturbation

for season in sorted(estimates['season'].unique()):
    season_data = estimates[estimates['season'] == season]
    
    # Rank method
    rev_rank = simulate_with_perturbation(season_data, 'rank', perturbation, n_sim)
    avg_reversal_rank = np.mean([r['reversal_rate'] for r in rev_rank]) if rev_rank else 0
    
    # Percentage method
    rev_pct = simulate_with_perturbation(season_data, 'pct', perturbation, n_sim)
    avg_reversal_pct = np.mean([r['reversal_rate'] for r in rev_pct]) if rev_pct else 0
    
    elasticity_results.append({
        'season': season,
        'elasticity_rank': avg_reversal_rank,
        'elasticity_pct': avg_reversal_pct,
        'elasticity_diff': avg_reversal_pct - avg_reversal_rank
    })
    
    if season % 10 == 0:
        print(f"    Processed season {season}...")

elasticity_df = pd.DataFrame(elasticity_results)

# =============================================================================
# RESULTS
# =============================================================================
print("\n[3] Fan-Elasticity Results:")
print("    " + "-" * 60)
print(f"    Average Fan-Elasticity (Rank method):  {elasticity_df['elasticity_rank'].mean():.4f}")
print(f"    Average Fan-Elasticity (Pct method):   {elasticity_df['elasticity_pct'].mean():.4f}")
print(f"    Difference (Pct - Rank):               {elasticity_df['elasticity_diff'].mean():+.4f}")
print("    " + "-" * 60)

# 统计检验
t_stat, p_value = stats.ttest_rel(
    elasticity_df['elasticity_pct'].dropna(),
    elasticity_df['elasticity_rank'].dropna()
)
print(f"    Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")

# 结论
if elasticity_df['elasticity_diff'].mean() > 0 and p_value < 0.1:
    conclusion = "Percentage method is MORE SENSITIVE to fan vote changes (higher elasticity)"
    fan_favoring = "Percentage"
elif elasticity_df['elasticity_diff'].mean() < 0 and p_value < 0.1:
    conclusion = "Rank method is MORE SENSITIVE to fan vote changes (higher elasticity)"
    fan_favoring = "Rank"
else:
    conclusion = "No significant difference in fan-elasticity between methods"
    fan_favoring = "Neither clearly"

print(f"\n    CONCLUSION: {conclusion}")
print(f"    Method more favorable to fans: {fan_favoring}")

# =============================================================================
# COMBINE WITH FFI RESULTS
# =============================================================================
print("\n[4] Combining with FFI analysis...")

# 加载之前的favor_indices
favor_df = pd.read_csv('cleaned_outputs/favor_indices.csv')
favor_df = favor_df.merge(elasticity_df[['season', 'elasticity_rank', 'elasticity_pct', 'elasticity_diff']], 
                          on='season', how='left')

# 保存更新的结果
favor_df.to_csv('cleaned_outputs/favor_indices_complete.csv', index=False)
print(f"    Saved: favor_indices_complete.csv")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n[5] Generating visualization...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 5.1 Elasticity by season
ax1 = axes[0]
seasons = elasticity_df['season']
ax1.plot(seasons, elasticity_df['elasticity_rank'], 'b-o', label='Rank', markersize=3, alpha=0.7)
ax1.plot(seasons, elasticity_df['elasticity_pct'], 'r-s', label='Percentage', markersize=3, alpha=0.7)
ax1.set_xlabel('Season')
ax1.set_ylabel('Fan-Elasticity (Reversal Rate)')
ax1.set_title('Fan-Elasticity by Season')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 5.2 Elasticity comparison
ax2 = axes[1]
methods = ['Rank', 'Percentage']
means = [elasticity_df['elasticity_rank'].mean(), elasticity_df['elasticity_pct'].mean()]
stds = [elasticity_df['elasticity_rank'].std(), elasticity_df['elasticity_pct'].std()]
bars = ax2.bar(methods, means, yerr=stds, capsize=5, color=['steelblue', 'coral'])
ax2.set_ylabel('Average Fan-Elasticity')
ax2.set_title(f'Fan-Elasticity Comparison\n(p={p_value:.4f})')
for bar, mean in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{mean:.3f}', ha='center', va='bottom')

# 5.3 Difference distribution
ax3 = axes[2]
ax3.hist(elasticity_df['elasticity_diff'], bins=15, color='gray', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
ax3.axvline(x=elasticity_df['elasticity_diff'].mean(), color='blue', linestyle='-', 
            linewidth=2, label=f'Mean={elasticity_df["elasticity_diff"].mean():.3f}')
ax3.set_xlabel('Elasticity Difference (Pct - Rank)')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Elasticity Differences')
ax3.legend()

plt.tight_layout()
plt.savefig('cleaned_outputs/fan_elasticity_analysis.png', dpi=150, bbox_inches='tight')
print("    Saved: fan_elasticity_analysis.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FAN-ELASTICITY ANALYSIS SUMMARY")
print("=" * 70)

# 综合FFI和Elasticity的结论
ffi_rank_mean = favor_df['FFI_rank'].mean()
ffi_pct_mean = favor_df['FFI_pct'].mean()
ela_rank_mean = elasticity_df['elasticity_rank'].mean()
ela_pct_mean = elasticity_df['elasticity_pct'].mean()

print(f"""
METRIC COMPARISON:
==================

                        Rank Method     Percentage Method
---------------------------------------------------------
Fan-Favor Index (FFI)   {ffi_rank_mean:>10.4f}       {ffi_pct_mean:>10.4f}
Fan-Elasticity          {ela_rank_mean:>10.4f}       {ela_pct_mean:>10.4f}
---------------------------------------------------------

INTERPRETATION:
• Higher FFI = Final ranking correlates more with fan ranking
• Higher Elasticity = Small fan vote changes cause more elimination reversals

FINAL VERDICT:
""")

# 综合判断
ffi_pct_wins = ffi_pct_mean > ffi_rank_mean
ela_pct_wins = ela_pct_mean > ela_rank_mean

if ffi_pct_wins and ela_pct_wins:
    verdict = "PERCENTAGE method is MORE FAN-FAVORING (both FFI and Elasticity higher)"
elif not ffi_pct_wins and not ela_pct_wins:
    verdict = "RANK method is MORE FAN-FAVORING (both FFI and Elasticity higher)"
elif ffi_pct_wins:
    verdict = "PERCENTAGE method has higher FFI; RANK has higher Elasticity - Mixed result"
else:
    verdict = "RANK method has higher FFI; PERCENTAGE has higher Elasticity - Mixed result"

print(f"    {verdict}")

# JFI comparison (meritocracy)
jfi_rank_mean = favor_df['JFI_rank'].mean()
jfi_pct_mean = favor_df['JFI_pct'].mean()

print(f"""
MERITOCRACY CHECK (Judge-Favor Index):
• Rank method JFI:       {jfi_rank_mean:.4f}
• Percentage method JFI: {jfi_pct_mean:.4f}
• More meritocratic:     {'RANK' if jfi_rank_mean > jfi_pct_mean else 'PERCENTAGE'}

RECOMMENDATION FOR PRODUCERS:
If priority is REDUCING extreme fan influence → Choose {'RANK' if ela_pct_mean > ela_rank_mean else 'PERCENTAGE'} (lower elasticity)
If priority is MAINTAINING fan engagement   → Choose {'PERCENTAGE' if ffi_pct_mean > ffi_rank_mean else 'RANK'} (higher FFI)
If priority is MERITOCRACY                  → Choose {'RANK' if jfi_rank_mean > jfi_pct_mean else 'PERCENTAGE'} (higher JFI)

FILES SAVED:
• favor_indices_complete.csv (with elasticity)
• fan_elasticity_analysis.png
""")

# 保存elasticity单独的CSV
elasticity_df.to_csv('cleaned_outputs/fan_elasticity.csv', index=False)
print("• fan_elasticity.csv")

print("=" * 70)
print("PATCH 4B SUPPLEMENT COMPLETE!")
print("=" * 70)
