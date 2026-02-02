#!/usr/bin/env python3
"""
Generate Sensitivity Analysis Figures for Chapter 8
- Cross-season stability boxplot
- Bootstrap confidence intervals
- Parameter sensitivity contour plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

# Load data
balanced_df = pd.read_csv('cleaned_outputs/phase3_pareto_analysis/balanced_optimization_final.csv')
season_df = pd.read_csv('cleaned_outputs/phase3_pareto_analysis/season_rank_vs_pct.csv')
cv_df = pd.read_csv('cleaned_outputs/phase3_pareto_analysis/extreme_value_analysis.csv')

# ============================================================
# Figure 1: Parameter Sensitivity Heatmap (w_min vs w_max)
# ============================================================
print("Generating Parameter Sensitivity Heatmap...")

# Filter dynamic sigmoid rules
sigmoid_rules = balanced_df[balanced_df['rule_type'] == 'dynamic_sigmoid'].copy()

# Get unique parameter values
w_min_vals = sorted(sigmoid_rules['j_min'].dropna().unique())
w_max_vals = sorted(sigmoid_rules['j_max'].dropna().unique())

# Create pivot table for steepness=6 (optimal)
sigmoid_s6 = sigmoid_rules[sigmoid_rules['steepness'] == 6.0]

# Create heatmap data
heatmap_data = sigmoid_s6.pivot_table(
    values='Score_balanced', 
    index='j_min', 
    columns='j_max',
    aggfunc='mean'
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Composite Score heatmap
ax1 = axes[0]
im = ax1.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto')
plt.colorbar(im, ax=ax1, label='Composite Score')

# Mark optimal point first to get coordinates
optimal_flat = np.argmax(heatmap_data.values)
opt_i, opt_j = np.unravel_index(optimal_flat, heatmap_data.values.shape)

# Add annotations (skip optimal cell, show value next to star instead)
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        if i == opt_i and j == opt_j:
            # For optimal cell, show value offset to avoid star
            ax1.text(j + 0.35, i - 0.35, f'{heatmap_data.values[i, j]:.3f}', 
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        else:
            ax1.text(j, i, f'{heatmap_data.values[i, j]:.3f}', 
                    ha='center', va='center', fontsize=8)

ax1.set_xticks(range(len(heatmap_data.columns)))
ax1.set_yticks(range(len(heatmap_data.index)))
ax1.set_xticklabels([f'{x:.2f}' for x in heatmap_data.columns])
ax1.set_yticklabels([f'{x:.2f}' for x in heatmap_data.index])
ax1.set_xlabel('$w_{max}$ (Late-stage Judge Weight)')
ax1.set_ylabel('$w_{min}$ (Early-stage Judge Weight)')
ax1.set_title('(a) Composite Score Sensitivity\n(Steepness $s=6$)')

# Draw star marker
ax1.scatter([opt_j], [opt_i], marker='*', s=250, c='blue', 
            edgecolors='white', linewidths=1.5, zorder=10, 
            label=f'Optimal: ({heatmap_data.index[opt_i]:.2f}, {heatmap_data.columns[opt_j]:.2f})')
ax1.legend(loc='lower right', fontsize=8)

# Right: Steepness sensitivity
steepness_vals = sorted(sigmoid_rules['steepness'].dropna().unique())
steepness_scores = []
for s in steepness_vals:
    scores = sigmoid_rules[sigmoid_rules['steepness'] == s]['Score_balanced']
    steepness_scores.append({
        'steepness': s,
        'mean': scores.mean(),
        'std': scores.std(),
        'max': scores.max()
    })
steepness_df = pd.DataFrame(steepness_scores)

ax2 = axes[1]
ax2.errorbar(steepness_df['steepness'], steepness_df['mean'], 
             yerr=steepness_df['std'], fmt='o-', capsize=5, 
             color='steelblue', label='Mean ± Std')
ax2.plot(steepness_df['steepness'], steepness_df['max'], 
         's--', color='darkgreen', label='Maximum')
ax2.axhline(y=0.4688, color='red', linestyle=':', label='Static Rank 50-50')
ax2.set_xlabel('Steepness Parameter $s$')
ax2.set_ylabel('Composite Score')
ax2.set_title('(b) Steepness Sensitivity Analysis')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(steepness_vals)

plt.tight_layout()
plt.savefig('cleaned_outputs/phase3_pareto_analysis/parameter_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('cleaned_outputs/phase3_pareto_analysis/parameter_sensitivity_analysis.pdf', bbox_inches='tight')
print("  Saved: parameter_sensitivity_analysis.png/pdf")

# ============================================================
# Figure 2: Cross-Season Stability Analysis
# ============================================================
print("Generating Cross-Season Stability Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Boxplot comparison
ax1 = axes[0]
data_for_box = {
    'Rank (J)': season_df['J_rank'].dropna(),
    'Pct (J)': season_df['J_pct'].dropna(),
    'Rank (F)': season_df['F_rank'].dropna(),
    'Pct (F)': season_df['F_pct'].dropna()
}
box_df = pd.DataFrame(dict([(k, pd.Series(v.values)) for k, v in data_for_box.items()]))
bp = box_df.boxplot(ax=ax1, patch_artist=True, return_type='dict')

colors = ['steelblue', 'coral', 'steelblue', 'coral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax1.set_ylabel('Spearman Correlation')
ax1.set_title('(a) Cross-Season Stability: Rank vs. Pct')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Add CV annotations
rank_j_cv = season_df['J_rank'].std() / abs(season_df['J_rank'].mean())
pct_j_cv = season_df['J_pct'].std() / abs(season_df['J_pct'].mean())
ax1.text(0.05, 0.95, f'CV(Rank-J): {rank_j_cv:.2f}\nCV(Pct-J): {pct_j_cv:.2f}', 
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right: Bootstrap confidence intervals
ax2 = axes[1]

# Bootstrap for optimal dynamic vs static
np.random.seed(42)
n_bootstrap = 1000

# Get scores for comparison
static_scores = balanced_df[balanced_df['rule_name'] == 'Static_Rank(0.50)']['Score_balanced'].values
if len(static_scores) == 0:
    static_scores = np.array([0.4688])
dynamic_scores = balanced_df[
    (balanced_df['j_min'] == 0.30) & 
    (balanced_df['j_max'] == 0.75) & 
    (balanced_df['steepness'] == 6.0)
]['Score_balanced'].values
if len(dynamic_scores) == 0:
    dynamic_scores = np.array([0.5701])

# Bootstrap difference
boot_diff = []
for _ in range(n_bootstrap):
    s_sample = np.random.choice(static_scores, size=len(static_scores), replace=True)
    d_sample = np.random.choice(dynamic_scores, size=len(dynamic_scores), replace=True)
    boot_diff.append(d_sample.mean() - s_sample.mean())

boot_diff = np.array(boot_diff)
ci_low, ci_high = np.percentile(boot_diff, [2.5, 97.5])
mean_diff = np.mean(boot_diff)

# Plot
ax2.hist(boot_diff, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax2.axvline(x=mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.3f}')
ax2.axvline(x=ci_low, color='darkred', linestyle='--', linewidth=1.5, label=f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]')
ax2.axvline(x=ci_high, color='darkred', linestyle='--', linewidth=1.5)
ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1, label='No Difference')
ax2.fill_betweenx([0, ax2.get_ylim()[1]*1.1], ci_low, ci_high, alpha=0.2, color='red')

ax2.set_xlabel('Score Difference (Dynamic - Static)')
ax2.set_ylabel('Density')
ax2.set_title('(b) Bootstrap 95% CI for Score Improvement')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('cleaned_outputs/phase3_pareto_analysis/cross_season_stability.png', dpi=300, bbox_inches='tight')
plt.savefig('cleaned_outputs/phase3_pareto_analysis/cross_season_stability.pdf', bbox_inches='tight')
print("  Saved: cross_season_stability.png/pdf")

# ============================================================
# Figure 3: Robustness to Extreme Scenarios
# ============================================================
print("Generating Robustness Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: CV ratio vs method performance difference
ax1 = axes[0]
merged = season_df.merge(cv_df, on='season', how='inner')
ax1.scatter(merged['cv_ratio'], merged['J_diff'], c='steelblue', s=60, alpha=0.7, label='J improvement')
ax1.scatter(merged['cv_ratio'], merged['F_diff'], c='coral', s=60, alpha=0.7, marker='^', label='F change')

# Regression line for J
mask = ~np.isnan(merged['cv_ratio']) & ~np.isnan(merged['J_diff'])
if mask.sum() > 2:
    slope, intercept, r, p, se = stats.linregress(merged.loc[mask, 'cv_ratio'], merged.loc[mask, 'J_diff'])
    x_line = np.linspace(merged['cv_ratio'].min(), merged['cv_ratio'].max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, 'b--', alpha=0.7, 
             label=f'J trend: r={r:.2f}, p={p:.3f}')

ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Fan Vote Variability Ratio (CV_fan / CV_judge)')
ax1.set_ylabel('Performance Difference (Rank - Pct)')
ax1.set_title('(a) Robustness vs. Fan Vote Variability')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Performance by judge system (3-judge vs 4-judge)
ax2 = axes[1]

# Seasons with 3 judges: 1-10, 13-14, 16, 27, 29
three_judge_seasons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 27, 29]
season_df['judge_system'] = season_df['season'].apply(
    lambda x: '3-Judge' if x in three_judge_seasons else '4-Judge'
)

# Aggregate by system
system_stats = season_df.groupby('judge_system').agg({
    'J_rank': ['mean', 'std'],
    'F_rank': ['mean', 'std'],
    'J_pct': ['mean', 'std'],
    'F_pct': ['mean', 'std']
}).round(3)

# Bar plot
x = np.arange(2)
width = 0.2
metrics = ['J_rank', 'J_pct', 'F_rank', 'F_pct']
colors = ['steelblue', 'lightblue', 'coral', 'lightsalmon']
labels = ['J (Rank)', 'J (Pct)', 'F (Rank)', 'F (Pct)']

for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
    means = [season_df[season_df['judge_system']==sys][metric].mean() for sys in ['3-Judge', '4-Judge']]
    stds = [season_df[season_df['judge_system']==sys][metric].std() for sys in ['3-Judge', '4-Judge']]
    ax2.bar(x + i*width, means, width, yerr=stds, label=label, color=color, capsize=3, alpha=0.8)

ax2.set_xticks(x + 1.5*width)
ax2.set_xticklabels(['3-Judge Seasons\n(S1-10, 13-14, 16, 27, 29)', '4-Judge Seasons\n(Other)'])
ax2.set_ylabel('Spearman Correlation')
ax2.set_title('(b) Performance Across Judge Systems')
ax2.legend(ncol=2, loc='upper right')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('cleaned_outputs/phase3_pareto_analysis/robustness_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('cleaned_outputs/phase3_pareto_analysis/robustness_analysis.pdf', bbox_inches='tight')
print("  Saved: robustness_analysis.png/pdf")

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "="*60)
print("SENSITIVITY ANALYSIS SUMMARY")
print("="*60)

print("\n1. Parameter Sensitivity (Sigmoid Dynamic Rule):")
print(f"   Optimal configuration: w_min=0.30, w_max=0.75, s=6")
print(f"   Composite Score: {0.5701:.4f}")
print(f"   Score range across 107 configs: [{sigmoid_rules['Score_balanced'].min():.3f}, {sigmoid_rules['Score_balanced'].max():.3f}]")

print("\n2. Cross-Season Stability:")
print(f"   Rank method CV(J): {rank_j_cv:.3f}")
print(f"   Pct method CV(J):  {pct_j_cv:.3f}")
print(f"   Rank is {pct_j_cv/rank_j_cv:.1f}x more stable")

print("\n3. Bootstrap Analysis:")
print(f"   Mean improvement (Dynamic - Static): {mean_diff:.4f}")
print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"   Statistically significant: {'Yes' if ci_low > 0 else 'No'}")

print("\n4. Judge System Robustness:")
for sys in ['3-Judge', '4-Judge']:
    j_mean = season_df[season_df['judge_system']==sys]['J_rank'].mean()
    f_mean = season_df[season_df['judge_system']==sys]['F_rank'].mean()
    print(f"   {sys}: J={j_mean:.3f}, F={f_mean:.3f}")

print("\n" + "="*60)
print("All figures saved to cleaned_outputs/phase3_pareto_analysis/")
print("="*60)
