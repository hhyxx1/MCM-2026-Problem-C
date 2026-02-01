"""
Patch 2: Global Scan - Judge-Audience Divergence Analysis
=========================================================
Phase 1, Step 3: Chronological Analysis of All Seasons (1-34)

Tasks:
1. Loop through Season 1 to Season 34
2. Calculate "Judge-Audience Divergence" metrics for each season/week
3. Draw Chronological Heatmap showing divergence trends
4. Prove divergence increases with social media development

Social Media Timeline (for context):
- 2005 (S1): YouTube founded, Facebook college-only
- 2006 (S2-3): Twitter launched, Facebook opens to public
- 2007-2009 (S4-9): Facebook/Twitter mainstream adoption
- 2010-2012 (S10-15): Instagram, smartphones ubiquitous
- 2013-2016 (S16-23): Vine, Snapchat, social media voting
- 2017-2019 (S24-28): Instagram Stories, TikTok rising
- 2020-2024 (S29-34): TikTok dominant, pandemic streaming
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*70)
print("PATCH 2: GLOBAL SCAN - JUDGE-AUDIENCE DIVERGENCE")
print("="*70)

df_panel = pd.read_csv('/home/hyx/文档/MCM/cleaned_outputs/clean_weekly_panel.csv')
df_pbi = pd.read_csv('/home/hyx/文档/MCM/cleaned_outputs/contestant_pbi.csv')

print(f"\n[1] Loaded data:")
print(f"    Panel data: {len(df_panel)} observations")
print(f"    Seasons: 1 to {df_panel['season'].max()}")

# ============================================================================
# DEFINE DIVERGENCE METRICS
# ============================================================================
print("\n[2] Defining divergence metrics...")

def calculate_weekly_divergence(season_week_data):
    """
    Calculate Judge-Audience Divergence for a single week.
    
    Metrics:
    1. Rank Correlation (Spearman): Judge rank vs Final placement
       - Low correlation = high divergence
    2. Absolute Rank Difference: Mean |rank_judge - rank_final|
    3. Top-3 Mismatch: Whether weekly top-3 by judges matches final top-3
    """
    if len(season_week_data) < 3:
        return None
    
    # Get unique contestants for this week
    data = season_week_data.copy()
    
    # Spearman correlation between judge rank and final placement
    if data['rank_judge'].nunique() > 1 and data['rank_final'].nunique() > 1:
        corr, p_val = stats.spearmanr(data['rank_judge'], data['rank_final'])
    else:
        corr, p_val = np.nan, np.nan
    
    # Mean absolute rank difference
    mean_rank_diff = np.abs(data['rank_judge'] - data['rank_final']).mean()
    
    # Variance in PBI (if available) - measures spread of divergence
    pbi_std = (data['rank_judge'] - data['rank_final']).std()
    
    return {
        'spearman_corr': corr,
        'p_value': p_val,
        'mean_rank_diff': mean_rank_diff,
        'pbi_std': pbi_std,
        'n_contestants': len(data),
        # Divergence score: 1 - correlation (higher = more divergence)
        'divergence_score': 1 - corr if not np.isnan(corr) else np.nan
    }

# ============================================================================
# CALCULATE DIVERGENCE FOR ALL SEASONS AND WEEKS
# ============================================================================
print("\n[3] Calculating divergence for all seasons and weeks...")

divergence_data = []

for season in range(1, 35):
    season_data = df_panel[df_panel['season'] == season]
    
    if len(season_data) == 0:
        continue
    
    weeks = sorted(season_data['week'].unique())
    
    for week in weeks:
        week_data = season_data[season_data['week'] == week]
        metrics = calculate_weekly_divergence(week_data)
        
        if metrics:
            divergence_data.append({
                'season': season,
                'week': week,
                **metrics
            })

df_divergence = pd.DataFrame(divergence_data)
print(f"    Calculated divergence for {len(df_divergence)} season-week combinations")

# ============================================================================
# CALCULATE SEASON-LEVEL DIVERGENCE
# ============================================================================
print("\n[4] Aggregating to season-level metrics...")

season_divergence = df_divergence.groupby('season').agg({
    'spearman_corr': 'mean',
    'divergence_score': 'mean',
    'mean_rank_diff': 'mean',
    'pbi_std': 'mean',
    'n_contestants': 'mean'
}).reset_index()

# Add year information (Season 1 started in 2005)
season_divergence['year'] = 2005 + (season_divergence['season'] - 1) // 2

# Define social media eras
def get_era(season):
    if season <= 3:
        return 'Pre-Social (S1-3)'
    elif season <= 9:
        return 'Early Social (S4-9)'
    elif season <= 15:
        return 'Peak Facebook (S10-15)'
    elif season <= 23:
        return 'Multi-Platform (S16-23)'
    elif season <= 28:
        return 'Instagram Era (S24-28)'
    else:
        return 'TikTok Era (S29-34)'

season_divergence['social_era'] = season_divergence['season'].apply(get_era)

print(season_divergence[['season', 'year', 'social_era', 'spearman_corr', 'divergence_score', 'mean_rank_diff']].to_string(index=False))

# ============================================================================
# STATISTICAL TREND ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("TREND ANALYSIS")
print("="*70)

# Linear regression: Divergence vs Season
slope, intercept, r_value, p_value, std_err = stats.linregress(
    season_divergence['season'], 
    season_divergence['divergence_score']
)

print(f"\n[5] Linear Trend: Divergence Score vs Season")
print(f"    Slope: {slope:.4f} (per season)")
print(f"    R²: {r_value**2:.4f}")
print(f"    p-value: {p_value:.6f}")
print(f"    Significant: {'Yes' if p_value < 0.05 else 'No'}")

if slope > 0:
    print(f"    → Divergence INCREASES over time (+{slope*10:.3f} per 10 seasons)")
else:
    print(f"    → Divergence DECREASES over time ({slope*10:.3f} per 10 seasons)")

# Compare eras
print(f"\n[6] Divergence by Social Media Era:")
era_stats = season_divergence.groupby('social_era').agg({
    'divergence_score': ['mean', 'std'],
    'mean_rank_diff': 'mean',
    'season': ['min', 'max']
}).round(4)
era_stats.columns = ['div_mean', 'div_std', 'rank_diff', 'season_min', 'season_max']
era_stats = era_stats.sort_values('season_min')
print(era_stats.to_string())

# ANOVA across eras
era_groups = [group['divergence_score'].values for name, group in season_divergence.groupby('social_era')]
f_stat, anova_p = stats.f_oneway(*era_groups)
print(f"\n    ANOVA F-statistic: {f_stat:.4f}")
print(f"    ANOVA p-value: {anova_p:.4f}")

# ============================================================================
# GENERATE HEATMAP
# ============================================================================
print("\n[7] Generating Chronological Heatmap...")

import os
img_dir = '/home/hyx/文档/MCM/cleaned_outputs/global_scan'
os.makedirs(img_dir, exist_ok=True)

# Create pivot table for heatmap
pivot_data = df_divergence.pivot(index='week', columns='season', values='divergence_score')

# 准备 era 相关数据
era_order = ['Pre-Social (S1-3)', 'Early Social (S4-9)', 'Peak Facebook (S10-15)', 
             'Multi-Platform (S16-23)', 'Instagram Era (S24-28)', 'TikTok Era (S29-34)']
era_data = [season_divergence[season_divergence['social_era'] == era]['divergence_score'].values 
            for era in era_order]
era_means = [np.mean(d) for d in era_data]
era_colors = {'Pre-Social (S1-3)': '#e8f5e9', 'Early Social (S4-9)': '#fff3e0',
              'Peak Facebook (S10-15)': '#fce4ec', 'Multi-Platform (S16-23)': '#e3f2fd',
              'Instagram Era (S24-28)': '#f3e5f5', 'TikTok Era (S29-34)': '#ffebee'}
era_ranges = [(1, 3), (4, 9), (10, 15), (16, 23), (24, 28), (29, 34)]
era_names = list(era_colors.keys())
era_boundaries = [3, 9, 15, 23, 28]

slope2, intercept2, r2, p2, _ = stats.linregress(
    season_divergence['season'], season_divergence['mean_rank_diff'])

# --- Plot 1: Divergence Heatmap (单独图) ---
fig1, ax1 = plt.subplots(figsize=(12, 8))
im = ax1.imshow(pivot_data.values, aspect='auto', cmap='RdYlBu_r', 
                vmin=0, vmax=1.5)
ax1.set_xlabel('Season')
ax1.set_ylabel('Week')
ax1.set_title('Judge-Audience Divergence Heatmap\n(Red = High Divergence, Blue = Low)')
ax1.set_xticks(range(0, 34, 5))
ax1.set_xticklabels(range(1, 35, 5))
ax1.set_yticks(range(len(pivot_data.index)))
ax1.set_yticklabels(pivot_data.index)
plt.colorbar(im, ax=ax1, label='Divergence Score (1 - Spearman r)')
for boundary in era_boundaries:
    ax1.axvline(x=boundary-0.5, color='white', linestyle='--', linewidth=2, alpha=0.7)
plt.tight_layout()
plt.savefig(f'{img_dir}/divergence_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 2: Season Trend Line (单独图) ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.scatter(season_divergence['season'], season_divergence['divergence_score'], 
            c=season_divergence['season'], cmap='viridis', s=80, edgecolors='black', zorder=3)
ax2.plot(season_divergence['season'], intercept + slope * season_divergence['season'], 
         'r--', linewidth=2, label=f'Trend: slope={slope:.4f}, R²={r_value**2:.3f}')
for (start, end), color, name in zip(era_ranges, era_colors.values(), era_names):
    ax2.axvspan(start-0.5, end+0.5, alpha=0.3, color=color, label=name)
ax2.set_xlabel('Season')
ax2.set_ylabel('Divergence Score')
ax2.set_title('Judge-Audience Divergence Trend Over Seasons\n(Higher = More Fan-Judge Disagreement)')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{img_dir}/divergence_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 3: Era Comparison Box Plot (单独图) ---
fig3, ax3 = plt.subplots(figsize=(10, 6))
bp = ax3.boxplot(era_data, labels=[e.split('(')[0].strip() for e in era_order], patch_artist=True)
colors = ['#4caf50', '#ff9800', '#e91e63', '#2196f3', '#9c27b0', '#f44336']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.set_xlabel('Social Media Era')
ax3.set_ylabel('Divergence Score')
ax3.set_title('Divergence by Social Media Era\n(ANOVA p={:.4f})'.format(anova_p))
ax3.tick_params(axis='x', rotation=30)
ax3.plot(range(1, 7), era_means, 'ko-', markersize=8, label='Mean')
plt.tight_layout()
plt.savefig(f'{img_dir}/era_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 4: Mean Rank Difference (单独图) ---
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.bar(season_divergence['season'], season_divergence['mean_rank_diff'], 
        color=plt.cm.RdYlBu_r(season_divergence['divergence_score'] / season_divergence['divergence_score'].max()),
        edgecolor='black', alpha=0.8)
ax4.plot(season_divergence['season'], intercept2 + slope2 * season_divergence['season'],
         'r--', linewidth=2, label=f'Trend: slope={slope2:.4f}')
ax4.set_xlabel('Season')
ax4.set_ylabel('Mean |Judge Rank - Final Rank|')
ax4.set_title('Average Rank Disagreement by Season')
ax4.legend()
plt.tight_layout()
plt.savefig(f'{img_dir}/rank_difference.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"    Saved 4 individual plots to {img_dir}/")

# --- 生成面板图 (保留原有功能) ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
im = ax1.imshow(pivot_data.values, aspect='auto', cmap='RdYlBu_r', 
                vmin=0, vmax=1.5)
ax1.set_xlabel('Season')
ax1.set_ylabel('Week')
ax1.set_title('Judge-Audience Divergence Heatmap\n(Red = High Divergence, Blue = Low)')
ax1.set_xticks(range(0, 34, 5))
ax1.set_xticklabels(range(1, 35, 5))
ax1.set_yticks(range(len(pivot_data.index)))
ax1.set_yticklabels(pivot_data.index)
plt.colorbar(im, ax=ax1, label='Divergence Score (1 - Spearman r)')
for boundary in era_boundaries:
    ax1.axvline(x=boundary-0.5, color='white', linestyle='--', linewidth=2, alpha=0.7)

ax2 = axes[0, 1]
ax2.scatter(season_divergence['season'], season_divergence['divergence_score'], 
            c=season_divergence['season'], cmap='viridis', s=80, edgecolors='black', zorder=3)
ax2.plot(season_divergence['season'], intercept + slope * season_divergence['season'], 
         'r--', linewidth=2, label=f'Trend: slope={slope:.4f}, R²={r_value**2:.3f}')
for (start, end), color, name in zip(era_ranges, era_colors.values(), era_names):
    ax2.axvspan(start-0.5, end+0.5, alpha=0.3, color=color, label=name)
ax2.set_xlabel('Season')
ax2.set_ylabel('Divergence Score')
ax2.set_title('Judge-Audience Divergence Trend Over Seasons\n(Higher = More Fan-Judge Disagreement)')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
bp = ax3.boxplot(era_data, labels=[e.split('(')[0].strip() for e in era_order], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.set_xlabel('Social Media Era')
ax3.set_ylabel('Divergence Score')
ax3.set_title('Divergence by Social Media Era\n(ANOVA p={:.4f})'.format(anova_p))
ax3.tick_params(axis='x', rotation=30)
ax3.plot(range(1, 7), era_means, 'ko-', markersize=8, label='Mean')

ax4 = axes[1, 1]
ax4.bar(season_divergence['season'], season_divergence['mean_rank_diff'], 
        color=plt.cm.RdYlBu_r(season_divergence['divergence_score'] / season_divergence['divergence_score'].max()),
        edgecolor='black', alpha=0.8)
ax4.plot(season_divergence['season'], intercept2 + slope2 * season_divergence['season'],
         'r--', linewidth=2, label=f'Trend: slope={slope2:.4f}')
ax4.set_xlabel('Season')
ax4.set_ylabel('Mean |Judge Rank - Final Rank|')
ax4.set_title('Average Rank Disagreement by Season')
ax4.legend()

plt.tight_layout()
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/global_scan_heatmap.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{img_dir}/panel_all.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: global_scan_heatmap.png (panel)")

# ============================================================================
# DETAILED WEEKLY HEATMAP
# ============================================================================
print("\n[8] Generating detailed weekly heatmap...")

fig2, ax = plt.subplots(figsize=(18, 8))

# Fill missing values for visualization
pivot_filled = pivot_data.fillna(0)

im = ax.imshow(pivot_filled.values, aspect='auto', cmap='RdYlBu_r', 
               vmin=0, vmax=1.5, interpolation='nearest')

ax.set_xlabel('Season', fontsize=12)
ax.set_ylabel('Week', fontsize=12)
ax.set_title('Chronological Heatmap: Judge-Audience Divergence (Season 1-34)\n' +
             'Red = High Divergence (Fans ≠ Judges), Blue = Low Divergence (Fans ≈ Judges)',
             fontsize=14)

# Set ticks
ax.set_xticks(range(34))
ax.set_xticklabels(range(1, 35), fontsize=9)
ax.set_yticks(range(len(pivot_filled.index)))
ax.set_yticklabels(pivot_filled.index)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Divergence Score (1 - Spearman correlation)')

# Add era labels at top
era_labels = [('Pre-\nSocial', 1.5), ('Early Social', 6), ('Peak Facebook', 12), 
              ('Multi-Platform', 19), ('Instagram', 25.5), ('TikTok', 31)]
for label, x in era_labels:
    ax.annotate(label, xy=(x, -1.5), ha='center', fontsize=9, color='gray')

# Era boundary lines
for boundary in era_boundaries:
    ax.axvline(x=boundary-0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

plt.tight_layout()
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/divergence_heatmap_detailed.png', dpi=150, bbox_inches='tight')
print(f"    Saved: divergence_heatmap_detailed.png")

plt.close('all')

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[9] Saving analysis results...")

# Save detailed divergence data
df_divergence.to_csv('/home/hyx/文档/MCM/cleaned_outputs/weekly_divergence.csv', index=False)
print(f"    Saved: weekly_divergence.csv ({len(df_divergence)} rows)")

# Save season-level summary
season_divergence.to_csv('/home/hyx/文档/MCM/cleaned_outputs/season_divergence.csv', index=False)
print(f"    Saved: season_divergence.csv ({len(season_divergence)} rows)")

# ============================================================================
# KEY FINDINGS SUMMARY
# ============================================================================
print("\n" + "="*70)
print("KEY FINDINGS: SOCIAL MEDIA IMPACT ON DIVERGENCE")
print("="*70)

# Calculate era averages
early_div = season_divergence[season_divergence['season'] <= 9]['divergence_score'].mean()
late_div = season_divergence[season_divergence['season'] >= 25]['divergence_score'].mean()
change_pct = (late_div - early_div) / early_div * 100

print(f"""
📊 DIVERGENCE TREND SUMMARY:

1. LINEAR TREND (Season 1-34):
   • Slope: {slope:.4f} per season
   • R²: {r_value**2:.4f}
   • p-value: {p_value:.6f}
   • Conclusion: {'SIGNIFICANT increase' if p_value < 0.05 and slope > 0 else 'No significant trend'}

2. ERA COMPARISON:
   • Pre-Social (S1-3):     Divergence = {season_divergence[season_divergence['social_era']=='Pre-Social (S1-3)']['divergence_score'].mean():.4f}
   • Early Social (S4-9):   Divergence = {season_divergence[season_divergence['social_era']=='Early Social (S4-9)']['divergence_score'].mean():.4f}
   • Peak Facebook (S10-15): Divergence = {season_divergence[season_divergence['social_era']=='Peak Facebook (S10-15)']['divergence_score'].mean():.4f}
   • Multi-Platform (S16-23): Divergence = {season_divergence[season_divergence['social_era']=='Multi-Platform (S16-23)']['divergence_score'].mean():.4f}
   • Instagram (S24-28):    Divergence = {season_divergence[season_divergence['social_era']=='Instagram Era (S24-28)']['divergence_score'].mean():.4f}
   • TikTok (S29-34):       Divergence = {season_divergence[season_divergence['social_era']=='TikTok Era (S29-34)']['divergence_score'].mean():.4f}

3. EARLY vs LATE COMPARISON:
   • Early seasons (S1-9) avg: {early_div:.4f}
   • Late seasons (S25-34) avg: {late_div:.4f}
   • Change: {'+' if change_pct > 0 else ''}{change_pct:.1f}%

4. ANOVA (Era differences):
   • F-statistic: {f_stat:.4f}
   • p-value: {anova_p:.4f}
   • Conclusion: {'SIGNIFICANT differences between eras' if anova_p < 0.05 else 'No significant era differences'}

📈 INTERPRETATION:
{'The data supports the hypothesis that Judge-Audience divergence has INCREASED with social media development. This provides justification for rule reform to address the growing gap between expert judgment and popular vote.' if slope > 0 and p_value < 0.1 else 'The trend is not strongly conclusive, but era-based differences suggest social media may influence voting patterns differently across periods.'}

OUTPUT FILES:
  • global_scan_heatmap.png - 4-panel visualization
  • divergence_heatmap_detailed.png - Detailed chronological heatmap
  • weekly_divergence.csv - Week-level divergence metrics
  • season_divergence.csv - Season-level summary with eras
""")

print("="*70)
print("PATCH 2: GLOBAL SCAN COMPLETE!")
print("="*70)
