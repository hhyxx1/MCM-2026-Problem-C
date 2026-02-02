"""
Dancing with the Stars - Feature Engineering
=============================================
Phase 1, Step 2: Key Feature - Popularity Bias Index (Δ)

Tasks:
1. Calculate Δ = R^𝒥_i - R^*_i  (formerly PBI)
   数学定义: 人气偏差指数 = 评委排名 - 最终名次
   - Positive Δ: Fan favorite (ranked better by audience than judges)
   - Negative Δ: Judge favorite (ranked better by judges than audience)

2. Patch 1: Partner Impact Analysis
   - Calculate historical average Δ for each Professional Dancer
   - Identify "Star Makers" who consistently boost celebrity performance

3. Patch 1B: Celebrity Covariates Analysis
   - Analyze how Age, Industry, Region affect Judge Scores 𝒥
   - Prepare features for mixed-effects modeling

数学符号对应 (Symbol Mapping):
    PBI      -> Δ         人气偏差指数
    J_pct    -> 𝒥(i,t)    评委得分百分比
    rank_judge -> R^𝒥     评委排名
    rank_final -> R^*     最终名次
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# ============================================================================
# LOAD CLEANED DATA
# ============================================================================
print("="*70)
print("PHASE 1, STEP 2: POPULARITY BIAS INDEX (PBI) ANALYSIS")
print("="*70)

df_panel = pd.read_csv('/home/hyx/文档/MCM/cleaned_outputs/clean_weekly_panel.csv')
df_wide = pd.read_csv('/home/hyx/文档/MCM/cleaned_outputs/clean_judge_scores_wide.csv')
df_season = pd.read_csv('/home/hyx/文档/MCM/cleaned_outputs/season_summary.csv')

print(f"\n[1] Loaded data:")
print(f"    Panel data: {len(df_panel)} observations")
print(f"    Wide data: {len(df_wide)} contestants")

# ============================================================================
# STEP 1: CALCULATE WEEKLY JUDGE RANKINGS
# ============================================================================
print("\n[2] Calculating weekly judge rankings...")

def calculate_weekly_rankings(df_panel):
    """
    For each season-week, calculate:
    - Judge Rank: Based on J_pct (higher score = better rank = lower number)
    """
    df = df_panel.copy()
    
    # Calculate Judge Rank within each season-week (1 = best)
    df['rank_judge'] = df.groupby(['season', 'week'])['J_pct'].rank(ascending=False, method='min')
    
    return df

df_panel = calculate_weekly_rankings(df_panel)

# Verify rankings
sample = df_panel[(df_panel['season'] == 1) & (df_panel['week'] == 1)][['celebrity_name', 'J_pct', 'rank_judge']]
print("    Sample rankings (Season 1, Week 1):")
print(sample.sort_values('rank_judge').to_string(index=False))

# ============================================================================
# STEP 2: CALCULATE FINAL PLACEMENT RANK
# ============================================================================
print("\n[3] Calculating final placement ranks...")

# Final placement is already in the data (lower = better, 1 = winner)
# We need Rank_Final for each contestant
df_panel['rank_final'] = df_panel['placement']

# ============================================================================
# STEP 3: CALCULATE Δ (Popularity Bias Index, 人气偏差指数)
# ============================================================================
print("\n[4] Calculating Popularity Bias Index (Δ, formerly PBI)...")

# 数学公式: Δ = R^𝒥 - R^*
# Positive Δ: Contestant ranked better by audience (final) than by judges
# Negative Δ: Contestant ranked better by judges than by audience

# Calculate average judge rank across all weeks for each contestant
contestant_avg_judge_rank = df_panel.groupby(['season', 'contestant_id', 'celebrity_name', 
                                               'ballroom_partner', 'placement']).agg({
    'rank_judge': 'mean',
    'J_pct': 'mean',
    'week': 'max'  # last week competed
}).reset_index()

contestant_avg_judge_rank.columns = ['season', 'contestant_id', 'celebrity_name', 
                                      'ballroom_partner', 'rank_final', 
                                      'avg_rank_judge', 'avg_J_pct', 'last_week']

# Calculate PBI
contestant_avg_judge_rank['PBI'] = contestant_avg_judge_rank['avg_rank_judge'] - contestant_avg_judge_rank['rank_final']

print(f"    Calculated PBI for {len(contestant_avg_judge_rank)} contestants")
print(f"\n    PBI Statistics:")
print(f"    Mean PBI: {contestant_avg_judge_rank['PBI'].mean():.3f}")
print(f"    Std PBI: {contestant_avg_judge_rank['PBI'].std():.3f}")
print(f"    Min PBI: {contestant_avg_judge_rank['PBI'].min():.3f} (Most judge-favored)")
print(f"    Max PBI: {contestant_avg_judge_rank['PBI'].max():.3f} (Most fan-favored)")

# ============================================================================
# STEP 4: IDENTIFY EXTREME CASES
# ============================================================================
print("\n[5] Identifying extreme PBI cases...")

# Top 10 Fan Favorites (Highest positive PBI)
print("\n    TOP 10 FAN FAVORITES (High positive PBI - ranked better by fans):")
fan_favorites = contestant_avg_judge_rank.nlargest(10, 'PBI')[
    ['season', 'celebrity_name', 'ballroom_partner', 'rank_final', 'avg_rank_judge', 'PBI', 'avg_J_pct']
]
print(fan_favorites.to_string(index=False))

# Top 10 Judge Favorites (Lowest negative PBI)
print("\n    TOP 10 JUDGE FAVORITES (Low negative PBI - ranked better by judges):")
judge_favorites = contestant_avg_judge_rank.nsmallest(10, 'PBI')[
    ['season', 'celebrity_name', 'ballroom_partner', 'rank_final', 'avg_rank_judge', 'PBI', 'avg_J_pct']
]
print(judge_favorites.to_string(index=False))

# ============================================================================
# STEP 5: PATCH 1 - PARTNER IMPACT ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("PATCH 1: PARTNER IMPACT ANALYSIS")
print("="*70)

# Calculate average PBI for each professional dancer
partner_stats = contestant_avg_judge_rank.groupby('ballroom_partner').agg({
    'PBI': ['mean', 'std', 'count'],
    'avg_J_pct': 'mean',
    'rank_final': 'mean'
}).reset_index()

partner_stats.columns = ['ballroom_partner', 'avg_PBI', 'std_PBI', 'num_seasons', 'avg_J_pct', 'avg_placement']

# Only consider partners with at least 3 seasons
partner_stats_filtered = partner_stats[partner_stats['num_seasons'] >= 3].copy()
partner_stats_filtered = partner_stats_filtered.sort_values('avg_PBI', ascending=False)

print(f"\n[6] Partner statistics (min 3 seasons):")
print(f"    Total professional dancers: {len(partner_stats)}")
print(f"    Dancers with 3+ seasons: {len(partner_stats_filtered)}")

# Identify Star Makers (positive avg PBI - their celebrities outperform judge expectations)
print("\n    STAR MAKERS (Highest avg PBI - celebrities outperform judge expectations):")
star_makers = partner_stats_filtered.nlargest(10, 'avg_PBI')
print(star_makers.to_string(index=False))

# Identify Judge Favorites Partners (negative avg PBI)
print("\n    JUDGE-ALIGNED PARTNERS (Lowest avg PBI - consistent with judge rankings):")
judge_aligned = partner_stats_filtered.nsmallest(10, 'avg_PBI')
print(judge_aligned.to_string(index=False))

# Calculate partner "boost" coefficient
# Positive coefficient = Star Maker effect
partner_stats_filtered['star_maker_coefficient'] = (
    partner_stats_filtered['avg_PBI'] / partner_stats_filtered['avg_PBI'].std()
)

# ============================================================================
# STEP 6: DETAILED PARTNER ANALYSIS
# ============================================================================
print("\n[7] Detailed analysis of top partners...")

# Get detailed history for top star makers
top_partners = star_makers['ballroom_partner'].head(5).tolist()

for partner in top_partners:
    partner_history = contestant_avg_judge_rank[
        contestant_avg_judge_rank['ballroom_partner'] == partner
    ][['season', 'celebrity_name', 'rank_final', 'avg_rank_judge', 'PBI', 'avg_J_pct']]
    
    print(f"\n    {partner}'s History:")
    print(partner_history.sort_values('season').to_string(index=False))

# ============================================================================
# STEP 7: PATCH 1B - CELEBRITY COVARIATES ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("PATCH 1B: CELEBRITY COVARIATES ANALYSIS")
print("="*70)

# Merge PBI data with panel data for covariate analysis
df_analysis = df_panel.merge(
    contestant_avg_judge_rank[['contestant_id', 'PBI', 'avg_J_pct']], 
    on='contestant_id', 
    how='left'
)

# Analysis by Industry
print("\n[8] PBI by Industry:")
industry_pbi = df_analysis.groupby(['contestant_id', 'industry_std']).agg({
    'PBI': 'first',
    'J_pct': 'mean'
}).reset_index().groupby('industry_std').agg({
    'PBI': ['mean', 'std', 'count'],
    'J_pct': 'mean'
}).reset_index()

industry_pbi.columns = ['industry', 'avg_PBI', 'std_PBI', 'count', 'avg_J_pct']
industry_pbi = industry_pbi.sort_values('avg_PBI', ascending=False)
print(industry_pbi.to_string(index=False))

# Analysis by Age
print("\n[9] PBI by Age Group:")
age_pbi = df_analysis.groupby(['contestant_id', 'age_bin']).agg({
    'PBI': 'first',
    'J_pct': 'mean'
}).reset_index().groupby('age_bin').agg({
    'PBI': ['mean', 'std', 'count'],
    'J_pct': 'mean'
}).reset_index()

age_pbi.columns = ['age_bin', 'avg_PBI', 'std_PBI', 'count', 'avg_J_pct']
print(age_pbi.to_string(index=False))

# Analysis by Region (US vs Non-US)
print("\n[10] PBI by Region:")
region_pbi = df_analysis.groupby(['contestant_id', 'is_US']).agg({
    'PBI': 'first',
    'J_pct': 'mean'
}).reset_index().groupby('is_US').agg({
    'PBI': ['mean', 'std', 'count'],
    'J_pct': 'mean'
}).reset_index()

region_pbi.columns = ['is_US', 'avg_PBI', 'std_PBI', 'count', 'avg_J_pct']
region_pbi['region'] = region_pbi['is_US'].map({1: 'US', 0: 'Non-US'})
print(region_pbi[['region', 'avg_PBI', 'std_PBI', 'count', 'avg_J_pct']].to_string(index=False))

# Analysis by Season (temporal trend)
print("\n[11] PBI Trend by Season:")
season_pbi = contestant_avg_judge_rank.groupby('season').agg({
    'PBI': ['mean', 'std'],
    'avg_J_pct': 'mean'
}).reset_index()

season_pbi.columns = ['season', 'avg_PBI', 'std_PBI', 'avg_J_pct']
print(season_pbi.to_string(index=False))

# ============================================================================
# STEP 8: STATISTICAL TESTS
# ============================================================================
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

# Test if PBI differs significantly by industry
print("\n[12] ANOVA: PBI by Industry")
industry_groups = [group['PBI'].values for name, group in 
                   df_analysis.groupby(['contestant_id', 'industry_std']).agg({'PBI': 'first'}).reset_index().groupby('industry_std')]
f_stat, p_value = stats.f_oneway(*[g[1]['PBI'].values for g in 
                                    df_analysis.groupby(['contestant_id', 'industry_std']).agg({'PBI': 'first'}).reset_index().groupby('industry_std')])
print(f"    F-statistic: {f_stat:.4f}")
print(f"    p-value: {p_value:.4f}")
print(f"    Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

# Test if PBI differs by region
print("\n[13] T-test: PBI by Region (US vs Non-US)")
us_pbi = df_analysis[df_analysis['is_US'] == 1].groupby('contestant_id')['PBI'].first()
non_us_pbi = df_analysis[df_analysis['is_US'] == 0].groupby('contestant_id')['PBI'].first()
t_stat, p_value = stats.ttest_ind(us_pbi, non_us_pbi)
print(f"    t-statistic: {t_stat:.4f}")
print(f"    p-value: {p_value:.4f}")
print(f"    Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

# Correlation: PBI vs Average J%
print("\n[14] Correlation: PBI vs Average Judge Score")
corr, p_value = stats.pearsonr(contestant_avg_judge_rank['PBI'], contestant_avg_judge_rank['avg_J_pct'])
print(f"    Pearson r: {corr:.4f}")
print(f"    p-value: {p_value:.6f}")
print(f"    Interpretation: {'Negative correlation - lower scores get more fan boost' if corr < 0 else 'Positive correlation'}")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

import os
output_dir = '/home/hyx/文档/MCM/cleaned_outputs'

# Save PBI data for each contestant
contestant_avg_judge_rank.to_csv(f'{output_dir}/contestant_pbi.csv', index=False)
print(f"    Saved: contestant_pbi.csv ({len(contestant_avg_judge_rank)} rows)")

# Save partner statistics
partner_stats_filtered.to_csv(f'{output_dir}/partner_stats.csv', index=False)
print(f"    Saved: partner_stats.csv ({len(partner_stats_filtered)} rows)")

# Save weekly panel with rankings
df_panel.to_csv(f'{output_dir}/clean_weekly_panel.csv', index=False)
print(f"    Updated: clean_weekly_panel.csv with rank_judge column")

# Save covariate analysis summary
covariate_summary = {
    'industry': industry_pbi.to_dict('records'),
    'age': age_pbi.to_dict('records'),
    'region': region_pbi[['region', 'avg_PBI', 'std_PBI', 'count', 'avg_J_pct']].to_dict('records'),
    'season_trend': season_pbi.to_dict('records')
}

import json
with open(f'{output_dir}/covariate_analysis.json', 'w') as f:
    json.dump(covariate_summary, f, indent=2)
print(f"    Saved: covariate_analysis.json")

# ============================================================================
# STEP 10: GENERATE VISUALIZATIONS
# ============================================================================
print("\n[15] Generating visualizations...")

import os
img_dir = f'{output_dir}/feature_engineering'
os.makedirs(img_dir, exist_ok=True)

# Prepare data
industry_order = industry_pbi.sort_values('avg_PBI')['industry'].tolist()
industry_colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in industry_pbi.sort_values('avg_PBI')['avg_PBI']]
z_trend = np.polyfit(season_pbi['season'], season_pbi['avg_PBI'], 1)
p_trend = np.poly1d(z_trend)
top_n = 15
top_star_makers = partner_stats_filtered.nlargest(top_n, 'avg_PBI')
star_colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_star_makers['avg_PBI']]

# --- Individual Plots ---

# Plot 1: PBI Distribution (Individual)
fig1, ax1_ind = plt.subplots(figsize=(8, 6))
ax1_ind.hist(contestant_avg_judge_rank['PBI'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax1_ind.axvline(x=0, color='red', linestyle='--', linewidth=2, label=r'$\Delta$=0 (Judge=Fan)')
ax1_ind.axvline(x=contestant_avg_judge_rank['PBI'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean={contestant_avg_judge_rank["PBI"].mean():.2f}')
ax1_ind.set_xlabel(r'Popularity Bias Index ($\Delta$ = R$^{\mathcal{J}}$ - R*)')
ax1_ind.set_ylabel('Frequency')
ax1_ind.set_title(r'Distribution of $\Delta$' + '\n(Positive = Fan Favorite, Negative = Judge Favorite)')
ax1_ind.legend()
plt.tight_layout()
plt.savefig(f'{img_dir}/pbi_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/pbi_distribution.png")

# Plot 2: PBI by Industry (Individual)
fig2, ax2_ind = plt.subplots(figsize=(8, 6))
ax2_ind.barh(industry_order, industry_pbi.sort_values('avg_PBI')['avg_PBI'], color=industry_colors, edgecolor='black')
ax2_ind.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2_ind.set_xlabel('Average Δ')
ax2_ind.set_title('Δ by Industry\n(Green = Fan Boost, Red = Judge Aligned)')
plt.tight_layout()
plt.savefig(f'{img_dir}/pbi_by_industry.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/pbi_by_industry.png")

# Plot 3: PBI Trend over Seasons (Individual)
fig3, ax3_ind = plt.subplots(figsize=(8, 6))
ax3_ind.plot(season_pbi['season'], season_pbi['avg_PBI'], 'o-', color='steelblue', linewidth=2, markersize=6)
ax3_ind.fill_between(season_pbi['season'], 
                  season_pbi['avg_PBI'] - season_pbi['std_PBI'],
                  season_pbi['avg_PBI'] + season_pbi['std_PBI'],
                  alpha=0.3)
ax3_ind.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax3_ind.set_xlabel('Season (s)')
ax3_ind.set_ylabel('Average Δ')
ax3_ind.set_title('Δ Trend Over Seasons\n(Positive trend = increasing fan influence)')
ax3_ind.plot(season_pbi['season'], p_trend(season_pbi['season']), 'r--', alpha=0.8, label=f'Trend: slope={z_trend[0]:.3f}')
ax3_ind.legend()
plt.tight_layout()
plt.savefig(f'{img_dir}/pbi_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/pbi_trend.png")

# Plot 4: Top Star Makers (Individual)
fig4, ax4_ind = plt.subplots(figsize=(8, 6))
ax4_ind.barh(range(top_n), top_star_makers['avg_PBI'], color=star_colors, edgecolor='black')
ax4_ind.set_yticks(range(top_n))
ax4_ind.set_yticklabels(top_star_makers['ballroom_partner'])
ax4_ind.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4_ind.set_xlabel(r'Average $\Delta$')
ax4_ind.set_title(f'Top {top_n} "Star Makers"\n' + r'(Pro dancers with high $u_p$ effect)')
for i, (idx, row) in enumerate(top_star_makers.iterrows()):
    ax4_ind.annotate(f'n={int(row["num_seasons"])}', 
                 xy=(row['avg_PBI'] + 0.1, i),
                 va='center', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig(f'{img_dir}/star_makers.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/star_makers.png")

# Plot 5: PBI vs Judge Score scatter (Individual)
fig5, ax5_ind = plt.subplots(figsize=(10, 8))
scatter = ax5_ind.scatter(contestant_avg_judge_rank['avg_J_pct'], 
                     contestant_avg_judge_rank['PBI'],
                     c=contestant_avg_judge_rank['season'],
                     cmap='viridis', alpha=0.6, edgecolors='white', linewidth=0.5)
ax5_ind.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
z_score = np.polyfit(contestant_avg_judge_rank['avg_J_pct'], contestant_avg_judge_rank['PBI'], 1)
p_score = np.poly1d(z_score)
x_line = np.linspace(contestant_avg_judge_rank['avg_J_pct'].min(), contestant_avg_judge_rank['avg_J_pct'].max(), 100)
ax5_ind.plot(x_line, p_score(x_line), 'r-', linewidth=2, label=f'Trend: r={corr:.3f}')
plt.colorbar(scatter, label='Season (s)')
ax5_ind.set_xlabel(r'Average $\mathcal{J}$(i,t) (%)')
ax5_ind.set_ylabel(r'Popularity Bias Index ($\Delta$)')
ax5_ind.set_title(r'$\Delta$ vs $\mathcal{J}$' + '\nNegative correlation: Lower-scoring contestants get more fan boost')
ax5_ind.legend()
plt.tight_layout()
plt.savefig(f'{img_dir}/pbi_vs_score.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/pbi_vs_score.png")

# --- Panel Plot (Combined) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: PBI Distribution
ax1 = axes[0, 0]
ax1.hist(contestant_avg_judge_rank['PBI'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label=r'$\Delta$=0 (Judge=Fan)')
ax1.axvline(x=contestant_avg_judge_rank['PBI'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean={contestant_avg_judge_rank["PBI"].mean():.2f}')
ax1.set_xlabel(r'Popularity Bias Index ($\Delta$ = R$^{\mathcal{J}}$ - R*)')
ax1.set_ylabel('Frequency')
ax1.set_title(r'Distribution of $\Delta$' + '\n(Positive = Fan Favorite, Negative = Judge Favorite)')
ax1.legend()

# Plot 2: PBI by Industry
ax2 = axes[0, 1]
bars = ax2.barh(industry_order, industry_pbi.sort_values('avg_PBI')['avg_PBI'], color=industry_colors, edgecolor='black')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Average Δ')
ax2.set_title('Δ by Industry\n(Green = Fan Boost, Red = Judge Aligned)')

# Plot 3: PBI Trend over Seasons
ax3 = axes[1, 0]
ax3.plot(season_pbi['season'], season_pbi['avg_PBI'], 'o-', color='steelblue', linewidth=2, markersize=6)
ax3.fill_between(season_pbi['season'], 
                  season_pbi['avg_PBI'] - season_pbi['std_PBI'],
                  season_pbi['avg_PBI'] + season_pbi['std_PBI'],
                  alpha=0.3)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax3.set_xlabel('Season (s)')
ax3.set_ylabel('Average Δ')
ax3.set_title('Δ Trend Over Seasons\n(Positive trend = increasing fan influence)')
ax3.plot(season_pbi['season'], p_trend(season_pbi['season']), 'r--', alpha=0.8, label=f'Trend: slope={z_trend[0]:.3f}')
ax3.legend()

# Plot 4: Top Star Makers
ax4 = axes[1, 1]
bars = ax4.barh(range(top_n), top_star_makers['avg_PBI'], color=star_colors, edgecolor='black')
ax4.set_yticks(range(top_n))
ax4.set_yticklabels(top_star_makers['ballroom_partner'])
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel(r'Average $\Delta$')
ax4.set_title(f'Top {top_n} "Star Makers"\n' + r'(Pro dancers with high $u_p$ effect)')
for i, (idx, row) in enumerate(top_star_makers.iterrows()):
    ax4.annotate(f'n={int(row["num_seasons"])}', 
                 xy=(row['avg_PBI'] + 0.1, i),
                 va='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(f'{output_dir}/pbi_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{img_dir}/panel_all.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: pbi_analysis.png")
print(f"    Saved: {img_dir}/panel_all.png")

# Plot 5: PBI vs Judge Score scatter (Original location)
fig2, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(contestant_avg_judge_rank['avg_J_pct'], 
                     contestant_avg_judge_rank['PBI'],
                     c=contestant_avg_judge_rank['season'],
                     cmap='viridis', alpha=0.6, edgecolors='white', linewidth=0.5)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.plot(x_line, p_score(x_line), 'r-', linewidth=2, label=f'Trend: r={corr:.3f}')
plt.colorbar(scatter, label='Season (s)')
ax.set_xlabel(r'Average $\mathcal{J}$(i,t) (%)')
ax.set_ylabel(r'Popularity Bias Index ($\Delta$)')
ax.set_title(r'$\Delta$ vs $\mathcal{J}$' + '\nNegative correlation: Lower-scoring contestants get more fan boost')
ax.legend()

plt.savefig(f'{output_dir}/pbi_vs_score.png', dpi=150, bbox_inches='tight')
print(f"    Saved: pbi_vs_score.png")

plt.close('all')

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

print("""
KEY FINDINGS:

1. PBI (Popularity Bias Index) Analysis:
   - Formula: PBI = Rank_Judge - Rank_Final
   - Positive PBI = Fan favorite (outperforms judge expectations)
   - Negative PBI = Judge favorite (underperforms fan expectations)

2. Star Maker Partners:
   - Professional dancers with high average PBI consistently help
     celebrities outperform judge rankings
   - These partners have a "fan boost" effect

3. Industry Effects:
   - Different industries show different PBI patterns
   - This suggests fan voting is influenced by celebrity background

4. Temporal Trend:
   - PBI trend over seasons shows how fan influence has evolved
   - Important for understanding the need for rule reform

5. Correlation with Judge Score:
   - Negative correlation suggests fans tend to "rescue" lower-scoring
     contestants, creating judge-audience divergence

OUTPUT FILES:
  - contestant_pbi.csv: PBI for each contestant
  - partner_stats.csv: Star maker statistics
  - covariate_analysis.json: Industry/Age/Region analysis
  - pbi_analysis.png: 4-panel visualization
  - pbi_vs_score.png: PBI vs Judge Score scatter plot
""")

print("="*70)
print("PBI ANALYSIS COMPLETE!")
print("="*70)
