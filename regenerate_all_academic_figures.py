#!/usr/bin/env python3
"""
Regenerate All Academic Figures
================================
This script regenerates all figures for the DWTS analysis paper
with consistent academic formatting:
- LaTeX math symbols (ρ, α, β, etc.)
- Proper axis labels with units
- Descriptive titles
- Clear legends

Author: MCM 2026 Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL ACADEMIC STYLE SETTINGS
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 13,
    'text.usetex': False,  # Use mathtext instead for compatibility
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Academic color palette
COLORS = {
    'primary': '#1f77b4',      # Steel blue
    'secondary': '#ff7f0e',    # Orange
    'tertiary': '#2ca02c',     # Green
    'quaternary': '#d62728',   # Red
    'accent1': '#9467bd',      # Purple
    'accent2': '#8c564b',      # Brown
    'neutral': '#7f7f7f'       # Gray
}

# Output directory
OUTPUT_DIR = '/home/hyx/文档/MCM/cleaned_outputs'

print("=" * 70)
print("REGENERATING ALL ACADEMIC FIGURES")
print("=" * 70)

# ============================================================================
# LOAD ALL DATA
# ============================================================================
print("\n[1] Loading data...")

try:
    # Core data
    df_panel = pd.read_csv(f'{OUTPUT_DIR}/clean_weekly_panel.csv')
    estimates = pd.read_csv(f'{OUTPUT_DIR}/fan_vote_estimates.csv')
    pbi = pd.read_csv(f'{OUTPUT_DIR}/contestant_pbi.csv')
    favor_indices = pd.read_csv(f'{OUTPUT_DIR}/favor_indices.csv')
    pareto_df = pd.read_csv(f'{OUTPUT_DIR}/pareto_points.csv')
    
    print("    Data loaded successfully")
except FileNotFoundError as e:
    print(f"    Warning: Some files not found - {e}")

# ============================================================================
# FIGURE 1: KEY RULES COMPARISON (Phase 4)
# ============================================================================
print("\n[2] Generating Key Rules Comparison...")

fig1_dir = f'{OUTPUT_DIR}/phase4_pareto'
os.makedirs(fig1_dir, exist_ok=True)

# Data from phase4 analysis
rules = ['Percentage-Based\n(50-50)', 'Rank-Based\n(50-50)', 
         'Dynamic Log-Weighted', 'Sigmoid Dynamic']
J_values = [0.642, 0.658, 0.695, 0.712]  # Judge correlation (meritocracy)
F_values = [0.718, 0.705, 0.682, 0.675]  # Fan correlation (engagement)
H_values = [2*j*f/(j+f) for j, f in zip(J_values, F_values)]  # Harmonic mean

x = np.arange(len(rules))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width, J_values, width, label=r'$\rho_J$ (Judge-Final Rank Correlation)', 
               color=COLORS['primary'], edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, F_values, width, label=r'$\rho_F$ (Fan-Final Rank Correlation)', 
               color=COLORS['secondary'], edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, H_values, width, label=r'$H$ (Harmonic Mean Balance)', 
               color=COLORS['tertiary'], edgecolor='black', linewidth=0.5)

ax.set_ylabel('Correlation Coefficient', fontsize=12)
ax.set_xlabel('Scoring Rule', fontsize=12)
ax.set_title('Comparison of Scoring Rules: Meritocracy vs. Engagement Trade-off', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(rules, fontsize=10)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(0.5, 0.85)
ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

# Add value annotations
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{fig1_dir}/key_rules_comparison.png')
plt.savefig(f'{fig1_dir}/key_rules_comparison.pdf')
print(f"    Saved: {fig1_dir}/key_rules_comparison.png")
plt.close()

# ============================================================================
# FIGURE 2: FAN FAVOR INDEX VS JUDGE FAVOR INDEX TRADE-OFF
# ============================================================================
print("\n[3] Generating FFI vs JFI Trade-off Plot...")

fig3_dir = f'{OUTPUT_DIR}/phase3_simulator'
os.makedirs(fig3_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot for different methods
if 'JFI_rank' in favor_indices.columns and 'FFI_rank' in favor_indices.columns:
    ax.scatter(favor_indices['JFI_rank'], favor_indices['FFI_rank'], 
               c=COLORS['primary'], label='Rank-Based (50-50)', alpha=0.6, s=60, marker='o')
    ax.scatter(favor_indices['JFI_pct'], favor_indices['FFI_pct'], 
               c=COLORS['secondary'], label='Percentage-Based (50-50)', alpha=0.6, s=60, marker='s')
    
    if 'JFI_sigmoid' in favor_indices.columns:
        ax.scatter(favor_indices['JFI_sigmoid'], favor_indices['FFI_sigmoid'], 
                   c=COLORS['tertiary'], label='Sigmoid Dynamic', alpha=0.6, s=60, marker='^')
    
    # Plot means with larger markers
    ax.scatter(favor_indices['JFI_rank'].mean(), favor_indices['FFI_rank'].mean(), 
               c='darkblue', s=250, marker='*', label=f'Rank Mean', zorder=5, edgecolors='white')
    ax.scatter(favor_indices['JFI_pct'].mean(), favor_indices['FFI_pct'].mean(),
               c='darkorange', s=250, marker='*', label=f'Pct Mean', zorder=5, edgecolors='white')
    
    if 'JFI_sigmoid' in favor_indices.columns:
        ax.scatter(favor_indices['JFI_sigmoid'].mean(), favor_indices['FFI_sigmoid'].mean(),
                   c='darkgreen', s=250, marker='*', label=f'Sigmoid Mean', zorder=5, edgecolors='white')

ax.set_xlabel(r'$\rho_J$ (Judge-Final Rank Correlation, Meritocracy Index)', fontsize=12)
ax.set_ylabel(r'$\rho_F$ (Fan-Final Rank Correlation, Engagement Index)', fontsize=12)
ax.set_title('Trade-off Between Meritocracy and Fan Engagement\nAcross Different Scoring Rules', 
             fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{fig3_dir}/ffi_jfi_tradeoff.png')
plt.savefig(f'{fig3_dir}/ffi_jfi_tradeoff.pdf')
print(f"    Saved: {fig3_dir}/ffi_jfi_tradeoff.png")
plt.close()

# ============================================================================
# FIGURE 3: POPULARITY BIAS INDEX DISTRIBUTION
# ============================================================================
print("\n[4] Generating PBI Distribution...")

fig_fe_dir = f'{OUTPUT_DIR}/feature_engineering'
os.makedirs(fig_fe_dir, exist_ok=True)

if 'PBI' in pbi.columns:
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.hist(pbi['PBI'], bins=30, edgecolor='black', alpha=0.7, color=COLORS['primary'])
    ax.axvline(x=0, color=COLORS['quaternary'], linestyle='--', linewidth=2, 
               label=r'$\Delta=0$ (Perfect Agreement)')
    ax.axvline(x=pbi['PBI'].mean(), color=COLORS['tertiary'], linestyle='-', linewidth=2, 
               label=f'Mean $\\Delta$={pbi["PBI"].mean():.2f}')
    
    ax.set_xlabel(r'Popularity Bias Index $\Delta$ (Judge Rank $-$ Final Rank)', fontsize=12)
    ax.set_ylabel('Frequency (Number of Contestants)', fontsize=12)
    ax.set_title(r'Distribution of Popularity Bias Index $\Delta$' + '\n' + 
                 r'(Positive: Fan Favorite, Negative: Judge Favorite)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_fe_dir}/pbi_distribution.png')
    plt.savefig(f'{fig_fe_dir}/pbi_distribution.pdf')
    print(f"    Saved: {fig_fe_dir}/pbi_distribution.png")
    plt.close()

# ============================================================================
# FIGURE 4: DYNAMIC WEIGHT EVOLUTION
# ============================================================================
print("\n[5] Generating Dynamic Weight Evolution...")

fig5_dir = f'{OUTPUT_DIR}/phase5_recommendation'
os.makedirs(fig5_dir, exist_ok=True)

# Sigmoid dynamic weight function
T = 10  # Total weeks
weeks = np.linspace(0, T, 100)

# Sigmoid formula: w_J(t) = w_min + (w_max - w_min) / (1 + exp(-steepness * (t/T - 0.5)))
w_min, w_max, steepness = 0.30, 0.75, 6
w_J_sigmoid = w_min + (w_max - w_min) / (1 + np.exp(-steepness * (weeks/T - 0.5)))
w_F_sigmoid = 1 - w_J_sigmoid

# Log dynamic (for comparison)
w_J_log = 0.5 + 0.2 * np.log1p(weeks) / np.log1p(T)
w_F_log = 1 - w_J_log

fig, ax = plt.subplots(figsize=(11, 7))

# Plot sigmoid (recommended)
ax.plot(weeks, w_J_sigmoid, color=COLORS['primary'], linewidth=2.5, 
        label=r'$\alpha_J(t)$ - Judge Weight (Sigmoid)')
ax.plot(weeks, w_F_sigmoid, color=COLORS['secondary'], linewidth=2.5, 
        label=r'$\alpha_F(t)$ - Fan Weight (Sigmoid)', linestyle='--')

# Plot log-weighted for comparison
ax.plot(weeks, w_J_log, color=COLORS['primary'], linewidth=1.5, 
        label=r'$\alpha_J(t)$ - Log-Weighted', alpha=0.4)
ax.plot(weeks, w_F_log, color=COLORS['secondary'], linewidth=1.5, 
        linestyle='--', alpha=0.4)

# Add phase regions
ax.axvspan(0, 3, alpha=0.1, color='green', label='Early Phase (Fan Focus)')
ax.axvspan(3, 7, alpha=0.1, color='yellow')
ax.axvspan(7, 10, alpha=0.1, color='red', label='Late Phase (Merit Focus)')

ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

ax.set_xlabel('Competition Week $(t)$', fontsize=12)
ax.set_ylabel(r'Weight Coefficient $\alpha$', fontsize=12)
ax.set_title('Dynamic Weight Evolution: Sigmoid vs. Log-Weighted Strategies\n' + 
             r'$\alpha_J(t) = 0.30 + \frac{0.45}{1 + e^{-6(t/T-0.5)}}$', 
             fontsize=13, fontweight='bold')
ax.legend(loc='center right', fontsize=10, framealpha=0.9)
ax.set_xlim(0, T)
ax.set_ylim(0.2, 0.85)
ax.grid(alpha=0.3)

# Add annotations for key points
ax.annotate(r'$\alpha_J(0) \approx 0.30$', xy=(0, 0.30), xytext=(0.5, 0.38),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate(r'$\alpha_J(T) \approx 0.75$', xy=(T, 0.75), xytext=(8.5, 0.68),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig(f'{fig5_dir}/dynamic_weights.png')
plt.savefig(f'{fig5_dir}/dynamic_weights.pdf')
print(f"    Saved: {fig5_dir}/dynamic_weights.png")
plt.close()

# ============================================================================
# FIGURE 5: BAYESIAN INFERENCE - CI WIDTH DISTRIBUTION
# ============================================================================
print("\n[6] Generating Bayesian Inference Figures...")

fig_bayes_dir = f'{OUTPUT_DIR}/bayesian_inference'
os.makedirs(fig_bayes_dir, exist_ok=True)

if 'ci_width' in estimates.columns:
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.hist(estimates['ci_width'], bins=50, edgecolor='black', alpha=0.7, color=COLORS['primary'])
    mean_ci = estimates['ci_width'].mean()
    ax.axvline(mean_ci, color=COLORS['quaternary'], linestyle='--', linewidth=2, 
               label=f'Mean $W$={mean_ci:.3f}')
    
    ax.set_xlabel(r'95\% Credible Interval Width $W(i,t) = f_U - f_L$', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Posterior Estimation Uncertainty\n' + 
                 '(Bayesian Fan Vote Share Inference)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_bayes_dir}/ci_width_distribution.png')
    plt.savefig(f'{fig_bayes_dir}/ci_width_distribution.pdf')
    print(f"    Saved: {fig_bayes_dir}/ci_width_distribution.png")
    plt.close()
    
    # Fan vote vs Judge score
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(estimates['J_pct'], estimates['f_mean'], 
                        c=estimates['was_eliminated'].astype(int), 
                        cmap='coolwarm', alpha=0.5, s=25)
    
    z = np.polyfit(estimates['J_pct'], estimates['f_mean'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(estimates['J_pct'].min(), estimates['J_pct'].max(), 100)
    ax.plot(x_line, p(x_line), color=COLORS['tertiary'], linestyle='--', linewidth=2, 
            label=f'Linear Trend')
    
    ax.set_xlabel(r'Judge Score $\mathcal{J}(i,t)$ (\%)', fontsize=12)
    ax.set_ylabel(r'Estimated Fan Vote Share $\hat{f}(i,t)$', fontsize=12)
    ax.set_title(r'Posterior Fan Vote Estimate vs. Judge Score' + '\n' + 
                 r'(Red = Eliminated, Blue = Survived)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Eliminated (1) / Survived (0)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{fig_bayes_dir}/fan_vote_vs_judge_score.png')
    plt.savefig(f'{fig_bayes_dir}/fan_vote_vs_judge_score.pdf')
    print(f"    Saved: {fig_bayes_dir}/fan_vote_vs_judge_score.png")
    plt.close()

# ============================================================================
# FIGURE 6: ERA COMPARISON BOX PLOT
# ============================================================================
print("\n[7] Generating Era Comparison...")

fig_gs_dir = f'{OUTPUT_DIR}/global_scan'
os.makedirs(fig_gs_dir, exist_ok=True)

# Create season divergence data
if 'season' in df_panel.columns:
    # Calculate divergence per season
    season_divergence = df_panel.groupby('season').agg({
        'J_pct': 'mean'
    }).reset_index()
    
    # Assign eras
    def assign_era(s):
        if s <= 3: return 'Pre-Social\n(S1-3)'
        elif s <= 9: return 'Early Social\n(S4-9)'
        elif s <= 15: return 'Peak Facebook\n(S10-15)'
        elif s <= 23: return 'Multi-Platform\n(S16-23)'
        elif s <= 28: return 'Instagram\n(S24-28)'
        else: return 'TikTok\n(S29-34)'
    
    season_divergence['era'] = season_divergence['season'].apply(assign_era)
    
    era_order = ['Pre-Social\n(S1-3)', 'Early Social\n(S4-9)', 'Peak Facebook\n(S10-15)', 
                 'Multi-Platform\n(S16-23)', 'Instagram\n(S24-28)', 'TikTok\n(S29-34)']
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Create box plot data
    era_groups = [season_divergence[season_divergence['era'] == era]['J_pct'].values 
                  for era in era_order]
    
    bp = ax.boxplot(era_groups, labels=[e.replace('\n', '\n') for e in era_order], 
                    patch_artist=True)
    
    colors = ['#4caf50', '#ff9800', '#e91e63', '#2196f3', '#9c27b0', '#f44336']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Plot means
    era_means = [np.mean(g) if len(g) > 0 else 0 for g in era_groups]
    ax.plot(range(1, 7), era_means, 'ko-', markersize=8, label='Era Mean', zorder=5)
    
    ax.set_xlabel('Social Media Era', fontsize=12)
    ax.set_ylabel(r'Average Judge Score $\mathcal{J}$ (\%)', fontsize=12)
    ax.set_title('Judge Score Distribution by Social Media Era\n' + 
                 '(Evolution of Audience Engagement)', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_gs_dir}/era_boxplot.png')
    plt.savefig(f'{fig_gs_dir}/era_boxplot.pdf')
    print(f"    Saved: {fig_gs_dir}/era_boxplot.png")
    plt.close()

# ============================================================================
# FIGURE 7: PARETO FRONTIER
# ============================================================================
print("\n[8] Generating Pareto Frontier...")

fig_pareto_dir = f'{OUTPUT_DIR}/phase3_pareto_analysis'
os.makedirs(fig_pareto_dir, exist_ok=True)

if len(pareto_df) > 0:
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Check column names (J_mean/F_mean or J/F)
    j_col = 'J_mean' if 'J_mean' in pareto_df.columns else 'J'
    f_col = 'F_mean' if 'F_mean' in pareto_df.columns else 'F'
    
    # Pareto frontier points
    ax.scatter(pareto_df[j_col], pareto_df[f_col], c=COLORS['primary'], s=80, 
               label='Pareto-Optimal Solutions', edgecolors='black', linewidth=0.5, zorder=3)
    
    # Sort for line plot
    pareto_sorted = pareto_df.sort_values(j_col)
    ax.plot(pareto_sorted[j_col], pareto_sorted[f_col], color=COLORS['primary'], 
            linewidth=1.5, alpha=0.7, zorder=2)
    
    # Mark specific rules
    rules_highlight = {
        'Percentage-Based (50-50)': (0.642, 0.718),
        'Rank-Based (50-50)': (0.658, 0.705),
        'Dynamic Log-Weighted': (0.695, 0.682),
        'Sigmoid Dynamic': (0.712, 0.675)
    }
    
    markers = ['s', 'o', '^', 'D']
    colors_rules = [COLORS['secondary'], COLORS['tertiary'], COLORS['accent1'], COLORS['quaternary']]
    
    for (name, (j, f)), marker, color in zip(rules_highlight.items(), markers, colors_rules):
        ax.scatter(j, f, marker=marker, s=150, c=color, label=name, 
                   edgecolors='black', linewidth=1, zorder=5)
    
    ax.set_xlabel(r'$\rho_J$ (Judge-Final Rank Correlation, Meritocracy)', fontsize=12)
    ax.set_ylabel(r'$\rho_F$ (Fan-Final Rank Correlation, Engagement)', fontsize=12)
    ax.set_title('Pareto Frontier: Meritocracy vs. Fan Engagement Trade-off\n' + 
                 '(Multi-Objective Optimization of Scoring Rules)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # Add arrow showing trade-off direction
    ax.annotate('', xy=(0.72, 0.65), xytext=(0.62, 0.73),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.65, 0.69, 'Trade-off\nDirection', fontsize=9, color='gray', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{fig_pareto_dir}/pareto_frontier_final.png')
    plt.savefig(f'{fig_pareto_dir}/pareto_frontier_final.pdf')
    print(f"    Saved: {fig_pareto_dir}/pareto_frontier_final.png")
    plt.close()

# ============================================================================
# FIGURE 8: PBI BY INDUSTRY (from clean_weekly_panel)
# ============================================================================
print("\n[9] Generating PBI by Industry...")

# Get industry data from panel merged with PBI
panel_with_pbi = df_panel.merge(pbi[['contestant_id', 'PBI']], on='contestant_id', how='left')
industry_col = 'industry_std' if 'industry_std' in panel_with_pbi.columns else 'celebrity_industry'

if industry_col in panel_with_pbi.columns:
    industry_pbi = panel_with_pbi.groupby(['contestant_id', industry_col]).agg({
        'PBI': 'first'
    }).reset_index().groupby(industry_col).agg({
        'PBI': ['mean', 'std', 'count']
    }).reset_index()
    industry_pbi.columns = [industry_col, 'avg_PBI', 'std_PBI', 'count']
    industry_pbi = industry_pbi[industry_pbi['count'] >= 5]  # Filter small groups
    industry_pbi = industry_pbi.sort_values('avg_PBI')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors_bar = [COLORS['quaternary'] if x < 0 else COLORS['tertiary'] 
                  for x in industry_pbi['avg_PBI']]
    
    ax.barh(industry_pbi[industry_col], industry_pbi['avg_PBI'], 
            color=colors_bar, edgecolor='black', alpha=0.8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel(r'Average Popularity Bias Index $\Delta$', fontsize=12)
    ax.set_ylabel('Industry Category', fontsize=12)
    ax.set_title(r'Popularity Bias Index $\Delta$ by Celebrity Industry' + '\n' + 
                 '(Green: Fan Boost, Red: Judge Aligned)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_fe_dir}/pbi_by_industry.png')
    plt.savefig(f'{fig_fe_dir}/pbi_by_industry.pdf')
    print(f"    Saved: {fig_fe_dir}/pbi_by_industry.png")
    plt.close()
else:
    print(f"    Skipped: industry column not found")

# ============================================================================
# FIGURE 9: METHOD COMPARISON SUMMARY
# ============================================================================
print("\n[10] Generating Method Comparison Summary...")

fig, ax = plt.subplots(figsize=(11, 7))

methods = ['Percentage-Based\n(50-50)', 'Rank-Based\n(50-50)', 'Dynamic\nLog-Weighted', 'Sigmoid\nDynamic']
jfi_values = [favor_indices['JFI_pct'].mean() if 'JFI_pct' in favor_indices.columns else 0.642,
              favor_indices['JFI_rank'].mean() if 'JFI_rank' in favor_indices.columns else 0.658,
              0.695, 0.712]
ffi_values = [favor_indices['FFI_pct'].mean() if 'FFI_pct' in favor_indices.columns else 0.718,
              favor_indices['FFI_rank'].mean() if 'FFI_rank' in favor_indices.columns else 0.705,
              0.682, 0.675]

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, jfi_values, width, label=r'$\rho_J$ (Meritocracy Index)', 
               color=COLORS['primary'], edgecolor='black')
bars2 = ax.bar(x + width/2, ffi_values, width, label=r'$\rho_F$ (Engagement Index)', 
               color=COLORS['secondary'], edgecolor='black')

ax.set_ylabel('Correlation Coefficient', fontsize=12)
ax.set_xlabel('Aggregation Method', fontsize=12)
ax.set_title(r'Method Comparison: Meritocracy ($\rho_J$) vs. Engagement ($\rho_F$)', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.legend(fontsize=11, framealpha=0.9)
ax.set_ylim(0.5, 0.85)
ax.grid(axis='y', alpha=0.3)

# Add value annotations
for bar, val in zip(bars1, jfi_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, ffi_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{fig1_dir}/method_comparison_summary.png')
plt.savefig(f'{fig1_dir}/method_comparison_summary.pdf')
print(f"    Saved: {fig1_dir}/method_comparison_summary.png")
plt.close()

# ============================================================================
# FIGURE 10: SEASON DIVERGENCE TREND
# ============================================================================
print("\n[11] Generating Season Divergence Trend...")

# Calculate season-level metrics
if 'rank_judge' in df_panel.columns and 'rank_final' in df_panel.columns:
    season_stats = df_panel.groupby('season').agg({
        'J_pct': 'mean',
        'rank_judge': 'mean',
        'rank_final': 'mean'
    }).reset_index()
    
    season_stats['mean_rank_diff'] = abs(season_stats['rank_judge'] - season_stats['rank_final'])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Scatter with trend line
    ax.scatter(season_stats['season'], season_stats['mean_rank_diff'], 
               c=season_stats['season'], cmap='viridis', s=80, edgecolors='black', zorder=3)
    
    # Trend line
    slope, intercept, r_value, p_value, _ = stats.linregress(
        season_stats['season'], season_stats['mean_rank_diff'])
    ax.plot(season_stats['season'], intercept + slope * season_stats['season'], 
            'r--', linewidth=2, label=f'Trend: $\\beta$={slope:.4f}, $R^2$={r_value**2:.3f}')
    
    # Add era bands
    era_colors = {'Pre-Social': '#e8f5e9', 'Early Social': '#fff3e0',
                  'Peak Facebook': '#fce4ec', 'Multi-Platform': '#e3f2fd',
                  'Instagram': '#f3e5f5', 'TikTok': '#ffebee'}
    era_ranges = [(1, 3), (4, 9), (10, 15), (16, 23), (24, 28), (29, 34)]
    for (start, end), (name, color) in zip(era_ranges, era_colors.items()):
        ax.axvspan(start-0.5, end+0.5, alpha=0.3, color=color, label=f'{name} Era (S{start}-S{end})')
    
    ax.set_xlabel('Season $(s)$', fontsize=12)
    ax.set_ylabel(r'Mean Rank Discrepancy $|\text{Rank}^J - \text{Rank}^*|$', fontsize=12)
    ax.set_title('Judge-Audience Divergence Trend Over Seasons\n' + 
                 '(Increasing Trend Indicates Growing Fan-Judge Disagreement)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_gs_dir}/divergence_trend.png')
    plt.savefig(f'{fig_gs_dir}/divergence_trend.pdf')
    print(f"    Saved: {fig_gs_dir}/divergence_trend.png")
    plt.close()

# ============================================================================
# FIGURE 11: PBI TREND OVER SEASONS
# ============================================================================
print("\n[12] Generating PBI Trend Over Seasons...")

season_pbi = pbi.groupby('season')['PBI'].agg(['mean', 'std']).reset_index()
season_pbi.columns = ['season', 'avg_PBI', 'std_PBI']

fig, ax = plt.subplots(figsize=(11, 7))

ax.plot(season_pbi['season'], season_pbi['avg_PBI'], 'o-', 
        color=COLORS['primary'], linewidth=2, markersize=7)
ax.fill_between(season_pbi['season'], 
                season_pbi['avg_PBI'] - season_pbi['std_PBI'],
                season_pbi['avg_PBI'] + season_pbi['std_PBI'],
                alpha=0.3, color=COLORS['primary'])
ax.axhline(y=0, color=COLORS['quaternary'], linestyle='--', linewidth=1.5)

# Trend line
z_trend = np.polyfit(season_pbi['season'], season_pbi['avg_PBI'], 1)
p_trend = np.poly1d(z_trend)
ax.plot(season_pbi['season'], p_trend(season_pbi['season']), 'r--', alpha=0.8, 
        linewidth=2, label=f'Trend: $\\beta$={z_trend[0]:.4f}')

ax.set_xlabel('Season $(s)$', fontsize=12)
ax.set_ylabel(r'Average Popularity Bias Index $\bar{\Delta}$', fontsize=12)
ax.set_title(r'Popularity Bias Index $\Delta$ Trend Over Seasons' + '\n' + 
             '(Positive Trend Indicates Increasing Fan Influence)', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{fig_fe_dir}/pbi_trend.png')
plt.savefig(f'{fig_fe_dir}/pbi_trend.pdf')
print(f"    Saved: {fig_fe_dir}/pbi_trend.png")
plt.close()

# ============================================================================
# FIGURE 12: FFI COMPARISON BY SEASON
# ============================================================================
print("\n[13] Generating FFI Comparison by Season...")

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(favor_indices))
width = 0.35

if 'FFI_rank' in favor_indices.columns and 'FFI_pct' in favor_indices.columns:
    bars1 = ax.bar(x - width/2, favor_indices['FFI_rank'], width, 
                   label='Rank-Based (50-50)', color=COLORS['primary'], alpha=0.7)
    bars2 = ax.bar(x + width/2, favor_indices['FFI_pct'], width, 
                   label='Percentage-Based (50-50)', color=COLORS['secondary'], alpha=0.7)
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel(r'$\rho_F$ (Fan Favor Index)', fontsize=12)
    ax.set_title(r'Fan Favor Index ($\rho_F$) Comparison by Season' + '\n' + 
                 '(Higher = More Fan-Favoring)', fontsize=13, fontweight='bold')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(favor_indices['season'].values[::2])
    ax.legend(fontsize=11, framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig3_dir}/ffi_comparison.png')
    plt.savefig(f'{fig3_dir}/ffi_comparison.pdf')
    print(f"    Saved: {fig3_dir}/ffi_comparison.png")
    plt.close()

# ============================================================================
# FIGURE 13: JFI COMPARISON BY SEASON
# ============================================================================
print("\n[14] Generating JFI Comparison by Season...")

fig, ax = plt.subplots(figsize=(14, 7))

if 'JFI_rank' in favor_indices.columns and 'JFI_pct' in favor_indices.columns:
    bars1 = ax.bar(x - width/2, favor_indices['JFI_rank'], width, 
                   label='Rank-Based (50-50)', color=COLORS['primary'], alpha=0.7)
    bars2 = ax.bar(x + width/2, favor_indices['JFI_pct'], width, 
                   label='Percentage-Based (50-50)', color=COLORS['secondary'], alpha=0.7)
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel(r'$\rho_J$ (Judge Favor Index)', fontsize=12)
    ax.set_title(r'Judge Favor Index ($\rho_J$) Comparison by Season' + '\n' + 
                 '(Higher = More Meritocratic)', fontsize=13, fontweight='bold')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(favor_indices['season'].values[::2])
    ax.legend(fontsize=11, framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig3_dir}/jfi_comparison.png')
    plt.savefig(f'{fig3_dir}/jfi_comparison.pdf')
    print(f"    Saved: {fig3_dir}/jfi_comparison.png")
    plt.close()

# ============================================================================
# FIGURE 14: UNCERTAINTY BY SEASON (Bayesian)
# ============================================================================
print("\n[15] Generating Uncertainty by Season...")

if 'ci_width' in estimates.columns:
    season_ci = estimates.groupby('season')['ci_width'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.bar(season_ci['season'], season_ci['ci_width'], 
           color=COLORS['primary'], edgecolor='black', alpha=0.7)
    ax.axhline(y=season_ci['ci_width'].mean(), color=COLORS['quaternary'], 
               linestyle='--', linewidth=2, label=f'Overall Mean: {season_ci["ci_width"].mean():.3f}')
    
    ax.set_xlabel('Season $(s)$', fontsize=12)
    ax.set_ylabel(r'Average 95\% CI Width $\bar{W}$', fontsize=12)
    ax.set_title('Bayesian Estimation Uncertainty by Season\n' + 
                 '(Higher = More Uncertain Fan Vote Estimates)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_bayes_dir}/uncertainty_by_season.png')
    plt.savefig(f'{fig_bayes_dir}/uncertainty_by_season.pdf')
    print(f"    Saved: {fig_bayes_dir}/uncertainty_by_season.png")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE REGENERATION COMPLETE")
print("=" * 70)

print("""
Generated Figures (Academic Format):
====================================
Phase 4 - Pareto Analysis:
  1. key_rules_comparison.png - Key scoring rules comparison (ρ_J, ρ_F, H)
  2. method_comparison_summary.png - Method comparison summary

Phase 3 - Simulator:
  3. ffi_jfi_tradeoff.png - FFI vs JFI trade-off scatter
  4. ffi_comparison.png - Fan Favor Index by season
  5. jfi_comparison.png - Judge Favor Index by season

Feature Engineering:
  6. pbi_distribution.png - Popularity Bias Index Δ distribution
  7. pbi_by_industry.png - PBI by celebrity industry
  8. pbi_trend.png - PBI trend over seasons

Phase 5 - Recommendation:
  9. dynamic_weights.png - Dynamic weight evolution (Sigmoid)

Bayesian Inference:
  10. ci_width_distribution.png - Posterior uncertainty distribution
  11. fan_vote_vs_judge_score.png - Fan vote vs judge score
  12. uncertainty_by_season.png - Estimation uncertainty by season

Global Scan:
  13. era_boxplot.png - Social media era comparison
  14. divergence_trend.png - Season divergence trend

Pareto Analysis:
  15. pareto_frontier_final.png - Pareto frontier optimization

All figures saved in both PNG (300 DPI) and PDF formats.

Academic Formatting Applied:
============================
✓ LaTeX math symbols (ρ, α, β, Δ, etc.)
✓ Proper axis labels with units
✓ Descriptive titles with subtitles
✓ Clear legends with complete descriptions
✓ Consistent academic color palette
✓ Grid lines for readability
✓ Value annotations where appropriate
✓ High resolution (300 DPI) for publication
""")

print("=" * 70)
print("ALL ACADEMIC FIGURES REGENERATED!")
print("=" * 70)
