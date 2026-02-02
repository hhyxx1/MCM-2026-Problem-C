#!/usr/bin/env python3
"""Update divergence_trend.png with Era (S赛季) in legend"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'mathtext.fontset': 'stix'
})

df = pd.read_csv('cleaned_outputs/clean_weekly_panel.csv')
fig_gs_dir = 'cleaned_outputs/global_scan'

# Calculate season stats
season_stats = df.groupby(['season', 'contestant_id']).agg({
    'J_pct': 'mean',
    'elimination_week': 'max'
}).reset_index()
season_stats['rank_judge'] = season_stats.groupby('season')['J_pct'].rank(ascending=False)
season_stats['rank_final'] = season_stats.groupby('season')['elimination_week'].rank(ascending=True, method='max')
season_stats = season_stats.groupby('season').agg({'rank_judge': 'mean', 'rank_final': 'mean'}).reset_index()
season_stats['mean_rank_diff'] = abs(season_stats['rank_judge'] - season_stats['rank_final'])

fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(season_stats['season'], season_stats['mean_rank_diff'], 
           c=season_stats['season'], cmap='viridis', s=80, edgecolors='black', zorder=3)

slope, intercept, r_value, p_value, _ = stats.linregress(
    season_stats['season'], season_stats['mean_rank_diff'])
ax.plot(season_stats['season'], intercept + slope * season_stats['season'], 
        'r--', linewidth=2, label=fr'Trend: $\beta$={slope:.4f}, $R^2$={r_value**2:.3f}')

# Era bands with (S赛季) format
era_colors = {'Pre-Social': '#e8f5e9', 'Early Social': '#fff3e0',
              'Peak Facebook': '#fce4ec', 'Multi-Platform': '#e3f2fd',
              'Instagram': '#f3e5f5', 'TikTok': '#ffebee'}
era_ranges = [(1, 3), (4, 9), (10, 15), (16, 23), (24, 28), (29, 34)]
for (start, end), (name, color) in zip(era_ranges, era_colors.items()):
    ax.axvspan(start-0.5, end+0.5, alpha=0.3, color=color, label=f'{name} Era (S{start}-S{end})')

ax.set_xlabel('Season $(s)$', fontsize=12)
ax.set_ylabel(r'Mean Rank Discrepancy $|\mathrm{Rank}^J - \mathrm{Rank}^*|$', fontsize=12)
ax.set_title('Judge-Audience Divergence Trend Over Seasons\n(Increasing Trend Indicates Growing Fan-Judge Disagreement)', 
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{fig_gs_dir}/divergence_trend.png', dpi=300)
plt.savefig(f'{fig_gs_dir}/divergence_trend.pdf')
print(f'Saved: {fig_gs_dir}/divergence_trend.png')
plt.close()
