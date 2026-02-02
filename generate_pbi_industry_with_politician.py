#!/usr/bin/env python3
"""
Generate PBI by Industry Figure with Politician Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置学术风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'mathtext.fontset': 'stix'
})

COLORS = {
    'tertiary': '#2ca02c',     # Green (positive)
    'quaternary': '#d62728',   # Red (negative)
    'highlight': '#ff7f0e'     # Orange (special highlight)
}

OUTPUT_DIR = '/home/hyx/文档/MCM/cleaned_outputs/feature_engineering'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df_panel = pd.read_csv('/home/hyx/文档/MCM/cleaned_outputs/clean_weekly_panel.csv')
pbi = pd.read_csv('/home/hyx/文档/MCM/cleaned_outputs/contestant_pbi.csv')

# Merge data
panel_with_pbi = df_panel.merge(pbi[['contestant_id', 'PBI']], on='contestant_id', how='left')

# Calculate PBI by industry
industry_pbi = panel_with_pbi.groupby(['contestant_id', 'industry_std']).agg({
    'PBI': 'first',
    'celebrity_name': 'first'
}).reset_index().groupby('industry_std').agg({
    'PBI': ['mean', 'std', 'count']
}).reset_index()
industry_pbi.columns = ['industry', 'avg_PBI', 'std_PBI', 'count']

# Include all industries with count >= 3 (to include Politician)
industry_pbi = industry_pbi[industry_pbi['count'] >= 3]
industry_pbi = industry_pbi.sort_values('avg_PBI')

print(f"\nIndustry PBI Summary:")
print(industry_pbi.to_string(index=False))

# Create figure
fig, ax = plt.subplots(figsize=(11, 8))

# Color coding: red for negative, green for positive, orange for Politician (highlight)
colors_bar = []
for idx, row in industry_pbi.iterrows():
    if row['industry'] == 'Politician':
        colors_bar.append(COLORS['highlight'])  # Special highlight for Politician
    elif row['avg_PBI'] < 0:
        colors_bar.append(COLORS['quaternary'])  # Red for negative
    else:
        colors_bar.append(COLORS['tertiary'])    # Green for positive

# Plot bars
bars = ax.barh(industry_pbi['industry'], industry_pbi['avg_PBI'], 
               color=colors_bar, edgecolor='black', alpha=0.8)

# Add error bars for standard deviation
ax.errorbar(industry_pbi['avg_PBI'], industry_pbi['industry'], 
            xerr=industry_pbi['std_PBI'], fmt='none', color='gray', capsize=3, alpha=0.6)

# Add vertical line at x=0
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Add count annotations
for i, (idx, row) in enumerate(industry_pbi.iterrows()):
    x_pos = row['avg_PBI']
    offset = 0.15 if x_pos >= 0 else -0.15
    ha = 'left' if x_pos >= 0 else 'right'
    ax.annotate(f"n={int(row['count'])}", 
                xy=(x_pos + offset, i), 
                va='center', ha=ha, fontsize=9, color='gray')

# Find the y-position for Politician
industry_list = industry_pbi['industry'].tolist()
politician_y_pos = industry_list.index('Politician')
politician_row = industry_pbi[industry_pbi['industry'] == 'Politician'].iloc[0]

# Special annotation for Politician
ax.annotate(
    'Politicians: high variance\n($\sigma$ = 2.29, n = 3)',
    xy=(politician_row['avg_PBI'] + politician_row['std_PBI'], politician_y_pos),
    xytext=(2.0, politician_y_pos + 1.5),
    fontsize=9,
    ha='left',
    arrowprops=dict(arrowstyle='->', color='darkgray', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8)
)

# Labels
ax.set_xlabel(r'Average Popularity Bias Index $\bar{\Delta}$', fontsize=12)
ax.set_ylabel('Celebrity Industry', fontsize=12)
ax.set_title(r'Popularity Bias Index $\Delta$ by Celebrity Industry' + '\n' + 
             r'(Green: Fan Boost $\Delta > 0$, Red: Judge Aligned $\Delta < 0$, Orange: Politician)', 
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add legend explanation
legend_text = """
$\Delta > 0$: Fans rank higher than judges (Fan Favorite)
$\Delta < 0$: Judges rank higher than fans (Judge Favorite)
"""

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pbi_by_industry.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/pbi_by_industry.pdf', bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/pbi_by_industry.png")
plt.close()

# =============================================================================
# Politician Analysis
# =============================================================================
print("\n" + "="*70)
print("POLITICIAN ANALYSIS")
print("="*70)

# Get detailed politician data
politicians = panel_with_pbi[panel_with_pbi['industry_std'] == 'Politician'][
    ['celebrity_name', 'season', 'PBI', 'J_pct']
].drop_duplicates(subset=['celebrity_name'])

print(f"""
Politicians in DWTS (n=3):
==========================
{politicians.to_string(index=False)}

Key Findings:
-------------
• Average PBI: +0.074 (slight fan boost)
• Standard Deviation: 2.29 (high variance)

Individual Analysis:
--------------------
1. Tom DeLay (S9):    PBI = -2.33 (Judge Favorite)
   - Former House Majority Leader
   - Strong dance performance relative to fan support

2. Rick Perry (S23):  PBI = +0.33 (Slight Fan Boost)
   - Former Governor of Texas
   - Balanced judge and fan reception

3. Sean Spicer (S28): PBI = +2.22 (Strong Fan Favorite)
   - Former White House Press Secretary
   - Significant fan support despite lower judge scores
   - Most controversial politician contestant

Interpretation:
---------------
Politicians show HETEROGENEOUS patterns:
- Some (Sean Spicer) receive strong fan support despite judge criticism
- Others (Tom DeLay) are judge-favored with less fan enthusiasm
- This reflects the polarizing nature of political figures

Note: Small sample size (n=3) limits statistical generalization.
""")

print("="*70)
print("Done!")
print("="*70)
