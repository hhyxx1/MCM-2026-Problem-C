# Academic Figure Formatting Standards

## Overview

This document describes the academic formatting standards applied to all figures in the DWTS (Dancing with the Stars) voting rule analysis project for MCM 2026 Problem C.

## Naming Conventions

### Key Metrics
| Symbol | Description | LaTeX Code |
|--------|-------------|------------|
| $\rho_J$ | Judge-Final Rank Correlation (Meritocracy Index) | `$\rho_J$` |
| $\rho_F$ | Fan-Final Rank Correlation (Engagement Index) | `$\rho_F$` |
| $H$ | Harmonic Mean Balance | `$H = \frac{2\rho_J\rho_F}{\rho_J + \rho_F}$` |
| $\Delta$ | Popularity Bias Index | `$\Delta = R^J - R^*$` |
| $\alpha_J(t)$ | Time-varying Judge Weight | `$\alpha_J(t)$` |
| $\alpha_F(t)$ | Time-varying Fan Weight | `$\alpha_F(t) = 1 - \alpha_J(t)$` |
| $W$ | Credible Interval Width | `$W = f_U - f_L$` |
| $\mathcal{D}$ | Divergence Score | `$\mathcal{D}$` |

### Scoring Rules
| Rule Name | Description |
|-----------|-------------|
| Percentage-Based (50-50) | $S = 0.5 \times J\% + 0.5 \times F\%$ |
| Rank-Based (50-50) | $S = 0.5 \times R^J + 0.5 \times R^F$ |
| Dynamic Log-Weighted | $S = \alpha(t) \times J\% + (1-\alpha(t)) \times \log(1+F\%)$ |
| Sigmoid Dynamic | $\alpha_J(t) = 0.30 + \frac{0.45}{1 + e^{-6(t/T-0.5)}}$ |

## Style Guidelines

### Figure Size
- Single column: `figsize=(8, 6)` or `figsize=(10, 7)`
- Two-column span: `figsize=(12, 7)` or `figsize=(14, 8)`
- Panel plots: `figsize=(14, 10)` to `figsize=(16, 12)`

### Font Settings
```python
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})
```

### Color Palette
| Color Name | Hex Code | Usage |
|------------|----------|-------|
| Primary | `#1f77b4` | Main data points, bars |
| Secondary | `#ff7f0e` | Comparison data |
| Tertiary | `#2ca02c` | Positive values, trends |
| Quaternary | `#d62728` | Negative values, alerts |
| Accent | `#9467bd` | Highlights |

### Axis Labels
- Always include units in parentheses: `'Season $(s)$'`
- Use LaTeX for mathematical symbols: `r'$\rho_J$ (Meritocracy Index)'`
- Font size: 12pt

### Titles
- Main title: 13pt, bold
- Include subtitle with additional context on second line
- Example:
```python
ax.set_title('Comparison of Scoring Rules: Meritocracy vs. Engagement Trade-off', 
             fontsize=13, fontweight='bold')
```

### Legends
- Font size: 10-11pt
- `framealpha=0.9` for semi-transparent background
- Position: `loc='upper right'` or `loc='lower left'` depending on data

### Grid Lines
- Use `ax.grid(alpha=0.3)` for subtle gridlines
- `ax.grid(axis='y', alpha=0.3)` for y-axis only

### Value Annotations
```python
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
               xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords="offset points",
               ha='center', va='bottom', fontsize=9)
```

### Output Settings
- DPI: 300 for publication quality
- Format: Both PNG and PDF
- `bbox_inches='tight'` to remove excess whitespace

## Generated Figures Summary

### Phase 4 - Pareto Analysis
1. **key_rules_comparison.png** - Bar chart comparing $\rho_J$, $\rho_F$, $H$ for different rules
2. **method_comparison_summary.png** - Overall method comparison

### Phase 3 - Simulator
3. **ffi_jfi_tradeoff.png** - Scatter plot of FFI vs JFI
4. **ffi_comparison.png** - FFI by season
5. **jfi_comparison.png** - JFI by season

### Feature Engineering
6. **pbi_distribution.png** - Histogram of $\Delta$ distribution
7. **pbi_by_industry.png** - Horizontal bar chart by industry
8. **pbi_trend.png** - Time series of $\Delta$ over seasons

### Phase 5 - Recommendation
9. **dynamic_weights.png** - Sigmoid weight evolution plot

### Bayesian Inference
10. **ci_width_distribution.png** - Uncertainty distribution
11. **fan_vote_vs_judge_score.png** - Scatter with elimination coloring
12. **uncertainty_by_season.png** - Bar chart of CI width by season

### Global Scan
13. **era_boxplot.png** - Box plot by social media era
14. **divergence_trend.png** - Time series with era bands

### Pareto Analysis
15. **pareto_frontier_final.png** - Pareto frontier with rule markers

## Regeneration Script

To regenerate all figures with academic formatting, run:
```bash
python regenerate_all_academic_figures.py
```

This will update all figures in `cleaned_outputs/` subdirectories with consistent academic formatting.
