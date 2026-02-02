#!/usr/bin/env python3
"""MCM 2026 workflow flowchart generator - v5.

与论文章节对齐的工作流程图：
- Data Archaeology & Exploratory Analysis
- Bayesian Inverse Inference
- Pareto Optimization Model
- Rule Simulation & Mechanism Comparison
- Covariate Effect Analysis
- Sensitivity Analysis & Model Evaluation

Output:
  - cleaned_outputs/workflow_flowchart_v3.png
  - cleaned_outputs/workflow_flowchart_v3.pdf
  - context/necessary/workflow_flowchart_v3.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(16, 18))
ax.set_xlim(0, 16)
ax.set_ylim(0, 18)
ax.axis('off')
ax.set_aspect('equal')

# 颜色定义 - 6个section使用不同颜色
colors = {
    'sec3': {'bg': '#E3F2FD', 'border': '#1565C0', 'title': '#0D47A1'},   # 蓝色 - Data
    'sec4': {'bg': '#E8F5E9', 'border': '#2E7D32', 'title': '#1B5E20'},   # 绿色 - Bayesian
    'sec5': {'bg': '#FFF8E1', 'border': '#FF8F00', 'title': '#E65100'},   # 橙色 - Pareto
    'sec6': {'bg': '#F3E5F5', 'border': '#7B1FA2', 'title': '#4A148C'},   # 紫色 - Simulation
    'sec7': {'bg': '#E0F7FA', 'border': '#00838F', 'title': '#006064'},   # 青色 - Covariate
    'sec8': {'bg': '#FFEBEE', 'border': '#C62828', 'title': '#B71C1C'},   # 红色 - Sensitivity
    'input': '#BBDEFB',
    'process': '#FFF9C4',
    'core': '#C8E6C9',
    'output': '#E1BEE7',
    'decision': '#FFCDD2',
    'insight': '#FFE0B2',
}

def draw_box(ax, x, y, w, h, text, box_type='process', fontsize=8):
    """绘制带圆角的方框"""
    color_map = {
        'input': (colors['input'], '#1565C0'),
        'process': (colors['process'], '#F9A825'),
        'core': (colors['core'], '#2E7D32'),
        'output': (colors['output'], '#6A1B9A'),
        'decision': (colors['decision'], '#C62828'),
        'insight': (colors['insight'], '#EF6C00'),
    }
    facecolor, edgecolor = color_map.get(box_type, ('#FFFFFF', '#333333'))
    
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_diamond(ax, x, y, w, h, text, fontsize=8):
    """绘制菱形决策框"""
    diamond = plt.Polygon([(x + w/2, y + h), (x + w, y + h/2), (x + w/2, y), (x, y + h/2)],
                          facecolor=colors['decision'], edgecolor='#C62828', linewidth=2.5)
    ax.add_patch(diamond)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_section_box(ax, x, y, w, h, title, section_color):
    """绘制Section区域框"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.3",
                         facecolor=section_color['bg'],
                         edgecolor=section_color['border'],
                         linewidth=2.5,
                         alpha=0.3)
    ax.add_patch(box)
    ax.text(x + 0.15, y + h - 0.25, title, ha='left', va='top', 
            fontsize=9, fontweight='bold', color=section_color['title'])

def draw_arrow(ax, start, end, color='#455A64', style='-', lw=2.0, path=None, mutation_scale=18):
    """绘制箭头，支持正交路径（折线）"""
    arrow_style = '-|>'
    if path is None:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=arrow_style, color=color, lw=lw,
                                   linestyle=style, shrinkA=5, shrinkB=5,
                                   mutation_scale=mutation_scale))
    else:
        all_points = [start] + path + [end]
        for i in range(len(all_points) - 1):
            p1, p2 = all_points[i], all_points[i+1]
            if i == len(all_points) - 2:
                ax.annotate('', xy=p2, xytext=p1,
                            arrowprops=dict(arrowstyle=arrow_style, color=color, lw=lw,
                                           linestyle=style, shrinkA=0, shrinkB=5,
                                           mutation_scale=mutation_scale))
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, linestyle=style)

# ==================== Section 3: Data Archaeology ====================
draw_section_box(ax, 0.3, 14, 7.2, 3.5, 'Data Archaeology & Exploratory Analysis', colors['sec3'])

draw_box(ax, 0.6, 15.8, 2.0, 0.8, 'Raw Data\n421 contestants\n34 seasons', 'input', 7)
draw_box(ax, 2.9, 15.8, 2.0, 0.8, 'Cleaning\nJ% normalize\nN/A removal', 'process', 7)
draw_box(ax, 5.2, 15.8, 2.0, 0.8, 'Features\nPBI index\nCovariates', 'process', 7)
draw_box(ax, 1.0, 14.4, 2.5, 0.8, 'Panel (i,w)\n2,777 obs', 'output', 7)
draw_box(ax, 4.5, 14.4, 2.5, 0.8, 'Divergence\nTrend +57%', 'insight', 7)

# Section 3 内部箭头
draw_arrow(ax, (2.6, 16.2), (2.9, 16.2))
draw_arrow(ax, (4.9, 16.2), (5.2, 16.2))
draw_arrow(ax, (3.9, 15.8), (2.25, 15.2), path=[(3.9, 15.0), (2.25, 15.0)])
draw_arrow(ax, (6.2, 15.8), (5.75, 15.2), path=[(6.2, 15.0), (5.75, 15.0)])

# ==================== Section 4: Bayesian Inference ====================
draw_section_box(ax, 0.3, 10, 7.2, 3.5, 'Bayesian Inverse Inference Model', colors['sec4'])

draw_box(ax, 0.6, 11.8, 2.2, 0.8, 'Constraints\nElimination\nSimplex', 'process', 7)
draw_box(ax, 3.2, 11.8, 2.2, 0.8, 'MCMC\nHit-and-Run\nSampling', 'core', 7)
draw_box(ax, 5.7, 11.8, 1.5, 0.8, 'f(i,w)\nEstimates', 'output', 7)
draw_box(ax, 1.0, 10.4, 2.8, 0.8, 'Validation\nEMR=73.5%\nConsist=89.2%', 'insight', 7)
draw_box(ax, 4.3, 10.4, 2.8, 0.8, 'CI Width\nMean=0.182\nCertain', 'insight', 7)

# Section 4 内部箭头
draw_arrow(ax, (2.8, 12.2), (3.2, 12.2))
draw_arrow(ax, (5.4, 12.2), (5.7, 12.2))
draw_arrow(ax, (4.3, 11.8), (2.4, 11.2), path=[(4.3, 11.0), (2.4, 11.0)])
draw_arrow(ax, (4.3, 11.8), (5.7, 11.2), path=[(4.3, 11.0), (5.7, 11.0)])

# ==================== Section 5: Pareto Optimization (CORE) ====================
draw_section_box(ax, 8.3, 12, 7.4, 5.5, 'Pareto Optimization Model [CORE]', colors['sec5'])

draw_box(ax, 8.6, 15.6, 2.2, 0.8, 'Dual Objectives\nJ (Meritocracy)\nF (Engagement)', 'process', 7)
draw_box(ax, 11.2, 15.6, 2.2, 0.8, 'Rule Space\n107 configs\nStatic/Dynamic', 'process', 7)
draw_box(ax, 8.6, 14.2, 2.5, 0.8, 'Pareto Frontier\nTrade-off\nAnalysis', 'core', 7)
draw_box(ax, 11.5, 14.2, 2.5, 0.8, 'Multi-Phase\nEvaluation\nEarly F/Late J', 'core', 7)

# Core decision
draw_diamond(ax, 9.5, 12.5, 4.0, 1.2, 'Optimal Rule\nSigmoid(0.30,0.75,6)', 7)

# Section 5 内部箭头
draw_arrow(ax, (10.8, 16.0), (11.2, 16.0))
draw_arrow(ax, (9.85, 15.6), (9.85, 15.0))
draw_arrow(ax, (12.75, 15.6), (12.75, 15.0))
draw_arrow(ax, (9.85, 14.2), (11.5, 13.7), path=[(9.85, 13.7)], color='#E65100', lw=2)
draw_arrow(ax, (12.75, 14.2), (11.5, 13.7), path=[(12.75, 13.7)], color='#E65100', lw=2)

# ==================== Section 6: Rule Simulation ====================
draw_section_box(ax, 8.3, 7, 7.4, 4.5, 'Rule Simulation & Mechanism Comparison', colors['sec6'])

draw_box(ax, 8.6, 9.6, 2.2, 0.8, 'Simulator\nReplay all\n34 seasons', 'core', 7)
draw_box(ax, 11.2, 9.6, 2.5, 0.8, 'Rank vs Pct\nJFI=0.665\nFFI=0.704', 'process', 7)
draw_box(ax, 8.6, 8.0, 2.5, 0.8, '4 Case Studies\nPalin, Bones\nRice, Cyrus', 'process', 7)
draw_box(ax, 11.5, 8.0, 2.5, 0.8, 'Judges\' Save\nMechanism\nAnalysis', 'process', 7)
draw_box(ax, 9.8, 7.2, 3.2, 0.6, 'Rank 2.7× more robust | All cases corrected', 'insight', 6)

# Section 6 内部箭头
draw_arrow(ax, (10.8, 10.0), (11.2, 10.0))
draw_arrow(ax, (9.7, 9.6), (9.7, 8.8))
draw_arrow(ax, (12.45, 9.6), (12.75, 8.8))
draw_arrow(ax, (9.85, 8.0), (11.4, 7.8), path=[(9.85, 7.5)])
draw_arrow(ax, (12.75, 8.0), (11.4, 7.8), path=[(12.75, 7.5)])

# ==================== Section 7: Covariate Effects ====================
draw_section_box(ax, 0.3, 5.5, 7.2, 4.0, 'Covariate Effect Analysis', colors['sec7'])

draw_box(ax, 0.6, 7.6, 2.2, 0.8, 'Mixed Effects\nModel\nRandom/Fixed', 'process', 7)
draw_box(ax, 3.2, 7.6, 2.2, 0.8, 'Pro Dancer\nEffect\nVariance 28.6%', 'core', 7)
draw_box(ax, 5.7, 7.6, 1.5, 0.8, 'Industry\nAge Effects', 'process', 7)
draw_box(ax, 1.0, 5.9, 2.8, 0.8, 'Asymmetry\nJ-path ≠ F-path\nDifferential', 'insight', 7)
draw_box(ax, 4.3, 5.9, 2.8, 0.8, 'Variance\nDecomposition\nCeleb 52.6%', 'insight', 7)

# Section 7 内部箭头
draw_arrow(ax, (2.8, 8.0), (3.2, 8.0))
draw_arrow(ax, (5.4, 8.0), (5.7, 8.0))
draw_arrow(ax, (4.3, 7.6), (2.4, 6.7), path=[(4.3, 6.5), (2.4, 6.5)])
draw_arrow(ax, (4.3, 7.6), (5.7, 6.7), path=[(4.3, 6.5), (5.7, 6.5)])

# ==================== Section 8: Sensitivity Analysis ====================
draw_section_box(ax, 0.3, 1.5, 15.4, 3.5, 'Sensitivity Analysis & Model Evaluation', colors['sec8'])

draw_box(ax, 0.6, 3.2, 2.5, 0.8, 'Parameter\nSensitivity\n107 configs', 'process', 7)
draw_box(ax, 3.5, 3.2, 2.5, 0.8, 'Cross-Season\nStability\nCV: 2.4× stable', 'process', 7)
draw_box(ax, 6.4, 3.2, 2.5, 0.8, 'Bootstrap\n95% CI\n[0.089, 0.113]', 'core', 7)
draw_box(ax, 9.3, 3.2, 2.5, 0.8, 'Robustness\nExtreme cases\n3/4-judge', 'process', 7)
draw_box(ax, 12.2, 3.2, 3.2, 0.8, 'Strengths\n& Weaknesses\nModel Limits', 'insight', 7)

# Final recommendation box at bottom
draw_diamond(ax, 5.5, 1.7, 5.0, 1.2, 'FINAL: Sigmoid+Rank Rule\nF_early +52.7% | J_late +67.5%', 7)

# Section 8 内部箭头
draw_arrow(ax, (3.1, 3.6), (3.5, 3.6))
draw_arrow(ax, (6.0, 3.6), (6.4, 3.6))
draw_arrow(ax, (8.9, 3.6), (9.3, 3.6))
draw_arrow(ax, (11.8, 3.6), (12.2, 3.6))
draw_arrow(ax, (7.65, 3.2), (8.0, 2.9), color='#C62828', lw=2)

# ==================== Cross-Section Links (Main Workflow Backbone) ====================

# Section 3 -> Section 4
draw_arrow(ax, (2.25, 14.4), (2.25, 12.6), 
           path=[(2.25, 13.3), (0.15, 13.3), (0.15, 12.6), (0.6, 12.6)], 
           color='#37474F', lw=2.5)

# Section 4 -> Section 5
draw_arrow(ax, (6.45, 12.2), (9.85, 16.4), 
           path=[(7.5, 12.2), (7.5, 16.4)], 
           color='#37474F', lw=2.5)

# Section 5 -> Section 6
draw_arrow(ax, (11.5, 12.5), (11.5, 10.4), 
           path=[], 
           color='#37474F', lw=2.5)

# Section 6 -> Section 7 (via bottom-left)
draw_arrow(ax, (9.8, 7.2), (5.7, 6.7), 
           path=[(7.8, 7.0), (7.8, 5.0), (0.15, 5.0), (0.15, 8.0), (0.6, 8.0)], 
           color='#37474F', lw=2.5)

# Section 7 -> Section 8
draw_arrow(ax, (2.4, 5.9), (1.85, 4.0), 
           path=[(2.4, 5.2), (1.85, 5.2)], 
           color='#37474F', lw=2.5)

# Section 5 also feeds Section 8 (validation loop)
draw_arrow(ax, (13.5, 12.5), (13.8, 4.0), 
           path=[(15.8, 12.5), (15.8, 3.6), (13.8, 3.6)], 
           color='#7B1FA2', lw=2, style='--')

# ==================== Legend ====================
legend_y = 0.2
ax.text(0.5, legend_y + 0.6, 'Legend:', fontsize=9, fontweight='bold')

legend_items = [
    ('Input', colors['input'], '#1565C0'),
    ('Process', colors['process'], '#F9A825'),
    ('Core', colors['core'], '#2E7D32'),
    ('Output', colors['output'], '#6A1B9A'),
    ('Decision', colors['decision'], '#C62828'),
    ('Finding', colors['insight'], '#EF6C00'),
]

for i, (label, facecolor, edgecolor) in enumerate(legend_items):
    x = 0.5 + i * 2.5
    box = FancyBboxPatch((x, legend_y), 1.0, 0.4,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 0.5, legend_y + 0.2, label, ha='center', va='center', fontsize=7)

# ==================== 标题 ====================
ax.text(8, 17.6, 'MCM 2026 Problem C: Workflow Flowchart', 
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(8, 17.2, 'The Fairness-Engagement Equilibrium Model', 
        ha='center', va='center', fontsize=11, style='italic', color='#666666')

# 保存
plt.tight_layout()
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/workflow_flowchart_v3.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/workflow_flowchart_v3.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
# Also save to context/necessary for the paper
plt.savefig('/home/hyx/文档/MCM/context/necessary/workflow_flowchart_v3.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print('=' * 72)
print('FLOWCHART GENERATED (v5 - Aligned with Paper Sections)!')
print('=' * 72)
print('Output files:')
print('  - cleaned_outputs/workflow_flowchart_v3.png')
print('  - cleaned_outputs/workflow_flowchart_v3.pdf')
print('  - context/necessary/workflow_flowchart_v3.png')
print('=' * 72)
print('\nSection Structure:')
print('  - Data Archaeology & Exploratory Analysis')
print('  - Bayesian Inverse Inference Model')
print('  - Pareto Optimization Model [CORE]')
print('  - Rule Simulation & Mechanism Comparison')
print('  - Covariate Effect Analysis')
print('  - Sensitivity Analysis & Model Evaluation')
print('\nKey Result:')
print('  - Optimal Rule: Sigmoid(0.30, 0.75, 6) + Rank')
print('  - Impact: F_early +52.7%, J_late +67.5%')
print('=' * 72)
