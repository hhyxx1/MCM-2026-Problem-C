#!/usr/bin/env python3
"""MCM 2026 workflow flowchart generator using matplotlib.

目标：对齐 plan.md 的 Phase 1-5 逻辑，生成用于正文的流程图。

核心逻辑链 (来自 plan.md):
[Phase 1] 数据清洗 + 全局扫描 → 发现问题：社交媒体时代，评委-粉丝分歧加剧
[Phase 2] 贝叶斯逆推粉丝票 → 估计隐变量 f(i,w)，验证模型可靠性
[Phase 3] Pareto优化 + 动态加权规则 ← 核心方法论
[Phase 4] 规则模拟与案例验证 → 用历史数据验证新规则效果
[Phase 5] 最终建议与备忘录

Output:
  - cleaned_outputs/workflow_flowchart_v3.png
  - cleaned_outputs/workflow_flowchart_v3.pdf
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
fig, ax = plt.subplots(1, 1, figsize=(16, 20))
ax.set_xlim(0, 16)
ax.set_ylim(0, 20)
ax.axis('off')
ax.set_aspect('equal')

# 颜色定义
colors = {
    'phase1': {'bg': '#E3F2FD', 'border': '#1565C0', 'title': '#0D47A1'},
    'phase2': {'bg': '#E8F5E9', 'border': '#2E7D32', 'title': '#1B5E20'},
    'phase3': {'bg': '#FFF8E1', 'border': '#FF8F00', 'title': '#E65100'},
    'phase4': {'bg': '#F3E5F5', 'border': '#7B1FA2', 'title': '#4A148C'},
    'phase5': {'bg': '#FFEBEE', 'border': '#C62828', 'title': '#B71C1C'},
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

def draw_phase_box(ax, x, y, w, h, title, phase_color):
    """绘制Phase区域框"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.3",
                         facecolor=phase_color['bg'],
                         edgecolor=phase_color['border'],
                         linewidth=2.5,
                         alpha=0.3)
    ax.add_patch(box)
    ax.text(x + 0.2, y + h - 0.3, title, ha='left', va='top', 
            fontsize=10, fontweight='bold', color=phase_color['title'])

def draw_arrow(ax, start, end, color='#455A64', style='-', lw=2.0, path=None, mutation_scale=20):
    """绘制箭头，支持正交路径（折线）"""
    arrow_style = '-|>' # Filled arrow
    if path is None:
        # 直接连接
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=arrow_style, color=color, lw=lw,
                                   linestyle=style, shrinkA=5, shrinkB=5,
                                   mutation_scale=mutation_scale))
    else:
        # 正交路径：path是中间点列表
        all_points = [start] + path + [end]
        for i in range(len(all_points) - 1):
            p1, p2 = all_points[i], all_points[i+1]
            if i == len(all_points) - 2:
                # 最后一段带箭头
                ax.annotate('', xy=p2, xytext=p1,
                            arrowprops=dict(arrowstyle=arrow_style, color=color, lw=lw,
                                           linestyle=style, shrinkA=0, shrinkB=5,
                                           mutation_scale=mutation_scale))
            else:
                # 中间段不带箭头
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, linestyle=style)

# ==================== Phase 1: Data Archaeology & Global Scan ====================
draw_phase_box(ax, 0.3, 15, 7, 4.5, 'PHASE 1: Data Archaeology & Global Scan', colors['phase1'])

draw_box(ax, 0.8, 17.8, 2.2, 0.9, 'Raw Data\n421 contestants\n34 seasons', 'input', 7)
draw_box(ax, 3.3, 17.8, 1.8, 0.9, 'Cleaning\nJ% standardize\nwithdrawals', 'process', 7)
draw_box(ax, 5.4, 17.8, 1.6, 0.9, 'Features\nPBI\nAge/Industry', 'process', 7)
draw_box(ax, 1.5, 15.8, 2.0, 0.9, 'Panel (i,w)\n2,777 obs', 'output', 7)
draw_box(ax, 4.5, 15.8, 2.2, 0.9, 'Global Scan\nDivergence +57%', 'insight', 7)

# Phase 1 internal arrows (horizontal)
draw_arrow(ax, (3.0, 18.25), (3.3, 18.25))
draw_arrow(ax, (5.1, 18.25), (5.4, 18.25))
# Cleaning -> Panel (vertical down, then horizontal)
draw_arrow(ax, (4.2, 17.8), (2.5, 15.8), path=[(4.2, 16.25), (2.5, 16.25)])
# Features -> Global Scan (vertical down)
draw_arrow(ax, (6.2, 17.8), (5.6, 16.7), path=[(6.2, 16.25), (5.6, 16.25)])
# Panel -> Global Scan (horizontal)
draw_arrow(ax, (3.5, 16.25), (4.5, 16.25))

# ==================== Phase 2: Bayesian Inference & Validation ====================
draw_phase_box(ax, 0.3, 10.2, 7, 4.3, 'PHASE 2: Bayesian Inference & Validation', colors['phase2'])

draw_box(ax, 0.8, 12.5, 2.0, 0.9, 'Constraints\nBottom-k\nSum(f)=1', 'process', 7)
draw_box(ax, 3.3, 12.5, 2.2, 0.9, 'MCMC\n(Hit-and-Run)\nCI=0.073', 'core', 7)
draw_box(ax, 0.8, 10.8, 2.2, 0.9, 'Posterior f(i,w)\n2,777 estimates', 'output', 7)
draw_box(ax, 3.8, 10.8, 2.8, 0.9, 'Validation\nExact 96.6%\nP_bar=0.983', 'insight', 7)

# Phase 2 internal arrows
draw_arrow(ax, (2.8, 12.95), (3.3, 12.95), style='--')
# MCMC -> Posterior (down then left)
draw_arrow(ax, (3.3, 12.5), (1.9, 11.7), path=[(3.3, 11.25), (1.9, 11.25)])
# Posterior -> Validation (horizontal)
draw_arrow(ax, (3.0, 11.25), (3.8, 11.25))

# ==================== Phase 3: Pareto Optimization & Dynamic Weighting (CORE) ====================
draw_phase_box(ax, 8, 12.5, 7.5, 7, 'PHASE 3: Pareto Optimization [CORE]', colors['phase3'])

draw_box(ax, 8.5, 17.5, 2.5, 0.9, 'Bi-Objectives\nJ (Merit)\nF (Engagement)', 'process', 7)
draw_box(ax, 11.5, 17.5, 2.2, 0.9, 'Rule Space\nStatic vs\nDynamic', 'process', 7)
draw_box(ax, 8.5, 15.8, 2.5, 0.9, 'Pareto Frontier\n51 configurations', 'core', 7)
draw_box(ax, 11.5, 15.8, 2.8, 0.9, 'Multi-Phase Eval\nEarly F\nLate J', 'core', 7)

# Core decision diamond
draw_diamond(ax, 9.5, 13.8, 3.5, 1.4, 'Sigmoid Dynamic\nw_min=0.30\nw_max=0.75, s=6', 7)

draw_box(ax, 8.5, 12.8, 6.5, 0.8, 'Advantage: F_early +52.7% | J_late +67.5% | Composite +21.6%', 'insight', 7)

# Phase 3 internal arrows (horizontal and vertical)
draw_arrow(ax, (11.0, 17.95), (11.5, 17.95))
# Bi-Objectives -> Pareto (vertical down)
draw_arrow(ax, (9.75, 17.5), (9.75, 16.7))
# Rule Space -> Multi-Phase (vertical down)
draw_arrow(ax, (12.6, 17.5), (12.9, 16.7))
# Pareto -> Sigmoid (down then right)
draw_arrow(ax, (9.75, 15.8), (11.25, 15.2), path=[(9.75, 15.5), (11.25, 15.5)], color='#E65100', lw=2)
# Multi-Phase -> Sigmoid (down then left)
draw_arrow(ax, (12.9, 15.8), (11.25, 15.2), path=[(12.9, 15.5), (11.25, 15.5)], color='#E65100', lw=2)
# Sigmoid -> Advantage (vertical down)
draw_arrow(ax, (11.25, 13.8), (11.25, 13.6))

# ==================== Phase 4: Rule Simulation & Case Studies ====================
draw_phase_box(ax, 8, 6.5, 7.5, 5.5, 'PHASE 4: Rule Simulation & Case Studies', colors['phase4'])

draw_box(ax, 8.5, 10.2, 2.2, 0.9, 'Simulator\nReplay\n34 seasons', 'core', 7)
draw_box(ax, 11.2, 10.2, 2.5, 0.9, 'Rank vs Pct\nFFI delta+0.085', 'process', 7)
draw_box(ax, 8.5, 8.5, 2.5, 0.9, '4 Cases\nRice | Cyrus\nPalin | Bones', 'process', 7)
draw_box(ax, 11.5, 8.5, 2.2, 0.9, 'Effects Model\nPro Dancer\nCovariates', 'process', 7)
draw_box(ax, 9.5, 7, 3.5, 0.8, 'Evidence: Rank more robust | Save effective', 'insight', 7)

# Phase 4 internal arrows
draw_arrow(ax, (10.7, 10.65), (11.2, 10.65))
# Simulator -> 4 Cases (vertical down)
draw_arrow(ax, (9.6, 10.2), (9.6, 9.4))
# Rank vs Pct -> Effects (vertical down)
draw_arrow(ax, (12.6, 10.2), (12.6, 9.4))
# 4 Cases -> Evidence (down then right)
draw_arrow(ax, (9.75, 8.5), (11.25, 7.8), path=[(9.75, 7.8)])
# Effects -> Evidence (down then left)
draw_arrow(ax, (12.6, 8.5), (11.25, 7.8), path=[(12.6, 7.8)])

# ==================== Phase 5: Final Recommendation & Memo ====================
draw_phase_box(ax, 0.3, 5.5, 7, 4.2, 'PHASE 5: Final Recommendation & Memo', colors['phase5'])

# Core recommendation box
draw_diamond(ax, 1.5, 7.8, 4, 1.2, 'RECOMMEND\nSigmoid+Rank | Save Y', 7)

draw_box(ax, 0.8, 6.3, 3.2, 0.8, 'Impact\nEarly F 0.58->0.88\nLate J 0.55->0.91', 'insight', 7)
draw_box(ax, 4.5, 6.3, 2.3, 0.8, 'Memo\nProducer advice\nRisk notes', 'output', 7)

# Phase 5 internal arrows
# Recommend -> Impact (vertical down)
draw_arrow(ax, (3.5, 7.8), (2.4, 7.1), path=[(3.5, 7.1), (2.4, 7.1)])
# Impact -> Memo (horizontal)
draw_arrow(ax, (4.0, 6.7), (4.5, 6.7), color='#6A1B9A', lw=2)

# ==================== Cross-Phase Links (Main Workflow Backbone) ====================

# 1. Phase 1 (Panel Output) -> Phase 2 (Constraints Input)
# Path: Bottom of Panel (2.5, 15.8) -> Down -> Left -> Top of Constraints (1.8, 13.4)
draw_arrow(ax, (2.5, 15.8), (1.8, 13.4), 
           path=[(2.5, 14.5), (1.8, 14.5)], 
           color='#37474F', lw=2.5)

# 2. Phase 2 (Posterior Output) -> Phase 3 (Bi-Objectives Input) - The Bridge
# Path: Bottom of Posterior (1.9, 10.8) -> Down -> Right (Gap) -> Up -> Right -> Top of Bi-Obj
# Gap X is approx 7.5
draw_arrow(ax, (1.9, 10.8), (9.75, 18.4), 
           path=[(1.9, 10.0), (7.5, 10.0), (7.5, 18.8), (9.75, 18.8)], 
           color='#37474F', lw=2.5)

# 3. Phase 3 (Advantage Output) -> Phase 4 (Simulator Input)
# Path: Bottom of Advantage (9.6, 12.8) -> Vertical Down -> Top of Simulator (9.6, 11.1)
draw_arrow(ax, (9.6, 12.8), (9.6, 11.1), 
           path=[], 
           color='#37474F', lw=2.5)

# 4. Phase 4 (Evidence Output) -> Phase 5 (Recommendation Input)
# Path: Left of Evidence (9.5, 7.4) -> Left (Gap) -> Up/Down -> Right Tip of Recommend Diamond
# Recommend Diamond Right Tip is (5.5, 8.4)
# Evidence Left Center is (9.5, 7.4)
draw_arrow(ax, (9.5, 7.4), (5.5, 8.4), 
           path=[(7.5, 7.4), (7.5, 8.4)], 
           color='#37474F', lw=2.5)

# ==================== Legend ====================
legend_y = 1.5
ax.text(0.5, legend_y + 1.5, 'Legend:', fontsize=10, fontweight='bold')

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
    box = FancyBboxPatch((x, legend_y), 1.2, 0.5,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 0.6, legend_y + 0.25, label, ha='center', va='center', fontsize=8)

# ==================== 标题 ====================
ax.text(8, 19.5, 'MCM 2026 Problem C: Workflow Flowchart', 
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(8, 19.0, 'The Fairness-Engagement Equilibrium Model', 
        ha='center', va='center', fontsize=11, style='italic', color='#666666')

# 保存
plt.tight_layout()
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/workflow_flowchart_v3.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/workflow_flowchart_v3.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print('=' * 72)
print('FLOWCHART GENERATED (v3 - Updated per plan.md)!')
print('=' * 72)
print('PNG: cleaned_outputs/workflow_flowchart_v3.png')
print('PDF: cleaned_outputs/workflow_flowchart_v3.pdf')
print('=' * 72)
print('\nKey Features:')
print('  - Phase 1: 数据考古与全局扫描 (分歧趋势 +57%)')
print('  - Phase 2: 贝叶斯逆推与验证 (Exact 96.6%, P̄=0.983)')
print('  - Phase 3: Pareto优化与动态加权 ⭐核心')
print('  - Phase 4: 规则模拟与案例验证')
print('  - Phase 5: 最终建议与备忘录')
print('\nHighlighted:')
print('  - Sigmoid Dynamic Rule (w_min=0.30, w_max=0.75, s=6)')
print('  - Multi-Phase Evaluation Framework')
print('  - Impact: F_early +52.7%, J_late +67.5%, Composite +21.6%')
print('=' * 72)
