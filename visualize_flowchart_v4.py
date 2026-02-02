#!/usr/bin/env python3
"""MCM 2026 workflow flowchart generator using matplotlib.

目标：对齐 plan.md 的 Phase 1-5 逻辑，生成用于正文的流程图。
V4 Update: 紧凑布局，Phase标题简化，箭头仅接触框架边缘，去图例。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# 创建图形 - 更紧凑
fig, ax = plt.subplots(1, 1, figsize=(14, 12))
ax.set_xlim(0, 15)
ax.set_ylim(0, 12) # 压缩Y轴
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

def draw_phase_box(ax, x, y, w, h, title, phase_color, text_offset=0.3):
    """绘制Phase区域框"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.3",
                         facecolor=phase_color['bg'],
                         edgecolor=phase_color['border'],
                         linewidth=2.5,
                         alpha=0.3)
    ax.add_patch(box)
    ax.text(x + 0.2, y + h - text_offset, title, ha='left', va='top', 
            fontsize=24, fontweight='bold', color=phase_color['title'])

def draw_arrow(ax, start, end, color='#455A64', style='-', lw=2.0, path=None, mutation_scale=20):
    """绘制箭头，支持正交路径（折线）"""
    arrow_style = '-|>' # Filled arrow
    if path is None:
        # 直接连接
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=arrow_style, color=color, lw=lw,
                                   linestyle=style, shrinkA=0, shrinkB=0,
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
                                           linestyle=style, shrinkA=0, shrinkB=0,
                                           mutation_scale=mutation_scale))
            else:
                # 中间段不带箭头
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, linestyle=style)

# ==================== Phase 1: Top Left ====================
# Box: x=0.5, y=9.0, w=6, h=2.5
p1_x, p1_y, p1_w, p1_h = 0.5, 9.0, 6.0, 2.5
draw_phase_box(ax, p1_x, p1_y, p1_w, p1_h, 'PHASE 1', colors['phase1'], text_offset=0.1)

# Contents
draw_box(ax, 0.8, 10.2, 1.5, 0.8, 'Raw Data', 'input', 14)
draw_box(ax, 2.6, 10.2, 1.5, 0.8, 'Cleaning', 'process', 14)
draw_box(ax, 4.4, 10.2, 1.8, 0.8, 'Features', 'process', 14)
draw_box(ax, 1.5, 9.2, 1.8, 0.8, 'Panel (i,w)', 'output', 14)
draw_box(ax, 3.8, 9.2, 2.2, 0.8, 'Global Scan', 'insight', 14)

# Internal Arrows (Strict Edges)
# Raw(Right) -> Cleaning(Left)
draw_arrow(ax, (2.3, 10.6), (2.6, 10.6))
# Cleaning(Right) -> Features(Left)
draw_arrow(ax, (4.1, 10.6), (4.4, 10.6))
# Cleaning(Bottom) -> Panel(Top)
# Cleaning Bottom: (3.35, 10.2). Panel Top: (2.4, 10.0). 
draw_arrow(ax, (3.35, 10.2), (2.4, 10.0), path=[(3.35, 10.1), (2.4, 10.1)])

# Panel(Right) -> Global Scan(Left)
draw_arrow(ax, (3.3, 9.6), (3.8, 9.6))
# Features(Bottom) -> Global Scan(Top)
draw_arrow(ax, (5.3, 10.2), (4.9, 10.0), path=[(5.3, 10.1), (4.9, 10.1)])

# ==================== Phase 2: Mid Left (Below P1) ====================
# Box: x=0.5, y=5.5 -> 5.0 (Lower), w=6, h=2.8 -> 3.3 (Taller)
p2_x, p2_y, p2_w, p2_h = 0.5, 5.0, 6.0, 3.3
draw_phase_box(ax, p2_x, p2_y, p2_w, p2_h, 'PHASE 2', colors['phase2'])

# Contents
draw_box(ax, 0.8, 6.8, 1.8, 0.8, 'Constraints', 'process', 14)
draw_box(ax, 3.2, 6.8, 2.0, 0.8, 'MCMC', 'core', 14)
# Moved down from 5.8 to 5.3
draw_box(ax, 0.8, 5.3, 2.2, 0.8, 'Posterior f(i,w)', 'output', 14)
draw_box(ax, 3.6, 5.3, 2.2, 0.8, 'Validation', 'insight', 14)

# Internal
# Constraints -> MCMC
draw_arrow(ax, (2.6, 7.2), (3.2, 7.2), style='--')
# MCMC -> Posterior
# MCMC Bottom (4.2, 6.8) -> Posterior Top (1.9, 6.1) [Box at 5.3, h=0.8 => Top 6.1]
draw_arrow(ax, (4.2, 6.8), (1.9, 6.1), path=[(4.2, 6.45), (1.9, 6.45)]) 
# Posterior -> Validation
# Center y = 5.3 + 0.4 = 5.7
draw_arrow(ax, (3.0, 5.7), (3.6, 5.7))

# ==================== Phase 3: Top Right ====================
# Box: x=7.5, y=7.0, w=7.0, h=4.5
p3_x, p3_y, p3_w, p3_h = 7.5, 7.0, 7.0, 4.5
draw_phase_box(ax, p3_x, p3_y, p3_w, p3_h, 'PHASE 3', colors['phase3'])

# Contents
draw_box(ax, 8.0, 10.0, 2.5, 0.8, 'Bi-Objectives', 'process', 14)
draw_box(ax, 11.2, 10.0, 2.5, 0.8, 'Rule Space', 'process', 14)
draw_box(ax, 8.0, 8.8, 2.5, 0.8, 'Pareto Frontier', 'core', 14)
draw_box(ax, 11.2, 8.8, 2.5, 0.8, 'Multi-Phase', 'core', 14)
draw_diamond(ax, 9.0, 7.3, 4.0, 1.2, 'Sigmoid Dynamic Rule', 14)

# Internal
# Bi-Obj -> Pareto (Down)
draw_arrow(ax, (9.25, 10.0), (9.25, 9.6))
# Rule Space -> Multi-Phase (Down)
draw_arrow(ax, (12.45, 10.0), (12.45, 9.6))
# Pareto -> Sigmoid (Diamond Top Left-ish)
# Pareto Bottom (9.25, 8.8). Diamond Top (11.0, 8.5)
draw_arrow(ax, (9.25, 8.8), (11.0, 8.5), path=[(9.25, 8.65), (11.0, 8.65)])
# Multi-Phase -> Sigmoid
draw_arrow(ax, (12.45, 8.8), (11.0, 8.5), path=[(12.45, 8.65), (11.0, 8.65)])

# ==================== Phase 4: Mid Right (Below P3) ====================
# Box: x=7.5, y=2.5, w=7.0, h=4.0
p4_x, p4_y, p4_w, p4_h = 7.5, 2.5, 7.0, 4.0
draw_phase_box(ax, p4_x, p4_y, p4_w, p4_h, 'PHASE 4', colors['phase4'])

# Contents
draw_box(ax, 8.0, 5.0, 2.5, 0.8, 'Simulator', 'core', 14)
draw_box(ax, 11.5, 5.0, 2.2, 0.8, 'Rank vs Pct', 'process', 14)
draw_box(ax, 8.0, 3.8, 2.5, 0.8, '4 Cases', 'process', 14)
draw_box(ax, 11.5, 3.8, 2.2, 0.8, 'Effects Model', 'process', 14)
draw_box(ax, 9.0, 2.8, 4.0, 0.7, 'Evidence Summary', 'insight', 14)

# Internal
# Simulator -> Cases
draw_arrow(ax, (9.25, 5.0), (9.25, 4.6))
# Rank -> Effects
draw_arrow(ax, (12.6, 5.0), (12.6, 4.6))
# Cases -> Evidence
draw_arrow(ax, (9.25, 3.8), (11.0, 3.5), path=[(9.25, 3.65), (11.0, 3.65)])
# Effects -> Evidence
draw_arrow(ax, (12.6, 3.8), (11.0, 3.5), path=[(12.6, 3.65), (11.0, 3.65)])

# ==================== Phase 5: Bottom Left (Below P2) ====================
# Box: x=0.5, y=0.5, w=6.0, h=4.0
p5_x, p5_y, p5_w, p5_h = 0.5, 0.5, 6.0, 4.0
draw_phase_box(ax, p5_x, p5_y, p5_w, p5_h, 'PHASE 5', colors['phase5'])

# Contents
draw_diamond(ax, 1.5, 2.5, 4.0, 1.2, 'RECOMMENDATION', 14)
draw_box(ax, 1.0, 1.2, 2.0, 0.8, 'Impact', 'insight', 14)
draw_box(ax, 3.5, 1.2, 2.0, 0.8, 'Memo', 'output', 14)

# Internal
# Rec -> Impact
draw_arrow(ax, (3.5, 2.5), (2.0, 2.0), path=[(3.5, 2.25), (2.0, 2.25)])
# Rec -> Memo
draw_arrow(ax, (3.5, 2.5), (4.5, 2.0), path=[(3.5, 2.25), (4.5, 2.25)])

# ==================== BACKBONE LINKS (Strictly Frame-to-Frame) ====================

# P1 (Bottom) -> P2 (Top)
# P1 Frame Bottom: y=9.0. P2 Frame Top: y=8.3.
# Exit P1 at x=3.5. Enter P2 at x=3.5.
draw_arrow(ax, (3.5, 9.0), (3.5, 8.3), lw=2.5, color='#37474F')

# P2 (Right) -> P3 (Left)
# P2 Frame Right: x=6.5. P3 Frame Left: x=7.5.
# Exit P2 at y=6.9. Enter P3 at y=9.25.
draw_arrow(ax, (6.5, 6.9), (7.5, 9.25), 
           path=[(7.0, 6.9), (7.0, 9.25)], 
           lw=2.5, color='#37474F')

# P3 (Bottom) -> P4 (Top)
# P3 Frame Bottom: y=7.0. P4 Frame Top: y=6.5.
# Exit P3 at x=11.0. Enter P4 at x=11.0.
draw_arrow(ax, (11.0, 7.0), (11.0, 6.5), lw=2.5, color='#37474F')

# P4 (Left) -> P5 (Right)
# P4 Frame Left: x=7.5. P5 Frame Right: x=6.5.
# Exit P4 at y=4.5 (Top of P4). Enter P5 at y=2.5 (Top of P5 approx).
# Let's align on y=3.5? No, P5 Top is 4.5.
# P4 y range: 2.5 to 6.5. Center 4.5.
# P5 y range: 0.5 to 4.5. Center 2.5.
# Connect P4 Left Center (7.5, 4.5) to P5 Right Center (6.5, 3.5)?
# Path from (7.5, 4.5) -> (7.0, 4.5) -> (7.0, 3.5) -> (6.5, 3.5).
draw_arrow(ax, (7.5, 4.5), (6.5, 3.5), 
           path=[(7.0, 4.5), (7.0, 3.5)], 
           lw=2.5, color='#37474F')

# ==================== Main Title (Removed) ====================
# ax.text(14.8, 0.2, 'MCM 2026 Problem C: Workflow', 
#         ha='right', va='bottom', fontsize=28, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/workflow_flowchart_v3.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
