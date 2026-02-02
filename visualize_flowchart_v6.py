#!/usr/bin/env python3
"""MCM 2026 workflow flowchart generator - v6.

改进版：
- 箭头横平竖直，不穿过框
- 箭头头部更大
- 从左到右、从上到下布局
- 阶段间连接清晰
- 紧凑布局
- 无legend、无标题
- 字体放大
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# 创建图形 - 更紧凑
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_aspect('equal')

# 颜色定义
colors = {
    'sec3': {'bg': '#E3F2FD', 'border': '#1565C0', 'title': '#0D47A1'},
    'sec4': {'bg': '#E8F5E9', 'border': '#2E7D32', 'title': '#1B5E20'},
    'sec5': {'bg': '#FFF8E1', 'border': '#FF8F00', 'title': '#E65100'},
    'sec6': {'bg': '#F3E5F5', 'border': '#7B1FA2', 'title': '#4A148C'},
    'sec7': {'bg': '#E0F7FA', 'border': '#00838F', 'title': '#006064'},
    'sec8': {'bg': '#FFEBEE', 'border': '#C62828', 'title': '#B71C1C'},
    'input': '#BBDEFB',
    'process': '#FFF9C4',
    'core': '#C8E6C9',
    'output': '#E1BEE7',
    'insight': '#FFE0B2',
}

def draw_box(ax, x, y, w, h, text, box_type='process', fontsize=14):
    """绘制带圆角的方框"""
    color_map = {
        'input': (colors['input'], '#1565C0'),
        'process': (colors['process'], '#F9A825'),
        'core': (colors['core'], '#2E7D32'),
        'output': (colors['output'], '#6A1B9A'),
        'insight': (colors['insight'], '#EF6C00'),
    }
    facecolor, edgecolor = color_map.get(box_type, ('#FFFFFF', '#333333'))
    
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_section_box(ax, x, y, w, h, title, section_color, fontsize=12):
    """绘制Section区域框"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.2",
                         facecolor=section_color['bg'],
                         edgecolor=section_color['border'],
                         linewidth=2.5,
                         alpha=0.4)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.22, title, ha='center', va='top', 
            fontsize=fontsize, fontweight='bold', color=section_color['title'])

def draw_arrow_ortho(ax, start, end, color='#37474F', lw=2.5):
    """绘制正交箭头（只有水平和垂直线段），箭头更大"""
    x1, y1 = start
    x2, y2 = end
    
    # 确定路径：先水平后垂直，或先垂直后水平
    if abs(x2 - x1) > abs(y2 - y1):
        # 主要是水平移动：先水平到x2，再垂直到y2
        mid_x = x2
        mid_y = y1
    else:
        # 主要是垂直移动：先垂直到y2，再水平到x2
        mid_x = x1
        mid_y = y2
    
    # 画线段（不带箭头）
    ax.plot([x1, mid_x], [y1, mid_y], color=color, lw=lw, solid_capstyle='round')
    # 最后一段带箭头
    ax.annotate('', xy=(x2, y2), xytext=(mid_x, mid_y),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                               shrinkA=0, shrinkB=0, mutation_scale=25))

def draw_simple_arrow(ax, start, end, color='#455A64', lw=2):
    """绘制简单直线箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                               shrinkA=0, shrinkB=0, mutation_scale=20))

# ==================== 布局参数 ====================
# Row 1: Section 3 (Data) and Section 4 (Bayesian)
# Row 2: Section 5 (Pareto) and Section 6 (Simulation)  
# Row 3: Section 7 (Covariate) and Section 8 (Sensitivity)

row1_y = 6.8
row2_y = 3.6
row3_y = 0.4
col1_x = 0.2
col2_x = 7.2
sec_w = 6.6
sec_h = 2.8

# ==================== Section 3: Data Archaeology ====================
draw_section_box(ax, col1_x, row1_y, sec_w, sec_h, 'Data Archaeology', colors['sec3'])

draw_box(ax, 0.4, 7.4, 1.8, 0.7, 'Raw Data\n34 seasons', 'input', 12)
draw_box(ax, 2.5, 7.4, 1.8, 0.7, 'Cleaning\nJ% normalize', 'process', 12)
draw_box(ax, 4.5, 7.4, 2.1, 0.7, 'PBI Index\nDivergence+57%', 'insight', 12)

# Section 3 内部箭头
draw_simple_arrow(ax, (2.2, 7.75), (2.5, 7.75))
draw_simple_arrow(ax, (4.3, 7.75), (4.5, 7.75))

# ==================== Section 4: Bayesian Inference ====================
draw_section_box(ax, col2_x, row1_y, sec_w, sec_h, 'Bayesian Inference', colors['sec4'])

draw_box(ax, 7.4, 7.4, 1.8, 0.7, 'Constraints\nElimination', 'process', 12)
draw_box(ax, 9.5, 7.4, 1.8, 0.7, 'MCMC\nHit-and-Run', 'core', 12)
draw_box(ax, 11.6, 7.4, 2.1, 0.7, 'f(i,w) Est.\nConsist=89%', 'output', 12)

# Section 4 内部箭头
draw_simple_arrow(ax, (9.2, 7.75), (9.5, 7.75))
draw_simple_arrow(ax, (11.3, 7.75), (11.6, 7.75))

# ==================== Section 5: Pareto Optimization ====================
draw_section_box(ax, col1_x, row2_y, sec_w, sec_h, 'Pareto Optimization [CORE]', colors['sec5'])

draw_box(ax, 0.4, 4.2, 1.8, 0.7, 'Dual Obj.\nJ & F', 'process', 12)
draw_box(ax, 2.5, 4.2, 2.0, 0.7, 'Multi-Phase\nEvaluation', 'core', 12)
draw_box(ax, 4.7, 4.2, 2.0, 0.7, 'Sigmoid Rule\n(0.30,0.75,6)', 'insight', 12)

# Section 5 内部箭头
draw_simple_arrow(ax, (2.2, 4.55), (2.5, 4.55))
draw_simple_arrow(ax, (4.5, 4.55), (4.7, 4.55))

# ==================== Section 6: Rule Simulation ====================
draw_section_box(ax, col2_x, row2_y, sec_w, sec_h, 'Rule Simulation', colors['sec6'])

draw_box(ax, 7.4, 4.2, 1.8, 0.7, 'Simulator\n34 seasons', 'core', 12)
draw_box(ax, 9.5, 4.2, 1.8, 0.7, 'Rank vs Pct\nJFI=0.665', 'process', 12)
draw_box(ax, 11.6, 4.2, 2.1, 0.7, '4 Cases\nAll corrected', 'insight', 12)

# Section 6 内部箭头
draw_simple_arrow(ax, (9.2, 4.55), (9.5, 4.55))
draw_simple_arrow(ax, (11.3, 4.55), (11.6, 4.55))

# ==================== Section 7: Covariate Effects ====================
draw_section_box(ax, col1_x, row3_y, sec_w, sec_h, 'Covariate Effects', colors['sec7'])

draw_box(ax, 0.4, 1.0, 1.8, 0.7, 'Mixed Model\nRandom Eff.', 'process', 12)
draw_box(ax, 2.5, 1.0, 2.0, 0.7, 'Pro Dancer\nVar=28.6%', 'core', 12)
draw_box(ax, 4.7, 1.0, 2.0, 0.7, 'Asymmetry\nJ≠F path', 'insight', 12)

# Section 7 内部箭头
draw_simple_arrow(ax, (2.2, 1.35), (2.5, 1.35))
draw_simple_arrow(ax, (4.5, 1.35), (4.7, 1.35))

# ==================== Section 8: Sensitivity Analysis ====================
draw_section_box(ax, col2_x, row3_y, sec_w, sec_h, 'Sensitivity Analysis', colors['sec8'])

draw_box(ax, 7.4, 1.0, 1.8, 0.7, 'Parameter\nSensitivity', 'process', 12)
draw_box(ax, 9.5, 1.0, 1.8, 0.7, 'Bootstrap\n95% CI', 'core', 12)
draw_box(ax, 11.6, 1.0, 2.1, 0.7, 'Robust\n2.4× stable', 'insight', 12)

# Section 8 内部箭头
draw_simple_arrow(ax, (9.2, 1.35), (9.5, 1.35))
draw_simple_arrow(ax, (11.3, 1.35), (11.6, 1.35))

# ==================== 阶段间连接箭头 ====================
# Section 3 -> Section 4 (水平向右)
draw_simple_arrow(ax, (6.8, 7.75), (7.2, 7.75), color='#37474F', lw=3)

# Section 4 -> Section 5 (向下再向左)
# 从Section 4底部中点出发，向下，再向左到Section 5右边
ax.plot([10.5, 10.5], [6.8, 5.7], color='#37474F', lw=3, solid_capstyle='round')
ax.plot([10.5, 6.9], [5.7, 5.7], color='#37474F', lw=3, solid_capstyle='round')
ax.annotate('', xy=(6.8, 5.7), xytext=(6.9, 5.7),
            arrowprops=dict(arrowstyle='-|>', color='#37474F', lw=3,
                           shrinkA=0, shrinkB=0, mutation_scale=25))

# Section 5 -> Section 6 (水平向右)
draw_simple_arrow(ax, (6.8, 4.55), (7.2, 4.55), color='#37474F', lw=3)

# Section 6 -> Section 7 (向下再向左)
ax.plot([10.5, 10.5], [3.6, 2.5], color='#37474F', lw=3, solid_capstyle='round')
ax.plot([10.5, 6.9], [2.5, 2.5], color='#37474F', lw=3, solid_capstyle='round')
ax.annotate('', xy=(6.8, 2.5), xytext=(6.9, 2.5),
            arrowprops=dict(arrowstyle='-|>', color='#37474F', lw=3,
                           shrinkA=0, shrinkB=0, mutation_scale=25))

# Section 7 -> Section 8 (水平向右)
draw_simple_arrow(ax, (6.8, 1.35), (7.2, 1.35), color='#37474F', lw=3)

# 保存
plt.tight_layout()
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/workflow_flowchart_v3.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/workflow_flowchart_v3.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/home/hyx/文档/MCM/context/necessary/workflow_flowchart_v3.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print('=' * 60)
print('FLOWCHART GENERATED (v6 - Clean & Compact)!')
print('=' * 60)
print('Output: cleaned_outputs/workflow_flowchart_v3.png')
print('        context/necessary/workflow_flowchart_v3.png')
print('=' * 60)
