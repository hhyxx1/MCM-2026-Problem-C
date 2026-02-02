#!/usr/bin/env python3
"""MCM 2026 workflow flowchart generator - v7.

蛇形排布版本：
- 第一行：1 → 2 → 3 (从左到右)
- 第二行：6 ← 5 ← 4 (从右到左)
- 外框更紧凑
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 创建图形 - 更紧凑的蛇形布局
fig, ax = plt.subplots(1, 1, figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
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

def draw_box(ax, x, y, w, h, text, box_type='process', fontsize=11):
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
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         linewidth=1.8)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_section_box(ax, x, y, w, h, title, section_color, fontsize=11):
    """绘制Section区域框"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=section_color['bg'],
                         edgecolor=section_color['border'],
                         linewidth=2.2,
                         alpha=0.4)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.18, title, ha='center', va='top', 
            fontsize=fontsize, fontweight='bold', color=section_color['title'])

def draw_simple_arrow(ax, start, end, color='#455A64', lw=1.8):
    """绘制简单直线箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                               shrinkA=0, shrinkB=0, mutation_scale=18))

# ==================== 蛇形布局参数 ====================
# Row 1: Section 3 → 4 → 5 (从左到右)
# Row 2: Section 8 ← 7 ← 6 (从右到左，实际顺序是6→7→8)

row1_y = 3.8  # 第一行Y坐标
row2_y = 0.4  # 第二行Y坐标
sec_w = 3.9   # Section宽度
sec_h = 2.6   # Section高度
gap = 0.25    # Section间隙

# 第一行X坐标
col1_x = 0.2
col2_x = col1_x + sec_w + gap
col3_x = col2_x + sec_w + gap

# ==================== Row 1: 从左到右 ====================

# Section 3: Data Archaeology
draw_section_box(ax, col1_x, row1_y, sec_w, sec_h, 'Sec 3: Data Archaeology', colors['sec3'])
draw_box(ax, 0.35, 4.3, 1.1, 0.55, 'Raw Data\n34 seasons', 'input', 9)
draw_box(ax, 1.55, 4.3, 1.1, 0.55, 'Cleaning\nNormalize', 'process', 9)
draw_box(ax, 2.75, 4.3, 1.2, 0.55, 'PBI Index\nDiv+57%', 'insight', 9)
draw_simple_arrow(ax, (1.45, 4.57), (1.55, 4.57), lw=1.5)
draw_simple_arrow(ax, (2.65, 4.57), (2.75, 4.57), lw=1.5)

# Section 4: Bayesian Inference
draw_section_box(ax, col2_x, row1_y, sec_w, sec_h, 'Sec 4: Bayesian Inference', colors['sec4'])
draw_box(ax, 4.5, 4.3, 1.1, 0.55, 'Constrain\nElimination', 'process', 9)
draw_box(ax, 5.7, 4.3, 1.1, 0.55, 'MCMC\nHit-Run', 'core', 9)
draw_box(ax, 6.9, 4.3, 1.2, 0.55, 'f(i,w) Est\n89% cons.', 'output', 9)
draw_simple_arrow(ax, (5.6, 4.57), (5.7, 4.57), lw=1.5)
draw_simple_arrow(ax, (6.8, 4.57), (6.9, 4.57), lw=1.5)

# Section 5: Pareto Optimization [CORE]
draw_section_box(ax, col3_x, row1_y, sec_w, sec_h, 'Sec 5: Pareto [CORE]', colors['sec5'])
draw_box(ax, 8.65, 4.3, 1.1, 0.55, 'Dual Obj\nJ & F', 'process', 9)
draw_box(ax, 9.85, 4.3, 1.1, 0.55, 'Multi-Ph\nEval', 'core', 9)
draw_box(ax, 11.05, 4.3, 1.2, 0.55, 'Sigmoid\n(0.3,0.75,6)', 'insight', 9)
draw_simple_arrow(ax, (9.75, 4.57), (9.85, 4.57), lw=1.5)
draw_simple_arrow(ax, (10.95, 4.57), (11.05, 4.57), lw=1.5)

# ==================== Row 2: 从右到左显示，但逻辑上是 6→7→8 ====================

# Section 6: Rule Simulation (右边)
draw_section_box(ax, col3_x, row2_y, sec_w, sec_h, 'Sec 6: Rule Simulation', colors['sec6'])
draw_box(ax, 8.65, 0.9, 1.1, 0.55, 'Simulator\n34 seasons', 'core', 9)
draw_box(ax, 9.85, 0.9, 1.1, 0.55, 'Rank-Pct\nJFI=0.665', 'process', 9)
draw_box(ax, 11.05, 0.9, 1.2, 0.55, '4 Cases\nCorrected', 'insight', 9)
draw_simple_arrow(ax, (9.75, 1.17), (9.85, 1.17), lw=1.5)
draw_simple_arrow(ax, (10.95, 1.17), (11.05, 1.17), lw=1.5)

# Section 7: Covariate Effects (中间)
draw_section_box(ax, col2_x, row2_y, sec_w, sec_h, 'Sec 7: Covariate Effects', colors['sec7'])
draw_box(ax, 4.5, 0.9, 1.1, 0.55, 'Mixed\nModel', 'process', 9)
draw_box(ax, 5.7, 0.9, 1.1, 0.55, 'Pro Dance\n28.6%var', 'core', 9)
draw_box(ax, 6.9, 0.9, 1.2, 0.55, 'Asymmetry\nJ≠F path', 'insight', 9)
draw_simple_arrow(ax, (5.6, 1.17), (5.7, 1.17), lw=1.5)
draw_simple_arrow(ax, (6.8, 1.17), (6.9, 1.17), lw=1.5)

# Section 8: Sensitivity Analysis (左边)
draw_section_box(ax, col1_x, row2_y, sec_w, sec_h, 'Sec 8: Sensitivity', colors['sec8'])
draw_box(ax, 0.35, 0.9, 1.1, 0.55, 'Param\nSensitivity', 'process', 9)
draw_box(ax, 1.55, 0.9, 1.1, 0.55, 'Bootstrap\n95% CI', 'core', 9)
draw_box(ax, 2.75, 0.9, 1.2, 0.55, 'Robust\n2.4× stable', 'insight', 9)
draw_simple_arrow(ax, (1.45, 1.17), (1.55, 1.17), lw=1.5)
draw_simple_arrow(ax, (2.65, 1.17), (2.75, 1.17), lw=1.5)

# ==================== 阶段间连接箭头（蛇形） ====================

# Row 1: Section 3 → 4 (水平向右)
draw_simple_arrow(ax, (col1_x + sec_w, 5.1), (col2_x, 5.1), color='#37474F', lw=2.5)

# Row 1: Section 4 → 5 (水平向右)
draw_simple_arrow(ax, (col2_x + sec_w, 5.1), (col3_x, 5.1), color='#37474F', lw=2.5)

# Section 5 → 6 (向下，蛇形转折)
# 从Section 5底部中点向下
mid_x = col3_x + sec_w/2
ax.plot([mid_x, mid_x], [row1_y, row2_y + sec_h], color='#37474F', lw=2.5, solid_capstyle='round')
ax.annotate('', xy=(mid_x, row2_y + sec_h), xytext=(mid_x, row2_y + sec_h + 0.1),
            arrowprops=dict(arrowstyle='-|>', color='#37474F', lw=2.5,
                           shrinkA=0, shrinkB=0, mutation_scale=22))

# Row 2: Section 6 → 7 (水平向左)
ax.annotate('', xy=(col2_x + sec_w, 1.5), xytext=(col3_x, 1.5),
            arrowprops=dict(arrowstyle='-|>', color='#37474F', lw=2.5,
                           shrinkA=0, shrinkB=0, mutation_scale=22))

# Row 2: Section 7 → 8 (水平向左)
ax.annotate('', xy=(col1_x + sec_w, 1.5), xytext=(col2_x, 1.5),
            arrowprops=dict(arrowstyle='-|>', color='#37474F', lw=2.5,
                           shrinkA=0, shrinkB=0, mutation_scale=22))

# 保存
plt.tight_layout()
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/workflow_flowchart_v4.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/home/hyx/文档/MCM/cleaned_outputs/workflow_flowchart_v4.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('/home/hyx/文档/MCM/context/necessary/workflow_flowchart_v4.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print('=' * 60)
print('FLOWCHART GENERATED (v7 - Snake Layout)!')
print('=' * 60)
print('Output: cleaned_outputs/workflow_flowchart_v4.png')
print('        cleaned_outputs/workflow_flowchart_v4.pdf')
print('        context/necessary/workflow_flowchart_v4.png')
print('=' * 60)
