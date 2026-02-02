#!/usr/bin/env python3
"""MCM 2026 workflow flowchart generator.

目标：对齐 plan.md 的 Phase 1-5 逻辑，并生成更美观、可用于正文的流程图。

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

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from graphviz import Digraph


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


ROOT = Path(__file__).resolve().parent
STATS = _safe_read_json(ROOT / 'cleaned_outputs' / 'key_statistics.json')
REC = _safe_read_json(ROOT / 'cleaned_outputs' / 'recommended_rule.json')

try:
    inference_stats = pd.read_csv(ROOT / 'cleaned_outputs' / 'inference_season_stats.csv')
    avg_ci_width = float(inference_stats['avg_ci_width'].mean())
except Exception:
    avg_ci_width = float('nan')

total_seasons = int(STATS.get('total_seasons', 34))
total_contestants = int(STATS.get('total_contestants', 421))
total_obs = int(STATS.get('total_observations', 2777))

cv_mean = _safe_float(STATS.get('supplementary', {}).get('cv_mean'))
p_bar = _safe_float(STATS.get('supplementary', {}).get('posterior_consistency_P_bar'))
exact_match = _safe_float(STATS.get('prediction_accuracy'))

rank_ffi = _safe_float(STATS.get('rank_ffi'))
pct_ffi = _safe_float(STATS.get('pct_ffi'))
ffi_delta = pct_ffi - rank_ffi

knee = REC.get('knee_point', {})
J_star = _safe_float(knee.get('J_meritocracy'))
F_star = _safe_float(knee.get('F_engagement'))
rank_knee = _safe_float(knee.get('rank_knee_distance'))
pct_knee = _safe_float(knee.get('pct_knee_distance'))

current_rule = REC.get('current_rule', {})
J_current = _safe_float(current_rule.get('J_meritocracy'))
F_current = _safe_float(current_rule.get('F_engagement'))

final_rec = REC.get('final_recommendation', {})
rec_method = str(final_rec.get('method', 'rank')).upper()
rec_weights = str(final_rec.get('weights', '50-50'))
rec_save = bool(final_rec.get('include_judges_save', True))

J_gain = (J_star - J_current) if (J_star and J_current) else 0.0
J_gain_pct = (J_gain / J_current * 100.0) if J_current else 0.0

# 新方案关键指标 (来自 plan.md Phase 3)
F_early_static = 0.5754
F_early_dynamic = 0.8785
J_late_static = 0.5451
J_late_dynamic = 0.9133
dynamic_pattern_score = 1.5546
composite_score_static = 0.4681
composite_score_dynamic = 0.5693


def _node(dot: Digraph, name: str, label: str, kind: str, **kwargs) -> None:
    palette = {
        'input': '#E3F2FD',
        'process': '#FFF8E1',
        'core': '#E8F5E9',
        'output': '#F3E5F5',
        'decision': '#FFEBEE',
        'insight': '#FFF3E0',
    }
    stroke = {
        'input': '#1565C0',
        'process': '#F9A825',
        'core': '#2E7D32',
        'output': '#6A1B9A',
        'decision': '#C62828',
        'insight': '#EF6C00',
    }

    dot.node(
        name,
        label,
        shape=kwargs.pop('shape', 'box'),
        style=kwargs.pop('style', 'rounded,filled'),
        fillcolor=kwargs.pop('fillcolor', palette.get(kind, '#FFFFFF')),
        color=kwargs.pop('color', stroke.get(kind, '#333333')),
        penwidth=kwargs.pop('penwidth', '2'),
        **kwargs,
    )


dot = Digraph(comment='MCM 2026 Workflow Flowchart', format='png')
dot.attr(
    rankdir='TB',
    splines='spline',
    nodesep='0.15',
    ranksep='0.25',
)
dot.attr(
    'graph',
    bgcolor='white',
    pad='0.02',
    margin='0',
    dpi='280',
    ratio='compress',
    newrank='true',
    pack='true',
    packmode='clust',
)
dot.attr('node', fontname='DejaVu Sans', fontsize='9', margin='0.10,0.08')
dot.attr('edge', fontname='DejaVu Sans', fontsize='8', arrowsize='0.65', color='#455A64')


def _anchor(dot_: Digraph, name: str) -> None:
    dot_.node(
        name,
        label='',
        shape='point',
        style='invis',
        width='0.01',
        height='0.01',
        fixedsize='true',
    )


# ----------------------------- Phase 1 -----------------------------
with dot.subgraph(name='cluster_phase1') as c:
    c.attr(
        label='PHASE 1  数据考古与全局扫描',
        labelloc='t',
        labeljust='l',
        style='rounded,filled',
        color='#90CAF9',
        fillcolor='#F7FBFF',
        fontsize='10.5',
    )

    _anchor(c, 'a_p1')

    _node(
        c,
        'raw',
        f'Raw Data\n{total_contestants} contestants • {total_seasons} seasons',
        'input',
        shape='cylinder',
    )
    _node(c, 'clean', 'Cleaning\nJ% 标准化 • 退赛处理', 'process')
    _node(c, 'feat', 'Features\nPBI + Age/Industry/Pro', 'process')
    _node(c, 'panel', f'Panel (i,w)\n{total_obs:,} obs', 'output', shape='note')
    _node(c, 'div', 'Global Scan\n分歧趋势 +57%', 'insight')

    c.edge('a_p1', 'raw', style='invis')

    c.edge('raw', 'clean')
    c.edge('clean', 'feat')
    c.edge('clean', 'panel')
    c.edge('feat', 'panel')
    c.edge('panel', 'div')


# ----------------------------- Phase 2 -----------------------------
with dot.subgraph(name='cluster_phase2') as c:
    c.attr(
        label='PHASE 2  贝叶斯逆推与验证',
        labelloc='t',
        labeljust='l',
        style='rounded,filled',
        color='#A5D6A7',
        fillcolor='#F7FFF8',
        fontsize='10.5',
    )

    _anchor(c, 'a_p2')

    _node(c, 'constraints', 'Constraints\nBottom-k • Σf=1', 'process')
    _node(c, 'mcmc', f'MCMC (Hit-and-Run)\nAvg CI: {avg_ci_width:.3f}', 'core')
    _node(c, 'post', f'Posterior f(i,w)\n{total_obs:,} estimates', 'output', shape='note')
    _node(
        c,
        'valid',
        f'Validation\nCI {avg_ci_width:.3f} • Exact {exact_match*100:.1f}% • P̄ {p_bar:.3f}',
        'insight',
    )

    c.edge('a_p2', 'constraints', style='invis')

    c.edge('constraints', 'mcmc', style='dashed', color='#6D4C41')
    c.edge('mcmc', 'post')
    c.edge('post', 'valid')


# ----------------------------- Phase 3 (核心方法论) -----------------------------
with dot.subgraph(name='cluster_phase3') as c:
    c.attr(
        label='PHASE 3  Pareto优化与动态加权 ⭐核心',
        labelloc='t',
        labeljust='l',
        style='rounded,filled',
        color='#FFCC80',
        fillcolor='#FFFDF7',
        fontsize='10.5',
    )

    _anchor(c, 'a_p3')

    _node(c, 'objectives', 'Bi-Objectives\nJ (Merit) • F (Engagement)', 'process')
    _node(c, 'rule_space', 'Rule Space\nStatic vs Dynamic', 'process')
    _node(c, 'pareto', 'Pareto Frontier\n51 configurations', 'core')
    _node(c, 'multi_phase', 'Multi-Phase Eval\nEarly F • Late J', 'core', style='rounded,filled,bold')
    _node(
        c,
        'sigmoid',
        f'Sigmoid Dynamic ⭐\nw_min=0.30 • w_max=0.75 • s=6',
        'decision',
        shape='diamond',
        style='filled',
        penwidth='2.6',
    )
    _node(
        c,
        'advantage',
        f'Advantage\nF_early +52.7% • J_late +67.5%\nComposite +21.6%',
        'insight',
    )

    c.edge('a_p3', 'objectives', style='invis')

    c.edge('objectives', 'rule_space')
    c.edge('rule_space', 'pareto')
    c.edge('pareto', 'multi_phase')
    c.edge('multi_phase', 'sigmoid', penwidth='2.2', color='#E65100')
    c.edge('sigmoid', 'advantage')


# ----------------------------- Phase 4 -----------------------------
with dot.subgraph(name='cluster_phase4') as c:
    c.attr(
        label='PHASE 4  规则模拟与案例验证',
        labelloc='t',
        labeljust='l',
        style='rounded,filled',
        color='#CE93D8',
        fillcolor='#FDF7FF',
        fontsize='10.5',
    )

    _anchor(c, 'a_p4')

    _node(c, 'sim', f'Simulator\nReplay {total_seasons} seasons', 'core')
    _node(c, 'rank_pct', f'Rank vs Pct\nFFI Δ{ffi_delta:+.3f}', 'process')
    _node(c, 'cases', '4 Cases\nRice • Cyrus • Palin • Bones', 'process')
    _node(c, 'effects', 'Effects Model\nPro Dancer + Covariates', 'process')
    _node(c, 'evidence', 'Evidence\nRank更稳健 • Save有效', 'insight')

    c.edge('a_p4', 'sim', style='invis')

    c.edge('sim', 'rank_pct')
    c.edge('sim', 'cases')
    c.edge('rank_pct', 'evidence')
    c.edge('cases', 'evidence')


# ----------------------------- Phase 5 -----------------------------
with dot.subgraph(name='cluster_phase5') as c:
    c.attr(
        label='PHASE 5  最终建议与备忘录',
        labelloc='t',
        labeljust='l',
        style='rounded,filled',
        color='#FFAB91',
        fillcolor='#FFF7F4',
        fontsize='10.5',
    )

    _anchor(c, 'a_p5')

    _node(
        c,
        'rec',
        f'RECOMMEND\nSigmoid+Rank • Save {"Y" if rec_save else "N"}',
        'decision',
        style='rounded,filled,bold',
        penwidth='3.0',
    )
    _node(
        c,
        'compare',
        f'Impact\n早期F 0.58→0.88 • 后期J 0.55→0.91',
        'insight',
    )
    _node(c, 'memo', 'Memo\n制片人建议 + 风险提示', 'output', shape='note')

    c.edge('a_p5', 'rec', style='invis')

    c.edge('rec', 'compare')
    c.edge('compare', 'memo', penwidth='2.2', color='#6A1B9A')


# ----------------------------- Cross-phase links (narrative) -----------------------------
# Phase 1 → Phase 2: Panel数据流入贝叶斯模型
dot.edge('panel', 'mcmc', penwidth='2.0', color='#2E7D32')

# Phase 2 → Phase 3: 粉丝票估计流入Pareto优化
dot.edge('post', 'objectives', penwidth='2.2', color='#2E7D32', xlabel='f(i,w)')

# Phase 1 Divergence → Phase 3: 分歧趋势证明改革必要性
dot.edge('div', 'objectives', xlabel='reform rationale', style='bold', color='#1565C0', penwidth='2.2')

# Phase 3 → Phase 4: 最优规则进入模拟验证
dot.edge('sigmoid', 'sim', penwidth='2.2', color='#E65100', xlabel='optimal rule')

# Phase 2 → Phase 4: 粉丝票用于案例分析
dot.edge('post', 'cases', style='dashed', color='#546E7A')
dot.edge('feat', 'effects', style='dashed', color='#546E7A')
dot.edge('post', 'effects', style='dashed', color='#546E7A')

# Phase 3 Advantage → Phase 5: 优势论证流入最终建议
dot.edge('advantage', 'rec', style='bold', color='#C62828', penwidth='2.2')

# Phase 4 Evidence → Phase 5: 案例证据支持备忘录
dot.edge('evidence', 'memo', style='dashed', color='#546E7A')
dot.edge('effects', 'memo', style='dashed', color='#546E7A')


# ----------------------------- Compact legend node -----------------------------
_node(
    dot,
    'legend',
    'Legend\nInput • Process • Core • Decision • Finding',
    'output',
    shape='note',
    fontsize='8',
    penwidth='1.6',
)

_anchor(dot, 'a_leg')
dot.edge('a_leg', 'legend', style='invis')


# ----------------------------- Macro layout (2 columns x 3 rows) -----------------------------
# Row alignment
with dot.subgraph() as r1:
    r1.attr(rank='same')
    r1.node('a_p1')
    r1.node('a_p3')

with dot.subgraph() as r2:
    r2.attr(rank='same')
    r2.node('a_p2')
    r2.node('a_p4')

with dot.subgraph() as r3:
    r3.attr(rank='same')
    r3.node('a_p5')
    r3.node('a_leg')

# Column ordering (left -> right within a row)
dot.edge('a_p1', 'a_p3', style='invis', weight='50')
dot.edge('a_p2', 'a_p4', style='invis', weight='50')
dot.edge('a_p5', 'a_leg', style='invis', weight='50')

# Vertical ordering (top -> bottom within a column)
dot.edge('a_p1', 'a_p2', style='invis', weight='50')
dot.edge('a_p2', 'a_p5', style='invis', weight='50')
dot.edge('a_p3', 'a_p4', style='invis', weight='50')
dot.edge('a_p4', 'a_leg', style='invis', weight='50')


out_base = ROOT / 'cleaned_outputs' / 'workflow_flowchart_v3'
dot.format = 'png'
dot.render(str(out_base), cleanup=True)

dot.format = 'pdf'
dot.render(str(out_base), cleanup=True)

print('=' * 72)
print('FLOWCHART GENERATED (v3 - Updated per plan.md)!')
print('=' * 72)
print(f'PNG: {out_base}.png')
print(f'PDF: {out_base}.pdf')
print('=' * 72)
print('\nKey Changes from v2:')
print('  - Phase 3 now: Pareto优化与动态加权 (核心方法论)')
print('  - Phase 4 now: 规则模拟与案例验证')
print('  - Highlighted: Sigmoid Dynamic Rule (w_min=0.30, w_max=0.75, s=6)')
print('  - Multi-Phase Evaluation Framework emphasized')
print('  - Impact metrics: F_early +52.7%, J_late +67.5%, Composite +21.6%')
print('=' * 72)
