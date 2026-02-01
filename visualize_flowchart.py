#!/usr/bin/env python3
"""MCM 2026 workflow flowchart generator.

目标：对齐 plan.md 的 Phase 1-5 逻辑，并生成更美观、可用于正文的流程图。

Output:
  - cleaned_outputs/workflow_flowchart_v2.png
  - cleaned_outputs/workflow_flowchart_v2.pdf
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
    splines='ortho',
    nodesep='0.12',
    ranksep='0.18',
)
dot.attr(
    'graph',
    bgcolor='white',
    pad='0.02',
    margin='0',
    dpi='280',
    size='7,7!',
    ratio='fill',
    pack='true',
    packmode='clust',
)
dot.attr('node', fontname='DejaVu Sans', fontsize='9', margin='0.10,0.08')
dot.attr('edge', fontname='DejaVu Sans', fontsize='8', arrowsize='0.65', color='#455A64')


# ----------------------------- Phase 1 -----------------------------
with dot.subgraph(name='cluster_phase1') as c:
    c.attr(
        label='PHASE 1  Data & Global Scan',
        labelloc='t',
        labeljust='l',
        style='rounded,filled',
        color='#90CAF9',
        fillcolor='#F7FBFF',
        fontsize='10.5',
    )

    _node(
        c,
        'raw',
        f'Raw Data\n{total_contestants} contestants • {total_seasons} seasons',
        'input',
        shape='cylinder',
    )
    _node(c, 'clean', 'Cleaning\nJ% std • withdrawals/N/A', 'process')
    _node(c, 'feat', 'Features\nPBI + covariates', 'process')
    _node(c, 'panel', f'Panel (i,w)\n{total_obs:,} obs', 'output', shape='note')
    _node(c, 'div', 'Global Scan\nDivergence trend', 'insight')

    c.edge('raw', 'clean')
    c.edge('clean', 'feat')
    c.edge('clean', 'panel')
    c.edge('feat', 'panel')
    c.edge('panel', 'div')


# ----------------------------- Phase 2 -----------------------------
with dot.subgraph(name='cluster_phase2') as c:
    c.attr(
        label='PHASE 2  Bayesian Inference',
        labelloc='t',
        labeljust='l',
        style='rounded,filled',
        color='#A5D6A7',
        fillcolor='#F7FFF8',
        fontsize='10.5',
    )

    _node(c, 'constraints', 'Constraints\nBottom-k • Σf=1', 'process')
    _node(c, 'mcmc', f'MCMC (Hit-and-Run)\nAvg CI: {avg_ci_width:.3f}', 'core')
    _node(c, 'post', f'Posterior f(i,w)\n{total_obs:,} + 95% CI', 'output', shape='note')
    _node(
        c,
        'valid',
        f'Validation\nCI {avg_ci_width:.3f} • CV {cv_mean:.3f} • Exact {exact_match*100:.1f}% • P̄ {p_bar:.3f}',
        'insight',
    )

    c.edge('constraints', 'mcmc', style='dashed', color='#6D4C41')
    c.edge('mcmc', 'post')
    c.edge('post', 'valid')


# ----------------------------- Phase 3 -----------------------------
with dot.subgraph(name='cluster_phase3') as c:
    c.attr(
        label='PHASE 3  Simulator & Evidence',
        labelloc='t',
        labeljust='l',
        style='rounded,filled',
        color='#CE93D8',
        fillcolor='#FDF7FF',
        fontsize='10.5',
    )

    _node(c, 'sim', f'Simulator\nreplay {total_seasons} seasons', 'core')
    _node(c, 'rank', f'Rank\nFFI {rank_ffi:.3f}', 'process')
    _node(c, 'pct', f'Pct\nFFI {pct_ffi:.3f}', 'process')
    _node(c, 'bias', f'Bias\nΔFFI {ffi_delta:+.3f}', 'insight')
    _node(c, 'cases', '4 Cases\nRice • Cyrus • Palin • Bones', 'process')
    _node(c, 'supp', 'Effects\nPro + covariates', 'process')

    c.edge('sim', 'rank')
    c.edge('sim', 'pct')
    c.edge('rank', 'bias')
    c.edge('pct', 'bias')


# ----------------------------- Phase 4 -----------------------------
with dot.subgraph(name='cluster_phase4') as c:
    c.attr(
        label='PHASE 4  Pareto Optimization',
        labelloc='t',
        labeljust='l',
        style='rounded,filled',
        color='#81C784',
        fillcolor='#F6FFF7',
        fontsize='10.5',
    )

    _node(c, 'obj', f'Objectives\nJ {J_star:.3f} • F {F_star:.3f}', 'process')
    _node(c, 'pareto', 'Frontier\n51 weights', 'core')
    _node(c, 'knee', f'Knee\nR {rank_knee:.3f} • P {pct_knee:.3f}', 'decision', shape='diamond', style='filled', penwidth='2.6')
    _node(c, 'save', "Save\n+J at modest F cost", 'insight')

    c.edge('obj', 'pareto')
    c.edge('pareto', 'knee', penwidth='2.2', color='#C62828')
    c.edge('knee', 'save')


# ----------------------------- Phase 5 -----------------------------
with dot.subgraph(name='cluster_phase5') as c:
    c.attr(
        label='PHASE 5  Recommendation',
        labelloc='t',
        labeljust='l',
        style='rounded,filled',
        color='#FFAB91',
        fillcolor='#FFF7F4',
        fontsize='10.5',
    )

    _node(
        c,
        'rec',
        f'RECOMMEND\n{rec_method} • {rec_weights} • Save {"Y" if rec_save else "N"}',
        'decision',
        style='rounded,filled,bold',
        penwidth='3.0',
    )
    _node(
        c,
        'compare',
        f'Impact\nJ {J_current:.3f}→{J_star:.3f} ({J_gain_pct:+.0f}%) • F {F_current:.3f}→{F_star:.3f}',
        'insight',
    )
    _node(c, 'memo', 'Memo\nimplementation + risks', 'output', shape='note')

    c.edge('rec', 'compare')
    c.edge('compare', 'memo', penwidth='2.2', color='#6A1B9A')


# ----------------------------- Cross-phase links (narrative) -----------------------------
dot.edge('panel', 'mcmc', penwidth='2.0', color='#2E7D32')
dot.edge('post', 'sim', penwidth='2.2', color='#2E7D32')
dot.edge('feat', 'supp', style='dashed')
dot.edge('post', 'supp', style='dashed')
dot.edge('post', 'cases')

dot.edge('div', 'obj', xlabel='reform rationale', style='bold', color='#1565C0', penwidth='2.2')
dot.edge('bias', 'obj', xlabel='trade-off', style='dashed', color='#1565C0')
dot.edge('save', 'rec', style='bold', color='#C62828', penwidth='2.2')
dot.edge('cases', 'memo', style='dashed', color='#546E7A')
dot.edge('supp', 'memo', style='dashed', color='#546E7A')


# ----------------------------- Compact legend node -----------------------------
_node(
    dot,
    'legend',
    'Legend\nInput • Process • Core • Output • Decision • Finding',
    'output',
    shape='note',
    fontsize='8',
    penwidth='1.6',
)


out_base = ROOT / 'cleaned_outputs' / 'workflow_flowchart_v2'
dot.format = 'png'
dot.render(str(out_base), cleanup=True)

dot.format = 'pdf'
dot.render(str(out_base), cleanup=True)

print('=' * 72)
print('FLOWCHART GENERATED (v2)!')
print('=' * 72)
print(f'PNG: {out_base}.png')
print(f'PDF: {out_base}.pdf')
print('=' * 72)
