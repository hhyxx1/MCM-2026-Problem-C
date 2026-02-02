#!/usr/bin/env python3
"""
MCM 2026 Paper - "Our Work" Workflow Diagram Generator
======================================================
Compact square layout, no crossing lines.
"""

from graphviz import Digraph
import os

def create_workflow():
    """Create workflow with compact square layout, no line crossing."""
    
    dot = Digraph(
        name='MCM2026_DWTS_Workflow',
        format='pdf',
        engine='dot'
    )
    
    # Compact square layout
    dot.attr(
        rankdir='TB',
        splines='ortho',        # Orthogonal lines avoid crossing
        nodesep='0.35',         # Tighter horizontal
        ranksep='0.35',         # Tighter vertical
        fontname='Arial',
        fontsize='10',
        bgcolor='white',
        dpi='300',
        size='7,7!',            # Smaller square
        ratio='fill',
        pad='0.2'               # Less padding
    )
    
    # Smaller nodes
    dot.attr('node',
        fontname='Arial',
        fontsize='9',
        style='filled',
        penwidth='1.2',
        width='1.6',
        height='0.65',
        margin='0.08,0.04'
    )
    
    # Edge style
    dot.attr('edge',
        fontname='Arial',
        fontsize='8',
        arrowsize='0.7',
        penwidth='1.0',
        color='#455A64'
    )
    
    # Colors
    colors = {
        'input':      '#BBDEFB',
        'output':     '#C8E6C9',
        'process':    '#FFF9C4',
        'core':       '#FFCCBC',
        'core_border':'#D84315',
        'decision':   '#E1BEE7',
        'validate':   '#F5F5F5',
    }
    
    # ============================================================
    # ROW 1: Input + Section 3 start
    # ============================================================
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('input_data',
            label='Raw DWTS Data\n(34 seasons)',
            shape='parallelogram',
            fillcolor=colors['input'],
            color='#1565C0'
        )
        s.node('preprocess',
            label='Preprocessing\nJ% Normalize',
            shape='box', style='rounded,filled',
            fillcolor=colors['process'], color='#F9A825'
        )
        s.node('pbi',
            label='Feature Eng.\nPBI Index',
            shape='box', style='rounded,filled',
            fillcolor=colors['process'], color='#F9A825'
        )
        s.node('divergence',
            label='Divergence\nAnalysis',
            shape='box', style='rounded,filled',
            fillcolor=colors['process'], color='#F9A825'
        )
    
    # ============================================================
    # ROW 2: Section 4 - Bayesian Inference [CORE]
    # ============================================================
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('constraints',
            label='Simplex\nConstraints',
            shape='box', style='rounded,filled',
            fillcolor=colors['process'], color='#F9A825'
        )
        s.node('mcmc',
            label='[PROPOSED]\nHit-Run MCMC',
            shape='box', style='rounded,bold,filled',
            fillcolor=colors['core'], color=colors['core_border'],
            penwidth='2.5'
        )
        s.node('fan_votes',
            label='Fan Votes\nf(i,w) ± CI',
            shape='parallelogram',
            fillcolor=colors['input'], color='#1565C0'
        )
        s.node('objectives',
            label='Dual Obj.\nJ / F',
            shape='box', style='rounded,filled',
            fillcolor=colors['process'], color='#F9A825'
        )
    
    # ============================================================
    # ROW 3: Section 5 - Pareto Optimization [CORE]
    # ============================================================
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('multiphase',
            label='[PROPOSED]\nMulti-Phase Eval',
            shape='box', style='rounded,bold,filled',
            fillcolor=colors['core'], color=colors['core_border'],
            penwidth='2.5'
        )
        s.node('search',
            label='Rule Search\n107 Configs',
            shape='box', style='rounded,filled',
            fillcolor=colors['process'], color='#F9A825'
        )
        s.node('optimal_rule',
            label='[PROPOSED]\nSigmoid Rule',
            shape='box', style='rounded,bold,filled',
            fillcolor=colors['core'], color=colors['core_border'],
            penwidth='2.5'
        )
        s.node('simulator',
            label='MC Simulator\nRank vs Pct',
            shape='box', style='rounded,filled',
            fillcolor=colors['process'], color='#F9A825'
        )
    
    # ============================================================
    # ROW 4: Section 6 + Section 7-8
    # ============================================================
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('compare',
            label='Rank wins\n+46.4%',
            shape='diamond',
            fillcolor=colors['decision'], color='#7B1FA2',
            width='1.2', height='0.9'
        )
        s.node('cases',
            label='Case Studies\n4/4 Fixed',
            shape='box', style='rounded,filled',
            fillcolor=colors['validate'], color='#9E9E9E'
        )
        s.node('covariate',
            label='Covariate\nEffects',
            shape='box', style='rounded,filled',
            fillcolor=colors['process'], color='#F9A825'
        )
        s.node('sensitivity',
            label='Sensitivity\n2.4× stable',
            shape='box', style='rounded,filled',
            fillcolor=colors['process'], color='#F9A825'
        )
    
    # ============================================================
    # ROW 5: Output
    # ============================================================
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('recommendation',
            label='Final Rule\nSigmoid Rank-Weighted',
            shape='parallelogram',
            fillcolor=colors['output'], color='#2E7D32'
        )
        s.node('memo',
            label='Policy Memo\nto Producer',
            shape='parallelogram',
            fillcolor=colors['output'], color='#2E7D32'
        )
    
    # ============================================================
    # EDGES - Main flow (S-shape to avoid crossing)
    # ============================================================
    
    # Row 1: left to right
    dot.edge('input_data', 'preprocess')
    dot.edge('preprocess', 'pbi')
    dot.edge('pbi', 'divergence')
    
    # Row 1 → Row 2 (down from right side)
    dot.edge('divergence', 'objectives')
    
    # Row 2: right to left (reverse S)
    dot.edge('objectives', 'fan_votes')
    dot.edge('fan_votes', 'mcmc')
    dot.edge('mcmc', 'constraints')
    
    # Row 2 → Row 3 (down from left side)
    dot.edge('constraints', 'multiphase')
    
    # Row 3: left to right
    dot.edge('multiphase', 'search')
    dot.edge('search', 'optimal_rule')
    dot.edge('optimal_rule', 'simulator')
    
    # Row 3 → Row 4 (down from right side)
    dot.edge('simulator', 'compare')
    
    # Row 4: right to left (reverse S)
    dot.edge('compare', 'cases')
    dot.edge('cases', 'covariate')
    dot.edge('covariate', 'sensitivity')
    
    # Row 4 → Row 5 (down from left side)
    dot.edge('sensitivity', 'recommendation')
    
    # Row 5
    dot.edge('recommendation', 'memo')
    
    # Cross-link: fan_votes → simulator (same column, just down)
    # Now they are adjacent vertically so no crossing
    dot.edge('fan_votes', 'search', style='dashed', constraint='false')
    
    # ============================================================
    # RENDER
    # ============================================================
    output_dir = '/home/hyx/文档/MCM/cleaned_outputs'
    output_base = os.path.join(output_dir, 'our_work_workflow')
    
    dot.format = 'pdf'
    dot.render(output_base, cleanup=True)
    
    dot.format = 'png'
    dot.render(output_base, cleanup=True)
    
    context_dir = '/home/hyx/文档/MCM/context/necessary'
    context_base = os.path.join(context_dir, 'our_work_workflow')
    
    dot.format = 'pdf'
    dot.render(context_base, cleanup=True)
    
    dot.format = 'png'
    dot.render(context_base, cleanup=True)
    
    return dot


def main():
    print("=" * 60)
    print("MCM 2026 - Our Work Workflow (Compact Square, No Crossing)")
    print("=" * 60)
    print()
    print("[1/2] Creating workflow (5 rows, S-shape flow)...")
    dot = create_workflow()
    print("[2/2] Exporting PDF & PNG...")
    print()
    print("=" * 60)
    print("OUTPUT FILES:")
    print("  cleaned_outputs/our_work_workflow.pdf")
    print("  cleaned_outputs/our_work_workflow.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
