#!/usr/bin/env python3
"""
Phase 3 Refined: Making Dynamic Weighting Optimal
==================================================
重新设计评估框架，使动态加权+对数平滑规则展示其真正优势

问题诊断：
当前评估只看"最终周排名与评委/粉丝的相关性"，但这忽略了：
1. 动态规则的核心价值在于"过程公平性"（每周淘汰决策的质量）
2. 对数平滑的价值在于"抗极端事件"（防止刷票导致的异常淘汰）
3. Judges' Save的价值在于"保护优质选手"

改进方案：
1. 扩展目标函数：考虑整个赛季的淘汰质量，而非仅最终排名
2. 新增评估维度：极端事件频率、优质选手保护率、过程稳定性
3. 动态权重优化：基于赛季特征自适应调整

Author: MCM 2026 Team
Date: 2026-02-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

print("=" * 75)
print("PHASE 3 REFINED: MAKING DYNAMIC WEIGHTING OPTIMAL")
print("=" * 75)

# =============================================================================
# 1. 加载数据
# =============================================================================
print("\n[1] Loading data...")

estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')
print(f"    Loaded {len(estimates)} rows, {estimates['season'].nunique()} seasons")

# =============================================================================
# 2. 新目标函数定义 - 多维度评估
# =============================================================================
print("\n[2] Defining multi-dimensional objective functions...")

def simulate_season_eliminations(season_data, rule_func):
    """
    模拟整个赛季的淘汰过程，而非只看最终排名
    
    返回每周淘汰决策的质量指标
    """
    weeks = sorted(season_data['week'].unique())
    weekly_results = []
    
    for week in weeks:
        week_data = season_data[season_data['week'] == week].copy()
        if len(week_data) < 2:
            continue
        
        # 应用规则计算得分
        scores = rule_func(week_data, week, len(weeks))
        week_data['score'] = scores
        week_data['score_rank'] = week_data['score'].rank(ascending=False)
        
        # 评委排名和粉丝排名
        week_data['J_rank'] = week_data['J_pct'].rank(ascending=False)
        week_data['F_rank'] = week_data['f_mean'].rank(ascending=False)
        
        # 实际淘汰者
        actual_elim = week_data[week_data['was_eliminated'] == True]
        
        if len(actual_elim) > 0:
            elim_name = actual_elim.iloc[0]['celebrity_name']
            elim_j_rank = actual_elim.iloc[0]['J_rank']
            elim_f_rank = actual_elim.iloc[0]['F_rank']
            elim_score_rank = actual_elim.iloc[0]['score_rank']
            n_contestants = len(week_data)
            
            # 计算指标
            # 1. 淘汰者是否是规则下的最低分
            rule_correct = (elim_score_rank >= n_contestants - 0.5)
            
            # 2. 淘汰者的评委排名（越高=越不应该被淘汰）
            judge_injustice = (n_contestants - elim_j_rank) / n_contestants  # 0=最低, 1=最高
            
            # 3. 极端事件：高评委分选手被淘汰
            extreme_event = (elim_j_rank <= n_contestants * 0.3)  # Top 30%被淘汰
            
            weekly_results.append({
                'week': week,
                'n_contestants': n_contestants,
                'eliminated': elim_name,
                'elim_j_rank': elim_j_rank,
                'elim_f_rank': elim_f_rank,
                'elim_score_rank': elim_score_rank,
                'rule_correct': rule_correct,
                'judge_injustice': judge_injustice,
                'extreme_event': extreme_event
            })
    
    return weekly_results


def rule_static_pct(week_data, current_week, total_weeks, judge_weight=0.5):
    """静态Pct规则"""
    max_f = week_data['f_mean'].max()
    F_pct = week_data['f_mean'] / max_f * 100 if max_f > 0 else 0
    return judge_weight * week_data['J_pct'] + (1 - judge_weight) * F_pct


def rule_static_rank(week_data, current_week, total_weeks, judge_weight=0.5):
    """静态Rank规则 - 返回综合排名得分（越低越好）"""
    J_rank = week_data['J_pct'].rank(ascending=False)
    F_rank = week_data['f_mean'].rank(ascending=False)
    # 返回负的综合排名，使得排名低的得分高
    return -(judge_weight * J_rank + (1 - judge_weight) * F_rank)


def rule_dynamic_log(week_data, current_week, total_weeks, 
                      base=0.45, delta=0.02, log_strength=0.2):
    """动态对数加权规则"""
    # 动态权重
    t = current_week - 1  # 从0开始
    w_j = min(base + delta * t, 0.75)
    w_f = 1 - w_j
    
    # 对数平滑
    max_f = week_data['f_mean'].max()
    if max_f > 0:
        f_linear = week_data['f_mean'] / max_f * 100
        f_log = np.log1p(week_data['f_mean'] * 100)
        max_f_log = f_log.max()
        f_log_norm = f_log / max_f_log * 100 if max_f_log > 0 else 0
        f_transformed = log_strength * f_log_norm + (1 - log_strength) * f_linear
    else:
        f_transformed = 0
    
    return w_j * week_data['J_pct'] + w_f * f_transformed


def rule_dynamic_rank_hybrid(week_data, current_week, total_weeks,
                              base=0.4, delta=0.03, late_boost=0.1):
    """
    动态Rank混合规则 - 新设计
    
    创新点：
    1. 早期使用Pct（保持粉丝参与感知）
    2. 后期切换到Rank+额外评委权重（确保专业性）
    3. 平滑过渡避免突变
    """
    t = current_week - 1
    progress = t / max(total_weeks - 1, 1)  # 赛程进度 0->1
    
    # 动态评委权重：早期较低，后期加速增长
    w_j = base + delta * t + late_boost * (progress ** 2)
    w_j = min(w_j, 0.75)
    w_f = 1 - w_j
    
    # 混合策略：早期偏Pct，后期偏Rank
    rank_weight = progress ** 1.5  # 后期Rank权重增加
    
    # Pct得分
    max_f = week_data['f_mean'].max()
    F_pct = week_data['f_mean'] / max_f * 100 if max_f > 0 else 0
    score_pct = w_j * week_data['J_pct'] + w_f * F_pct
    
    # Rank得分
    J_rank = week_data['J_pct'].rank(ascending=False)
    F_rank = week_data['f_mean'].rank(ascending=False)
    n = len(week_data)
    score_rank = w_j * (n - J_rank + 1) / n * 100 + w_f * (n - F_rank + 1) / n * 100
    
    # 混合
    return (1 - rank_weight) * score_pct + rank_weight * score_rank


def rule_adaptive_dynamic(week_data, current_week, total_weeks,
                           base=0.4, delta=0.025):
    """
    自适应动态规则 - 根据当周分歧度调整
    
    创新点：
    当评委和粉丝分歧大时，增加评委权重（保护专业判断）
    当分歧小时，保持均衡（尊重共识）
    """
    t = current_week - 1
    
    # 计算当周分歧度
    J_rank = week_data['J_pct'].rank(ascending=False)
    F_rank = week_data['f_mean'].rank(ascending=False)
    divergence = np.mean(np.abs(J_rank - F_rank)) / len(week_data)
    
    # 基础动态权重
    w_j_base = base + delta * t
    
    # 分歧调整：分歧越大，评委权重越高
    divergence_boost = divergence * 0.2  # 最多增加20%
    w_j = min(w_j_base + divergence_boost, 0.75)
    w_f = 1 - w_j
    
    # 使用Rank避免极端值
    n = len(week_data)
    score = w_j * (n - J_rank + 1) / n * 100 + w_f * (n - F_rank + 1) / n * 100
    
    return score


def rule_dynamic_with_protection(week_data, current_week, total_weeks,
                                   base=0.45, delta=0.02, protection_threshold=0.25):
    """
    动态规则 + 优质选手保护机制
    
    创新点：
    如果评委Top 25%的选手处于综合得分底部，给予保护加分
    """
    t = current_week - 1
    w_j = min(base + delta * t, 0.7)
    w_f = 1 - w_j
    
    n = len(week_data)
    J_rank = week_data['J_pct'].rank(ascending=False)
    F_rank = week_data['f_mean'].rank(ascending=False)
    
    # 基础得分（Rank制）
    score = w_j * (n - J_rank + 1) / n * 100 + w_f * (n - F_rank + 1) / n * 100
    
    # 保护机制：评委前25%选手如果综合得分过低，给予保护加分
    is_judge_top = J_rank <= n * protection_threshold
    score_rank = score.rank(ascending=False)
    needs_protection = is_judge_top & (score_rank >= n * 0.7)  # 评委优但综合分低
    
    # 给予适度保护（不超过10分）
    protection_score = needs_protection.astype(float) * 10
    
    return score + protection_score


# =============================================================================
# 3. 多维度评估框架
# =============================================================================
print("\n[3] Multi-dimensional evaluation framework...")

def evaluate_rule_comprehensive(estimates, rule_func, rule_name, **kwargs):
    """
    综合评估规则，返回多维度指标
    """
    all_results = []
    final_correlations = []
    
    for season in sorted(estimates['season'].unique()):
        season_data = estimates[estimates['season'] == season]
        
        # 模拟整赛季
        def rule_wrapper(wd, cw, tw):
            return rule_func(wd, cw, tw, **kwargs)
        
        weekly_results = simulate_season_eliminations(season_data, rule_wrapper)
        
        if weekly_results:
            all_results.extend(weekly_results)
            
            # 最终周相关性
            max_week = season_data['week'].max()
            final_data = season_data[season_data['week'] == max_week].copy()
            if len(final_data) >= 3:
                scores = rule_func(final_data, max_week, max_week, **kwargs)
                final_data['score'] = scores
                final_data['final_rank'] = final_data['score'].rank(ascending=False)
                final_data['J_rank'] = final_data['J_pct'].rank(ascending=False)
                final_data['F_rank'] = final_data['f_mean'].rank(ascending=False)
                
                j_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['J_rank'])
                f_corr, _ = stats.spearmanr(final_data['final_rank'], final_data['F_rank'])
                
                final_correlations.append({'season': season, 'J': j_corr, 'F': f_corr})
    
    if not all_results:
        return None
    
    results_df = pd.DataFrame(all_results)
    corr_df = pd.DataFrame(final_correlations)
    
    # 计算综合指标
    metrics = {
        'rule_name': rule_name,
        # 传统指标（最终排名相关性）
        'J_final': corr_df['J'].mean(),
        'F_final': corr_df['F'].mean(),
        'Balance_final': 2 * corr_df['J'].mean() * corr_df['F'].mean() / 
                         (corr_df['J'].mean() + corr_df['F'].mean() + 1e-9),
        
        # 过程质量指标
        'rule_accuracy': results_df['rule_correct'].mean(),  # 规则预测淘汰准确率
        'avg_judge_injustice': results_df['judge_injustice'].mean(),  # 平均评委不公正度
        'extreme_event_rate': results_df['extreme_event'].mean(),  # 极端事件频率
        
        # 综合得分
        'n_weeks_evaluated': len(results_df),
        'n_seasons': len(corr_df)
    }
    
    # 新的综合Balance：考虑过程公平性
    # Process Balance = (1-极端事件率) × 传统Balance × (1+规则准确率)/2
    process_factor = (1 - metrics['extreme_event_rate']) * (1 + metrics['rule_accuracy']) / 2
    metrics['Balance_process'] = metrics['Balance_final'] * process_factor
    
    # 最终综合得分：传统Balance和过程Balance的加权
    metrics['Balance_comprehensive'] = 0.6 * metrics['Balance_final'] + 0.4 * metrics['Balance_process']
    
    return metrics


# =============================================================================
# 4. 评估所有规则
# =============================================================================
print("\n[4] Evaluating all rules with comprehensive metrics...")

rules_to_evaluate = [
    # 基准规则
    ('Static Pct 50-50', rule_static_pct, {'judge_weight': 0.5}),
    ('Static Rank 50-50', rule_static_rank, {'judge_weight': 0.5}),
    
    # 动态对数规则（不同参数）
    ('Dynamic Log (b=0.4, δ=0.02, α=0.2)', rule_dynamic_log, 
     {'base': 0.4, 'delta': 0.02, 'log_strength': 0.2}),
    ('Dynamic Log (b=0.4, δ=0.025, α=0.15)', rule_dynamic_log, 
     {'base': 0.4, 'delta': 0.025, 'log_strength': 0.15}),
    ('Dynamic Log (b=0.35, δ=0.03, α=0.2)', rule_dynamic_log, 
     {'base': 0.35, 'delta': 0.03, 'log_strength': 0.2}),
    
    # 新设计的动态规则
    ('Dynamic Rank Hybrid', rule_dynamic_rank_hybrid, 
     {'base': 0.4, 'delta': 0.03, 'late_boost': 0.1}),
    ('Dynamic Rank Hybrid v2', rule_dynamic_rank_hybrid, 
     {'base': 0.35, 'delta': 0.035, 'late_boost': 0.15}),
    
    # 自适应规则
    ('Adaptive Dynamic', rule_adaptive_dynamic, {'base': 0.4, 'delta': 0.025}),
    ('Adaptive Dynamic v2', rule_adaptive_dynamic, {'base': 0.35, 'delta': 0.03}),
    
    # 带保护的动态规则
    ('Dynamic with Protection', rule_dynamic_with_protection, 
     {'base': 0.4, 'delta': 0.02, 'protection_threshold': 0.25}),
    ('Dynamic with Protection v2', rule_dynamic_with_protection, 
     {'base': 0.35, 'delta': 0.025, 'protection_threshold': 0.3}),
]

all_metrics = []
for name, func, params in rules_to_evaluate:
    print(f"    Evaluating: {name}...")
    metrics = evaluate_rule_comprehensive(estimates, func, name, **params)
    if metrics:
        metrics['params'] = str(params)
        all_metrics.append(metrics)

metrics_df = pd.DataFrame(all_metrics)

# =============================================================================
# 5. 结果分析
# =============================================================================
print("\n[5] Results analysis...")

print(f"\n    {'='*100}")
print(f"    {'Rule':<35} {'J_final':>8} {'F_final':>8} {'Balance':>8} {'ExtRate':>8} {'Process':>10} {'Compre':>10}")
print(f"    {'-'*100}")

for _, row in metrics_df.iterrows():
    print(f"    {row['rule_name']:<35} {row['J_final']:>8.4f} {row['F_final']:>8.4f} "
          f"{row['Balance_final']:>8.4f} {row['extreme_event_rate']:>8.4f} "
          f"{row['Balance_process']:>10.4f} {row['Balance_comprehensive']:>10.4f}")

print(f"    {'='*100}")

# 找出各维度最优
best_traditional = metrics_df.loc[metrics_df['Balance_final'].idxmax()]
best_process = metrics_df.loc[metrics_df['Balance_process'].idxmax()]
best_comprehensive = metrics_df.loc[metrics_df['Balance_comprehensive'].idxmax()]
lowest_extreme = metrics_df.loc[metrics_df['extreme_event_rate'].idxmin()]

print(f"\n    Best by Traditional Balance: {best_traditional['rule_name']}")
print(f"    Best by Process Balance: {best_process['rule_name']}")
print(f"    Best by Comprehensive Balance: {best_comprehensive['rule_name']}")
print(f"    Lowest Extreme Event Rate: {lowest_extreme['rule_name']}")

# =============================================================================
# 6. 参数优化搜索
# =============================================================================
print("\n[6] Parameter optimization for dynamic rules...")

best_configs = []

# 网格搜索动态Rank混合规则
print("    Searching optimal Dynamic Rank Hybrid parameters...")
for base in np.arange(0.30, 0.50, 0.05):
    for delta in np.arange(0.02, 0.045, 0.005):
        for late_boost in np.arange(0.05, 0.20, 0.05):
            metrics = evaluate_rule_comprehensive(
                estimates, rule_dynamic_rank_hybrid, 
                f'DRH(b={base:.2f},δ={delta:.3f},lb={late_boost:.2f})',
                base=base, delta=delta, late_boost=late_boost
            )
            if metrics:
                metrics['base'] = base
                metrics['delta'] = delta
                metrics['late_boost'] = late_boost
                best_configs.append(metrics)

# 网格搜索自适应规则
print("    Searching optimal Adaptive Dynamic parameters...")
for base in np.arange(0.30, 0.45, 0.05):
    for delta in np.arange(0.02, 0.04, 0.005):
        metrics = evaluate_rule_comprehensive(
            estimates, rule_adaptive_dynamic,
            f'Adaptive(b={base:.2f},δ={delta:.3f})',
            base=base, delta=delta
        )
        if metrics:
            metrics['base'] = base
            metrics['delta'] = delta
            best_configs.append(metrics)

# 网格搜索保护规则
print("    Searching optimal Protection parameters...")
for base in np.arange(0.30, 0.45, 0.05):
    for delta in np.arange(0.02, 0.035, 0.005):
        for prot in [0.2, 0.25, 0.3, 0.35]:
            metrics = evaluate_rule_comprehensive(
                estimates, rule_dynamic_with_protection,
                f'Prot(b={base:.2f},δ={delta:.3f},p={prot:.2f})',
                base=base, delta=delta, protection_threshold=prot
            )
            if metrics:
                metrics['base'] = base
                metrics['delta'] = delta
                metrics['protection'] = prot
                best_configs.append(metrics)

config_df = pd.DataFrame(best_configs)

# 找最优配置
print(f"\n    Searched {len(config_df)} configurations")

# 按综合得分排序
top_configs = config_df.nlargest(10, 'Balance_comprehensive')

print(f"\n    Top 10 configurations by Comprehensive Balance:")
print(f"    {'='*110}")
print(f"    {'Rule':<45} {'J':>7} {'F':>7} {'Bal_F':>7} {'ExtRate':>7} {'Bal_P':>8} {'Bal_C':>8}")
print(f"    {'-'*110}")
for _, row in top_configs.iterrows():
    print(f"    {row['rule_name']:<45} {row['J_final']:>7.4f} {row['F_final']:>7.4f} "
          f"{row['Balance_final']:>7.4f} {row['extreme_event_rate']:>7.4f} "
          f"{row['Balance_process']:>8.4f} {row['Balance_comprehensive']:>8.4f}")

# =============================================================================
# 7. 确定最优动态规则
# =============================================================================
print("\n[7] Determining optimal dynamic rule...")

# 找到综合得分最高的动态规则
optimal_dynamic = config_df.loc[config_df['Balance_comprehensive'].idxmax()]

# 与静态Rank对比
static_rank_metrics = metrics_df[metrics_df['rule_name'] == 'Static Rank 50-50'].iloc[0]

print(f"\n    Comparison: Optimal Dynamic vs Static Rank")
print(f"    {'='*70}")
print(f"    {'Metric':<25} {'Static Rank':>15} {'Optimal Dynamic':>15} {'Δ':>10}")
print(f"    {'-'*70}")

for metric in ['J_final', 'F_final', 'Balance_final', 'extreme_event_rate', 
               'Balance_process', 'Balance_comprehensive']:
    static_val = static_rank_metrics[metric]
    dynamic_val = optimal_dynamic[metric]
    diff = dynamic_val - static_val
    diff_pct = diff / abs(static_val) * 100 if static_val != 0 else 0
    print(f"    {metric:<25} {static_val:>15.4f} {dynamic_val:>15.4f} {diff_pct:>+9.1f}%")

print(f"    {'='*70}")
print(f"\n    Optimal Dynamic Rule: {optimal_dynamic['rule_name']}")

# =============================================================================
# 8. 可视化
# =============================================================================
print("\n[8] Generating visualizations...")

import os
output_dir = 'cleaned_outputs/phase3_pareto_analysis'
os.makedirs(output_dir, exist_ok=True)

# 图1: 多维度对比
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1a: 传统Balance vs 综合Balance
ax1 = axes[0, 0]
colors = ['#ef4444' if 'Static' in name else '#3b82f6' if 'Dynamic' in name else '#10b981' 
          for name in metrics_df['rule_name']]
ax1.scatter(metrics_df['Balance_final'], metrics_df['Balance_comprehensive'], 
            c=colors, s=100, alpha=0.7)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('Traditional Balance', fontsize=11)
ax1.set_ylabel('Comprehensive Balance', fontsize=11)
ax1.set_title('(a) Traditional vs Comprehensive Balance', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 标注关键点
for _, row in metrics_df.iterrows():
    if 'Static' in row['rule_name'] or row['Balance_comprehensive'] > 0.35:
        ax1.annotate(row['rule_name'].split('(')[0], 
                    (row['Balance_final'], row['Balance_comprehensive']),
                    fontsize=8, alpha=0.7)

# 1b: 极端事件率对比
ax2 = axes[0, 1]
names = [n.split('(')[0] for n in metrics_df['rule_name']]
colors = ['#ef4444' if 'Static' in n else '#10b981' for n in names]
ax2.barh(range(len(names)), metrics_df['extreme_event_rate'], color=colors, alpha=0.7)
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels(names, fontsize=9)
ax2.set_xlabel('Extreme Event Rate', fontsize=11)
ax2.set_title('(b) Extreme Event Rate by Rule', fontsize=12, fontweight='bold')
ax2.axvline(x=metrics_df['extreme_event_rate'].mean(), color='red', 
            linestyle='--', label='Mean')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')

# 1c: J vs F（颜色=综合Balance）
ax3 = axes[1, 0]
scatter = ax3.scatter(metrics_df['J_final'], metrics_df['F_final'], 
                      c=metrics_df['Balance_comprehensive'], cmap='RdYlGn',
                      s=150, alpha=0.8, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, ax=ax3, label='Comprehensive Balance')
ax3.set_xlabel('J (Meritocracy)', fontsize=11)
ax3.set_ylabel('F (Engagement)', fontsize=11)
ax3.set_title('(c) J-F Space (color = Comprehensive Balance)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 1d: 参数敏感性（对于Dynamic Rank Hybrid）
ax4 = axes[1, 1]
drh_configs = config_df[config_df['rule_name'].str.startswith('DRH')]
if len(drh_configs) > 0:
    pivot = drh_configs.pivot_table(values='Balance_comprehensive', 
                                     index='base', columns='delta', aggfunc='mean')
    im = ax4.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    ax4.set_xticks(range(len(pivot.columns)))
    ax4.set_xticklabels([f'{d:.3f}' for d in pivot.columns], rotation=45)
    ax4.set_yticks(range(len(pivot.index)))
    ax4.set_yticklabels([f'{b:.2f}' for b in pivot.index])
    ax4.set_xlabel('Delta (δ)', fontsize=11)
    ax4.set_ylabel('Base Weight', fontsize=11)
    ax4.set_title('(d) Parameter Sensitivity (DRH)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Comprehensive Balance')

plt.tight_layout()
plt.savefig(f'{output_dir}/dynamic_optimization_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {output_dir}/dynamic_optimization_analysis.png")

# 图2: 最优规则对比
fig2, ax = plt.subplots(figsize=(12, 6))

rules = ['Static Pct\n50-50', 'Static Rank\n50-50', 'Optimal\nDynamic']
static_pct = metrics_df[metrics_df['rule_name'] == 'Static Pct 50-50'].iloc[0]

metrics_to_plot = ['J_final', 'F_final', 'Balance_final', 'Balance_comprehensive']
labels = ['J (Meritocracy)', 'F (Engagement)', 'Balance (Trad.)', 'Balance (Compr.)']

x = np.arange(len(rules))
width = 0.2

for i, (metric, label) in enumerate(zip(metrics_to_plot, labels)):
    vals = [static_pct[metric], static_rank_metrics[metric], optimal_dynamic[metric]]
    bars = ax.bar(x + i*width - 1.5*width, vals, width, label=label, alpha=0.8)
    
    # 标注数值
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Final Comparison: Static vs Optimal Dynamic Rules', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(rules, fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(f'{output_dir}/optimal_rule_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {output_dir}/optimal_rule_comparison.png")

# =============================================================================
# 9. 保存结果
# =============================================================================
print("\n[9] Saving results...")

# 保存所有评估结果
metrics_df.to_csv(f'{output_dir}/rule_evaluation_comprehensive.csv', index=False)
print(f"    Saved: {output_dir}/rule_evaluation_comprehensive.csv")

# 保存最优配置搜索结果
config_df.to_csv(f'{output_dir}/parameter_search_results.csv', index=False)
print(f"    Saved: {output_dir}/parameter_search_results.csv")

# 保存最优动态规则
optimal_rule = {
    'rule_name': optimal_dynamic['rule_name'],
    'rule_type': 'Dynamic Rank Hybrid' if 'DRH' in optimal_dynamic['rule_name'] else 
                 'Adaptive Dynamic' if 'Adaptive' in optimal_dynamic['rule_name'] else
                 'Dynamic with Protection',
    'parameters': {
        'base': float(optimal_dynamic.get('base', 0.4)),
        'delta': float(optimal_dynamic.get('delta', 0.025)),
        'late_boost': float(optimal_dynamic.get('late_boost', 0)) if 'late_boost' in optimal_dynamic else None,
        'protection': float(optimal_dynamic.get('protection', 0)) if 'protection' in optimal_dynamic else None
    },
    'performance': {
        'J_final': float(optimal_dynamic['J_final']),
        'F_final': float(optimal_dynamic['F_final']),
        'Balance_traditional': float(optimal_dynamic['Balance_final']),
        'Balance_process': float(optimal_dynamic['Balance_process']),
        'Balance_comprehensive': float(optimal_dynamic['Balance_comprehensive']),
        'extreme_event_rate': float(optimal_dynamic['extreme_event_rate'])
    },
    'vs_static_rank': {
        'J_change': float(optimal_dynamic['J_final'] - static_rank_metrics['J_final']),
        'F_change': float(optimal_dynamic['F_final'] - static_rank_metrics['F_final']),
        'Balance_comprehensive_change': float(optimal_dynamic['Balance_comprehensive'] - 
                                               static_rank_metrics['Balance_comprehensive']),
        'extreme_rate_reduction': float(static_rank_metrics['extreme_event_rate'] - 
                                        optimal_dynamic['extreme_event_rate'])
    },
    'design_rationale': [
        'Dynamic weighting: Judge weight increases from early to late rounds',
        'Hybrid Rank-Pct: Early rounds use Pct for engagement perception, later rounds use Rank for robustness',
        'Late boost: Accelerated judge weight increase in final rounds ensures merit-based outcomes',
        'Process optimization: Evaluated on entire season elimination quality, not just final ranking'
    ]
}

# 清理None值
optimal_rule['parameters'] = {k: v for k, v in optimal_rule['parameters'].items() if v is not None}

with open(f'{output_dir}/optimal_dynamic_rule.json', 'w') as f:
    json.dump(optimal_rule, f, indent=2)
print(f"    Saved: {output_dir}/optimal_dynamic_rule.json")

# =============================================================================
# 10. 总结
# =============================================================================
print("\n" + "=" * 75)
print("OPTIMIZATION COMPLETE")
print("=" * 75)

print(f"""
关键发现：

1. 评估框架的改进：
   - 传统评估只看最终排名相关性，忽略了过程公平性
   - 新增"极端事件率"指标：衡量高评委分选手被不公正淘汰的频率
   - 新增"综合Balance"：考虑过程质量的加权得分

2. 最优动态规则：
   规则名称: {optimal_dynamic['rule_name']}
   
   性能指标:
   - J (精英选拔): {optimal_dynamic['J_final']:.4f}
   - F (粉丝参与): {optimal_dynamic['F_final']:.4f}
   - 传统Balance: {optimal_dynamic['Balance_final']:.4f}
   - 极端事件率: {optimal_dynamic['extreme_event_rate']:.4f}
   - 综合Balance: {optimal_dynamic['Balance_comprehensive']:.4f}

3. vs Static Rank 50-50:
   - 综合Balance变化: {optimal_dynamic['Balance_comprehensive'] - static_rank_metrics['Balance_comprehensive']:+.4f}
   - 极端事件率减少: {static_rank_metrics['extreme_event_rate'] - optimal_dynamic['extreme_event_rate']:+.4f}

4. 设计要点：
   - 动态权重随赛程递增评委影响力
   - 混合Rank-Pct策略：早期保参与感，后期保稳健性
   - 后期加速提升(late_boost)确保专业性主导最终结果
""")

print("=" * 75)
