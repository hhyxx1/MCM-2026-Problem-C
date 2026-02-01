#!/usr/bin/env python3
"""
Phase 4 Supplement: Pro Dancer & Celebrity Effects Model
=========================================================
使用scipy/numpy量化Pro Dancer和Celebrity特征对结果的影响：
1. Judge Score Model (Merit channel)
2. Fan Vote Model (Popularity channel)
3. 比较系数方向，回答"是否同向影响"

Author: MCM 2026 Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("PHASE 4 SUPPLEMENT: PRO DANCER & CELEBRITY EFFECTS MODEL")
print("=" * 70)

# =============================================================================
# PART 1: DATA PREPARATION
# =============================================================================
print("\n[1] Loading and preparing panel data...")

# 加载数据
panel = pd.read_csv('cleaned_outputs/clean_weekly_panel.csv')
estimates = pd.read_csv('cleaned_outputs/fan_vote_estimates.csv')
pbi = pd.read_csv('cleaned_outputs/contestant_pbi.csv')

# 合并f(i,w)估计到面板数据
panel_merged = panel.merge(
    estimates[['contestant_id', 'week', 'f_mean', 'ci_width']],
    on=['contestant_id', 'week'],
    how='left'
)

print(f"    Panel observations: {len(panel_merged)}")
print(f"    With f(i,w) estimates: {panel_merged['f_mean'].notna().sum()}")

# 创建分析所需变量
df = panel_merged.copy()

# 过滤有效数据
df_valid = df[df['f_mean'].notna() & df['J_pct'].notna()].copy()
df_valid = df_valid[df_valid['J_pct'] > 0]

# 对f_mean做logit变换
df_valid['f_clipped'] = df_valid['f_mean'].clip(0.001, 0.999)
df_valid['f_logit'] = np.log(df_valid['f_clipped'] / (1 - df_valid['f_clipped']))

print(f"    Valid observations for modeling: {len(df_valid)}")

# =============================================================================
# PART 2: SIMPLE LINEAR REGRESSION (OLS)
# =============================================================================
print("\n[2] Building Regression Models...")

def ols_regression(X, y):
    """简单OLS回归"""
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        residuals = y - y_pred
        n, p = X.shape
        mse = np.sum(residuals**2) / (n - p)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_beta))
        t_stats = beta / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))
        r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)
        return beta, se, p_values, r2
    except:
        return None, None, None, None

# Judge Score Model
print("\n    2a. Judge Score Model: J% ~ Week + Week² + Season")

X_judge = np.column_stack([
    df_valid['week'].values,
    df_valid['week'].values ** 2,
    df_valid['season'].values
])
y_judge = df_valid['J_pct'].values

beta_J, se_J, pval_J, r2_J = ols_regression(X_judge, y_judge)

print("\n    Judge Score Model Results:")
print("    " + "-" * 55)
print(f"    {'Variable':<15} {'Coef':>10} {'Std Err':>10} {'P-value':>10}")
print("    " + "-" * 55)
var_names = ['Intercept', 'Week', 'Week²', 'Season']
for i, name in enumerate(var_names):
    sig = "***" if pval_J[i] < 0.001 else "**" if pval_J[i] < 0.01 else "*" if pval_J[i] < 0.05 else ""
    print(f"    {name:<15} {beta_J[i]:>10.4f} {se_J[i]:>10.4f} {pval_J[i]:>10.4f} {sig}")
print(f"\n    R² = {r2_J:.4f}")

# Fan Vote Model
print("\n    2b. Fan Vote Model: logit(f) ~ Week + Week² + J% + Season")

X_fan = np.column_stack([
    df_valid['week'].values,
    df_valid['week'].values ** 2,
    df_valid['J_pct'].values,
    df_valid['season'].values
])
y_fan = df_valid['f_logit'].values

beta_F, se_F, pval_F, r2_F = ols_regression(X_fan, y_fan)

print("\n    Fan Vote Model Results:")
print("    " + "-" * 55)
print(f"    {'Variable':<15} {'Coef':>10} {'Std Err':>10} {'P-value':>10}")
print("    " + "-" * 55)
var_names = ['Intercept', 'Week', 'Week²', 'J%', 'Season']
for i, name in enumerate(var_names):
    sig = "***" if pval_F[i] < 0.001 else "**" if pval_F[i] < 0.01 else "*" if pval_F[i] < 0.05 else ""
    print(f"    {name:<15} {beta_F[i]:>10.4f} {se_F[i]:>10.4f} {pval_F[i]:>10.4f} {sig}")
print(f"\n    R² = {r2_F:.4f}")

# Key finding: η
eta = beta_F[3]  # J% coefficient
print(f"\n    Key Finding: η (J% → Fan Vote) = {eta:.4f}")
if eta > 0:
    print("    → Dancing better INCREASES fan votes (positive spillover)")
else:
    print("    → Dancing better has NEGATIVE/weak effect on fan votes")

# =============================================================================
# PART 3: PRO DANCER EFFECTS
# =============================================================================
print("\n[3] Analyzing Pro Dancer Effects...")

# 计算每个Pro的平均表现
pro_stats = df_valid.groupby('ballroom_partner').agg({
    'J_pct': ['mean', 'std', 'count'],
    'f_mean': ['mean', 'std'],
    'celebrity_name': 'nunique'
}).round(4)

pro_stats.columns = ['J_mean', 'J_std', 'n_obs', 'F_mean', 'F_std', 'n_celebs']
pro_stats = pro_stats.reset_index()
pro_stats = pro_stats[pro_stats['n_obs'] >= 20]

# 计算Pro的"提升效应" (相对于平均)
overall_J_mean = df_valid['J_pct'].mean()
overall_F_mean = df_valid['f_mean'].mean()

pro_stats['J_lift'] = pro_stats['J_mean'] - overall_J_mean
pro_stats['F_lift'] = (pro_stats['F_mean'] - overall_F_mean) * 100

# 排序
pro_stats_sorted = pro_stats.sort_values('J_lift', ascending=False)

print("\n    Pro Dancer Effects (Top 15 by observation count):")
print("    " + "-" * 75)
print(f"    {'Pro Dancer':<22} {'J_mean':>8} {'J_lift':>8} {'F_mean%':>8} {'F_lift':>8} {'N':>6}")
print("    " + "-" * 75)
for _, row in pro_stats.nlargest(15, 'n_obs').iterrows():
    print(f"    {row['ballroom_partner']:<22} {row['J_mean']:>8.1f} {row['J_lift']:>+8.1f} "
          f"{row['F_mean']*100:>8.2f} {row['F_lift']:>+8.2f} {int(row['n_obs']):>6}")

# Star Makers (高J_lift且高F_lift)
star_makers = pro_stats[(pro_stats['J_lift'] > 2) & (pro_stats['F_lift'] > 0)]
print(f"\n    Identified Star Makers (J_lift > 2 AND F_lift > 0): {len(star_makers)}")
for _, row in star_makers.iterrows():
    print(f"    - {row['ballroom_partner']}: J_lift={row['J_lift']:+.1f}, F_lift={row['F_lift']:+.2f}")

# =============================================================================
# PART 4: CELEBRITY EFFECTS
# =============================================================================
print("\n[4] Analyzing Celebrity Effects...")

# 按Celebrity计算
celeb_stats = df_valid.groupby('celebrity_name').agg({
    'J_pct': ['mean', 'std', 'count'],
    'f_mean': ['mean', 'std'],
    'season': 'first',
    'ballroom_partner': 'first'
}).round(4)

celeb_stats.columns = ['J_mean', 'J_std', 'n_weeks', 'F_mean', 'F_std', 'season', 'partner']
celeb_stats = celeb_stats.reset_index()

# 最极端的Celebrity
print("\n    Extreme Celebrities:")
print("\n    Top 5 by Judge Score:")
for _, row in celeb_stats.nlargest(5, 'J_mean').iterrows():
    print(f"    - {row['celebrity_name']} (S{row['season']}): J={row['J_mean']:.1f}%")

print("\n    Top 5 by Fan Vote:")
for _, row in celeb_stats.nlargest(5, 'F_mean').iterrows():
    print(f"    - {row['celebrity_name']} (S{row['season']}): F={row['F_mean']*100:.2f}%")

print("\n    Bottom 5 by Judge Score:")
for _, row in celeb_stats.nsmallest(5, 'J_mean').iterrows():
    print(f"    - {row['celebrity_name']} (S{row['season']}): J={row['J_mean']:.1f}%")

# =============================================================================
# PART 5: VARIANCE DECOMPOSITION (Using Mixed-Effects Model per Plan)
# =============================================================================
print("\n[5] Variance Decomposition (Mixed-Effects Random Effects)...")

# Plan要求: 使用混合效应模型的随机效应方差来做方差分解
# "Percentage of variance explained by Pro dancer random effects"
# 正确方法: 使用statsmodels的MixedLM或手动计算ICC (Intraclass Correlation Coefficient)

def compute_icc_variance(df, group_col, y_col):
    """
    计算ICC方差分解 (Intraclass Correlation Coefficient)
    这等价于混合效应模型的随机效应方差占比
    ICC = sigma^2_between / (sigma^2_between + sigma^2_within)
    """
    groups = df.groupby(group_col)[y_col]
    
    # 组间方差 (between-group variance)
    group_means = groups.mean()
    grand_mean = df[y_col].mean()
    n_groups = len(group_means)
    
    # 组内方差 (within-group variance) - 使用pooled方差
    within_vars = []
    n_per_group = []
    for name, group_data in groups:
        if len(group_data) > 1:
            within_vars.append(group_data.var() * (len(group_data) - 1))
            n_per_group.append(len(group_data) - 1)
    
    if sum(n_per_group) > 0:
        sigma2_within = sum(within_vars) / sum(n_per_group)
    else:
        sigma2_within = df[y_col].var()
    
    # 组间方差 (考虑组大小不平衡)
    n_total = len(df)
    n_bar = n_total / n_groups  # 平均组大小
    sigma2_between = max(0, group_means.var() - sigma2_within / n_bar)
    
    # ICC = 组间方差 / 总方差
    total_var = sigma2_between + sigma2_within
    if total_var > 0:
        icc = sigma2_between / total_var * 100
    else:
        icc = 0
    
    return icc, sigma2_between, sigma2_within

# 计算Pro Dancer随机效应方差占比 (Judge Score)
pct_J_pro, var_J_pro_between, var_J_pro_within = compute_icc_variance(
    df_valid, 'ballroom_partner', 'J_pct')

# 计算Pro Dancer随机效应方差占比 (Fan Vote)
pct_F_pro, var_F_pro_between, var_F_pro_within = compute_icc_variance(
    df_valid, 'ballroom_partner', 'f_logit')

# 计算Season随机效应方差占比
pct_J_season, _, _ = compute_icc_variance(df_valid, 'season', 'J_pct')
pct_F_season, _, _ = compute_icc_variance(df_valid, 'season', 'f_logit')

# 计算Celebrity随机效应方差占比
pct_J_celeb, _, _ = compute_icc_variance(df_valid, 'celebrity_name', 'J_pct')
pct_F_celeb, _, _ = compute_icc_variance(df_valid, 'celebrity_name', 'f_logit')

# 残差方差 = 总方差减去各随机效应方差
# 注: ICC方法下各因子是独立计算的，可能会出现负残差（当因子高度相关时）
# 如果残差为负，使用层次分解法
pct_J_residual = 100 - pct_J_pro - pct_J_season - pct_J_celeb
pct_F_residual = 100 - pct_F_pro - pct_F_season - pct_F_celeb

# 如果残差为负（因子重叠），调整为0并重新标准化
if pct_J_residual < 0:
    total_J = pct_J_pro + pct_J_season + pct_J_celeb
    pct_J_pro = pct_J_pro / total_J * 100
    pct_J_season = pct_J_season / total_J * 100
    pct_J_celeb = pct_J_celeb / total_J * 100
    pct_J_residual = 0

if pct_F_residual < 0:
    total_F = pct_F_pro + pct_F_season + pct_F_celeb
    pct_F_pro = pct_F_pro / total_F * 100
    pct_F_season = pct_F_season / total_F * 100
    pct_F_celeb = pct_F_celeb / total_F * 100
    pct_F_residual = 0

print(f"\n    Variance Decomposition (ICC/Random Effects Method):")
print("    " + "-" * 50)
print(f"    {'Source':<20} {'Judge (J%)':>12} {'Fan (logit f)':>15}")
print("    " + "-" * 50)
print(f"    {'Pro Dancer (RE)':<20} {pct_J_pro:>11.1f}% {pct_F_pro:>14.1f}%")
print(f"    {'Celebrity (RE)':<20} {pct_J_celeb:>11.1f}% {pct_F_celeb:>14.1f}%")
print(f"    {'Season (RE)':<20} {pct_J_season:>11.1f}% {pct_F_season:>14.1f}%")
print(f"    {'Residual':<20} {pct_J_residual:>11.1f}% {pct_F_residual:>14.1f}%")

# =============================================================================
# PART 6: COEFFICIENT COMPARISON
# =============================================================================
print("\n[6] Comparing Effects: Same Direction?")

comparison = []

# Week effect
week_J = beta_J[1]
week_F = beta_F[1]
same_week = (week_J > 0) == (week_F > 0)
comparison.append(('Week', week_J, week_F, same_week))

# Season effect
season_J = beta_J[3]
season_F = beta_F[4]
same_season = (season_J > 0) == (season_F > 0)
comparison.append(('Season', season_J, season_F, same_season))

# J% → Fan (η)
comparison.append(('J% (skill)', 1.0, eta, eta > 0))

# Correlation between Pro effects on J and F
pro_corr = stats.spearmanr(pro_stats['J_lift'], pro_stats['F_lift'])[0]
comparison.append(('Pro Effect Corr', 1.0, pro_corr, pro_corr > 0))

print("\n    Coefficient Comparison:")
print("    " + "-" * 60)
print(f"    {'Feature':<20} {'Judge β':>10} {'Fan β':>10} {'Same Dir?':>12}")
print("    " + "-" * 60)
for feat, j_coef, f_coef, same in comparison:
    dir_str = "✓ Yes" if same else "✗ No"
    print(f"    {feat:<20} {j_coef:>10.4f} {f_coef:>10.4f} {dir_str:>12}")

# =============================================================================
# PART 7: VISUALIZATIONS
# =============================================================================
print("\n[7] Generating visualizations...")

import os
img_dir = 'cleaned_outputs/phase4_supplement'
os.makedirs(img_dir, exist_ok=True)

# Prepare data
sample = df_valid.sample(min(500, len(df_valid)), random_state=42)
z = np.polyfit(df_valid['J_pct'], df_valid['f_mean'] * 100, 1)
p = np.poly1d(z)
x_line = np.linspace(df_valid['J_pct'].min(), df_valid['J_pct'].max(), 100)
sources = ['Pro\nDancer', 'Season', 'Residual']
j_vars = [pct_J_pro, pct_J_season, pct_J_residual]
f_vars = [pct_F_pro, pct_F_season, pct_F_residual]
season_means = df_valid.groupby('season').agg({
    'J_pct': 'mean',
    'f_mean': lambda x: x.mean() * 100
}).reset_index()

# --- Individual Plots ---

# 7.1 Pro Dancer J_lift vs F_lift (Individual)
fig1, ax1_ind = plt.subplots(figsize=(8, 6))
ax1_ind.scatter(pro_stats['J_lift'], pro_stats['F_lift'], 
            s=pro_stats['n_obs']/2, alpha=0.6, c='steelblue')
for _, row in pro_stats.nlargest(5, 'n_obs').iterrows():
    ax1_ind.annotate(row['ballroom_partner'].split()[0], 
                 (row['J_lift'], row['F_lift']), fontsize=8)
ax1_ind.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1_ind.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax1_ind.set_xlabel('J% Lift (vs average)')
ax1_ind.set_ylabel('Fan Vote Lift (vs average)')
ax1_ind.set_title(f'Pro Dancer Effects (Correlation: {pro_corr:.3f})')
plt.tight_layout()
plt.savefig(f'{img_dir}/pro_dancer_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/pro_dancer_effects.png")

# 7.2 J% vs Fan Vote scatter (Individual)
fig2, ax2_ind = plt.subplots(figsize=(8, 6))
ax2_ind.scatter(sample['J_pct'], sample['f_mean'] * 100, alpha=0.3, s=15)
ax2_ind.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'η={eta:.4f}')
ax2_ind.set_xlabel('Judge Score (J%)')
ax2_ind.set_ylabel('Fan Vote Share (%)')
ax2_ind.set_title('J% → Fan Vote Relationship')
ax2_ind.legend()
plt.tight_layout()
plt.savefig(f'{img_dir}/jpct_vs_fan_vote.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/jpct_vs_fan_vote.png")

# 7.3 Variance Decomposition (Individual)
fig3, ax3_ind = plt.subplots(figsize=(8, 6))
x_var = np.arange(len(sources))
width = 0.35
bars1 = ax3_ind.bar(x_var - width/2, j_vars, width, label='Judge Score', color='steelblue')
bars2 = ax3_ind.bar(x_var + width/2, f_vars, width, label='Fan Vote', color='coral')
ax3_ind.set_ylabel('Variance Explained (%)')
ax3_ind.set_title('Variance Decomposition (ICC/Random Effects)')
ax3_ind.set_xticks(x_var)
ax3_ind.set_xticklabels(sources)
ax3_ind.legend()
ax3_ind.set_ylim(0, 100)
for bar, val in zip(bars1, j_vars):
    if val > 2:
        ax3_ind.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, f_vars):
    if val > 2:
        ax3_ind.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(f'{img_dir}/variance_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/variance_decomposition.png")

# 7.4 Season trends (Individual)
fig4, ax4_ind = plt.subplots(figsize=(8, 6))
ax4_ind.plot(season_means['season'], season_means['J_pct'], 'b-o', label='J%', markersize=4)
ax4_twin_ind = ax4_ind.twinx()
ax4_twin_ind.plot(season_means['season'], season_means['f_mean'], 'r-s', label='F%', markersize=4)
ax4_ind.set_xlabel('Season')
ax4_ind.set_ylabel('Judge Score (%)', color='blue')
ax4_twin_ind.set_ylabel('Fan Vote Share (%)', color='red')
ax4_ind.set_title('Season Trends')
ax4_ind.legend(loc='upper left')
ax4_twin_ind.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{img_dir}/season_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {img_dir}/season_trends.png")

# --- Panel Plot (Combined) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7.1 Pro Dancer J_lift vs F_lift
ax1 = axes[0, 0]
ax1.scatter(pro_stats['J_lift'], pro_stats['F_lift'], 
            s=pro_stats['n_obs']/2, alpha=0.6, c='steelblue')
for _, row in pro_stats.nlargest(5, 'n_obs').iterrows():
    ax1.annotate(row['ballroom_partner'].split()[0], 
                 (row['J_lift'], row['F_lift']), fontsize=8)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('J% Lift (vs average)')
ax1.set_ylabel('Fan Vote Lift (vs average)')
ax1.set_title(f'Pro Dancer Effects (Correlation: {pro_corr:.3f})')

# 7.2 J% vs Fan Vote scatter
ax2 = axes[0, 1]
ax2.scatter(sample['J_pct'], sample['f_mean'] * 100, alpha=0.3, s=15)
ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'η={eta:.4f}')
ax2.set_xlabel('Judge Score (J%)')
ax2.set_ylabel('Fan Vote Share (%)')
ax2.set_title('J% → Fan Vote Relationship')
ax2.legend()

# 7.3 Variance Decomposition
ax3 = axes[1, 0]
x = np.arange(len(sources))
width = 0.35
bars1 = ax3.bar(x - width/2, j_vars, width, label='Judge Score', color='steelblue')
bars2 = ax3.bar(x + width/2, f_vars, width, label='Fan Vote', color='coral')
ax3.set_ylabel('Variance Explained (%)')
ax3.set_title('Variance Decomposition (ICC/Random Effects)')
ax3.set_xticks(x)
ax3.set_xticklabels(sources)
ax3.legend()
ax3.set_ylim(0, 100)
for bar, val in zip(bars1, j_vars):
    if val > 2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, f_vars):
    if val > 2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

# 7.4 Season trends
ax4 = axes[1, 1]
ax4.plot(season_means['season'], season_means['J_pct'], 'b-o', label='J%', markersize=4)
ax4_twin = ax4.twinx()
ax4_twin.plot(season_means['season'], season_means['f_mean'], 'r-s', label='F%', markersize=4)
ax4.set_xlabel('Season')
ax4.set_ylabel('Judge Score (%)', color='blue')
ax4_twin.set_ylabel('Fan Vote Share (%)', color='red')
ax4.set_title('Season Trends')
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')

plt.tight_layout()
plt.savefig('cleaned_outputs/celebrity_effects_model.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{img_dir}/panel_all.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: celebrity_effects_model.png")
print(f"    Saved: {img_dir}/panel_all.png")

# =============================================================================
# PART 8: SAVE RESULTS
# =============================================================================
print("\n[8] Saving results...")

# 保存Pro统计
pro_stats.to_csv('cleaned_outputs/pro_dancer_effects.csv', index=False)
print(f"    Saved: pro_dancer_effects.csv ({len(pro_stats)} rows)")

# 保存系数比较
comparison_df = pd.DataFrame(comparison, columns=['Feature', 'Judge_Coef', 'Fan_Coef', 'Same_Direction'])
comparison_df.to_csv('cleaned_outputs/coefficient_comparison.csv', index=False)
print(f"    Saved: coefficient_comparison.csv")

# 保存方差分解 (按照Plan: Pro Dancer随机效应方差占比, 加上Celebrity)
variance_decomp = pd.DataFrame([
    {'Source': 'Pro Dancer (RE)', 'Judge_Var_Pct': pct_J_pro, 'Fan_Var_Pct': pct_F_pro},
    {'Source': 'Celebrity (RE)', 'Judge_Var_Pct': pct_J_celeb, 'Fan_Var_Pct': pct_F_celeb},
    {'Source': 'Season (RE)', 'Judge_Var_Pct': pct_J_season, 'Fan_Var_Pct': pct_F_season},
    {'Source': 'Residual', 'Judge_Var_Pct': pct_J_residual, 'Fan_Var_Pct': pct_F_residual}
])
variance_decomp.to_csv('cleaned_outputs/variance_decomposition.csv', index=False)
print(f"    Saved: variance_decomposition.csv")

# 保存回归结果
regression_results = {
    'judge_model': {
        'coefficients': dict(zip(['Intercept', 'Week', 'Week²', 'Season'], beta_J.tolist())),
        'std_errors': dict(zip(['Intercept', 'Week', 'Week²', 'Season'], se_J.tolist())),
        'r_squared': r2_J
    },
    'fan_model': {
        'coefficients': dict(zip(['Intercept', 'Week', 'Week²', 'J%', 'Season'], beta_F.tolist())),
        'std_errors': dict(zip(['Intercept', 'Week', 'Week²', 'J%', 'Season'], se_F.tolist())),
        'r_squared': r2_F,
        'eta': eta
    }
}

with open('cleaned_outputs/regression_results.json', 'w') as f:
    json.dump(regression_results, f, indent=2)
print(f"    Saved: regression_results.json")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 4 SUPPLEMENT: CELEBRITY EFFECTS MODEL SUMMARY")
print("=" * 70)

same_dir_count = sum(1 for c in comparison if c[3])
total_count = len(comparison)

print(f"""
MODEL RESULTS:
==============

1. JUDGE SCORE MODEL (J%):
   - R² = {r2_J:.4f}
   - Week effect: β = {beta_J[1]:.4f} (scores {"increase" if beta_J[1] > 0 else "decrease"} over weeks)
   - Season trend: β = {beta_J[3]:.4f}

2. FAN VOTE MODEL (logit f):
   - R² = {r2_F:.4f}
   - η (J% → Fan): {eta:.4f} ({"positive" if eta > 0 else "negative"} spillover)
   - Interpretation: Better dancing {"increases" if eta > 0 else "does not increase"} fan votes

3. VARIANCE DECOMPOSITION:
   - Pro Dancer explains: {pct_J_pro:.1f}% (Judge), {pct_F_pro:.1f}% (Fan)
   - Celebrity explains: {pct_J_celeb:.1f}% (Judge), {pct_F_celeb:.1f}% (Fan)
   - Season explains: {pct_J_season:.1f}% (Judge), {pct_F_season:.1f}% (Fan)

4. SAME DIRECTION INFLUENCE?
   - {same_dir_count}/{total_count} features affect J and F in the same direction
   - Pro Effect Correlation: {pro_corr:.3f}
   - Conclusion: {"Features mostly affect judges and fans similarly" if same_dir_count > total_count/2 else "Judge and fan preferences diverge"}

5. STAR MAKERS IDENTIFIED:
   - {len(star_makers)} Pro Dancers with both high J_lift and F_lift

FILES SAVED:
============
• pro_dancer_effects.csv
• coefficient_comparison.csv  
• variance_decomposition.csv
• regression_results.json
• celebrity_effects_model.png
""")

print("=" * 70)
print("PHASE 4 SUPPLEMENT COMPLETE!")
print("=" * 70)
