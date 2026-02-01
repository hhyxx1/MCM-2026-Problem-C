"""
Phase 2: Bayesian Inverse Inference for Fan Vote Estimation
==========================================================

Goal: Estimate fan vote share f(i,w) for each contestant i in week w

Constraints:
- f(i,w) >= 0 for all i
- sum_i f(i,w) = 1 for each week w
- Elimination constraint: eliminated contestants must be in Bottom-k by Combined Score

Method: MCMC sampling with inequality constraints using Hit-and-Run algorithm
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*70)
print("PHASE 2: BAYESIAN INVERSE INFERENCE")
print("="*70)

df_panel = pd.read_csv('/home/hyx/文档/MCM/cleaned_outputs/clean_weekly_panel.csv')
df_elim = pd.read_csv('/home/hyx/文档/MCM/cleaned_outputs/elimination_summary.csv')
df_season = pd.read_csv('/home/hyx/文档/MCM/cleaned_outputs/season_summary.csv')

print(f"\n[1] Data loaded:")
print(f"    Panel observations: {len(df_panel)}")
print(f"    Seasons: 1-{df_panel['season'].max()}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_week_contestants(df_panel, season, week):
    """Get contestants competing in a specific week"""
    data = df_panel[(df_panel['season'] == season) & (df_panel['week'] == week)]
    return data[['contestant_id', 'celebrity_name', 'J_pct', 'placement']].copy()

def get_elimination_info(df_panel, season, week):
    """
    Get who was eliminated in a specific week.
    Returns list of contestant_ids eliminated, or empty if no elimination.
    """
    season_data = df_panel[df_panel['season'] == season]
    
    # Find contestants who competed this week but not the next
    this_week = set(season_data[season_data['week'] == week]['contestant_id'])
    next_week = set(season_data[season_data['week'] == week + 1]['contestant_id'])
    
    # Also check elimination_week column
    elim_this_week = season_data[season_data['elimination_week'] == str(week)]['contestant_id'].tolist()
    
    # Combine both methods
    eliminated = list(this_week - next_week)
    
    # Filter out those who withdrew (not eliminated by votes)
    withdrew = season_data[season_data['elimination_week'] == 'withdrew']['contestant_id'].tolist()
    eliminated = [e for e in eliminated if e not in withdrew]
    
    return eliminated

def combined_score_rank(j_pct, f_vote, alpha=0.5):
    """
    Rank-based aggregation: Average of judge rank and fan rank
    Lower combined rank = better (safer from elimination)
    """
    n = len(j_pct)
    j_rank = stats.rankdata(-j_pct)  # Higher J% = better rank (1)
    f_rank = stats.rankdata(-f_vote)  # Higher vote = better rank (1)
    return alpha * j_rank + (1 - alpha) * f_rank

def combined_score_percentage(j_pct, f_vote, alpha=0.5):
    """
    Percentage-based aggregation: Weighted average of J% and F%
    Higher combined score = better (safer from elimination)
    """
    # Normalize f_vote to percentage (0-100)
    f_pct = f_vote * 100
    return alpha * j_pct + (1 - alpha) * f_pct

def check_elimination_constraint(f_vote, j_pct, eliminated_idx, method='rank'):
    """
    Check if the fan vote allocation satisfies elimination constraints.
    
    For rank method: eliminated contestants should have worst combined rank
    For percentage method: eliminated contestants should have lowest combined score
    """
    n = len(j_pct)
    k = len(eliminated_idx)
    
    if k == 0:
        return True
    
    if method == 'rank':
        combined = combined_score_rank(j_pct, f_vote)
        # Higher rank number = worse = should be eliminated
        threshold = sorted(combined)[-k]  # k-th worst
        for idx in eliminated_idx:
            if combined[idx] < threshold:
                return False
    else:  # percentage
        combined = combined_score_percentage(j_pct, f_vote)
        # Lower score = worse = should be eliminated
        threshold = sorted(combined)[k-1]  # k-th worst
        for idx in eliminated_idx:
            if combined[idx] > threshold:
                return False
    
    return True

# ============================================================================
# MCMC SAMPLER: Hit-and-Run on Simplex with Constraints
# ============================================================================

def sample_dirichlet_with_constraints(j_pct, eliminated_idx, n_samples=5000, 
                                       burn_in=1000, method='rank', 
                                       prior_alpha=1.0):
    """
    Sample fan vote distribution f using MCMC with elimination constraints.
    
    Uses Metropolis-Hastings with Dirichlet proposals, rejecting samples
    that violate elimination constraints.
    
    Args:
        j_pct: array of judge scores for contestants this week
        eliminated_idx: indices of eliminated contestants (0-indexed)
        n_samples: number of posterior samples
        burn_in: burn-in period
        method: 'rank' or 'percentage' aggregation
        prior_alpha: Dirichlet prior concentration (1.0 = uniform)
    
    Returns:
        samples: (n_samples, n_contestants) array of f allocations
        acceptance_rate: proportion of accepted proposals
    """
    n = len(j_pct)
    k = len(eliminated_idx)
    
    # Initialize with uniform
    f_current = np.ones(n) / n
    
    # If no elimination, just sample from prior
    if k == 0:
        samples = np.random.dirichlet(np.ones(n) * prior_alpha, size=n_samples)
        return samples, 1.0
    
    # Find a valid starting point
    max_init_attempts = 1000
    for _ in range(max_init_attempts):
        f_current = np.random.dirichlet(np.ones(n) * prior_alpha)
        if check_elimination_constraint(f_current, j_pct, eliminated_idx, method):
            break
    else:
        # If can't find valid start, use heuristic: give eliminated low votes
        f_current = np.ones(n)
        for idx in eliminated_idx:
            f_current[idx] = 0.01
        f_current = f_current / f_current.sum()
    
    samples = []
    accepted = 0
    total = 0
    
    # MCMC sampling
    for i in range(n_samples + burn_in):
        # Propose new f using Dirichlet centered on current
        # Concentration controls step size
        concentration = 50  # Higher = smaller steps
        alpha_proposal = f_current * concentration + 1
        f_proposal = np.random.dirichlet(alpha_proposal)
        
        # Check constraint
        if check_elimination_constraint(f_proposal, j_pct, eliminated_idx, method):
            # Compute acceptance ratio (symmetric proposal for simplicity)
            # With Dirichlet prior, acceptance is just based on constraint
            f_current = f_proposal
            accepted += 1
        
        total += 1
        
        if i >= burn_in:
            samples.append(f_current.copy())
    
    return np.array(samples), accepted / total

def estimate_fan_votes_for_week(df_panel, season, week, method='rank', n_samples=3000):
    """
    Estimate fan vote distribution for a specific season-week.
    
    Returns:
        dict with contestant_id -> {mean, median, ci_low, ci_high, std}
        Also includes week-level P_w (posterior consistency probability)
    """
    # Get contestants and their judge scores
    contestants = get_week_contestants(df_panel, season, week)
    if len(contestants) == 0:
        return None
    
    contestant_ids = contestants['contestant_id'].tolist()
    j_pct = contestants['J_pct'].values
    
    # Get elimination info
    eliminated = get_elimination_info(df_panel, season, week)
    eliminated_idx = [contestant_ids.index(e) for e in eliminated if e in contestant_ids]
    k = len(eliminated_idx)
    
    # Sample posterior
    samples, acc_rate = sample_dirichlet_with_constraints(
        j_pct, eliminated_idx, n_samples=n_samples, method=method
    )
    
    # =========================================================================
    # COMPUTE P_w: Posterior Consistency (per Plan Patch 3 requirement)
    # P_w = Prob(E_w is Bottom-k | posterior), estimated via posterior sampling
    # =========================================================================
    n_correct = 0
    if k > 0:
        for sample_f in samples:
            # Calculate combined score for this sample
            if method == 'rank':
                combined = combined_score_rank(j_pct, sample_f)
                # Higher rank = worse; get indices of k highest combined ranks
                predicted_bottom_k = set(np.argsort(combined)[-k:])
            else:
                combined = combined_score_percentage(j_pct, sample_f)
                # Lower score = worse; get indices of k lowest combined scores
                predicted_bottom_k = set(np.argsort(combined)[:k])
            
            # Check if predicted bottom-k matches actual eliminated
            if predicted_bottom_k == set(eliminated_idx):
                n_correct += 1
        
        P_w = n_correct / len(samples)
    else:
        P_w = 1.0  # No elimination this week, trivially consistent
    
    # Compute statistics
    results = {}
    for i, cid in enumerate(contestant_ids):
        f_samples = samples[:, i]
        results[cid] = {
            'contestant_id': cid,
            'celebrity_name': contestants.iloc[i]['celebrity_name'],
            'season': season,
            'week': week,
            'J_pct': j_pct[i],
            'f_mean': np.mean(f_samples),
            'f_median': np.median(f_samples),
            'f_ci_low': np.percentile(f_samples, 2.5),
            'f_ci_high': np.percentile(f_samples, 97.5),
            'f_std': np.std(f_samples),
            'ci_width': np.percentile(f_samples, 97.5) - np.percentile(f_samples, 2.5),
            'was_eliminated': cid in eliminated,
            'n_contestants': len(contestant_ids),
            'acceptance_rate': acc_rate,
            'P_w': P_w  # Posterior consistency for this week
        }
    
    return results

# ============================================================================
# RUN INFERENCE FOR ALL SEASONS
# ============================================================================
print("\n[2] Running Bayesian inference for all seasons...")
print("    (This may take a few minutes)")

all_estimates = []
season_stats = []

# Process each season
for season in range(1, 35):
    season_data = df_panel[df_panel['season'] == season]
    if len(season_data) == 0:
        continue
    
    weeks = sorted(season_data['week'].unique())
    season_estimates = []
    
    print(f"\n    Season {season}: {len(weeks)} weeks, ", end="")
    
    for week in weeks:
        results = estimate_fan_votes_for_week(df_panel, season, week, 
                                               method='rank', n_samples=2000)
        if results:
            for cid, est in results.items():
                all_estimates.append(est)
                season_estimates.append(est)
    
    # Season summary
    if season_estimates:
        df_season_est = pd.DataFrame(season_estimates)
        avg_ci = df_season_est['ci_width'].mean()
        avg_acc = df_season_est['acceptance_rate'].mean()
        print(f"avg CI width: {avg_ci:.4f}, acceptance: {avg_acc:.2%}")
        
        season_stats.append({
            'season': season,
            'n_weeks': len(weeks),
            'n_estimates': len(season_estimates),
            'avg_ci_width': avg_ci,
            'avg_acceptance': avg_acc
        })

# Create DataFrame
df_estimates = pd.DataFrame(all_estimates)
df_season_stats = pd.DataFrame(season_stats)

print(f"\n[3] Inference complete!")
print(f"    Total estimates: {len(df_estimates)}")
print(f"    Seasons processed: {len(df_season_stats)}")

# ============================================================================
# COMPUTE POSTERIOR STATISTICS
# ============================================================================
print("\n[4] Computing posterior statistics...")

# Credible Interval Width (Certainty measure)
print("\n    Certainty (CI Width) Summary:")
print(f"    Mean CI Width: {df_estimates['ci_width'].mean():.4f}")
print(f"    Std CI Width: {df_estimates['ci_width'].std():.4f}")
print(f"    Min CI Width: {df_estimates['ci_width'].min():.4f}")
print(f"    Max CI Width: {df_estimates['ci_width'].max():.4f}")

# Most uncertain estimates
print("\n    Top 10 Most Uncertain (Widest CI):")
uncertain = df_estimates.nlargest(10, 'ci_width')[
    ['season', 'week', 'celebrity_name', 'f_mean', 'ci_width', 'n_contestants']
]
print(uncertain.to_string(index=False))

# Most certain estimates
print("\n    Top 10 Most Certain (Narrowest CI):")
certain = df_estimates.nsmallest(10, 'ci_width')[
    ['season', 'week', 'celebrity_name', 'f_mean', 'ci_width', 'n_contestants']
]
print(certain.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[5] Saving results...")

output_dir = '/home/hyx/文档/MCM/cleaned_outputs'

df_estimates.to_csv(f'{output_dir}/fan_vote_estimates.csv', index=False)
print(f"    Saved: fan_vote_estimates.csv ({len(df_estimates)} rows)")

df_season_stats.to_csv(f'{output_dir}/inference_season_stats.csv', index=False)
print(f"    Saved: inference_season_stats.csv ({len(df_season_stats)} rows)")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n[6] Generating visualizations...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: CI Width Distribution
ax1 = axes[0, 0]
ax1.hist(df_estimates['ci_width'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(df_estimates['ci_width'].mean(), color='red', linestyle='--', 
            label=f'Mean={df_estimates["ci_width"].mean():.3f}')
ax1.set_xlabel('95% Credible Interval Width')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Estimation Uncertainty')
ax1.legend()

# Plot 2: CI Width by Week
ax2 = axes[0, 1]
week_ci = df_estimates.groupby('week')['ci_width'].agg(['mean', 'std']).reset_index()
ax2.errorbar(week_ci['week'], week_ci['mean'], yerr=week_ci['std'], 
             fmt='o-', capsize=3, color='steelblue')
ax2.set_xlabel('Week')
ax2.set_ylabel('Mean CI Width')
ax2.set_title('Estimation Uncertainty by Week\n(Later weeks have fewer contestants)')

# Plot 3: f_mean vs J_pct
ax3 = axes[1, 0]
ax3.scatter(df_estimates['J_pct'], df_estimates['f_mean'], 
            c=df_estimates['was_eliminated'].astype(int), 
            cmap='coolwarm', alpha=0.5, s=20)
ax3.set_xlabel('Judge Score (%)')
ax3.set_ylabel('Estimated Fan Vote Share')
ax3.set_title('Fan Vote Share vs Judge Score\n(Red=Eliminated, Blue=Survived)')

# Add trend line
z = np.polyfit(df_estimates['J_pct'], df_estimates['f_mean'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_estimates['J_pct'].min(), df_estimates['J_pct'].max(), 100)
ax3.plot(x_line, p(x_line), 'g--', linewidth=2, label=f'Trend')
ax3.legend()

# Plot 4: Season-level Statistics
ax4 = axes[1, 1]
ax4.bar(df_season_stats['season'], df_season_stats['avg_ci_width'], 
        color='steelblue', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Season')
ax4.set_ylabel('Average CI Width')
ax4.set_title('Estimation Uncertainty by Season')

plt.tight_layout()
plt.savefig(f'{output_dir}/bayesian_inference_summary.png', dpi=150, bbox_inches='tight')
print(f"    Saved: bayesian_inference_summary.png")

plt.close()

# ============================================================================
# SAMPLE OUTPUT: Show estimates for a specific season
# ============================================================================
print("\n" + "="*70)
print("SAMPLE OUTPUT: Season 27 (Bobby Bones season)")
print("="*70)

s27 = df_estimates[df_estimates['season'] == 27].sort_values(['week', 'f_mean'], ascending=[True, False])
for week in sorted(s27['week'].unique()):
    week_data = s27[s27['week'] == week]
    print(f"\n  Week {week}:")
    for _, row in week_data.iterrows():
        elim_mark = " [ELIMINATED]" if row['was_eliminated'] else ""
        print(f"    {row['celebrity_name']:20s}: f={row['f_mean']:.4f} "
              f"[{row['f_ci_low']:.4f}, {row['f_ci_high']:.4f}] "
              f"J%={row['J_pct']:.1f}{elim_mark}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("BAYESIAN INFERENCE SUMMARY")
print("="*70)

print(f"""
✓ COMPLETED: Fan vote share estimation for all season-weeks

METHOD:
  • MCMC sampling with Dirichlet proposals
  • Elimination constraints enforced (Bottom-k must be eliminated)
  • Rank-based aggregation: Combined = 0.5 * J_rank + 0.5 * F_rank

OUTPUT:
  • f(i,w): Estimated fan vote share for contestant i in week w
  • 95% Credible Intervals for each estimate
  • Uncertainty measured by CI width

KEY STATISTICS:
  • Total estimates: {len(df_estimates)}
  • Average CI width: {df_estimates['ci_width'].mean():.4f}
  • Average acceptance rate: {df_estimates['acceptance_rate'].mean():.2%}

INTERPRETATION:
  • Wider CI = More uncertainty about true fan vote share
  • CI width varies by week (fewer contestants = narrower CI)
  • Eliminated contestants constrained to have low combined score

FILES SAVED:
  • fan_vote_estimates.csv - All estimates with CIs
  • inference_season_stats.csv - Season-level summary
  • bayesian_inference_summary.png - Visualization
""")

print("="*70)
print("PHASE 2.1 BAYESIAN INFERENCE COMPLETE!")
print("="*70)
