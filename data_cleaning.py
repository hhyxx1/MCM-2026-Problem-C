"""
Dancing with the Stars Data Cleaning Script
==========================================
Phase 1: Data Archeology & Global Scan

Tasks:
1. Standardization: Unify judge scores from all seasons into percentage 𝒥(i,t)
   - Handle 30-point system (3 judges × 10 max) and 40-point system (4 judges × 10 max)
2. Withdrawal Handling: Exclude N/A and 0-point data
3. Convert wide table to long format panel data (i, t) for mixed-effects models
4. Extract celebrity covariates for analysis

数学符号对应 (Symbol Mapping):
    J_pct    -> 𝒥(i,t)    评委得分百分比 (Judge percentage)
    i        -> 选手索引 (contestant index)
    t        -> 周次 (week/time)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD RAW DATA
# ============================================================================
print("="*70)
print("DANCING WITH THE STARS - DATA CLEANING")
print("="*70)

df_raw = pd.read_csv('/home/hyx/文档/MCM/2026_MCM_Problem_C_Data.csv')
print(f"\n[1] Loaded raw data: {df_raw.shape[0]} contestants, {df_raw.shape[1]} columns")
print(f"    Seasons covered: {df_raw['season'].min()} to {df_raw['season'].max()}")

# ============================================================================
# IDENTIFY SCORE COLUMNS AND WEEKS
# ============================================================================
# Find all judge score columns (week{N}_judge{M}_score)
score_cols = [col for col in df_raw.columns if 'judge' in col and 'score' in col]
weeks = sorted(list(set([int(col.split('_')[0].replace('week', '')) for col in score_cols])))
print(f"    Weeks available: {min(weeks)} to {max(weeks)}")
print(f"    Score columns: {len(score_cols)}")

# ============================================================================
# FUNCTION: Calculate Judge Percentage (J%)
# ============================================================================
def calculate_judge_percentage(row, week):
    """
    Calculate normalized judge score percentage 𝒥(i,t) for a given week.
    数学定义: 𝒥 = (实际得分 / 最高可能得分) × 100

    Handles:
    - 3-judge system (judge4 is N/A): max = 30
    - 4-judge system: max = 40
    - N/A values: treated as missing
    - 0 values after elimination: excluded
    - Multi-dance weeks: scores may exceed 10 per judge (accumulated/averaged)
    
    Returns:
    - J% (0-100 scale) or np.nan if no valid scores
    """
    judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
    scores = []
    
    for col in judge_cols:
        if col in row.index:
            val = row[col]
            # Handle N/A strings
            if pd.isna(val) or val == 'N/A' or val == 'n/a':
                continue
            try:
                score = float(val)
                # Skip 0 scores (indicates eliminated or withdrew)
                if score > 0:
                    scores.append(score)
            except (ValueError, TypeError):
                continue
    
    if len(scores) == 0:
        return np.nan
    
    # Calculate total score and determine max possible
    total_score = sum(scores)
    
    # Detect multi-dance weeks: if any individual judge score > 10, it's a multi-dance week
    # In multi-dance weeks, scores are averaged or accumulated
    # We normalize by finding the implied max (highest score indicates the ceiling)
    max_single_score = max(scores)
    
    if max_single_score > 10:
        # Multi-dance week: estimate number of dances from the score ceiling
        # Typical patterns: 2 dances (max ~20), 3 dances (max ~30)
        # Use the maximum score to infer the ceiling
        implied_max_per_judge = 10 * np.ceil(max_single_score / 10)
        max_score = len(scores) * implied_max_per_judge
    else:
        # Standard single-dance week: max = num_judges * 10
        max_score = len(scores) * 10
    
    return (total_score / max_score) * 100


# ============================================================================
# STEP 1: CREATE CLEAN JUDGE SCORES (Wide Format)
# ============================================================================
print("\n[2] Computing Judge Score Percentages (J%)...")

# Create a new dataframe with J% for each week
df_clean = df_raw[['celebrity_name', 'ballroom_partner', 'celebrity_industry', 
                   'celebrity_homestate', 'celebrity_homecountry/region',
                   'celebrity_age_during_season', 'season', 'results', 'placement']].copy()

# Calculate J% for each week
for week in weeks:
    df_clean[f'week{week}_J_pct'] = df_raw.apply(lambda row: calculate_judge_percentage(row, week), axis=1)

print(f"    Created J% columns for weeks 1-{max(weeks)}")

# ============================================================================
# STEP 2: EXTRACT ELIMINATION INFORMATION
# ============================================================================
print("\n[3] Extracting elimination information...")

def parse_elimination_week(results_str):
    """
    Parse the results column to extract elimination week.
    
    Returns:
    - Elimination week number, or None for finalists/withdrew
    """
    if pd.isna(results_str):
        return None
    
    results_str = str(results_str).lower()
    
    # Check for specific patterns
    if 'withdrew' in results_str:
        return 'withdrew'
    if '1st place' in results_str or '2nd place' in results_str or '3rd place' in results_str:
        return 'finalist'
    if 'eliminated week' in results_str:
        try:
            week = int(results_str.replace('eliminated week', '').strip())
            return week
        except:
            return None
    return None

df_clean['elimination_week'] = df_raw['results'].apply(parse_elimination_week)

# Count eliminations by week
elim_counts = df_clean[df_clean['elimination_week'].apply(lambda x: isinstance(x, int))]['elimination_week'].value_counts().sort_index()
print(f"    Eliminated contestants by week:")
for week, count in elim_counts.items():
    print(f"      Week {week}: {count} eliminations")

# ============================================================================
# STEP 3: CONVERT TO LONG FORMAT (Panel Data)
# ============================================================================
print("\n[4] Converting to long format panel data (i, w)...")

# Melt the wide format to long format
j_pct_cols = [col for col in df_clean.columns if '_J_pct' in col]

# Create unique contestant ID
df_clean['contestant_id'] = df_clean['celebrity_name'] + '_S' + df_clean['season'].astype(str)

# Prepare for melting
id_vars = ['contestant_id', 'celebrity_name', 'ballroom_partner', 'celebrity_industry',
           'celebrity_homestate', 'celebrity_homecountry/region', 'celebrity_age_during_season',
           'season', 'results', 'placement', 'elimination_week']

# Melt the J% columns
df_long = pd.melt(df_clean, 
                  id_vars=id_vars,
                  value_vars=j_pct_cols,
                  var_name='week_col',
                  value_name='J_pct')

# Extract week number
df_long['week'] = df_long['week_col'].str.extract(r'week(\d+)').astype(int)
df_long = df_long.drop('week_col', axis=1)

# Remove rows with NaN J% (contestant didn't compete that week)
df_long = df_long.dropna(subset=['J_pct'])

# Sort for readability
df_long = df_long.sort_values(['season', 'week', 'contestant_id']).reset_index(drop=True)

print(f"    Long format data: {len(df_long)} observations")
print(f"    Unique contestants: {df_long['contestant_id'].nunique()}")
print(f"    Seasons × Weeks combinations: {df_long.groupby(['season', 'week']).ngroups}")

# ============================================================================
# STEP 4: STANDARDIZE CELEBRITY COVARIATES
# ============================================================================
print("\n[5] Standardizing celebrity covariates...")

# Industry one-hot encoding categories
print("    Industry categories found:")
industry_counts = df_clean['celebrity_industry'].value_counts()
for industry, count in industry_counts.items():
    print(f"      {industry}: {count}")

# Standardize industry categories (group similar ones)
industry_mapping = {
    'Actor/Actress': 'Actor',
    'Singer/Rapper': 'Musician',
    'Athlete': 'Athlete',
    'Model': 'Model',
    'TV Personality': 'TV_Personality',
    'News Anchor': 'TV_Personality',
    'Sports Broadcaster': 'TV_Personality',
    'Racing Driver': 'Athlete',
    'Beauty Pagent': 'Model',
    'Politician': 'Politician',
    'Comedian': 'Comedian',
    'Musician': 'Musician',
    'Reality Star': 'Reality_Star',
    'Social Media Star': 'Reality_Star',
    'YouTube Star': 'Reality_Star',
    'Disney Star': 'Actor',
    'Figure Skater': 'Athlete',
    'Dancer': 'Dancer',
    'WWE': 'Athlete',
    'Olympian': 'Athlete'
}

df_long['industry_std'] = df_long['celebrity_industry'].map(
    lambda x: industry_mapping.get(x, 'Other') if pd.notna(x) else 'Unknown'
)

# Region: US vs Non-US
df_long['is_US'] = df_long['celebrity_homecountry/region'].apply(
    lambda x: 1 if x == 'United States' else 0 if pd.notna(x) else np.nan
)

# Age binning
df_long['age_bin'] = pd.cut(df_long['celebrity_age_during_season'], 
                            bins=[0, 25, 35, 45, 55, 100],
                            labels=['18-25', '26-35', '36-45', '46-55', '55+'])

print(f"    Standardized industry categories: {df_long['industry_std'].nunique()}")

# ============================================================================
# STEP 5: CREATE SEASON-LEVEL SUMMARY
# ============================================================================
print("\n[6] Creating season-level summary...")

# Detect scoring system per season
def detect_scoring_system(season_data):
    """Detect if season uses 3 or 4 judge system"""
    # Check if judge4 scores exist and are not N/A
    judge4_cols = [col for col in df_raw.columns if 'judge4' in col]
    season_rows = df_raw[df_raw['season'] == season_data]
    
    for col in judge4_cols:
        valid_scores = season_rows[col].apply(lambda x: x not in ['N/A', 'n/a', np.nan] and pd.notna(x) and x != 0)
        if valid_scores.any():
            return 4
    return 3

season_summary = []
for season in sorted(df_clean['season'].unique()):
    season_data = df_clean[df_clean['season'] == season]
    long_data = df_long[df_long['season'] == season]
    
    num_judges = detect_scoring_system(season)
    num_contestants = len(season_data)
    num_weeks = long_data['week'].max() if len(long_data) > 0 else 0
    
    # Get elimination info
    elim_data = season_data[season_data['elimination_week'].apply(lambda x: isinstance(x, int))]
    
    season_summary.append({
        'season': season,
        'num_contestants': num_contestants,
        'num_weeks': num_weeks,
        'num_judges': num_judges,
        'max_score_per_week': num_judges * 10,
        'num_eliminations': len(elim_data),
        'finalists': len(season_data[season_data['elimination_week'] == 'finalist']),
        'withdrawals': len(season_data[season_data['elimination_week'] == 'withdrew'])
    })

df_season_summary = pd.DataFrame(season_summary)
print(df_season_summary.to_string(index=False))

# ============================================================================
# STEP 6: CREATE ELIMINATION TIMELINE
# ============================================================================
print("\n[7] Creating elimination timeline...")

# For each season/week, identify who was eliminated
elimination_timeline = []

for season in sorted(df_clean['season'].unique()):
    season_data = df_clean[df_clean['season'] == season]
    
    for _, row in season_data.iterrows():
        if isinstance(row['elimination_week'], int):
            elimination_timeline.append({
                'season': season,
                'week': row['elimination_week'],
                'contestant_id': row['contestant_id'],
                'celebrity_name': row['celebrity_name'],
                'placement': row['placement'],
                'J_pct_final': df_long[(df_long['contestant_id'] == row['contestant_id']) & 
                                       (df_long['week'] == row['elimination_week'])]['J_pct'].values[0] if len(df_long[(df_long['contestant_id'] == row['contestant_id']) & (df_long['week'] == row['elimination_week'])]) > 0 else np.nan
            })
        elif row['elimination_week'] == 'finalist':
            # Finalists - find their last week
            last_week = df_long[df_long['contestant_id'] == row['contestant_id']]['week'].max()
            elimination_timeline.append({
                'season': season,
                'week': last_week,
                'contestant_id': row['contestant_id'],
                'celebrity_name': row['celebrity_name'],
                'placement': row['placement'],
                'J_pct_final': df_long[(df_long['contestant_id'] == row['contestant_id']) & 
                                       (df_long['week'] == last_week)]['J_pct'].values[0] if len(df_long[(df_long['contestant_id'] == row['contestant_id']) & (df_long['week'] == last_week)]) > 0 else np.nan
            })

df_elimination = pd.DataFrame(elimination_timeline)
df_elimination = df_elimination.sort_values(['season', 'week', 'placement']).reset_index(drop=True)

# ============================================================================
# SAVE CLEANED DATA
# ============================================================================
print("\n[8] Saving cleaned data...")

import os
output_dir = '/home/hyx/文档/MCM/cleaned_outputs'
os.makedirs(output_dir, exist_ok=True)

# Save long format panel data
df_long.to_csv(f'{output_dir}/clean_weekly_panel.csv', index=False)
print(f"    Saved: clean_weekly_panel.csv ({len(df_long)} rows)")

# Save season summary
df_season_summary.to_csv(f'{output_dir}/season_summary.csv', index=False)
print(f"    Saved: season_summary.csv ({len(df_season_summary)} rows)")

# Save elimination timeline
df_elimination.to_csv(f'{output_dir}/elimination_summary.csv', index=False)
print(f"    Saved: elimination_summary.csv ({len(df_elimination)} rows)")

# Save wide format with J%
df_clean.to_csv(f'{output_dir}/clean_judge_scores_wide.csv', index=False)
print(f"    Saved: clean_judge_scores_wide.csv ({len(df_clean)} rows)")

# ============================================================================
# DATA QUALITY REPORT
# ============================================================================
print("\n" + "="*70)
print("DATA QUALITY REPORT")
print("="*70)

print(f"\n[A] Coverage Summary:")
print(f"    Total contestants: {len(df_clean)}")
print(f"    Total season-week observations: {len(df_long)}")
print(f"    Seasons with 3 judges: {len(df_season_summary[df_season_summary['num_judges'] == 3])}")
print(f"    Seasons with 4 judges: {len(df_season_summary[df_season_summary['num_judges'] == 4])}")

print(f"\n[B] J% Score Statistics:")
print(f"    Mean J%: {df_long['J_pct'].mean():.2f}%")
print(f"    Std J%: {df_long['J_pct'].std():.2f}%")
print(f"    Min J%: {df_long['J_pct'].min():.2f}%")
print(f"    Max J%: {df_long['J_pct'].max():.2f}%")

print(f"\n[C] Missing Data:")
# Check for any issues
missing_industry = df_long['celebrity_industry'].isna().sum()
missing_region = df_long['celebrity_homecountry/region'].isna().sum()
missing_age = df_long['celebrity_age_during_season'].isna().sum()
print(f"    Missing industry: {missing_industry} ({missing_industry/len(df_long)*100:.2f}%)")
print(f"    Missing region: {missing_region} ({missing_region/len(df_long)*100:.2f}%)")
print(f"    Missing age: {missing_age} ({missing_age/len(df_long)*100:.2f}%)")

print(f"\n[D] Industry Distribution (Standardized):")
industry_dist = df_long.groupby(['industry_std', 'contestant_id']).size().reset_index().groupby('industry_std').size()
for industry, count in industry_dist.sort_values(ascending=False).items():
    print(f"    {industry}: {count} contestants")

# ============================================================================
# PREVIEW OUTPUT
# ============================================================================
print("\n" + "="*70)
print("SAMPLE OUTPUT - Long Format Panel Data")
print("="*70)
print(df_long[['contestant_id', 'season', 'week', 'J_pct', 'industry_std', 'age_bin', 'is_US']].head(20).to_string())

print("\n" + "="*70)
print("DATA CLEANING COMPLETE!")
print("="*70)
print("""
Output files in cleaned_outputs/:
  1. clean_weekly_panel.csv    - Long format (i, w) panel data for modeling
  2. clean_judge_scores_wide.csv - Wide format with J% per week
  3. season_summary.csv        - Season-level statistics
  4. elimination_summary.csv   - Elimination timeline by season/week
  
Key Features:
  - J% normalized to 0-100 scale (handles 3/4 judge systems)
  - N/A and 0-scores properly excluded
  - Industry standardized into major categories
  - Region coded as US vs Non-US
  - Age binned for analysis
  - Ready for mixed-effects/hierarchical models
""")
