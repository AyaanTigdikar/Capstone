"""
forest_robustness.py
====================
Robustness check: subtract forest rents (and coal rents) from Total NR rents
to isolate hydrocarbon + mineral rents and re-run the main regression models.

Run from project root:
    python3 robustness-forest/forest_robustness.py

Outputs (all written to robustness-forest/outputs/):
    forest_rents.csv          — raw WB forest + coal rents fetched
    master_adj.csv            — master panel with adjusted rent column
    decomp_summary.csv        — forest-rent share per country (1995 & 2019)
    reg_coef_comparison.csv   — coefficient table: original vs adjusted spec
    reg_r2_comparison.csv     — R² table: original vs adjusted spec
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
import wbgapi as wb
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

OUT = os.path.join(ROOT, 'robustness-forest', 'outputs')
os.makedirs(OUT, exist_ok=True)

RENT_COL     = 'Total natural resources rents (% of GDP)'
RENT_ADJ_COL = 'NR rents excl. forest (% of GDP)'
ECI_COL      = 'Economic Complexity Index'
YEARS        = list(range(1995, 2020))

# ── Model 3 regressors (mirrors 6_Regressions_Unified.ipynb) ─────────────────
CORE_VARS = [
    'Human capital index',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP',
    'Political stability — estimate',
    'Rule of law index',
    'Total_Production_Value_Per_Capita',
    'Trade (% of GDP)',
]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Fetch forest + coal rents from World Bank
# ─────────────────────────────────────────────────────────────────────────────
print('Fetching WB forest rents (NY.GDP.FRST.RT.ZS) …')
forest_wide = wb.data.DataFrame('NY.GDP.FRST.RT.ZS', time=YEARS, labels=False)
forest_long = (forest_wide
               .reset_index()
               .rename(columns={'economy': 'Country Code'})
               .melt(id_vars='Country Code', var_name='Year', value_name='Forest_Rents_pct')
               .assign(Year=lambda d: d['Year'].str.replace('YR', '').astype(int)))

print('Fetching WB coal rents (NY.GDP.COAL.RT.ZS) …')
coal_wide = wb.data.DataFrame('NY.GDP.COAL.RT.ZS', time=YEARS, labels=False)
coal_long = (coal_wide
             .reset_index()
             .rename(columns={'economy': 'Country Code'})
             .melt(id_vars='Country Code', var_name='Year', value_name='Coal_Rents_pct')
             .assign(Year=lambda d: d['Year'].str.replace('YR', '').astype(int)))

forest_df = forest_long.merge(coal_long, on=['Country Code', 'Year'], how='outer')
forest_df['Forest_Rents_pct'] = forest_df['Forest_Rents_pct'].fillna(0)
forest_df['Coal_Rents_pct']   = forest_df['Coal_Rents_pct'].fillna(0)
forest_df['Forest_Coal_pct']  = forest_df['Forest_Rents_pct'] + forest_df['Coal_Rents_pct']

forest_df.to_csv(os.path.join(OUT, 'forest_rents.csv'), index=False)
print(f'  Saved forest_rents.csv  ({len(forest_df):,} rows)')

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Build adjusted master panel
# ─────────────────────────────────────────────────────────────────────────────
print('\nLoading master panel …')
master = pd.read_csv(os.path.join(ROOT, 'intermediary', 'Master.csv'),
                     dtype={'Country Code': str})
master['Year'] = master['Year'].astype(int)

master_adj = master.merge(
    forest_df[['Country Code', 'Year', 'Forest_Rents_pct', 'Coal_Rents_pct', 'Forest_Coal_pct']],
    on=['Country Code', 'Year'], how='left'
)
master_adj['Forest_Rents_pct'] = master_adj['Forest_Rents_pct'].fillna(0)
master_adj['Coal_Rents_pct']   = master_adj['Coal_Rents_pct'].fillna(0)
master_adj['Forest_Coal_pct']  = master_adj['Forest_Coal_pct'].fillna(0)

master_adj[RENT_ADJ_COL] = (
    master_adj[RENT_COL] - master_adj['Forest_Rents_pct']
).clip(lower=0)

# ── Derive per-capita production (needed for regression) ─────────────────────
master_adj['Total_Production_Value_Per_Capita'] = (
    master_adj['Total_Production_Value'] / master_adj['Population'].replace(0, np.nan)
)

# ── Decomposition summary (1995 & 2019) ──────────────────────────────────────
decomp = (master_adj[master_adj['Year'].isin([1995, 2019])]
          .groupby(['Country Code', 'Country Name', 'Year'])
          [[RENT_COL, 'Oil rents (% of GDP)', 'Natural gas rents (% of GDP)',
            'Mineral rents (% of GDP)', 'Forest_Rents_pct', 'Coal_Rents_pct',
            RENT_ADJ_COL]]
          .first().reset_index())
decomp['Forest_share_of_total'] = (
    decomp['Forest_Rents_pct'] / decomp[RENT_COL].replace(0, np.nan)
).fillna(0)

decomp.to_csv(os.path.join(OUT, 'decomp_summary.csv'), index=False)
print(f'  Saved decomp_summary.csv')

master_adj.to_csv(os.path.join(OUT, 'master_adj.csv'), index=False)
print(f'  Saved master_adj.csv  ({master_adj.shape})')

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Regression helper: Driscoll-Kraay (clustered HAC) via Newey-West
#     per-group — mirrors the notebook approach
# ─────────────────────────────────────────────────────────────────────────────
def run_model3(df, rent_col, label):
    """
    OLS with HAC-Newey-West SEs (mirrors Model 3a in the notebook).
    Returns a tidy coefficient DataFrame.
    """
    df = df.copy().dropna(subset=[ECI_COL, rent_col] + CORE_VARS)

    # Log-transforms (with floor to avoid log(0))
    df['log_HCI']  = np.log(df['Human capital index'].clip(lower=1e-6))
    df['log_GFCF'] = np.log(
        df['Gross fixed capital formation, all, Constant prices, Percent of GDP'].clip(lower=1e-6))
    df['log_Prod'] = np.log(df['Total_Production_Value_Per_Capita'].clip(lower=1))

    # Mean-centre interactions on full-sample means
    df['log_HCI_x_Prod']  = (df['log_HCI']  - df['log_HCI'].mean())  * df['log_Prod']
    df['log_GFCF_x_Prod'] = (df['log_GFCF'] - df['log_GFCF'].mean()) * df['log_Prod']

    regressors = [
        'log_HCI', 'log_GFCF',
        'Political stability — estimate',
        'Rule of law index',
        'log_Prod',
        'Trade (% of GDP)',
        rent_col,
        'log_HCI_x_Prod', 'log_GFCF_x_Prod',
    ]

    # Drop any regressors not present in df
    regressors = [r for r in regressors if r in df.columns]

    X = sm.add_constant(df[regressors])
    y = df[ECI_COL]

    ols = sm.OLS(y, X).fit()
    hac = ols.get_robustcov_results(cov_type='HAC', maxlags=2, use_correction=True)

    ci = hac.conf_int()
    res = pd.DataFrame({
        'Variable':  X.columns.tolist(),
        'Coef':      np.asarray(hac.params).flatten(),
        'SE':        np.asarray(hac.bse).flatten(),
        'p':         np.asarray(hac.pvalues).flatten(),
        'CI_lo':     np.asarray(ci)[:, 0],
        'CI_hi':     np.asarray(ci)[:, 1],
        'Spec':      label,
        'N':         hac.nobs,
        'R2':        hac.rsquared,
        'R2_adj':    hac.rsquared_adj,
    })
    return res, hac


# ── Import 54-country include list ────────────────────────────────────────────
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from viz_utils import INCLUDE_LIST
master_adj_54 = master_adj[master_adj['Country Code'].isin(INCLUDE_LIST)].copy()

print('\nRunning Model 3a — ORIGINAL spec (54 countries, total NR rents) …')
res_orig, fit_orig = run_model3(master_adj_54, RENT_COL,     'Original (54 ctry, total rents)')

print('Running Model 3a — ADJUSTED rent, same 54-country sample …')
res_adj,  fit_adj  = run_model3(master_adj_54, RENT_ADJ_COL, 'Adj. rent, same sample (54 ctry)')

# ── Spec 3: re-select sample — adjusted rents ≥ 3% OR Gulf state ─────────────
GULF = {'ARE', 'BHR', 'KWT', 'OMN', 'QAT', 'SAU', 'IRQ', 'IRN', 'YEM'}
qualifying_3pct = set(
    master_adj[(master_adj['Year'] == 1995) & (master_adj[RENT_ADJ_COL] >= 3.0)]
    ['Country Code'].tolist()
)
qualifying_adj = qualifying_3pct | GULF
master_adj_resamp = master_adj[master_adj['Country Code'].isin(qualifying_adj)].copy()
n_resamp = master_adj_resamp['Country Code'].nunique()
print(f'Running Model 3a — ADJUSTED rent + re-selected sample ({n_resamp} countries, ≥3% adj or Gulf) …')
res_adj2, fit_adj2 = run_model3(master_adj_resamp, RENT_ADJ_COL,
                                f'Adj. rent + re-selected ({n_resamp} ctry, ≥3% excl.forest)')

# ── Spec 4: re-select sample — adjusted rents ≥ 2% OR Gulf state ─────────────
qualifying_2pct = set(
    master_adj[(master_adj['Year'] == 1995) & (master_adj[RENT_ADJ_COL] >= 2.0)]
    ['Country Code'].tolist()
) | GULF
master_adj_resamp2 = master_adj[master_adj['Country Code'].isin(qualifying_2pct)].copy()
n_resamp2 = master_adj_resamp2['Country Code'].nunique()
print(f'Running Model 3a — ADJUSTED rent + re-selected ({n_resamp2} countries, ≥2% adj or Gulf) …')
res_adj3, fit_adj3 = run_model3(master_adj_resamp2, RENT_ADJ_COL,
                                f'Adj. rent + re-selected ({n_resamp2} ctry, ≥2% excl.forest)')

# ── Spec 5: re-select sample — adjusted rents ≥ 1% OR Gulf state ─────────────
qualifying_1pct = set(
    master_adj[(master_adj['Year'] == 1995) & (master_adj[RENT_ADJ_COL] >= 1.0)]
    ['Country Code'].tolist()
) | GULF
master_adj_resamp1 = master_adj[master_adj['Country Code'].isin(qualifying_1pct)].copy()
n_resamp1 = master_adj_resamp1['Country Code'].nunique()
print(f'Running Model 3a — ADJUSTED rent + re-selected ({n_resamp1} countries, ≥1% adj or Gulf) …')
res_adj4, fit_adj4 = run_model3(master_adj_resamp1, RENT_ADJ_COL,
                                f'Adj. rent + re-selected ({n_resamp1} ctry, ≥1% excl.forest)')

# Save re-selected country lists for map
name_lu = master_adj[['Country Code', 'Country Name']].drop_duplicates()
pd.DataFrame({'Country Code': sorted(qualifying_adj)}).merge(name_lu, on='Country Code').to_csv(
    os.path.join(OUT, 'reselected_countries_3pct.csv'), index=False)
pd.DataFrame({'Country Code': sorted(qualifying_2pct)}).merge(name_lu, on='Country Code').to_csv(
    os.path.join(OUT, 'reselected_countries_2pct.csv'), index=False)
pd.DataFrame({'Country Code': sorted(qualifying_1pct)}).merge(name_lu, on='Country Code').to_csv(
    os.path.join(OUT, 'reselected_countries_1pct.csv'), index=False)
print(f'  Saved reselected_countries_3pct.csv  ({n_resamp} countries)')
print(f'  Saved reselected_countries_2pct.csv  ({n_resamp2} countries)')
print(f'  Saved reselected_countries_1pct.csv  ({n_resamp1} countries)')

# ── Spec 6: all non-HIC countries (excl. Gulf exception) ─────────────────────
import wbgapi as _wb
_eco = _wb.economy.DataFrame()
_hic = set(_eco[_eco['incomeLevel'] == 'HIC'].index.tolist())
# Preserve: Gulf states + any country from the original 54-country list
_hic_exclude = _hic - GULF - set(INCLUDE_LIST)
master_adj_alldev = master_adj[~master_adj['Country Code'].isin(_hic_exclude)].copy()
n_alldev = master_adj_alldev['Country Code'].nunique()
print(f'Running Model 3a — ALL NON-HIC COUNTRIES ({n_alldev} countries, excl. Gulf) …')
res_all, fit_all = run_model3(master_adj_alldev, RENT_ADJ_COL,
                              f'All non-HIC countries ({n_alldev} ctry)')

# ── Combine coefficient tables ────────────────────────────────────────────────
coef_compare = pd.concat([res_orig, res_adj, res_adj2, res_adj3, res_adj4, res_all], ignore_index=True)
coef_compare.to_csv(os.path.join(OUT, 'reg_coef_comparison.csv'), index=False)
print(f'  Saved reg_coef_comparison.csv')

# ── R² summary ────────────────────────────────────────────────────────────────
r2_compare = pd.DataFrame([
    {'Spec': 'Original (54 ctry)',                               'N': int(fit_orig.nobs),  'R2': fit_orig.rsquared,  'R2_adj': fit_orig.rsquared_adj},
    {'Spec': 'Adj. rent, same sample (54 ctry)',                 'N': int(fit_adj.nobs),   'R2': fit_adj.rsquared,   'R2_adj': fit_adj.rsquared_adj},
    {'Spec': f'Adj. rent + re-selected ({n_resamp} ctry, ≥3%)', 'N': int(fit_adj2.nobs),  'R2': fit_adj2.rsquared,  'R2_adj': fit_adj2.rsquared_adj},
    {'Spec': f'Adj. rent + re-selected ({n_resamp2} ctry, ≥2%)', 'N': int(fit_adj3.nobs), 'R2': fit_adj3.rsquared,  'R2_adj': fit_adj3.rsquared_adj},
    {'Spec': f'Adj. rent + re-selected ({n_resamp1} ctry, ≥1%)', 'N': int(fit_adj4.nobs), 'R2': fit_adj4.rsquared,  'R2_adj': fit_adj4.rsquared_adj},
    {'Spec': f'All non-HIC countries ({n_alldev} ctry)',         'N': int(fit_all.nobs),   'R2': fit_all.rsquared,   'R2_adj': fit_all.rsquared_adj},
])
r2_compare.to_csv(os.path.join(OUT, 'reg_r2_comparison.csv'), index=False)
print(f'  Saved reg_r2_comparison.csv')

# ── Console summary ───────────────────────────────────────────────────────────
print('\n' + '='*60)
print('COEFFICIENT COMPARISON — NR rent variable only')
print('='*60)
rent_rows = coef_compare[coef_compare['Variable'].isin([RENT_COL, RENT_ADJ_COL, 'const'])]
print(rent_rows[['Spec', 'Variable', 'Coef', 'SE', 'p']].round(4).to_string(index=False))

print('\nR² COMPARISON')
print(r2_compare.round(4).to_string(index=False))

print('\n✓ forest_robustness.py complete — outputs in robustness-forest/outputs/')
