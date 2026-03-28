#!/usr/bin/env python3
"""
run_loco_regression.py
======================
Leave-One-Country-Out (LOCO) for OLS regression Models 3a and 3b.
Mirrors the NB6 specification (Driscoll-Kraay SEs on full sample for baseline;
plain OLS per LOO run since we track coefficient stability, not inference).

Run from:  /Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP
    python robustness/run_loco_regression.py

Outputs
-------
intermediary/loco/loco_reg_r2.csv          — R2 per LOO run, 3a and 3b
intermediary/loco/loco_reg_coefs_3a.csv    — 3a coefficients per LOO run
intermediary/loco/loco_reg_coefs_3b.csv    — 3b coefficients per LOO run
"""

import os, time, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from types import SimpleNamespace

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

LOCO_DIR = "intermediary/loco"
os.makedirs(LOCO_DIR, exist_ok=True)

INCLUDE = [
    'AGO','ARE','AZE','BFA','BHR','BOL','CHL','CIV','CMR',
    'COD','COG','DZA','ECU','EGY','ETH','GAB','GHA','GIN',
    'GNQ','IDN','IRN','IRQ','KAZ','KEN','KWT','LAO','LBR',
    'LBY','MDG','MLI','MMR','MNG','MOZ','MWI','MYS','NER',
    'NGA','OMN','PNG','QAT','RUS','RWA','SAU','TCD','TGO',
    'TTO','TZA','UGA','UZB','VEN','VNM','YEM','ZMB','ZWE',
]

# Model 3 regressors (from NB6)
REG3_INPUT    = [
    'log_HCI', 'log_GFCF',
    'Political stability — estimate',
    'Rule of law index',
    'log_Production_Value',
    'Trade (% of GDP)',
]
INTERACT_VARS = ['log_HCI_x_log_Production', 'log_GFCF_x_log_Production']
VARS_3A = REG3_INPUT + INTERACT_VARS
VARS_3B = VARS_3A + ['ECI_lag1']
TARGET  = 'Economic Complexity Index'

NAME_MAP = {
    'log_HCI':                              'Log HCI',
    'log_GFCF':                             'Log GFCF',
    'Political stability — estimate':       'Political Stability',
    'Rule of law index':                    'Rule of Law',
    'log_Production_Value':                 'Log Production Value',
    'Trade (% of GDP)':                     'Trade',
    'log_HCI_x_log_Production':             'HCI × Production',
    'log_GFCF_x_log_Production':            'GFCF × Production',
    'ECI_lag1':                             'Lagged ECI',
    'const':                                'Constant',
}

def sname(v):
    return NAME_MAP.get(v, v[:25])


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION  (mirrors NB6 / run_bootstrap.prepare_regression_df)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_df(filepath="intermediary/Master.csv"):
    master = pd.read_csv(filepath)
    df = master[master['Country Code'].isin(INCLUDE)].copy()
    if len(df) < 50:
        return None

    df['Total_Production_Value_Per_Capita'] = (
        df['Total_Production_Value'] / df['Population']
    )
    df['log_HCI']              = np.log1p(df['Human capital index'])
    df['log_GFCF']             = np.log1p(df['Gross fixed capital formation, all, Constant prices, Percent of GDP'])
    df['log_Production_Value'] = np.log1p(df['Total_Production_Value_Per_Capita'])

    for col in ['log_HCI', 'log_GFCF', 'log_Production_Value']:
        df[f'{col}_c'] = df[col] - df[col].mean()

    df['log_HCI_x_log_Production']  = df['log_HCI_c']  * df['log_Production_Value_c']
    df['log_GFCF_x_log_Production'] = df['log_GFCF_c'] * df['log_Production_Value_c']

    df = df.sort_values(['Country Code', 'Year']).reset_index(drop=True)
    df['ECI_lag1'] = df.groupby('Country Code')[TARGET].shift(1)
    return df


def fit_driscoll_kraay(y, X, time_arr, groups):
    """Full-sample DK SEs for the baseline model (matches NB6)."""
    raw = sm.OLS(y, X).fit()
    robust = raw.get_robustcov_results(
        cov_type='HAC-Groupsum', time=time_arr, groups=groups,
        maxlags=2, kernel='bartlett', use_correction=True,
    )
    return SimpleNamespace(
        params=pd.Series(robust.params, index=X.columns),
        bse=pd.Series(robust.bse, index=X.columns),
        tvalues=pd.Series(robust.tvalues, index=X.columns),
        pvalues=pd.Series(robust.pvalues, index=X.columns),
        nobs=robust.nobs, rsquared=robust.rsquared,
    )


def fit_ols_loo(reg_df, var_list):
    """Plain OLS for LOCO run (we want coefficient values, not SE inference)."""
    clean = reg_df[[TARGET] + var_list].dropna()
    if len(clean) < 20:
        return None
    y = clean[TARGET]
    X = sm.add_constant(clean[var_list])
    fit = sm.OLS(y, X).fit()
    row = {'n_obs': int(fit.nobs), 'r2': fit.rsquared}
    for var in fit.params.index:
        row[var] = fit.params[var]
    return row


# ─────────────────────────────────────────────────────────────────────────────
# FULL-SAMPLE BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def fit_full_sample(df):
    cols_3a = [TARGET, 'Country Code', 'Year'] + VARS_3A
    d3a = df[cols_3a].dropna()
    y3a, X3a = d3a[TARGET], sm.add_constant(d3a[VARS_3A])
    m3a = fit_driscoll_kraay(y3a, X3a, d3a['Year'].values, d3a['Country Code'].values)
    print(f"  Model 3a  N={int(m3a.nobs)}  R²={m3a.rsquared:.4f}")

    cols_3b = cols_3a + ['ECI_lag1']
    d3b = df[cols_3b].dropna()
    y3b, X3b = d3b[TARGET], sm.add_constant(d3b[VARS_3B])
    m3b = fit_driscoll_kraay(y3b, X3b, d3b['Year'].values, d3b['Country Code'].values)
    print(f"  Model 3b  N={int(m3b.nobs)}  R²={m3b.rsquared:.4f}")

    return m3a, m3b


# ─────────────────────────────────────────────────────────────────────────────
# LOCO LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_loco(df, m3a, m3b):
    countries = sorted(df['Country Code'].unique())
    N = len(countries)
    r2_rows, c3a_rows, c3b_rows = [], [], []
    t0 = time.time()

    for i, cc in enumerate(countries, 1):
        df_loo = df[df['Country Code'] != cc].copy()
        if df_loo['Country Code'].nunique() < 10:
            print(f"  WARNING: skipping {cc}")
            continue

        r3a = fit_ols_loo(df_loo, VARS_3A)
        r3b = fit_ols_loo(df_loo, VARS_3B)

        if r3a is None or r3b is None:
            continue

        r2_rows.append({
            'excluded_country': cc,
            'r2_3a': r3a['r2'], 'r2_3b': r3b['r2'],
            'n_obs_3a': r3a['n_obs'], 'n_obs_3b': r3b['n_obs'],
        })
        c3a_rows.append({'excluded_country': cc, **{k: v for k, v in r3a.items() if k not in ('n_obs','r2')}})
        c3b_rows.append({'excluded_country': cc, **{k: v for k, v in r3b.items() if k not in ('n_obs','r2')}})

        if i % 10 == 0 or i == 1:
            elapsed = time.time() - t0
            eta     = (N - i) / max(i, 1) * elapsed
            print(f"  {i:>2d}/{N}  excluded={cc}  "
                  f"R²_3a={r3a['r2']:.4f}  R²_3b={r3b['r2']:.4f}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    r2_df  = pd.DataFrame(r2_rows)
    df_3a  = pd.DataFrame(c3a_rows)
    df_3b  = pd.DataFrame(c3b_rows)

    r2_df.to_csv( os.path.join(LOCO_DIR, "loco_reg_r2.csv"),       index=False)
    df_3a.to_csv( os.path.join(LOCO_DIR, "loco_reg_coefs_3a.csv"), index=False)
    df_3b.to_csv( os.path.join(LOCO_DIR, "loco_reg_coefs_3b.csv"), index=False)

    print(f"\n  {len(r2_rows)} LOCO runs in {time.time()-t0:.1f}s — CSVs saved to {LOCO_DIR}/")
    return r2_df, df_3a, df_3b


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(r2_df, df_3a, df_3b, m3a, m3b):
    print(f"\n{'='*70}")
    print("  REGRESSION LOCO SUMMARY")
    print(f"{'='*70}")
    for label, col, m in [('3a','r2_3a',m3a), ('3b','r2_3b',m3b)]:
        vals = r2_df[col]
        print(f"  Model {label}  full R²={m.rsquared:.4f}  "
              f"LOCO mean={vals.mean():.4f}  range=[{vals.min():.4f},{vals.max():.4f}]  "
              f"SD={vals.std():.4f}")

    # Coefficient stability
    for label, coef_df, m, var_list in [
        ('3a', df_3a, m3a, VARS_3A),
        ('3b', df_3b, m3b, VARS_3B),
    ]:
        print(f"\n  Model {label} coefficient stability:")
        print(f"  {'Variable':<32} {'Full':>8} {'Mean':>8} {'SD':>7} {'Sign%':>6}")
        print("  " + "-"*65)
        for var in var_list:
            if var not in coef_df.columns:
                continue
            vals = coef_df[var].dropna()
            fc   = m.params.get(var, np.nan)
            sgn  = (vals > 0).mean() if fc > 0 else (vals < 0).mean()
            print(f"  {sname(var):<32} {fc:>+8.4f} {vals.mean():>+8.4f} "
                  f"{vals.std():>7.4f} {sgn:>5.0%}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir).lower() in ("robustness", "scripts"):
        os.chdir(os.path.dirname(script_dir))

    print("="*70)
    print("  REGRESSION LOCO — Models 3a and 3b")
    print(f"  Working dir: {os.getcwd()}")
    print("="*70)

    t_total = time.time()

    print("\nLoading data...")
    df = prepare_df()
    print(f"  {df['Country Code'].nunique()} countries, {len(df):,} obs")

    print("\nFull-sample baselines (Driscoll-Kraay):")
    m3a, m3b = fit_full_sample(df)

    print("\nRunning LOCO...")
    r2_df, df_3a, df_3b = run_loco(df, m3a, m3b)

    print_summary(r2_df, df_3a, df_3b, m3a, m3b)

    total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  DONE  |  {total:.1f}s")
    print("="*70)
