"""
reg_models.py
=============
Run five regression models matching the clean-code notebook specs across
four sample definitions and produce one coefficient forest plot per model.

Models (mirror Descriptive Statistics and Regressions (clean code).ipynb):
  3a — base, no lag         : log_HCI, log_GFCF, PolStab, RoL, log_Prod, Trade,
                               HCI×Prod, GFCF×Prod · DV: ECI (levels)
  3b — one lag              : 3a + ECI_lag1            · DV: ECI (levels)
  3c — all vars lagged      : all regressors at t-1 + ECI_lag1 + lagged interactions
                                                        · DV: ECI (levels)
  3d — extended controls    : 3b + Electricity + resource-type dummies
                                                        · DV: ECI (levels)
  3e — first differences    : same regressors as 3d + ECI_lag1 (error-correction)
                                                        · DV: ΔECI

Samples (original vs 1/2/3% re-selected, forest excl.):
  1 — Original 54 countries (total NR rents, baseline)
  2 — Re-selected ≥3% adj. rents + Gulf
  3 — Re-selected ≥2% adj. rents + Gulf
  4 — Re-selected ≥1% adj. rents + Gulf

SE estimator: clustered by Country Code (mirrors the notebook).

Run from project root:
    python3 robustness-forest/reg_models.py

Output: robustness-forest/outputs/regression/
    coef_model3a/3b/3c/3d/3e  (.csv / .html / .png)
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import wbgapi as wb
import plotly.graph_objects as go
from viz_utils import FONT, BG, NAVY, GRID, base_layout, save, INCLUDE_LIST

OUT = os.path.join(ROOT, 'robustness-forest', 'outputs', 'regression')
os.makedirs(OUT, exist_ok=True)

GULF     = {'ARE', 'BHR', 'KWT', 'OMN', 'QAT', 'SAU', 'IRQ', 'IRN', 'YEM'}
RENT_ADJ = 'NR rents excl. forest (% of GDP)'
ECI_COL  = 'Economic Complexity Index'

# ── HIC exclusion list ────────────────────────────────────────────────────────
_eco    = wb.economy.DataFrame()
_hic    = set(_eco[_eco['incomeLevel'] == 'HIC'].index)
HIC_EXC = _hic - GULF - set(INCLUDE_LIST)

# ── load master_adj ───────────────────────────────────────────────────────────
master = pd.read_csv(os.path.join(ROOT, 'robustness-forest', 'outputs', 'master_adj.csv'),
                     dtype={'Country Code': str})
master['Year'] = master['Year'].astype(int)
master = master.sort_values(['Country Code', 'Year']).reset_index(drop=True)

# ── sample definitions ────────────────────────────────────────────────────────
d95 = master[master['Year'] == 1995].set_index('Country Code')
s3  = (set(d95[d95[RENT_ADJ] >= 3.0].index) | GULF) & set(d95.index)
s4  = (set(d95[d95[RENT_ADJ] >= 2.0].index) | GULF) & set(d95.index)
s5  = (set(d95[d95[RENT_ADJ] >= 1.0].index) | GULF) & set(d95.index)

SAMPLES = {
    'Original 54\n(total rents)': master[master['Country Code'].isin(INCLUDE_LIST)].copy(),
    'Adj. ≥3%':                   master[master['Country Code'].isin(s3)].copy(),
    'Adj. ≥2%':                   master[master['Country Code'].isin(s4)].copy(),
    'Adj. ≥1%':                   master[master['Country Code'].isin(s5)].copy(),
}

# ── colour per spec ───────────────────────────────────────────────────────────
SPEC_COLORS = ['#4a6fa5', '#2e7d4a', '#e8a838', '#d4853b']

DISPLAY_LABELS = {
    'log_HCI':                                 'Human Capital (log)',
    'log_GFCF':                                'GFCF (log)',
    'Political stability — estimate':          'Political Stability',
    'Rule of law index':                       'Rule of Law',
    'log_Production_Value':                    'NR Production (log)',
    'Trade (% of GDP)':                        'Trade',
    'log_HCI_x_log_Production':                'HCI × Production',
    'log_GFCF_x_log_Production':               'GFCF × Production',
    'ECI_lag1':                                'ECI (t−1)',
    'log_HCI_lag1':                            'Human Capital (t−1)',
    'log_GFCF_lag1':                           'GFCF (t−1)',
    'Political stability — estimate_lag1':     'Political Stability (t−1)',
    'Rule of law index_lag1':                  'Rule of Law (t−1)',
    'log_Production_Value_lag1':               'NR Production (t−1)',
    'Trade (% of GDP)_lag1':                   'Trade (t−1)',
    'log_HCI_x_log_Production_lag1':           'HCI × Production (t−1)',
    'log_GFCF_x_log_Production_lag1':          'GFCF × Production (t−1)',
    'Access to electricity (% of population)': 'Electricity access',
    'Hydrocarbons_Dominant':                   'Hydrocarbons dominant',
    'Subsoil_Metals_Dominant':                 'Subsoil metals dominant',
    'Precious_Metals_Dominant':                'Precious metals dominant',
}

BASE_INDEP = [
    'log_HCI', 'log_GFCF',
    'Political stability — estimate',
    'Rule of law index',
    'log_Production_Value',
    'Trade (% of GDP)',
]
EXTRA_CONTROLS = [
    'Hydrocarbons_Dominant',
    'Subsoil_Metals_Dominant',
    'Precious_Metals_Dominant',
    'Access to electricity (% of population)',
]


# ── feature engineering (per-sample so means are sample-specific) ─────────────
def prepare(df):
    df = df.copy().sort_values(['Country Code', 'Year']).reset_index(drop=True)

    df['Total_Production_Value_Per_Capita'] = (
        df['Total_Production_Value'] / df['Population'].replace(0, np.nan)
    )
    df['log_HCI']              = np.log1p(df['Human capital index'].clip(lower=0))
    df['log_GFCF']             = np.log1p(
        df['Gross fixed capital formation, all, Constant prices, Percent of GDP'].clip(lower=0))
    df['log_Production_Value'] = np.log1p(df['Total_Production_Value_Per_Capita'].clip(lower=0))

    # Lagged ECI and delta ECI
    df['ECI_lag1']   = df.groupby('Country Code')[ECI_COL].shift(1)
    df['delta_ECI']  = df[ECI_COL] - df['ECI_lag1']

    # Lagged versions of all base regressors
    for var in BASE_INDEP:
        df[f'{var}_lag1'] = df.groupby('Country Code')[var].shift(1)

    # ── Centred interactions (current period) ────────────────────────────────
    hci_c  = df['log_HCI']             - df['log_HCI'].mean()
    gfcf_c = df['log_GFCF']            - df['log_GFCF'].mean()
    prod_c = df['log_Production_Value'] - df['log_Production_Value'].mean()
    df['log_HCI_x_log_Production']  = hci_c  * prod_c
    df['log_GFCF_x_log_Production'] = gfcf_c * prod_c

    # ── Centred interactions (lagged) ─────────────────────────────────────────
    hci_l_c  = df['log_HCI_lag1']              - df['log_HCI_lag1'].mean()
    gfcf_l_c = df['log_GFCF_lag1']             - df['log_GFCF_lag1'].mean()
    prod_l_c = df['log_Production_Value_lag1']  - df['log_Production_Value_lag1'].mean()
    df['log_HCI_x_log_Production_lag1']  = hci_l_c  * prod_l_c
    df['log_GFCF_x_log_Production_lag1'] = gfcf_l_c * prod_l_c

    return df


# ── regression helper (clustered SE by country) ───────────────────────────────
def run_reg(df, dv, regressors, label):
    req = [dv] + regressors + ['Country Code']
    sub = df.dropna(subset=req).copy()
    X   = sm.add_constant(sub[regressors])
    y   = sub[dv]
    res = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': sub['Country Code']},
    )
    ci = res.conf_int()
    return pd.DataFrame({
        'Variable': X.columns.tolist(),
        'Coef':     np.asarray(res.params).flatten(),
        'CI_lo':    np.asarray(ci)[:, 0],
        'CI_hi':    np.asarray(ci)[:, 1],
        'p':        np.asarray(res.pvalues).flatten(),
        'Spec':     label,
        'N':        int(res.nobs),
        'R2':       round(res.rsquared, 4),
    })


# ── model variable sets ───────────────────────────────────────────────────────
VARS_3A = BASE_INDEP + ['log_HCI_x_log_Production', 'log_GFCF_x_log_Production']

VARS_3B = BASE_INDEP + [
    'ECI_lag1',
    'log_HCI_x_log_Production', 'log_GFCF_x_log_Production',
]

LAGGED_INDEP = [f'{v}_lag1' for v in BASE_INDEP]
VARS_3C = LAGGED_INDEP + [
    'ECI_lag1',
    'log_HCI_x_log_Production_lag1', 'log_GFCF_x_log_Production_lag1',
]

VARS_3D = BASE_INDEP + EXTRA_CONTROLS + [
    'ECI_lag1',
    'log_HCI_x_log_Production', 'log_GFCF_x_log_Production',
]

VARS_3E = VARS_3D   # same regressors, DV = delta_ECI

MODEL_SPECS = [
    ('3a', ECI_COL,    VARS_3A, 'Model 3a — Base (no lag)',
     'Clustered SE (country) · ECI = β₀ + β₁log(HCI) + β₂log(GFCF) + β₃PolStab + β₄RoL + β₅log(Prod) + β₆Trade + interactions'),
    ('3b', ECI_COL,    VARS_3B, 'Model 3b — Base + Lagged ECI',
     'Clustered SE (country) · Model 3a + ECI_{t−1}'),
    ('3c', ECI_COL,    VARS_3C, 'Model 3c — All Regressors Lagged',
     'Clustered SE (country) · All independent variables at t−1 + ECI_{t−1} + lagged interactions'),
    ('3d', ECI_COL,    VARS_3D, 'Model 3d — Extended Controls (Lagged ECI)',
     'Clustered SE (country) · Model 3b + Electricity + Hydrocarbons / Subsoil / Precious dummies'),
    ('3e', 'delta_ECI', VARS_3E, 'Model 3e — First Differences (ΔECI)',
     'Clustered SE (country) · DV = ECI_t − ECI_{t−1} · same regressors as 3d + ECI_{t−1} as error-correction'),
]


# ── run all models × samples ──────────────────────────────────────────────────
all_results = {mname: [] for mname, *_ in MODEL_SPECS}

for spec_label, df_raw in SAMPLES.items():
    df = prepare(df_raw)
    short = spec_label.replace('\n', ' ')
    print(f'\n{"="*55}\n{short}')
    for mname, dv, regressors, *_ in MODEL_SPECS:
        r = run_reg(df, dv, regressors, spec_label)
        n  = r['N'].iloc[0]
        r2 = r['R2'].iloc[0]
        print(f'  Model {mname}: N={n:,}  R²={r2:.4f}')
        all_results[mname].append(r)

# Save CSVs
for mname, *_ in MODEL_SPECS:
    df_out = pd.concat(all_results[mname], ignore_index=True)
    df_out.to_csv(os.path.join(OUT, f'coef_model{mname}.csv'), index=False)
print(f'\n  Saved coef_model 3a/3b/3c/3d/3e .csv')


# ── forest plot helper ────────────────────────────────────────────────────────
def forest_plot(coef_df, var_list, title, subtitle, fname):
    cf = coef_df[coef_df['Variable'].isin(var_list)].copy()
    cf['Label'] = cf['Variable'].map(DISPLAY_LABELS).fillna(cf['Variable'])

    var_order = [DISPLAY_LABELS.get(v, v) for v in var_list]
    y_pos     = {v: i for i, v in enumerate(var_order)}

    specs   = cf['Spec'].unique().tolist()
    n_specs = len(specs)
    offsets = np.linspace(-0.22, 0.22, n_specs)

    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color='#777', width=1.2, dash='dash'))

    for i, spec in enumerate(specs):
        sub = cf[cf['Spec'] == spec].copy()
        sub['y'] = sub['Label'].map(y_pos) + offsets[i]
        col = SPEC_COLORS[i]
        leg = spec.replace('\n', ' ')
        fig.add_trace(go.Scatter(
            x=sub['Coef'], y=sub['y'],
            mode='markers',
            marker=dict(size=9, color=col, line=dict(color='white', width=1.2)),
            error_x=dict(
                type='data', symmetric=False,
                array=(sub['CI_hi'] - sub['Coef']).values,
                arrayminus=(sub['Coef'] - sub['CI_lo']).values,
                color=col, thickness=2, width=4,
            ),
            name=leg,
            hovertemplate='%{text}<br>β=%{x:.4f}<extra>' + leg + '</extra>',
            text=sub['Label'],
        ))

    n_vars = len(var_order)
    fig.update_layout(**base_layout(
        height=max(480, n_vars * 58 + 260),
        margin=dict(l=60, r=60, t=140, b=60),
        xaxis=dict(title='Coefficient (95% CI, clustered SE by country)',
                   gridcolor=GRID, gridwidth=0.5),
        yaxis=dict(
            tickvals=list(y_pos.values()),
            ticktext=list(y_pos.keys()),
            tickfont=dict(size=11),
            showgrid=True, gridcolor=GRID, gridwidth=0.5,
            range=[-0.6, n_vars - 0.4],
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.06,
            xanchor='center', x=0.5, font=dict(size=10),
            bgcolor='rgba(255,255,255,0.0)',
        ),
        title=dict(
            text=f'{title}<br><sup>{subtitle}</sup>',
            font=dict(size=14), x=0.5, y=0.97, yanchor='top',
        ),
    ))
    save(fig, fname, OUT, w=1100, h=max(480, n_vars * 58 + 160))
    print(f'✓ {fname}')


# ── produce forest plots ──────────────────────────────────────────────────────
for mname, dv, var_list, title, subtitle in MODEL_SPECS:
    coef_df = pd.concat(all_results[mname], ignore_index=True)
    forest_plot(coef_df, var_list, title, subtitle, f'coef_model{mname}')

print(f'\n✓ reg_models.py complete — charts in robustness-forest/outputs/regression/')
