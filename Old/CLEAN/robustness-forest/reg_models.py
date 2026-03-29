"""
reg_models.py
=============
Run all regression models matching NB6 specs across four sample definitions.

Models:
  1  — kitchen sink (~40 vars)  · DV: log(ECI) · SE: HAC-panel (country, Bartlett k=3)
  2  — AR baseline (ECI_lag1)   · DV: ECI      · SE: HAC-panel (country, Bartlett k=3)
  3a — base, no lag             · DV: ECI      · SE: HAC-panel (country, Bartlett k=3)
  3b — one lag                  · DV: ECI      · SE: HAC-panel (country, Bartlett k=3)
  3c — all vars lagged          · DV: ECI      · SE: HAC-panel (country, Bartlett k=3)
  3d — extended controls        · DV: ECI      · SE: HAC-panel (country, Bartlett k=3)
  3e — first differences (ΔECI) · DV: ΔECI     · SE: HAC-panel (country, Bartlett k=3)

Samples (original vs 1/2/3% re-selected, forest excl.):
  1 — Original 47 countries (k=5 sample, baseline)
  2 — Re-selected ≥3% adj. rents + Gulf
  3 — Re-selected ≥2% adj. rents + Gulf
  4 — Re-selected ≥1% adj. rents + Gulf

SE estimator: HAC-panel (Driscoll-Kraay-style) grouped by Country Code.
  cov_type='hac-panel', Bartlett kernel, maxlags=3 (Newey-West T^(1/3) rule, T=25).
  Robust to both heteroskedasticity and within-country serial autocorrelation.

Run from project root:
    python3 robustness-forest/reg_models.py

Output: robustness-forest/outputs/regression/
    coef_model3a/3b/3c/3d/3e  (.csv / .html / .png)
    reg_table_main.csv / reg_table_main.html  (formatted regression table)
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

# Merge k=5 cluster IDs for Model 1 SE (clustered by cluster)
_clusters = pd.read_csv(os.path.join(ROOT, 'intermediary', 'clusters_k5_agg.csv'),
                        dtype={'Country Code': str},
                        usecols=['Country Code', 'Cluster', 'ClusterLabels'])
_clusters = _clusters.drop_duplicates('Country Code')
master = master.merge(_clusters, on='Country Code', how='left')

# ── sample definitions ────────────────────────────────────────────────────────
d95 = master[master['Year'] == 1995].set_index('Country Code')
s3  = (set(d95[d95[RENT_ADJ] >= 3.0].index) | GULF) & set(d95.index)
s4  = (set(d95[d95[RENT_ADJ] >= 2.0].index) | GULF) & set(d95.index)
s5  = (set(d95[d95[RENT_ADJ] >= 1.0].index) | GULF) & set(d95.index)

SAMPLES = {
    'Original 47\n(k=5 sample)': master[master['Country Code'].isin(INCLUDE_LIST)].copy(),
    'Adj. ≥3%':                  master[master['Country Code'].isin(s3)].copy(),
    'Adj. ≥2%':                  master[master['Country Code'].isin(s4)].copy(),
    'Adj. ≥1%':                  master[master['Country Code'].isin(s5)].copy(),
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

    # log(ECI) shifted to be positive — used in Model 1 only
    eci_min = df[ECI_COL].min()
    df['log_ECI'] = np.log(df[ECI_COL] - eci_min + 1)

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


# ── Kitchen Sink vars (Model 1) ───────────────────────────────────────────────
KITCHEN_SINK_VARS = [
    'Access to electricity (% of population)',
    'Adjusted savings: gross savings (% of GNI)',
    'Agriculture',
    'Capital depreciation rate',
    'Clientelism index',
    'Death rates, crude per 1000 people',
    'Domestic credit to private sector (% of GDP)',
    'GDP per capita (constant prices, PPP)',
    'Government revenue',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP',
    'Human capital index',
    'Industry',
    'Inflation, consumer prices (annual %)',
    'Landlocked',
    'Lending interest rate (%)',
    'Life expectancy at birth, total (years)',
    'Manufacturing',
    'Mineral rents (% of GDP)',
    'Mobile cellular subscriptions (per 100 people)',
    'Natural gas rents (% of GDP)',
    'Oil rents (% of GDP)',
    'Political corruption index',
    'Political stability — estimate',
    'Primary net lending, General government, Percent of GDP',
    'Property rights',
    'Real interest rate (%)',
    'Rule of law index',
    'Services',
    'Share of consumption in GDP',
    'Share of government spending in GDP',
    'Share of investment in GDP',
    'Total natural resources rents (% of GDP)',
    'Trade (% of GDP)',
    'Urban population (% of total population)',
    'Use of IMF credit (DOD, current US$)',
    'Total_Production_Value_Per_Capita',
    'Total_Reserves_Value_Per_Capita',
    'Hydrocarbons_Dominant',
    'Subsoil_Metals_Dominant',
    'Precious_Metals_Dominant',
]


# HAC bandwidth: T^(1/3) for T=25 years → 3 lags (Newey-West rule)
HAC_MAXLAGS = 3

# ── regression helper (HAC-panel SE — Bartlett kernel, grouped by Country Code) ─
def run_reg(df, dv, regressors, label, cluster_by=None):
    # cluster_by kept for signature compatibility but HAC always uses Country Code
    regressors = [r for r in regressors if r in df.columns]
    req = [dv] + regressors + ['Country Code']
    sub = df.dropna(subset=req).copy()
    X   = sm.add_constant(sub[regressors])
    y   = sub[dv]
    res = sm.OLS(y, X).fit(
        cov_type='hac-panel',
        cov_kwds={
            'groups':         sub['Country Code'],
            'maxlags':        HAC_MAXLAGS,
            'use_correction': True,
        },
    )
    ci = res.conf_int()
    return pd.DataFrame({
        'Variable': X.columns.tolist(),
        'Coef':     np.asarray(res.params).flatten(),
        'SE':       np.asarray(res.bse).flatten(),
        'tstat':    np.asarray(res.tvalues).flatten(),
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

HAC_SUBTITLE = f'HAC-panel SE (Bartlett kernel, {HAC_MAXLAGS} lags, country panel)'

MODEL_SPECS = [
    # (mname, dv, var_list, title, subtitle)
    ('1',  'log_ECI',   KITCHEN_SINK_VARS, 'Model 1 — Kitchen Sink',
     f'{HAC_SUBTITLE} · DV = log(ECI) · ~40 regressors · exploratory'),
    ('2',  ECI_COL,     ['ECI_lag1'],       'Model 2 — AR Baseline',
     f'{HAC_SUBTITLE} · DV = ECI · IV = ECI_{{t-1}} only · benchmark'),
    ('3a', ECI_COL,    VARS_3A, 'Model 3a — Base (no lag)',
     f'{HAC_SUBTITLE} · ECI = log(HCI) + log(GFCF) + PolStab + RoL + log(Prod) + Trade + interactions'),
    ('3b', ECI_COL,    VARS_3B, 'Model 3b — Base + Lagged ECI',
     f'{HAC_SUBTITLE} · Model 3a + ECI_{{t-1}}'),
    ('3c', ECI_COL,    VARS_3C, 'Model 3c — All Regressors Lagged',
     f'{HAC_SUBTITLE} · All independent variables at t-1 + ECI_{{t-1}} + lagged interactions'),
    ('3d', ECI_COL,    VARS_3D, 'Model 3d — Extended Controls (Lagged ECI)',
     f'{HAC_SUBTITLE} · Model 3b + Electricity + Hydrocarbons / Subsoil / Precious dummies'),
    ('3e', 'delta_ECI', VARS_3E, 'Model 3e — First Differences (ΔECI)',
     f'{HAC_SUBTITLE} · DV = ECI_t - ECI_{{t-1}} · same regressors as 3d + ECI_{{t-1}} as error-correction'),
]


# ── run all models × samples ──────────────────────────────────────────────────
all_results = {mname: [] for mname, *_ in MODEL_SPECS}

for spec_label, df_raw in SAMPLES.items():
    df = prepare(df_raw)
    short = spec_label.replace('\n', ' ')
    print(f'\n{"="*55}\n{short}')
    for mname, dv, regressors, title, subtitle in MODEL_SPECS:
        # Model 1 only on the original k=5 sample
        if mname == '1' and 'Original 47' not in spec_label:
            continue
        r = run_reg(df, dv, regressors, spec_label)
        n  = r['N'].iloc[0]
        r2 = r['R2'].iloc[0]
        print(f'  Model {mname}: N={n:,}  R²={r2:.4f}')
        all_results[mname].append(r)

# Save CSVs
for mname, *_ in MODEL_SPECS:
    if not all_results[mname]:
        continue
    df_out = pd.concat(all_results[mname], ignore_index=True)
    df_out.to_csv(os.path.join(OUT, f'coef_model{mname}.csv'), index=False)
print(f'\n  Saved coef_model 1/2/3a/3b/3c/3d/3e .csv')


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
        xaxis=dict(title=f'Coefficient (95% CI, HAC-panel SE, Bartlett k={HAC_MAXLAGS})',
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
    if not all_results[mname]:
        continue
    coef_df = pd.concat(all_results[mname], ignore_index=True)
    forest_plot(coef_df, var_list, title, subtitle, f'coef_model{mname}')

# ── regression table (primary sample, models 3a–3e) ──────────────────────────
def make_reg_table(results_by_model, primary_label, model_names, var_display):
    """
    Returns a formatted DataFrame: variables as rows, models as columns.
    Each cell shows coefficient with significance stars, (SE) on the next row.
    Bottom rows: N and R².
    """
    def stars(p):
        if p < 0.01:  return '***'
        if p < 0.05:  return '**'
        if p < 0.10:  return '*'
        return ''

    # Collect results for primary sample only
    model_dfs = {}
    for mname in model_names:
        if not results_by_model[mname]:
            continue
        df_all = pd.concat(results_by_model[mname], ignore_index=True)
        sub = df_all[df_all['Spec'] == primary_label]
        if not sub.empty:
            model_dfs[mname] = sub.set_index('Variable')

    if not model_dfs:
        print('  [reg_table] No results for primary sample — skipping table.')
        return None

    # Union of all variables (excluding constant for body; constant added last)
    all_vars = []
    seen = set()
    for mname in model_names:
        if mname not in model_dfs:
            continue
        for v in model_dfs[mname].index:
            if v != 'const' and v not in seen:
                all_vars.append(v)
                seen.add(v)

    rows = []
    for v in all_vars + ['const']:
        label = var_display.get(v, v)
        coef_row = {'Variable': label}
        se_row   = {'Variable': ''}
        for mname in model_names:
            if mname not in model_dfs or v not in model_dfs[mname].index:
                coef_row[f'M{mname}'] = ''
                se_row[f'M{mname}']   = ''
            else:
                r  = model_dfs[mname].loc[v]
                st = stars(r['p'])
                coef_row[f'M{mname}'] = f"{r['Coef']:.4f}{st}"
                se_row[f'M{mname}']   = f"({r['SE']:.4f})"
        rows.append(coef_row)
        rows.append(se_row)

    # Footer rows
    n_row  = {'Variable': 'N'}
    r2_row = {'Variable': 'R²'}
    for mname in model_names:
        if mname not in model_dfs:
            n_row[f'M{mname}']  = ''
            r2_row[f'M{mname}'] = ''
        else:
            s = model_dfs[mname].iloc[0]
            n_row[f'M{mname}']  = str(int(s['N']))
            r2_row[f'M{mname}'] = f"{s['R2']:.4f}"
    rows += [{'Variable': '─' * 20}, n_row, r2_row]
    rows.insert(0, {'Variable': 'SE estimator',
                    **{f'M{m}': f'HAC-panel (k={HAC_MAXLAGS})' for m in model_names}})
    rows.insert(1, {'Variable': '─' * 20})

    tbl = pd.DataFrame(rows).fillna('')
    tbl.columns = [''] + [f'({m})' for m in model_names]
    return tbl


TABLE_MODELS = ['3a', '3b', '3c', '3d', '3e']
PRIMARY_LABEL = 'Original 47\n(k=5 sample)'

tbl = make_reg_table(all_results, PRIMARY_LABEL, TABLE_MODELS, DISPLAY_LABELS)
if tbl is not None:
    csv_path  = os.path.join(OUT, 'reg_table_main.csv')
    html_path = os.path.join(OUT, 'reg_table_main.html')
    tbl.to_csv(csv_path, index=False)

    # HTML with minimal styling
    html = tbl.to_html(index=False, border=0, classes='reg-table')
    css  = ('<style>'
            'body{font-family:monospace;font-size:13px;padding:16px}'
            '.reg-table{border-collapse:collapse;width:100%}'
            '.reg-table th,.reg-table td{text-align:right;padding:3px 10px;border-bottom:1px solid #ddd}'
            '.reg-table td:first-child,.reg-table th:first-child{text-align:left}'
            '</style>\n')
    note = ('<p style="font-size:11px;color:#555;margin-top:8px">'
            f'Notes: HAC-panel SE (Bartlett kernel, maxlags={HAC_MAXLAGS}). '
            'Sample: Original 47 (k=5). '
            '* p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01.</p>')
    with open(html_path, 'w') as f:
        f.write(css + html + note)
    print(f'✓ reg_table_main.csv / .html')

    # Print to terminal
    print(f'\n{"="*70}')
    print(f'REGRESSION TABLE — {PRIMARY_LABEL.replace(chr(10), " ")}')
    print(f'HAC-panel SE (Bartlett kernel, maxlags={HAC_MAXLAGS})')
    print('* p<0.10  ** p<0.05  *** p<0.01')
    print('=' * 70)
    print(tbl.to_string(index=False))
    print('=' * 70)

print(f'\n✓ reg_models.py complete — charts in robustness-forest/outputs/regression/')
