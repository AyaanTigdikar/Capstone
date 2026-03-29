"""
forest_robustness_viz.py
========================
Visualisations for the forest-rents robustness check.
Must be run AFTER forest_robustness.py has populated robustness-forest/outputs/.

Run from project root:
    python3 robustness-forest/forest_robustness_viz.py

Charts produced (robustness-forest/outputs/):
    00_forest_decomp_top_countries   — stacked bar: who has forest rents (top 20, 1995)
    01_coef_comparison               — coefficient forest plot: 4 specs
    02_r2_comparison                 — R² bar chart: 4 specs
    03_sample_map                    — choropleth: original 54 vs 3% vs 1% sample
    04_rent_adj_vs_orig_scatter      — scatter: total rents vs adjusted rents
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from viz_utils import FONT, BG, NAVY, GRID, base_layout, save, INCLUDE_LIST

OUT          = os.path.join(ROOT, 'robustness-forest', 'outputs')
RENT_COL     = 'Total natural resources rents (% of GDP)'
RENT_ADJ_COL = 'NR rents excl. forest (% of GDP)'

# ── palette ───────────────────────────────────────────────────────────────────
C_OIL    = '#4a6fa5'
C_GAS    = '#3a8fa5'
C_MIN    = '#2e7d4a'
C_FOREST = '#8b6914'
C_COAL   = '#999999'

# one colour per spec (order matches forest_robustness.py output)
SPEC_COLORS = ['#4a6fa5', '#c23a3a', '#2e7d4a', '#e8a838', '#d4853b', '#7a5c9e']

# ── load outputs ──────────────────────────────────────────────────────────────
decomp   = pd.read_csv(os.path.join(OUT, 'decomp_summary.csv'))
coef     = pd.read_csv(os.path.join(OUT, 'reg_coef_comparison.csv'))
r2       = pd.read_csv(os.path.join(OUT, 'reg_r2_comparison.csv'))
master   = pd.read_csv(os.path.join(OUT, 'master_adj.csv'), dtype={'Country Code': str})
ctry_3   = pd.read_csv(os.path.join(OUT, 'reselected_countries_3pct.csv'), dtype={'Country Code': str})
ctry_2   = pd.read_csv(os.path.join(OUT, 'reselected_countries_2pct.csv'), dtype={'Country Code': str})
ctry_1   = pd.read_csv(os.path.join(OUT, 'reselected_countries_1pct.csv'), dtype={'Country Code': str})

# ══════════════════════════════════════════════════════════════════════════════
# CHART 00 — Stacked bar: rent decomposition, top-20 forest-rent countries 1995
# ══════════════════════════════════════════════════════════════════════════════
d95 = decomp[decomp['Year'] == 1995].copy()
d95 = d95.sort_values('Forest_Rents_pct', ascending=False).head(20)

fig00 = go.Figure()
stacks = [
    ('Oil rents (% of GDP)',         C_OIL,    'Oil'),
    ('Natural gas rents (% of GDP)', C_GAS,    'Gas'),
    ('Mineral rents (% of GDP)',     C_MIN,    'Minerals'),
    ('Forest_Rents_pct',             C_FOREST, 'Forest'),
    ('Coal_Rents_pct',               C_COAL,   'Coal'),
]
for col, color, name in stacks:
    if col not in d95.columns:
        continue
    fig00.add_trace(go.Bar(
        y=d95['Country Name'], x=d95[col], orientation='h',
        name=name, marker_color=color,
        hovertemplate=f'%{{y}} — {name}: %{{x:.1f}}% GDP<extra></extra>',
    ))

fig00.update_layout(**base_layout(
    barmode='stack', height=580,
    margin=dict(l=160, r=60, t=70, b=60),
    xaxis=dict(title='% of GDP', gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(tickfont=dict(size=11), autorange='reversed'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                xanchor='center', x=0.5, font=dict(size=11)),
    title=dict(text='NR Rent Decomposition — Top 20 Countries by Forest Rents (1995)',
               font=dict(size=13), x=0.5),
))
save(fig00, '00_forest_decomp_top_countries', OUT, w=1000, h=580)
print('✓ Chart 00')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 01 — Coefficient forest plot: 4 specs
# ══════════════════════════════════════════════════════════════════════════════
coef['VarLabel'] = coef['Variable'].replace({
    RENT_COL:     'NR rents',
    RENT_ADJ_COL: 'NR rents',
})

KEEP = ['log_HCI', 'log_GFCF', 'Political stability — estimate',
        'Rule of law index', 'log_Prod', 'Trade (% of GDP)',
        'NR rents', 'log_HCI_x_Prod', 'log_GFCF_x_Prod']
LABELS = {
    'log_HCI':                          'Human Capital (log)',
    'log_GFCF':                         'GFCF (log)',
    'Political stability — estimate':   'Political Stability',
    'Rule of law index':                'Rule of Law',
    'log_Prod':                         'NR Production (log)',
    'Trade (% of GDP)':                 'Trade',
    'NR rents':                         'NR Rents',
    'log_HCI_x_Prod':                   'HCI × Production',
    'log_GFCF_x_Prod':                  'GFCF × Production',
}

cf = coef[coef['VarLabel'].isin(KEEP)].copy()
cf['DisplayLabel'] = cf['VarLabel'].map(LABELS)

specs   = cf['Spec'].unique().tolist()
colors  = {s: SPEC_COLORS[i] for i, s in enumerate(specs)}
n_specs = len(specs)
offsets = {s: np.linspace(-0.2, 0.2, n_specs)[i] for i, s in enumerate(specs)}

var_order = [LABELS[k] for k in KEEP if k in LABELS]
y_pos     = {v: i for i, v in enumerate(var_order)}

fig01 = go.Figure()
fig01.add_vline(x=0, line=dict(color='#777', width=1.2, dash='dash'))

for spec in specs:
    sub = cf[cf['Spec'] == spec].copy()
    sub['y'] = sub['DisplayLabel'].map(y_pos) + offsets[spec]
    col = colors[spec]
    fig01.add_trace(go.Scatter(
        x=sub['Coef'], y=sub['y'],
        mode='markers',
        marker=dict(size=9, color=col, line=dict(color='white', width=1.2)),
        error_x=dict(
            type='data', symmetric=False,
            array=(sub['CI_hi'] - sub['Coef']).values,
            arrayminus=(sub['Coef'] - sub['CI_lo']).values,
            color=col, thickness=2, width=4,
        ),
        name=spec,
        hovertemplate='%{text}<br>Coef: %{x:.4f}<extra>' + spec + '</extra>',
        text=sub['DisplayLabel'],
    ))

fig01.update_layout(**base_layout(
    height=600,
    margin=dict(l=60, r=60, t=110, b=60),
    xaxis=dict(title='Coefficient (95% CI)', gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(
        tickvals=list(y_pos.values()),
        ticktext=list(y_pos.keys()),
        tickfont=dict(size=11),
        showgrid=True, gridcolor=GRID, gridwidth=0.5,
    ),
    legend=dict(orientation='h', yanchor='top', y=1.0,
                xanchor='center', x=0.5, font=dict(size=10),
                tracegroupgap=4),
    title=dict(
        text='Regression Robustness — NR Rents Coefficient<br>'
             '<sup>HAC-OLS (Newey-West, maxlags=2) · Dependent variable: ECI · 5 specifications</sup>',
        font=dict(size=14), x=0.5, y=0.98, yanchor='top',
    ),
))
save(fig01, '01_coef_comparison', OUT, w=1100, h=560)
print('✓ Chart 01')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 02 — R² grouped bar: 4 specs
# ══════════════════════════════════════════════════════════════════════════════
# Short spec labels for x-axis
_spec_short_defaults = [
    'Original (54)',
    'Adj. rent (54)',
    'Adj. ≥3%',
    'Adj. ≥2%',
    'Adj. ≥1%',
    'All countries',
]
spec_short = {row['Spec']: _spec_short_defaults[i]
              for i, (_, row) in enumerate(r2.iterrows())}

r2['SpecShort'] = r2['Spec'].map(spec_short).fillna(r2['Spec'])
r2['N_label']   = r2['N'].apply(lambda n: f'N={n:,}')

fig02 = go.Figure()
for metric, col, label in [('R2', '#4a6fa5', 'R²'), ('R2_adj', '#c23a3a', 'Adj. R²')]:
    fig02.add_trace(go.Bar(
        x=r2['SpecShort'], y=r2[metric],
        name=label, marker_color=col, opacity=0.88,
        text=[f'{v:.4f}' for v in r2[metric]],
        textposition='outside', textfont=dict(size=11, color=NAVY),
        hovertemplate='%{x}: %{y:.4f}<extra>' + label + '</extra>',
    ))

fig02.update_layout(**base_layout(
    barmode='group', height=440,
    margin=dict(l=60, r=60, t=90, b=80),
    xaxis=dict(tickfont=dict(size=11), tickangle=0),
    yaxis=dict(title='R²', gridcolor=GRID, gridwidth=0.5,
               range=[0, r2[['R2', 'R2_adj']].max().max() * 1.22]),
    legend=dict(orientation='h', yanchor='bottom', y=1.03,
                xanchor='center', x=0.5, font=dict(size=11)),
    title=dict(
        text='Regression Model Fit Across Sample Specifications<br>'
             '<sup>HAC-OLS (Newey-West) · Dependent variable: ECI</sup>',
        x=0.5, font=dict(size=13),
    ),
))
# N= annotation under each bar group
for i, row in r2.iterrows():
    fig02.add_annotation(
        x=row['SpecShort'], y=-0.06, xref='x', yref='paper',
        text=row['N_label'], showarrow=False,
        font=dict(size=10, color='#666'), align='center',
    )
save(fig02, '02_r2_comparison', OUT, w=900, h=420)
print('✓ Chart 02')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 03 — Choropleth map: sample membership
#   Original 54  — blue
#   In 3% sample — green (subset of 54-clean)
#   In 1% sample — teal  (wider)
#   Dropped      — red   (in original 54 but not in either adjusted sample)
# ══════════════════════════════════════════════════════════════════════════════
orig54  = set(INCLUDE_LIST)
s3      = set(ctry_3['Country Code'])
s2      = set(ctry_2['Country Code'])
s1      = set(ctry_1['Country Code'])

# Classification (hierarchical: broadest threshold takes precedence for membership)
def classify(cc):
    if cc in s3:
        return '≥ 3% (clean)'
    if cc in s2 and cc not in s3:
        return '2% only'
    if cc in s1 and cc not in s2:
        return '1% only'
    if cc in orig54:
        return 'Dropped (forest-driven)'
    return None

# All relevant countries
all_cc = orig54 | s1
name_lu = master[['Country Code', 'Country Name']].drop_duplicates().set_index('Country Code')['Country Name']

rows = []
for cc in all_cc:
    cat = classify(cc)
    if cat is None:
        continue
    rows.append({'Country Code': cc,
                 'Country Name': name_lu.get(cc, cc),
                 'Category': cat})
map_df = pd.DataFrame(rows)

# Colour map
cat_colors = {
    '≥ 3% (clean)':            '#2e7d4a',
    '2% only':                 '#e8a838',
    '1% only':                 '#3a8fa5',
    'Dropped (forest-driven)': '#c23a3a',
}
cat_order = ['≥ 3% (clean)', '2% only', '1% only', 'Dropped (forest-driven)']

fig03 = go.Figure()
for cat in cat_order:
    sub = map_df[map_df['Category'] == cat]
    if sub.empty:
        continue
    fig03.add_trace(go.Choropleth(
        locations=sub['Country Code'],
        z=[1] * len(sub),
        locationmode='ISO-3',
        colorscale=[[0, cat_colors[cat]], [1, cat_colors[cat]]],
        showscale=False,
        name=cat,
        text=sub['Country Name'],
        hovertemplate='<b>%{text}</b><br>' + cat + '<extra></extra>',
        marker_line_color='white', marker_line_width=0.5,
    ))

fig03.update_layout(**base_layout(
    height=480,
    margin=dict(l=0, r=0, t=70, b=20),
    geo=dict(
        showframe=False, showcoastlines=True,
        coastlinecolor='#ccc', landcolor='#f5f5f5',
        showocean=True, oceancolor='#eaf3fb',
        projection_type='natural earth',
    ),
    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                xanchor='center', x=0.5, font=dict(size=11)),
    title=dict(
        text='Sample Re-selection After Forest Rent Exclusion',
        font=dict(size=13), x=0.5,
    ),
))
save(fig03, '03_sample_map', OUT, w=1100, h=480)
print('✓ Chart 03')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 04 — Scatter: total NR rents vs adjusted rents (1995), labelled
# ══════════════════════════════════════════════════════════════════════════════
s95 = master[(master['Year'] == 1995)].dropna(subset=[RENT_COL, RENT_ADJ_COL]).copy()
s95['Category'] = s95['Country Code'].apply(
    lambda cc: '≥ 3% (clean)' if cc in s3
    else ('2% only' if cc in s2
    else ('1% only' if cc in s1
    else ('Dropped (forest-driven)' if cc in orig54
    else 'Not in any sample')))
)
s95 = s95[s95['Category'] != 'Not in any sample']
lim = [0, s95[RENT_COL].max() * 1.06]

fig04 = go.Figure()
# 45° identity line
fig04.add_trace(go.Scatter(
    x=lim, y=lim, mode='lines',
    line=dict(color='#aaa', width=1, dash='dash'),
    showlegend=False, hoverinfo='skip',
))

for cat in ['≥ 3% (clean)', '2% only', '1% only', 'Dropped (forest-driven)']:
    sub = s95[s95['Category'] == cat]
    col = cat_colors[cat]
    # label the most displaced (forest share > 30%)
    sub['forest_share'] = sub['Forest_Rents_pct'] / sub[RENT_COL].replace(0, np.nan)
    show_label = sub['forest_share'] > 0.30
    fig04.add_trace(go.Scatter(
        x=sub[RENT_COL], y=sub[RENT_ADJ_COL],
        mode='markers+text',
        marker=dict(size=8, color=col, opacity=0.85,
                    line=dict(color='white', width=0.8)),
        text=sub['Country Name'].where(show_label, ''),
        textposition='top center', textfont=dict(size=8),
        name=cat,
        customdata=sub[['Country Name', 'Forest_Rents_pct']].values,
        hovertemplate='<b>%{customdata[0]}</b><br>'
                      'Total: %{x:.1f}%<br>Adjusted: %{y:.1f}%<br>'
                      'Forest: %{customdata[1]:.1f}%<extra></extra>',
    ))

fig04.update_layout(**base_layout(
    height=520,
    margin=dict(l=70, r=50, t=80, b=60),
    xaxis=dict(title='Total NR rents % GDP (incl. forest)', range=lim,
               gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(title='Adjusted NR rents % GDP (excl. forest)', range=lim,
               gridcolor=GRID, gridwidth=0.5),
    legend=dict(orientation='h', yanchor='bottom', y=1.03,
                xanchor='center', x=0.5, font=dict(size=11)),
    title=dict(text='Total vs Adjusted NR Rents by Sample Category (1995)',
               font=dict(size=13), x=0.5),
))
save(fig04, '04_rent_adj_vs_orig_scatter', OUT, w=1000, h=520)
print('✓ Chart 04')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 05 — ML test R² across 4 samples × 4 models
# ══════════════════════════════════════════════════════════════════════════════
ml_r2  = pd.read_csv(os.path.join(OUT, 'ml_r2_comparison.csv'))
ml_imp = pd.read_csv(os.path.join(OUT, 'ml_feature_importance.csv'))

MODEL_COLORS = {
    'LASSO':         '#4a6fa5',
    'Ridge':         '#c23a3a',
    'Elastic Net':   '#2e7d4a',
    'Random Forest': '#d4853b',
}
SAMPLE_SHORT = {
    'A — Original 54 (NB4)':              'Original (54)',
    'B — Adj. rent, 3% sample (38 ctry)': 'Adj. ≥3%',
    'C — Adj. rent, 2% sample':           'Adj. ≥2%',
    'D — Adj. rent, 1% sample (58 ctry)': 'Adj. ≥1%',
    'E — All non-HIC countries':          'All non-HIC',
}
ml_r2['SampleShort'] = ml_r2['Sample'].map(SAMPLE_SHORT).fillna(ml_r2['Sample'])
sample_order = list(SAMPLE_SHORT.values())

fig05 = go.Figure()
for mname, mcol in MODEL_COLORS.items():
    sub = ml_r2[ml_r2['Model'] == mname]
    sub = sub.set_index('SampleShort').reindex(sample_order).reset_index()
    fig05.add_trace(go.Bar(
        x=sub['SampleShort'], y=sub['R2_test'],
        name=mname, marker_color=mcol, opacity=0.88,
        text=[f'{v:.3f}' if pd.notna(v) else '' for v in sub['R2_test']],
        textposition='outside', textfont=dict(size=10, color=NAVY),
        hovertemplate='%{x}<br>Test R²: %{y:.4f}<extra>' + mname + '</extra>',
    ))

fig05.update_layout(**base_layout(
    barmode='group', height=440,
    margin=dict(l=60, r=60, t=80, b=90),
    xaxis=dict(tickfont=dict(size=11)),
    yaxis=dict(title='Test R² (2015–2019)', gridcolor=GRID, gridwidth=0.5,
               range=[0, ml_r2['R2_test'].max() * 1.15]),
    legend=dict(orientation='h', yanchor='bottom', y=1.03,
                xanchor='center', x=0.5, font=dict(size=11)),
    title=dict(text='ML Out-of-Sample Performance Across Sample Definitions\n'
                    '(LASSO / Ridge / Elastic Net / RF — identical hyperparameters)',
               font=dict(size=13), x=0.5),
))
save(fig05, '05_ml_r2_comparison', OUT, w=1000, h=440)
print('✓ Chart 05')

# ── shared setup for charts 06-08 ────────────────────────────────────────────
SAMPLE_SHORT_06 = {k: v for k, v in SAMPLE_SHORT.items() if k != 'E — All non-HIC countries'}
s_colors_06     = ['#4a6fa5', '#2e7d4a', '#e8a838', '#3a8fa5']


def ml_importance_chart(metric_col, x_title, chart_title, fname, use_abs=False):
    """
    Grouped horizontal bar: top-10 features by mean |metric| across samples.
    use_abs=True  → rank and display absolute values (for signed coef metrics)
    use_abs=False → rank and display raw values (for RF importance, always ≥0)
    """
    imp_col = ml_imp[metric_col].abs() if use_abs else ml_imp[metric_col]
    feat_mean = (imp_col.groupby(ml_imp['Feature_short'])
                 .mean().sort_values(ascending=False).head(10))
    top10_f = feat_mean.index.tolist()

    fig = go.Figure()
    for i, (skey, slabel) in enumerate(SAMPLE_SHORT_06.items()):
        sub_df = ml_imp[ml_imp['Sample'] == skey].set_index('Feature_short')
        if use_abs:
            vals = [abs(sub_df[metric_col].get(f, 0)) for f in top10_f]
        else:
            vals = [sub_df[metric_col].get(f, 0) for f in top10_f]
        fig.add_trace(go.Bar(
            y=top10_f[::-1], x=[vals[j] for j in range(len(top10_f)-1, -1, -1)],
            orientation='h',
            name=slabel.replace('\n', ' '),
            marker_color=s_colors_06[i], opacity=0.82,
            hovertemplate='%{y}: %{x:.4f}<extra>' + slabel.replace('\n', ' ') + '</extra>',
        ))

    fig.update_layout(**base_layout(
        barmode='group', height=520,
        margin=dict(l=130, r=60, t=80, b=60),
        xaxis=dict(title=x_title, gridcolor=GRID, gridwidth=0.5),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation='h', yanchor='bottom', y=1.03,
                    xanchor='center', x=0.5, font=dict(size=10)),
        title=dict(text=chart_title, font=dict(size=13), x=0.5),
    ))
    save(fig, fname, OUT, w=1000, h=520)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 06 — Random Forest feature importance
# ══════════════════════════════════════════════════════════════════════════════
ml_importance_chart(
    metric_col  = 'RF_imp',
    x_title     = 'RF Feature Importance',
    chart_title = 'Top-10 RF Feature Importances — Stability Across Sample Definitions',
    fname       = '06_ml_feature_importance_rf',
    use_abs     = False,
)
print('✓ Chart 06 (RF)')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 07 — LASSO coefficients (absolute value)
# ══════════════════════════════════════════════════════════════════════════════
ml_importance_chart(
    metric_col  = 'LASSO_coef',
    x_title     = '|LASSO Coefficient|',
    chart_title = 'Top-10 LASSO Feature Coefficients (Absolute) — Stability Across Sample Definitions',
    fname       = '07_ml_feature_importance_lasso',
    use_abs     = True,
)
print('✓ Chart 07 (LASSO)')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 08 — Elastic Net coefficients (absolute value)
# ══════════════════════════════════════════════════════════════════════════════
ml_importance_chart(
    metric_col  = 'EN_coef',
    x_title     = '|Elastic Net Coefficient|',
    chart_title = 'Top-10 Elastic Net Feature Coefficients (Absolute) — Stability Across Sample Definitions',
    fname       = '08_ml_feature_importance_en',
    use_abs     = True,
)
print('✓ Chart 08 (Elastic Net)')

print('\n✓ forest_robustness_viz.py complete — charts in robustness-forest/outputs/')
