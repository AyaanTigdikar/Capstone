# Auto-extracted from 6_Regressions_Unified.ipynb
import matplotlib; matplotlib.use('Agg')
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # Descriptive Statistics & Regressions — Economic Complexity and Natural Resources
# 
# **Moody's Ratings Capstone — Industrial Upgrading in Emerging Markets**
# **NB6 of 6 · Pipeline Step: Econometric Analysis (Unified)**
# 
# Dependent variable: Economic Complexity Index (ECI, raw), 1995–2019.
# Sample: 54 resource-dependent developing countries (NR rents ≥ 5 % of GDP in 1995).
# Inputs: `intermediary/Master.csv`, `intermediary/clusters1995.csv` (from NB3 and NB4).
# 
# ---
# 
# ### Changes from the previous regression notebook
# 
# This notebook consolidates and revises the earlier regression analysis in
# `Descriptive_Statistics_and_Regressions_(clean_code).ipynb`. Changes are methodological,
# not merely organisational.
# 
# **1. Dependent variable: raw ECI throughout.**
# The old notebook uses `log_ECI = log(ECI - min + 1)` for the kitchen-sink and AR models,
# then switches to raw ECI levels for the interaction specifications. The shift-and-log
# transformation compresses the distribution and produces coefficients with no direct
# interpretation. NB6 uses raw ECI in every model.
# 
# **2. Lagged predictor in AR model corrected.**
# The old AR baseline regresses `log_ECI` on `log_ECI_lag1` — a lagged log of the shifted
# variable. NB6 regresses raw ECI on `ECI_lag1`. This is consistent with change 1 and
# removes the implicit unit-change interpretation problem from the lag coefficient.
# 
# **3. Kitchen-sink SE clustering corrected: K-means groups → countries.**
# The old Model 1 passes `groups=Cluster` (4 K-means IDs) to the clustered sandwich
# estimator. With only 4 groups the estimator is degenerate regardless of any small-sample
# correction. NB6 clusters by `Country Code` (54 groups). The kitchen-sink is still
# over-parameterised and should not be used for inference, but at least the clustering
# dimension is meaningful.
# 
# **4. Standard errors for core models: Driscoll-Kraay replaces country-clustered.**
# The old notebook uses `cov_type='cluster'` (country-level) in all regressions. This
# corrects serial correlation within countries but not cross-sectional dependence from
# common shocks (commodity price cycles). Models 3a and 3b use Driscoll-Kraay SEs
# (`HAC-Groupsum`, Bartlett kernel, bandwidth = 2 = floor(25^0.25)). Models 1 and 2
# retain clustered SEs. A DK-vs-clustered comparison toggle (`SHOW_CLUSTERED_SE_COMPARISON`)
# is included for robustness checks.
# 
# **5. Systematic mean-centring of interaction terms.**
# The old notebook centres logged variables before computing interactions only in the
# "no log variables" specification, not in the lagged-all-variables model. NB6 applies
# grand-mean centring consistently in every interaction model, so main-effect coefficients
# are always interpretable at the sample mean of the interacted variable.
# 
# **6. Residual diagnostics added.**
# NB6 adds QQ plots and Breusch-Pagan tests for Models 3a and 3b. DK SEs are robust to
# heteroskedasticity and serial correlation, so these diagnostics do not invalidate
# inference but are reported for econometric credibility.
# 
# **7. Full-sample comparison added.**
# NB6 re-estimates Models 3a and 3b on the full `Master.csv` (all available countries)
# and compares coefficients. Interaction terms are mean-centred separately for each
# sample; magnitudes are not directly comparable but sign and significance are.
# 
# **8. `Landlocked` removed from Model 1.**
# Time-invariant in pooled OLS without country fixed effects, so its coefficient
# conflates geography with all unobserved between-country differences.
# 
# **9. Dropped exploratory specifications.**
# 
# - *All-variables-lagged*: lags every structural regressor by one year, halving the
#   usable sample with no theoretical justification for uniform one-year delays. Lagged
#   ECI alone is retained as a persistence control (Model 3b).
# - *Resource-type dummies* (`Hydrocarbons_Dominant`, `Subsoil_Metals_Dominant`,
#   `Precious_Metals_Dominant`): collinear with per-capita production value; the
#   interaction term already captures how resource type modifies the complexity return.
# - *Delta-ECI regression*: equivalent to Model 3b with the lag coefficient constrained
#   to 1. Model 3b estimates that coefficient freely.
# 
# **10. Descriptive statistics section restructured.**
# The old notebook included a thematic variable grouping table, an ECI mean/IQR
# time-series chart (noted inline as "not useful because ECI is normalised each year"),
# a correlation heatmap grouped by theme, resource-type summary stats, and a top-10
# winners/losers table. NB6 replaces these with four focused charts tied directly to the
# regression: ECI distribution shift (1995 vs 2019), median ECI trajectory by cluster,
# correlation matrix for regression variables, and the HCI-production quartile scatter.
# The winners/losers table and resource-type summary are dropped.
# 
# **11. HCI-ECI median-split chart replaced.**
# The old notebook fit separate linear slopes for above/below-median production per
# capita groups and exported two static charts. NB6 replaces this with a
# production-quartile-coloured scatter (section 7b), which conveys the same interaction
# hypothesis without an arbitrary median-split model.
# 
# **12. Model consolidation and portable paths.**
# The old notebook had approximately eight regression cells with duplicated variable
# lists and data-cleaning steps, and hardcoded Windows paths
# (`C:/Users/emili/OneDrive/...`). NB6 consolidates to four numbered models using shared
# helper functions and reads from relative paths under `intermediary/`.
# 
# ---
# 
# ### Specification summary
# 
# | Model | Regressors | SE type | Purpose |
# |---|---|---|---|
# | **Model 1** | 44 controls (kitchen-sink) | Clustered (country) | Sign-checking only |
# | **Model 2** | Lagged ECI only | Clustered (country) | Persistence benchmark |
# | **Model 3a** | 6 vars + 2 interactions | Driscoll-Kraay | Core specification (no lag) |
# | **Model 3b** | 6 vars + 2 interactions + lagged ECI | Driscoll-Kraay | Core specification (with lag) |

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.stattools import durbin_watson
import scipy.stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from types import SimpleNamespace
warnings.filterwarnings('ignore')

# ── Output directory ─────────────────────────────────────────────────────────
OUT = os.path.join('Final', 'NB6')
os.makedirs(OUT, exist_ok=True)

# ── Shared style (matches NB5) ───────────────────────────────────────────────
STYLE = {
    'font_family':      'IBM Plex Sans, -apple-system, BlinkMacSystemFont, sans-serif',
    'tick_size':        11,
    'axis_title_size':  13,
    'legend_size':      11,
    'annotation_size':  12,
    'title_color':      '#1a2744',
    'template':         'plotly_white',
    'plot_bg':          '#fafafa',
    'paper_bg':         '#fafafa',
    'chart_height':     550,
    'chart_height_tall':700,
    'margin':           dict(l=60, r=40, t=10, b=50),
    'margin_bar':       dict(l=220, r=100, t=10, b=50),
    'grid_color':       '#e5e7eb',
    'grid_width':       0.5,
    'zero_line_color':  '#c9cfd6',
}

PALETTE = {
    'blue':        '#4a6fa5',
    'red':         '#c23a3a',
    'green':       '#2e7d4a',
    'orange':      '#d4853b',
    'light_blue':  '#7a9dc4',
    'light_red':   '#d46b6b',
    'light_green': '#5aa87a',
    'dark':        '#3d4f5f',
    'grey':        '#999999',
    'gold':        '#e6b980',
}

CLUSTER_COLORS = ['#4a6fa5', '#c23a3a', '#2e7d4a', '#d4853b']

WRITE_CONFIG = {'displayModeBar': False, 'responsive': True}

plt.rcParams.update({
    'font.family':       'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        120,
})


def base_layout(**kwargs) -> dict:
    layout = dict(
        template=STYLE['template'],
        plot_bgcolor=STYLE['plot_bg'],
        paper_bgcolor=STYLE['paper_bg'],
        font=dict(family=STYLE['font_family'], size=STYLE['tick_size'],
                  color=STYLE['title_color']),
        margin=STYLE['margin'],
        height=STYLE['chart_height'],
    )
    layout.update(kwargs)
    return layout


_WRITE_CONFIG = {'displayModeBar': False, 'responsive': True}

def save_chart(fig, path_no_ext: str, width: int = 1100, height: int = 600):
    """Write HTML + PNG for a Plotly figure."""
    os.makedirs(os.path.dirname(path_no_ext) or '.', exist_ok=True)
    fig.write_html(f"{path_no_ext}.html", config=_WRITE_CONFIG)
    print(f"  Saved: {path_no_ext}.html")
    try:
        fig.write_image(f"{path_no_ext}.png", width=width, height=height, scale=2)
        print(f"  Saved: {path_no_ext}.png")
    except Exception as e:
        print(f"  PNG skipped ({e})")

# ## 0. Data Loading
# 
# The master panel was assembled in NB1-NB3: NB1 merged raw sources (World Bank WDI,
# Penn World Table, V-Dem, Energy Institute Statistical Review, USGS mineral data),
# NB2 diagnosed missingness patterns, and NB3 imputed gaps using three-year linear
# interpolation (capped) followed by KNN for remaining holes.
# 
# The 54-country sample is defined by a **baseline threshold**: total natural resource
# rents ≥ 5 % of GDP in the year 1995. The threshold is applied only at the baseline to
# avoid endogeneity. If it were applied at every time period, countries that successfully
# diversified away from resources would exit the sample, biasing estimates.
# 
# Cluster assignments come from NB4, where PCA on per-capita log-transformed production
# data was followed by K-means clustering (k = 4). These clusters group countries by
# resource profile (broadly: hydrocarbon-dominated, subsoil-metals, mixed mineral,
# low-production) and are used only for descriptive purposes in this notebook.

# ── Load pipeline outputs ─────────────────────────────────────────────────────
master       = pd.read_csv('intermediary/Master.csv')
cluster_1995 = pd.read_csv('intermediary/clusters1995.csv')

print(f"Master:   {len(master):,} obs, {master['Country Code'].nunique()} countries, "
      f"{master['Year'].nunique()} years ({int(master['Year'].min())}–{int(master['Year'].max())})")
print(f"Clusters: {len(cluster_1995)} country assignments")

# ── 54-country sample (same include_list as NB4) ──────────────────────────────
INCLUDE = [
    'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
    'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
    'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
    'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
    'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
    'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
]

df = master[master['Country Code'].isin(INCLUDE)].copy()

# ── Per-capita production/reserves values ─────────────────────────────────────
df['Total_Production_Value_Per_Capita'] = df['Total_Production_Value'] / df['Population']
df['Total_Reserves_Value_Per_Capita']   = df['Total_Reserves_Value']   / df['Population']

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# ── Merge cluster labels (1995 assignments, time-invariant) ───────────────────
cluster_1995 = cluster_1995[['Country Code', 'Cluster']]
df = df.merge(cluster_1995, on='Country Code', how='left')

print(f"\nSample: {df['Country Code'].nunique()} countries × {df['Year'].nunique()} years "
      f"= {len(df):,} obs")
print(f"ECI range: {df['Economic Complexity Index'].min():.3f} – "
      f"{df['Economic Complexity Index'].max():.3f}")

df.to_csv('intermediary/high_resource_countries.csv', index=False)
print(f"  ✓ intermediary/high_resource_countries.csv")

# ## 1. Descriptive Statistics
# 
# Summary statistics for key variables across the 54-country sample.

# ── Key variables for descriptive table ──────────────────────────────────────
DESC_VARS = {
    'Economic Complexity Index': 'ECI',
    'Human capital index':                          'Human capital index',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'GFCF (% GDP)',
    'Domestic credit to private sector (% of GDP)': 'Domestic credit (% GDP)',
    'Access to electricity (% of population)':      'Electricity access (%)',
    'Rule of law index':                            'Rule of law',
    'Political stability — estimate':               'Political stability',
    'Total natural resources rents (% of GDP)':     'NR rents (% GDP)',
    'Total_Production_Value_Per_Capita':            'Prod. value per capita (USD)',
    'Trade (% of GDP)':                             'Trade (% GDP)',
}

desc = df[list(DESC_VARS.keys())].rename(columns=DESC_VARS)
print(desc.describe().round(3).to_string())
print(f"\nN obs: {len(df):,}  |  N countries: {df['Country Code'].nunique()}")

# ### 1a. ECI Distribution Change: 1995 vs 2019

yr_95 = df[df['Year'] == 1995]['Economic Complexity Index'].dropna().sort_values().values
yr_19 = df[df['Year'] == 2019]['Economic Complexity Index'].dropna().sort_values().values

fig = go.Figure()
for vals, yr, col in [(yr_95, 1995, PALETTE['blue']), (yr_19, 2019, PALETTE['red'])]:
    pcts = np.linspace(0, 100, len(vals))
    fig.add_trace(go.Scatter(
        x=pcts, y=vals,
        mode='lines', name=str(yr),
        line=dict(color=col, width=2.5),
    ))

fig.update_layout(**base_layout(
    height=STYLE['chart_height'],
    xaxis=dict(
        title=dict(text='Percentile', font=dict(size=STYLE['axis_title_size'])),
        gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
    ),
    yaxis=dict(
        title=dict(text='Economic Complexity Index', font=dict(size=STYLE['axis_title_size'])),
        gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
        zeroline=True, zerolinecolor=STYLE['zero_line_color'], zerolinewidth=1,
    ),
    legend=dict(font=dict(size=STYLE['legend_size'])),
))

save_chart(fig, os.path.join(OUT, 'eci_distribution_comparison'))

# ### 1b. Median ECI Trajectory by Resource-Profile Cluster

traj = (df.groupby(['Year', 'Cluster'])['Economic Complexity Index']
          .median().reset_index())

clusters_present = sorted(traj['Cluster'].dropna().unique())

fig = go.Figure()
for i, cl in enumerate(clusters_present):
    sub = traj[traj['Cluster'] == cl]
    fig.add_trace(go.Scatter(
        x=sub['Year'], y=sub['Economic Complexity Index'],
        mode='lines+markers', name=str(cl),
        line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], width=2),
        marker=dict(size=5),
    ))

fig.update_layout(**base_layout(
    xaxis=dict(
        title=dict(text='Year', font=dict(size=STYLE['axis_title_size'])),
        gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
        dtick=5,
    ),
    yaxis=dict(
        title=dict(text='Median ECI', font=dict(size=STYLE['axis_title_size'])),
        gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
        zeroline=True, zerolinecolor=STYLE['zero_line_color'], zerolinewidth=1,
    ),
    legend=dict(title=dict(text='Cluster'), font=dict(size=STYLE['legend_size'])),
))

save_chart(fig, os.path.join(OUT, 'eci_cluster_trajectories'))

# ### 1c. Correlation Matrix — Key Variables

corr_cols = list(DESC_VARS.keys())
corr_df   = df[corr_cols].rename(columns=DESC_VARS).corr().round(2)

labels = list(corr_df.columns)
z      = corr_df.values

fig = go.Figure(go.Heatmap(
    z=z, x=labels, y=labels,
    colorscale=[
        [0.0, PALETTE['red']], [0.5, '#fafafa'], [1.0, PALETTE['blue']]
    ],
    zmid=0, zmin=-1, zmax=1,
    text=z.round(2), texttemplate='%{text}',
    textfont=dict(size=9, family=STYLE['font_family']),
    hovertemplate='%{x} × %{y}: %{z:.2f}<extra></extra>',
    colorbar=dict(thickness=14, len=0.9,
                  tickfont=dict(size=STYLE['tick_size'],
                                family=STYLE['font_family'])),
))

fig.update_layout(**base_layout(
    height=620,
    margin=dict(l=180, r=60, t=10, b=180),
    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
    yaxis=dict(tickfont=dict(size=9)),
))

save_chart(fig, os.path.join(OUT, 'eci_correlation_heatmap'), width=900, height=700)

# ## 2. Regression Setup
# 
# Four OLS specifications are estimated. All use **raw ECI** as the dependent variable.
# None include country or year fixed effects; the pooled design captures both
# between-country and within-country variation.
# 
# **Standard errors.**
# Models 1 and 2 use clustered standard errors at the country level (54 groups).
# Models 3a and 3b use **Driscoll-Kraay** standard errors (`HAC-Groupsum`, Bartlett
# kernel, bandwidth = 2), which are robust to both within-country serial correlation
# and cross-country dependence within the same year.
# 
# **On including lagged ECI.**
# Model 3b adds last year's ECI as a right-hand-side variable. This controls for the
# strong persistence in economic complexity: because ECI barely moves from year to year,
# the remaining coefficients in 3b reflect associations with the *change* in ECI
# conditional on last year's level. If the interaction coefficients are stable between
# 3a and 3b, they capture structural relationships rather than proxying for omitted
# dynamic factors.
# 
# Note that this approach is more flexible than regressing the year-on-year change
# (ΔECI = ECI_t minus ECI_{t-1}) directly on the regressors. The delta-ECI approach
# implicitly constrains the lagged-ECI coefficient to equal exactly 1. Model 3b
# estimates that coefficient freely from the data.

# ── Full variable list (Model 1) ──────────────────────────────────────────────
# Note: 'Landlocked' removed. It is time-invariant and conflates geography with
# other between-country variation in pooled OLS.
INDEP_VARS = [
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
    'Total_Production',
    'Total_Reserves',
    'Total_Production_Value',
    'Total_Reserves_Value',
    'Total_Production_Value_Per_Capita',
    'Total_Reserves_Value_Per_Capita',
    'Hydrocarbons_Dominant',
    'Subsoil_Metals_Dominant',
    'Precious_Metals_Dominant',
    'Population',
]

# ── Parsimonious variable list (Models 3a, 3b) ───────────────────────────────
# Selected on theoretical grounds from the resource-curse literature:
#   - Human capital and institutional quality: most-cited channels for
#     resource wealth affecting diversification (Gylfason 2001, Mehlum et al. 2006)
#   - Per-capita production value: resource extraction intensity relative to
#     population size, closer to the standard resource-dependence measure
#   - Trade openness: exposure to international competition and learning
PARSIMONIOUS_VARS = [
    'Human capital index',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP',
    'Political stability — estimate',
    'Rule of law index',
    'Total_Production_Value_Per_Capita',
    'Trade (% of GDP)',
]

# ── Log transforms ────────────────────────────────────────────────────────────
# log1p(x) = ln(1 + x) handles zeros safely (ln(0) is undefined).
# Applied to HCI, GFCF, and per-capita production value because these span
# several orders of magnitude; the log reduces outlier influence and allows
# coefficients to be interpreted in terms of proportional changes.
df['log_HCI']              = np.log1p(df['Human capital index'])
df['log_GFCF']             = np.log1p(df['Gross fixed capital formation, all, Constant prices, Percent of GDP'])
df['log_Production_Value'] = np.log1p(df['Total_Production_Value_Per_Capita'])

# ── Mean-centred logs for interactions ────────────────────────────────────────
# Centring subtracts the grand (sample-wide) mean from each log-transformed
# variable before computing the product. This reduces multicollinearity between
# the interaction term and its constituent main effects. After centring, the
# main-effect coefficients are interpretable as the association at the sample
# mean of the interacted variable.
for col in ['log_HCI', 'log_GFCF', 'log_Production_Value']:
    df[f'{col}_c'] = df[col] - df[col].mean()

df['log_HCI_x_log_Production']  = df['log_HCI_c']  * df['log_Production_Value_c']
df['log_GFCF_x_log_Production'] = df['log_GFCF_c'] * df['log_Production_Value_c']

# ── Lagged ECI (raw) ─────────────────────────────────────────────────────────
df = df.sort_values(['Country Code', 'Year'])
df['ECI_lag1'] = df.groupby('Country Code')['Economic Complexity Index'].shift(1)

print("Variables ready.")
print(f"  Full set (Model 1):        {len(INDEP_VARS)} vars")
print(f"  Parsimonious (Models 3a/b): {len(PARSIMONIOUS_VARS)} vars + 2 interactions")
print(f"  Lagged ECI available:       {df['ECI_lag1'].notna().sum():,} obs")

# ## 3. Model 1 — Pooled OLS, Full Variable Set
# 
# **Note on over-parameterisation:** With 44 regressors and 54 country clusters, the clustered covariance matrix approaches rank deficiency. Treat this as a sign-checking reference only. The clustered SEs are unreliable in this configuration and should not be used for inference. If reviewers challenge this specification, the defensible response is either to drop it entirely or reduce to a 15–20 variable intermediate specification motivated by the same theoretical channels as Models 3a/3b.
# 
# All 44 controls, clustered standard errors (by country). Sign-checking reference only.
# 

reg1_cols = INDEP_VARS + ['Economic Complexity Index', 'Country Code']
reg1_df   = df[reg1_cols].dropna()

y1 = reg1_df['Economic Complexity Index']
X1 = sm.add_constant(reg1_df[INDEP_VARS])

m1 = sm.OLS(y1, X1).fit(
    cov_type='cluster',
    cov_kwds={'groups': reg1_df['Country Code']},
)

print("=" * 70)
print("MODEL 1 — Pooled OLS, Full Variable Set (Clustered SE by Country)")
print("=" * 70)
print(f"  N obs:        {int(m1.nobs):,}")
print(f"  N countries:  {reg1_df['Country Code'].nunique()}")
print(f"  R²:           {m1.rsquared:.4f}")
print(f"  Adj. R²:      {m1.rsquared_adj:.4f}")
print(f"  Durbin-Watson: {durbin_watson(m1.resid):.3f}")
print()
print(f"{'Variable':<52} {'Coef':>9} {'SE':>9} {'t':>7} {'p':>7}")
print("-" * 90)
for v in [c for c in m1.params.index if c != 'const']:
    coef, se, t, p = m1.params[v], m1.bse[v], m1.tvalues[v], m1.pvalues[v]
    sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
    if p < 0.1:
        print(f"{v:<52} {coef:>9.4f} {se:>9.4f} {t:>7.2f} {p:>7.4f} {sig}")

# ## 4. Model 2 — AR Baseline (Lagged ECI)
# 
# Autoregressive baseline: only lagged ECI as predictor. The high R² (typically ~0.97)
# reflects the strong persistence of economic complexity. A country's ECI today is
# almost entirely determined by its ECI last year. This baseline matters because any
# model claiming to explain ECI must be benchmarked against this inherent persistence.

reg2_cols = ['Economic Complexity Index', 'ECI_lag1', 'Country Code']
reg2_df   = df[reg2_cols].dropna()

y2 = reg2_df['Economic Complexity Index']
X2 = sm.add_constant(reg2_df[['ECI_lag1']])

m2 = sm.OLS(y2, X2).fit(
    cov_type='cluster',
    cov_kwds={'groups': reg2_df['Country Code']},
)

print("=" * 70)
print("MODEL 2 — AR Baseline: Lagged ECI (Clustered SE by Country)")
print("=" * 70)
print(f"  N obs:        {int(m2.nobs):,}")
print(f"  N countries:  {reg2_df['Country Code'].nunique()}")
print(f"  R²:           {m2.rsquared:.4f}")
print(f"  Adj. R²:      {m2.rsquared_adj:.4f}")
print(f"  Durbin-Watson: {durbin_watson(m2.resid):.3f}")
print()
for v in m2.params.index:
    coef, se, t, p = m2.params[v], m2.bse[v], m2.tvalues[v], m2.pvalues[v]
    sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
    print(f"  {v:<30} coef={coef:>8.4f}  SE={se:>8.4f}  t={t:>7.2f}  p={p:>7.4f} {sig}")

# ## 5. Model 3 — Interaction Model (HCI × Production, GFCF × Production)
# 
# Parsimonious specification with two interaction terms, estimated with and without
# lagged ECI as a control. Variables are mean-centred (grand mean) before computing
# interactions to reduce multicollinearity.
# 
# **Model 3a**: without lagged ECI. Captures the total cross-sectional and longitudinal
# association between the regressors and ECI.
# 
# **Model 3b**: with lagged ECI. Coefficients now reflect associations with ECI
# conditional on last year's level. If the interaction coefficients are stable across
# 3a and 3b, they capture structural relationships rather than proxying for omitted
# dynamic factors.
# 
# Driscoll-Kraay standard errors (HAC-Groupsum, Bartlett kernel, bandwidth = 2).

# ── Driscoll-Kraay fitting helper ────────────────────────────────────────────
INTERACT_VARS = ['log_HCI_x_log_Production', 'log_GFCF_x_log_Production']

reg3_input = ['log_HCI', 'log_GFCF', 'Political stability — estimate',
              'Rule of law index', 'log_Production_Value', 'Trade (% of GDP)']


def fit_driscoll_kraay(y, X, time, groups, label=''):
    """Fit OLS with Driscoll-Kraay SEs. Returns (SimpleNamespace, raw_fit)."""
    raw = sm.OLS(y, X).fit()
    robust = raw.get_robustcov_results(
        cov_type='HAC-Groupsum',
        time=time, groups=groups,
        maxlags=2,  # floor(T^(1/4)) = floor(25^0.25) ≈ 2.2 → bw=2 is defensible
        kernel='bartlett', use_correction=True,
    )

    ns = SimpleNamespace(
        params       = pd.Series(robust.params,  index=X.columns),
        bse          = pd.Series(robust.bse,     index=X.columns),
        tvalues      = pd.Series(robust.tvalues, index=X.columns),
        pvalues      = pd.Series(robust.pvalues, index=X.columns),
        nobs         = robust.nobs,
        rsquared     = robust.rsquared,
        rsquared_adj = robust.rsquared_adj,
        cov_params   = robust.cov_params,
    )

    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")
    print(f"  N obs:        {int(ns.nobs):,}")
    print(f"  N countries:  {groups.nunique()}")
    print(f"  R²:           {ns.rsquared:.4f}")
    print(f"  Adj. R²:      {ns.rsquared_adj:.4f}")
    print(f"  Durbin-Watson: {durbin_watson(raw.resid):.3f}")
    print()
    print(f"  {'Variable':<43} {'Coef':>9} {'SE':>9} {'t':>7} {'p':>7}")
    print("  " + "-" * 76)
    for v in ns.params.index:
        coef, se, t, p = ns.params[v], ns.bse[v], ns.tvalues[v], ns.pvalues[v]
        sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f"  {v:<43} {coef:>9.4f} {se:>9.4f} {t:>7.2f} {p:>7.4f} {sig}")
    return ns, raw


# ── Model 3a: without lagged ECI ─────────────────────────────────────────────
reg3_cols  = reg3_input + INTERACT_VARS + ['Economic Complexity Index', 'Country Code', 'Year']
reg3_df    = df[reg3_cols].dropna()

y3a = reg3_df['Economic Complexity Index']
X3a = sm.add_constant(reg3_df[reg3_input + INTERACT_VARS])

m3a, m3a_raw = fit_driscoll_kraay(
    y3a, X3a, reg3_df['Year'], reg3_df['Country Code'],
    label='MODEL 3a — Interaction Model, NO lag (Driscoll-Kraay SE)')


# ── Model 3b: with lagged ECI ────────────────────────────────────────────────
reg3b_cols = reg3_cols + ['ECI_lag1']
reg3b_df   = df[reg3b_cols].dropna()

y3b = reg3b_df['Economic Complexity Index']
X3b = sm.add_constant(reg3b_df[reg3_input + INTERACT_VARS + ['ECI_lag1']])

m3b, m3b_raw = fit_driscoll_kraay(
    y3b, X3b, reg3b_df['Year'], reg3b_df['Country Code'],
    label='MODEL 3b — Interaction Model, WITH lag (Driscoll-Kraay SE)')

# ── Quick comparison ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON: Model 3a vs 3b")
print("=" * 70)
print(f"  {'':30} {'3a (no lag)':>14} {'3b (with lag)':>14}")
print(f"  {'R²':<30} {m3a.rsquared:>14.4f} {m3b.rsquared:>14.4f}")
for v in reg3_input + INTERACT_VARS:
    c_a = m3a.params.get(v, np.nan)
    c_b = m3b.params.get(v, np.nan)
    s_a = '***' if m3a.pvalues.get(v,1)<0.01 else '**' if m3a.pvalues.get(v,1)<0.05 else '*' if m3a.pvalues.get(v,1)<0.1 else ''
    s_b = '***' if m3b.pvalues.get(v,1)<0.01 else '**' if m3b.pvalues.get(v,1)<0.05 else '*' if m3b.pvalues.get(v,1)<0.1 else ''
    print(f"  {v:<30} {c_a:>+10.4f}{s_a:<4} {c_b:>+10.4f}{s_b:<4}")
if 'ECI_lag1' in m3b.params:
    c_lag = m3b.params['ECI_lag1']
    s_lag = '***' if m3b.pvalues['ECI_lag1']<0.01 else '**' if m3b.pvalues['ECI_lag1']<0.05 else '*' if m3b.pvalues['ECI_lag1']<0.1 else ''
    print(f"  {'ECI_lag1':<30} {'':>14} {c_lag:>+10.4f}{s_lag:<4}")

# ── Clustered SE comparison (toggle) ────────────────────────────────────────
# Set SHOW_CLUSTERED_SE_COMPARISON = True to see how results change when
# using simpler country-clustered SEs instead of Driscoll-Kraay.
# DK is preferred (robust to cross-sectional dependence from commodity shocks)
# but reviewers may ask for this comparison to verify robustness.
SHOW_CLUSTERED_SE_COMPARISON = False

if SHOW_CLUSTERED_SE_COMPARISON:
    def _fmt(coef, se, pval):
        s = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        return f'{coef:>+9.4f} ({se:.4f}){s}'

    print('\n' + '=' * 80)
    print('SE ROBUSTNESS: Driscoll-Kraay vs Country-Clustered')
    print('=' * 80)
    for y_r, X_r, df_r, m_dk, spec in [
        (y3a, X3a, reg3_df,  m3a, '3a (no lag)'),
        (y3b, X3b, reg3b_df, m3b, '3b (with lag)'),
    ]:
        m_cl = sm.OLS(y_r, X_r).fit(
            cov_type='cluster', cov_kwds={'groups': df_r['Country Code']}
        )
        print(f'\nModel {spec}')
        print(f"  {'Variable':<40} {'DK SE':>26} {'Clustered SE':>26}")
        print('  ' + '-' * 95)
        for v in [c for c in m_dk.params.index if c != 'const']:
            dk_str  = _fmt(m_dk.params[v], m_dk.bse[v], m_dk.pvalues[v])
            cl_str  = _fmt(m_cl.params[v], m_cl.bse[v], m_cl.pvalues[v])
            print(f'  {v:<40} {dk_str:>26} {cl_str:>26}')


# ── Residual diagnostics — Models 3a / 3b ────────────────────────────────────
# Durbin-Watson is reported in the summary but does not diagnose
# heteroskedasticity or non-normality. Added:
#   1. QQ plot of residuals (visual normality check)
#   2. Breusch-Pagan test (formal heteroskedasticity test)
# Note: Driscoll-Kraay SEs are robust to both heteroskedasticity and serial
# correlation, so finding either does not invalidate the inference — but
# reporting these diagnostics strengthens econometric credibility.
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as stats

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, m_raw, label in [
    (axes[0], m3a_raw, 'Model 3a (no lag)'),
    (axes[1], m3b_raw, 'Model 3b (with lag)'),
]:
    resid = m_raw.resid
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist='norm')
    ax.scatter(osm, osr, s=18, alpha=0.6, color=PALETTE['blue'], edgecolor='none')
    ax.plot(
        [osm[0], osm[-1]],
        [osm[0] * slope + intercept, osm[-1] * slope + intercept],
        color=PALETTE['red'], linewidth=1.5, linestyle='--', label='Normal reference'
    )
    ax.set_xlabel('Theoretical quantiles', fontsize=11)
    ax.set_ylabel('Ordered residuals', fontsize=11)
    ax.set_title(f'Q-Q Plot — {label}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, linestyle='--')

plt.tight_layout()
plt.close("all")

# ── Breusch-Pagan test ────────────────────────────────────────────────────────
print('\nBreusch-Pagan test (H0: homoskedastic):')
for m_raw, X_reg, label in [
    (m3a_raw, X3a, 'Model 3a'),
    (m3b_raw, X3b, 'Model 3b'),
]:
    lm_stat, lm_pval, _, _ = het_breuschpagan(m_raw.resid, X_reg)
    verdict = 'reject H0 — heteroskedastic' if lm_pval < 0.05 else 'fail to reject H0'
    print(f'  {label}: LM={lm_stat:.3f}  p={lm_pval:.4f}  → {verdict}')
print('  (Driscoll-Kraay SEs remain valid regardless of this result)')


# ## 6. Sample Comparison — 54 Resource-Rich vs All Countries
# 
# The 54-country sample was selected based on NR rents ≥ 5 % of GDP in 1995.
# This section re-estimates Models 3a and 3b on the **full Master dataset** (all
# available countries) and compares coefficients. The purpose is to assess whether
# the interaction effects are specific to resource-dependent economies or hold more
# broadly.
# 
# **Note on centring**: the mean-centring of interaction terms is computed separately
# for each sample (using the respective grand mean), so interaction coefficients are
# not directly comparable in magnitude across samples. They are comparable in sign
# and significance.

# ── Prepare the full-sample dataframe ─────────────────────────────────────────
df_all = master.copy()

if 'Unnamed: 0' in df_all.columns:
    df_all = df_all.drop(columns=['Unnamed: 0'])

df_all['Total_Production_Value_Per_Capita'] = df_all['Total_Production_Value'] / df_all['Population']

df_all['log_HCI']              = np.log1p(df_all['Human capital index'])
df_all['log_GFCF']             = np.log1p(df_all['Gross fixed capital formation, all, Constant prices, Percent of GDP'])
df_all['log_Production_Value'] = np.log1p(df_all['Total_Production_Value_Per_Capita'])

# Centre df_all on the 54-country sample means (not the full-sample mean).
# This ensures main-effect coefficients are evaluated at the same point as
# in Models 3a/3b, making the coefficient deltas in the comparison table
# directly interpretable as shifts in effect size, not shifts in evaluation point.
_centre_means = {col: df[col].mean()
                 for col in ['log_HCI', 'log_GFCF', 'log_Production_Value']}
for col in ['log_HCI', 'log_GFCF', 'log_Production_Value']:
    df_all[f'{col}_c'] = df_all[col] - _centre_means[col]

df_all['log_HCI_x_log_Production']  = df_all['log_HCI_c']  * df_all['log_Production_Value_c']
df_all['log_GFCF_x_log_Production'] = df_all['log_GFCF_c'] * df_all['log_Production_Value_c']

df_all = df_all.sort_values(['Country Code', 'Year'])
df_all['ECI_lag1'] = df_all.groupby('Country Code')['Economic Complexity Index'].shift(1)

print(f"Full sample: {df_all['Country Code'].nunique()} countries, "
      f"{df_all['Year'].nunique()} years, {len(df_all):,} obs")


# ── Model 3a (full sample) ───────────────────────────────────────────────────
all_3a_cols = reg3_input + INTERACT_VARS + ['Economic Complexity Index', 'Country Code', 'Year']
all_3a_df   = df_all[all_3a_cols].dropna()

m3a_all, m3a_all_raw = fit_driscoll_kraay(
    all_3a_df['Economic Complexity Index'],
    sm.add_constant(all_3a_df[reg3_input + INTERACT_VARS]),
    all_3a_df['Year'], all_3a_df['Country Code'],
    label='FULL SAMPLE — Model 3a (no lag)')

# ── Model 3b (full sample) ───────────────────────────────────────────────────
all_3b_cols = all_3a_cols + ['ECI_lag1']
all_3b_df   = df_all[all_3b_cols].dropna()

m3b_all, m3b_all_raw = fit_driscoll_kraay(
    all_3b_df['Economic Complexity Index'],
    sm.add_constant(all_3b_df[reg3_input + INTERACT_VARS + ['ECI_lag1']]),
    all_3b_df['Year'], all_3b_df['Country Code'],
    label='FULL SAMPLE — Model 3b (with lag)')


# ── Comparison table ──────────────────────────────────────────────────────────
def sig(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '   '

core_vars = reg3_input + INTERACT_VARS + ['ECI_lag1']
var_labels = {
    'log_HCI':                       'log(HCI)',
    'log_GFCF':                      'log(GFCF)',
    'Political stability — estimate':'Pol. stability',
    'Rule of law index':             'Rule of law',
    'log_Production_Value':          'log(Prod. val. p.c.)',
    'Trade (% of GDP)':              'Trade (% GDP)',
    'log_HCI_x_log_Production':      'log(HCI) × log(Prod)',
    'log_GFCF_x_log_Production':     'log(GFCF) × log(Prod)',
    'ECI_lag1':                       'Lagged ECI',
}

pairs = [
    ('Model 3a',  m3a,     m3a_all),
    ('Model 3b',  m3b,     m3b_all),
]

print("\n" + "=" * 110)
print("SAMPLE COMPARISON: 54 Resource-Rich Countries vs All Countries")
print("=" * 110)

for spec_name, m_54, m_full in pairs:
    n54   = int(m_54.nobs)
    nfull = int(m_full.nobs)
    nc_full = df_all['Country Code'].nunique()
    print(f"\n{'─' * 110}")
    print(f"  {spec_name:<14} │  54 countries (N={n54:,})           │  All countries (N={nfull:,}, {nc_full} ctries)")
    print(f"  {'':14} │  R² = {m_54.rsquared:.4f}                    │  R² = {m_full.rsquared:.4f}")
    print(f"  {'Variable':<22} │ {'Coef':>9} {'SE':>9} {'':>4}  │ {'Coef':>9} {'SE':>9} {'':>4}  │ Δ coef")
    print(f"  {'─' * 22}─┼{'─' * 27}─┼{'─' * 27}─┼{'─' * 9}")
    for v in core_vars:
        label = var_labels.get(v, v[:22])
        c54  = m_54.params.get(v, None)
        c_all = m_full.params.get(v, None)
        if c54 is None and c_all is None:
            continue
        if c54 is not None:
            se54 = m_54.bse[v]
            s54  = sig(m_54.pvalues[v])
            c54_str = f"{c54:>+9.4f} {se54:>9.4f} {s54}"
        else:
            c54_str = f"{'':>24}"
            c54 = 0
        if c_all is not None:
            se_all = m_full.bse[v]
            s_all  = sig(m_full.pvalues[v])
            c_all_str = f"{c_all:>+9.4f} {se_all:>9.4f} {s_all}"
        else:
            c_all_str = f"{'':>24}"
            c_all = 0
        delta = c_all - c54 if (m_54.params.get(v) is not None and m_full.params.get(v) is not None) else None
        delta_str = f"{delta:>+9.4f}" if delta is not None else ""
        print(f"  {label:<22} │ {c54_str} │ {c_all_str} │ {delta_str}")

print(f"\n{'=' * 110}")
print("Note: 'All countries' = full Master.csv without the NR rents ≥ 5% filter.")
print("Interaction terms use grand-mean centring computed separately for each sample.")

# ## 7. Visualisations

# ### 7a. Coefficient Comparison — Model 3a vs 3b
# 
# Horizontal dot-and-whisker plot comparing point estimates and 95 % confidence
# intervals for the core variables across the two specifications. If a variable's
# confidence interval crosses zero, its association with ECI is not statistically
# significant at the 5 % level.

plot_vars = [v for v in reg3_input + INTERACT_VARS if v != 'const']

labels = [v.replace('log_', 'log(').replace('_x_', ') × log(') + (')' if 'log_' in v else '')
          for v in plot_vars]

models_to_plot = [
    (m3a,  PALETTE['blue'],      'Model 3a (no lag)'),
    (m3b,  PALETTE['light_blue'],'Model 3b (with lag)'),
]

fig = go.Figure()
for model, col, name in models_to_plot:
    coefs  = [model.params.get(v, np.nan)  for v in plot_vars]
    lowers = [model.params.get(v, np.nan) - 1.96 * model.bse.get(v, np.nan) for v in plot_vars]
    uppers = [model.params.get(v, np.nan) + 1.96 * model.bse.get(v, np.nan) for v in plot_vars]
    fig.add_trace(go.Scatter(
        y=labels, x=coefs,
        error_x=dict(
            type='data', symmetric=False,
            array=[u - c for c, u in zip(coefs, uppers)],
            arrayminus=[c - l for c, l in zip(coefs, lowers)],
            color=col, thickness=1.5, width=5,
        ),
        mode='markers',
        marker=dict(color=col, size=8, symbol='circle'),
        name=name,
    ))

fig.add_vline(x=0, line=dict(color=STYLE['zero_line_color'], width=1.5, dash='dash'))

fig.update_layout(**base_layout(
    height=STYLE['chart_height_tall'],
    margin=STYLE['margin_bar'],
    xaxis=dict(
        title=dict(text='Coefficient (95 % CI, Driscoll-Kraay)', font=dict(size=STYLE['axis_title_size'])),
        gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
        zeroline=False,
    ),
    yaxis=dict(tickfont=dict(size=STYLE['tick_size'])),
    legend=dict(font=dict(size=STYLE['legend_size']),
                orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
))

save_chart(fig, os.path.join(OUT, 'coef_comparison_3a_3b'), width=1100, height=700)

# ### 7b. ECI vs Human Capital — by Production Value Quartile
# 
# Scatter plot of log(HCI) against ECI, with points coloured by per-capita production
# value quartile. If the slope of HCI on ECI steepens at higher production quartiles,
# this provides visual evidence of the positive interaction effect.

plot_df = df[['log_HCI', 'Economic Complexity Index', 'log_Production_Value',
              'Country Code', 'Year']].dropna()

plot_df['Prod_quartile'] = pd.qcut(
    plot_df['log_Production_Value'], q=4,
    labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
)

q_colors = [PALETTE['light_blue'], PALETTE['blue'], PALETTE['orange'], PALETTE['red']]

fig = go.Figure()
for q, col in zip(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'], q_colors):
    sub = plot_df[plot_df['Prod_quartile'] == q]
    fig.add_trace(go.Scatter(
        x=sub['log_HCI'], y=sub['Economic Complexity Index'],
        mode='markers',
        marker=dict(color=col, size=5, opacity=0.65,
                    line=dict(width=0.3, color='white')),
        name=f'Production {q}',
        hovertemplate='%{customdata[0]} %{customdata[1]}<br>'
                      'log(HCI)=%{x:.2f}  ECI=%{y:.2f}<extra></extra>',
        customdata=sub[['Country Code', 'Year']].values,
    ))

fig.update_layout(**base_layout(
    xaxis=dict(
        title=dict(text='log(Human Capital Index)', font=dict(size=STYLE['axis_title_size'])),
        gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
    ),
    yaxis=dict(
        title=dict(text='Economic Complexity Index', font=dict(size=STYLE['axis_title_size'])),
        gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
    ),
    legend=dict(title=dict(text='Prod. Value p.c. Quartile'),
                font=dict(size=STYLE['legend_size'])),
))

save_chart(fig, os.path.join(OUT, 'eci_hci_production_interaction'), width=1000, height=600)

# ## Summary
# 
# | Item | Detail |
# |------|--------|
# | Sample | 54 countries × 25 years (actual obs vary by model after dropna) |
# | Dependent variable | Raw ECI |
# | Models | Pooled OLS (full), AR baseline, Interaction ×2 (±lag) |
# | SE type (Models 3a/b) | Driscoll-Kraay (HAC-Groupsum, Bartlett kernel, bw=2) |
# | Fixed effects | None (pooled design) |
# | Full-sample comparison | Yes (all countries in Master.csv, Models 3a and 3b) |
# 
# **Key results (expected):**
# 
# - `log(HCI) × log(Production p.c.)`: positive and significant across specifications.
#   Higher human capital amplifies the ECI returns to resource production.
# - `log(GFCF) × log(Production p.c.)`: small, not significant.
# - Adding lagged ECI (Model 3b) absorbs persistence and raises R² substantially.
#   If the other coefficients are stable after adding the lag, that confirms they
#   reflect genuine structural associations rather than proxying for ECI persistence.
# 
# **Methodological notes:**
# - DV is raw ECI (the previous log-of-shifted transformation compressed the
#   distribution and produced hard-to-interpret coefficients).
# - Production value is per-capita (captures resource intensity relative to
#   population rather than absolute scale).
# - No fixed effects: the pooled design retains between-country variation, which
#   is central to the research question.
# - Landlocked dummy removed: conflates geography with other between-country
#   variation in pooled OLS.

print("=" * 70)
print("NB6: REGRESSIONS SUMMARY (Unified)")
print("=" * 70)
print(f"  Sample:         {df['Country Code'].nunique()} countries × {df['Year'].nunique()} years")
print(f"  DV:             Raw ECI")
print()
print(f"  Model          N obs    R²      Adj R²   DW")
print(f"  {'Model 1':<13} {int(m1.nobs):>5}   {m1.rsquared:.4f}  {m1.rsquared_adj:.4f}   {durbin_watson(m1.resid):.3f}")
print(f"  {'Model 2 (AR)':<13} {int(m2.nobs):>5}   {m2.rsquared:.4f}  {m2.rsquared_adj:.4f}   {durbin_watson(m2.resid):.3f}")
print(f"  {'Model 3a':<13} {int(m3a.nobs):>5}   {m3a.rsquared:.4f}  {m3a.rsquared_adj:.4f}   {durbin_watson(m3a_raw.resid):.3f}")
print(f"  {'Model 3b':<13} {int(m3b.nobs):>5}   {m3b.rsquared:.4f}  {m3b.rsquared_adj:.4f}   {durbin_watson(m3b_raw.resid):.3f}")
print()
print("  Saved outputs:")
print(f"    intermediary/high_resource_countries.csv")
print("=" * 70)
