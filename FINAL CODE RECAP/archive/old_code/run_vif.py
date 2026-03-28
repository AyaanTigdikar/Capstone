"""
Standalone VIF analysis for:
  A) NB5 ML features  (all_features, 24 vars, StandardScaler-scaled X_train)
  B) NB6 Model-3a     (7 main + 4 interactions, no lag)
  C) NB6 Model-3b     (7 main + 4 interactions + ECI_lag1)
"""
import os, sys
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

FONT  = 'IBM Plex Sans, -apple-system, BlinkMacSystemFont, sans-serif'
BG    = '#ffffff'
NAVY  = '#1a2744'
GRID  = '#e5e7eb'

OUT_DIR = os.path.join('Final', 'charts', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

def _bar_color(vif):
    if vif > 10:   return '#c23a3a'   # red — high
    if vif > 5:    return '#d4853b'   # orange — moderate
    return '#4a6fa5'                  # blue — ok

def plot_vif(title, df, filename):
    df = df.sort_values('VIF')  # ascending for horizontal bar
    colors = [_bar_color(v) for v in df['VIF']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['VIF'], y=df['Feature'],
        orientation='h',
        marker_color=colors,
        text=[f'{v:.1f}' for v in df['VIF']],
        textposition='outside',
        textfont=dict(size=11, family=FONT, color=NAVY),
        cliponaxis=False,
    ))

    # threshold lines
    x_max = max(df['VIF'].max() * 1.25, 12)
    for thresh, label, col in [(5, 'VIF = 5', '#d4853b'), (10, 'VIF = 10', '#c23a3a')]:
        fig.add_vline(x=thresh, line_dash='dash', line_color=col, line_width=1.5,
                      annotation_text=label, annotation_font_size=10,
                      annotation_font_color=col, annotation_position='top right')

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family=FONT, color=NAVY), x=0.01, xanchor='left'),
        xaxis=dict(title=dict(text='VIF', font=dict(family=FONT, size=12)),
                   range=[0, x_max], gridcolor=GRID,
                   tickfont=dict(family=FONT, size=11)),
        yaxis=dict(tickfont=dict(family=FONT, size=11), automargin=True),
        plot_bgcolor=BG, paper_bgcolor=BG,
        margin=dict(l=10, r=80, t=50, b=40),
        height=max(300, 28 * len(df) + 80),
        width=750,
        showlegend=False,
    )

    path = os.path.join(OUT_DIR, filename)
    fig.write_html(path, config={'displayModeBar': False})
    print(f'  → saved {path}')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

INCLUDE = [
    'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
    'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
    'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
    'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
    'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
    'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
]

def vif_table(X_df):
    """Given a DataFrame of features, return a VIF DataFrame sorted descending."""
    mat = X_df.values.astype(float)
    vifs = [variance_inflation_factor(mat, i) for i in range(mat.shape[1])]
    return (pd.DataFrame({'Feature': X_df.columns, 'VIF': vifs})
              .sort_values('VIF', ascending=False)
              .reset_index(drop=True))

def print_vif(title, df):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"  N obs: {df.attrs.get('nobs', '?')}")
    print(f"{'='*70}")
    print(f"  {'Feature':<50} {'VIF':>8}  flag")
    print(f"  {'-'*60}")
    for _, row in df.iterrows():
        flag = ' *** HIGH' if row['VIF'] > 10 else (' * moderate' if row['VIF'] > 5 else '')
        print(f"  {row['Feature']:<50} {row['VIF']:>8.2f}  {flag}")

# ── Load master ────────────────────────────────────────────────────────────────
master = pd.read_csv('intermediary/Master.csv')
if 'Unnamed: 0' in master.columns:
    master = master.drop(columns=['Unnamed: 0'])

# ══════════════════════════════════════════════════════════════════════════════
# A) NB5 ML FEATURES
# ══════════════════════════════════════════════════════════════════════════════
df5 = master[
    (master['Year'] >= 1995) &
    (master['Year'] <= 2019) &
    (master['Country Code'].isin(INCLUDE))
].copy().sort_values(['Country Code', 'Year']).reset_index(drop=True)

# log1p transforms
log_cols5 = [
    'Human capital index',
    'Total_Production_Value',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP',
    'Government revenue',
    'Use of IMF credit (DOD, current US$)',
    'Forestry rents (% of GDP)'
]
df5[log_cols5] = np.log1p(df5[log_cols5]).replace([np.inf, -np.inf], np.nan)

# Rolling macro controls
df5['Inflation_roll5'] = (
    df5.groupby('Country Code')['Inflation, consumer prices (annual %)']
       .transform(lambda x: x.rolling(5, min_periods=3).mean())
)
df5['RealRate_roll5'] = (
    df5.groupby('Country Code')['Real interest rate (%)']
       .transform(lambda x: x.rolling(5, min_periods=3).mean())
)

# Resource HHI
rents_cols = ['Oil rents (% of GDP)', 'Natural gas rents (% of GDP)',
              'Mineral rents (% of GDP)', 'Forestry rents (% of GDP)']
total_rents = df5['Total natural resources rents (% of GDP)'].replace(0, np.nan)
df5['Resource_HHI'] = sum((df5[c] / total_rents) ** 2 for c in rents_cols)

# Interaction terms (mean-centred on the log-transformed features)
_hci_mean  = df5['Human capital index'].mean()
_prod_mean = df5['Total_Production_Value'].mean()
_rol_mean  = df5['Rule of law index'].mean()
df5['HCI_x_ProductionValue']       = (df5['Human capital index'] - _hci_mean) * \
                                      (df5['Total_Production_Value'] - _prod_mean)
df5['RuleOfLaw_x_ProductionValue'] = (df5['Rule of law index'] - _rol_mean) * \
                                      (df5['Total_Production_Value'] - _prod_mean)

# Lagged ECI
df5 = df5.sort_values(['Country Code', 'Year'])
df5['L1_ECI'] = df5.groupby('Country Code')['Economic Complexity Index'].shift(1)

base_features = [
    'Total_Production_Value',
    'Human capital index',
    'Rule of law index',
    'Political stability — estimate',
    'Trade (% of GDP)',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP',
    'Share of investment in GDP',
    'Domestic credit to private sector (% of GDP)',
    'Landlocked',
    'Urban population (% of total population)',
    'Government revenue',
    'Capital depreciation rate',
    'Use of IMF credit (DOD, current US$)',
    'Real interest rate (%)',
    'Inflation, consumer prices (annual %)',
    'Access to electricity (% of population)',
    'Adjusted savings: gross savings (% of GNI)',
    'L1_ECI',
    'Forestry rents (% of GDP)'
]
new_features         = ['Inflation_roll5', 'RealRate_roll5', 'Resource_HHI']
interaction_features = ['HCI_x_ProductionValue', 'RuleOfLaw_x_ProductionValue']
all_features         = base_features + new_features + interaction_features

# Drop rows with any missing feature
df5_clean = df5.dropna(subset=all_features)

# Train split (< 2015), standardised — same as notebook
train5 = df5_clean[df5_clean['Year'] < 2015]
scaler = StandardScaler()
X_train5 = pd.DataFrame(
    scaler.fit_transform(train5[all_features].values),
    columns=all_features
)

vif5 = vif_table(X_train5)
vif5.attrs['nobs'] = len(X_train5)
print_vif(f"NB5 — ML features ({len(all_features)} vars, train <2015, N={len(X_train5)})", vif5)
plot_vif(f'VIF — NB5 ML features (N={len(X_train5)}, train <2015)', vif5, 'vif_nb5_ml.html')

# ══════════════════════════════════════════════════════════════════════════════
# B & C) NB6 REGRESSION FEATURES (Model 3a / 3b)
# ══════════════════════════════════════════════════════════════════════════════
df6 = master[master['Country Code'].isin(INCLUDE)].copy()
df6['Total_Production_Value'] = df6['Total_Production_Value'] / df6['Population']
if 'Unnamed: 0' in df6.columns:
    df6 = df6.drop(columns=['Unnamed: 0'])

# Log transforms
df6['log_HCI']              = np.log(df6['Human capital index'] + 1)
df6['log_GFCF']             = np.log(df6['Gross fixed capital formation, all, Constant prices, Percent of GDP'] + 1)
df6['log_Production_Value'] = np.log(df6['Total_Production_Value'] + 1)

# Mean-centre for interactions
for col in ['log_HCI', 'log_GFCF', 'log_Production_Value']:
    df6[f'{col}_c'] = df6[col] - df6[col].mean()
df6['forestry_rents_c'] = df6['Forestry rents (% of GDP)'] - df6['Forestry rents (% of GDP)'].mean()

# Interaction terms
df6['log_HCI_x_log_Production']  = df6['log_HCI_c']  * df6['log_Production_Value_c']
df6['log_GFCF_x_log_Production'] = df6['log_GFCF_c'] * df6['log_Production_Value_c']
df6['log_HCI_x_forestry_rents']  = df6['log_HCI_c']  * df6['forestry_rents_c']
df6['log_GFCF_x_forestry_rents'] = df6['log_GFCF_c'] * df6['forestry_rents_c']

# Lagged ECI
df6 = df6.sort_values(['Country Code', 'Year'])
df6['ECI_lag1'] = df6.groupby('Country Code')['Economic Complexity Index'].shift(1)

reg3_input    = ['log_HCI', 'log_GFCF', 'Political stability — estimate',
                 'Rule of law index', 'log_Production_Value',
                 'Forestry rents (% of GDP)', 'Trade (% of GDP)']
INTERACT_VARS = ['log_HCI_x_log_Production', 'log_GFCF_x_log_Production',
                 'log_HCI_x_forestry_rents', 'log_GFCF_x_forestry_rents']

# Model 3a
vars_3a   = reg3_input + INTERACT_VARS
df6_3a    = df6[vars_3a].dropna()
vif6_3a   = vif_table(df6_3a)
vif6_3a.attrs['nobs'] = len(df6_3a)
print_vif(f"NB6 — Model 3a: 7 main + 4 interactions (N={len(df6_3a)})", vif6_3a)
plot_vif(f'VIF — NB6 Model 3a: 7 main + 4 interactions (N={len(df6_3a)})', vif6_3a, 'vif_nb6_3a.html')

# Model 3b
vars_3b   = reg3_input + INTERACT_VARS + ['ECI_lag1']
df6_3b    = df6[vars_3b].dropna()
vif6_3b   = vif_table(df6_3b)
vif6_3b.attrs['nobs'] = len(df6_3b)
print_vif(f"NB6 — Model 3b: 7 main + 4 interactions + ECI_lag1 (N={len(df6_3b)})", vif6_3b)
plot_vif(f'VIF — NB6 Model 3b: 7 main + 4 interactions + ECI_lag1 (N={len(df6_3b)})', vif6_3b, 'vif_nb6_3b.html')

print("\nDone.")
