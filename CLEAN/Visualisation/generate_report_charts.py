"""
generate_report_charts.py
=========================
Produces the 17 charts that appear in the main body of the capstone report.
Organised by report section so you can see which figure each chart feeds.

Run from project root (FINAL CODE RECAP/):
    python3 scripts/generate_report_charts.py

Dependencies:  viz_utils.py  (must be on sys.path or in scripts/)
Outputs:       Final/charts/{descriptive,clusters,ml,regression}/*.{html,png}

Chart → Report Figure mapping
------------------------------
  00  → Figure 1   (Data sources by macro-indicator)
  02  → Table 2    (Variable correlations with ECI)
  26  → Figure 3   (PCA resource loadings heatmap)
  03  → Figure 4   (PCA biplot, K-means in PCA space)
  04a → Figure 5   (Cluster world map, 1995)
  05  → Figure 6   (ECI vs GDP per capita animated, 1995-2019)
  14  →             (ECI distribution shift, 1995 vs 2019)
  15  →             (ECI median trajectory by cluster)
  16  →             (Regression coefficients, Model 3a vs 3b)
  17  →             (HCI x Production interaction scatter)
  30  →             (Regression variable correlation heatmap)
  07  → Figure 7   (ML feature importance consensus, 3 models)
  08  → Figure 8   (ML standardised coefficients)
  27  → Figure 11  (Random Forest feature importance)
  09  → Figure 9   (Train vs Test R², all models)
  10  → Figure 10  (Actual vs Predicted ECI, test set)
  11  → Figure Z   (ECI forecast, top improvers 2020-2030)
"""

# ── 0. Project root ──────────────────────────────────────────────────────────
import os, sys

def _find_root(marker='intermediary'):
    d = '/Users/leoss/Desktop/GitHub/Capstone/CLEAN'
    for _ in range(6):
        if os.path.isdir(os.path.join(d, marker)):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"Could not find project root (looking for '{marker}' dir).")

ROOT = _find_root()
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

# ── 1. Imports ────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings('ignore')
import math as _math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from viz_utils import (
    PALETTE, FONT, BG, NAVY, GRID, WRITE_CONFIG,
    base_layout, save,
    load_master, load_master_wide, load_clusters, load_nr, load_nb5, load_bootstrap,
    INCLUDE_LIST, resource_rich_codes, build_sample, shorten_feat,
    CLUSTER_LABELS, CLUSTER_COLORS, LABEL_TO_COLOR,
)

# ── 2. Output directories ────────────────────────────────────────────────────
_VIZ_BASE = '/Users/leoss/Desktop/GitHub/Capstone/CLEAN/Visualisation'
OUT_DESC = os.path.join(_VIZ_BASE, 'charts', 'descriptive')
OUT_CLUS = os.path.join(_VIZ_BASE, 'charts', 'clusters')
OUT_ML   = os.path.join(_VIZ_BASE, 'charts', 'ml')
OUT_REG  = os.path.join(_VIZ_BASE, 'charts', 'regression')
for _d in [OUT_DESC, OUT_CLUS, OUT_ML, OUT_REG]:
    os.makedirs(_d, exist_ok=True)

# Labels excluded from ML charts (lag + engineered rolling vars)
LABEL_EXCL = ['L1_ECI', 'Inflation_roll5', 'RealRate_roll5', 'Resource_HHI']

# ── 3. Clustering infrastructure (needed by charts 03, 04a, 05) ─────────────
_LABEL_COLORS_4K = {
    'Petrostates':       '#d4853b',
    'Oil Exporters':     '#4a6fa5',
    'Major Producers':   '#2e7d4a',
    'Limited Resources': '#c23a3a',
}

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def run_clustering(nr_data, year_filter=None, agg_years=None, n_clusters=4, random_state=42):
    """Full pipeline: pivot → per-capita → log1p → PCA(2) → KMeans(k)."""
    df = nr_data.copy()
    if year_filter is not None:
        df = df[df['Year'] == year_filter]
    elif agg_years is not None:
        df = df[df['Year'].isin(agg_years)]

    df_pivot = df.pivot_table(
        index=['Country', 'Country Code', 'Year', 'Population'],
        columns='Resource', values='Production_TotalValue',
    ).reset_index()

    resource_cols = df_pivot.columns.difference(['Country', 'Country Code', 'Year', 'Population'])
    df_pivot[resource_cols] = df_pivot[resource_cols].div(df_pivot['Population'], axis=0)
    df_pivot.drop(columns='Population', inplace=True)
    df_pivot = df_pivot.fillna(0)

    df_latest = (df_pivot.sort_values('Year', ascending=True)
                 .groupby(['Country', 'Country Code']).first().reset_index())

    feature_cols = [c for c in df_latest.columns if c not in ['Country', 'Country Code', 'Year']]
    X_log = np.log1p(df_latest[feature_cols].fillna(0))

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_log)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    clusters = kmeans.fit_predict(pca_components)

    pca_df = pd.DataFrame({
        'Country':      df_latest['Country'],
        'Country Code': df_latest['Country Code'],
        'Year':         df_latest['Year'],
        'PC1': pca_components[:, 0],
        'PC2': pca_components[:, 1],
        'Cluster': clusters,
    })

    centroids = kmeans.cluster_centers_
    pc1_rank  = list(np.argsort(-centroids[:, 0]))
    pc2_rank  = list(np.argsort(-centroids[:, 1]))

    label_map, labeled = {}, set()
    oil_id = pc1_rank[0]
    label_map[oil_id] = 'Petrostates'; labeled.add(oil_id)
    mineral_id = next(c for c in pc2_rank if c not in labeled)
    label_map[mineral_id] = 'Major Producers'; labeled.add(mineral_id)
    remaining = [c for c in pc1_rank if c not in labeled]
    label_map[remaining[0]] = 'Oil Exporters'
    label_map[remaining[1]] = 'Limited Resources'

    pca_df['ClusterLabels'] = pca_df['Cluster'].map(label_map)
    return pca_df, pca, feature_cols


def create_cluster_map(pca_df, nr_data, cluster_names_map=None,
                       label_colors=None, dominance_threshold=15.0):
    """Choropleth map with red borders for major global producers."""
    if cluster_names_map is None:
        cluster_names_map = dict(zip(pca_df['Cluster'].unique(), pca_df['ClusterLabels'].unique()))
    if label_colors is None:
        label_colors = _LABEL_COLORS_4K

    df_total = nr_data.pivot_table(
        index=['Country', 'Country Code'],
        columns='Resource', values='Production_TotalValue', aggfunc='sum',
    ).reset_index().fillna(0)

    prod_cols  = [c for c in df_total.columns if c not in ['Country', 'Country Code']]
    for col in prod_cols:
        total = df_total[col].sum()
        if total > 0:
            df_total[f'{col}_Share'] = (df_total[col] / total) * 100

    share_cols = [c for c in df_total.columns if c.endswith('_Share')]
    df_map = pca_df.merge(df_total[['Country Code'] + share_cols], on='Country Code', how='left')
    df_map['Is_Dominant']       = (df_map[share_cols] >= dominance_threshold).any(axis=1)
    df_map['Dominant_Resources'] = df_map.apply(
        lambda row: [sc.replace('_Share', '') for sc in share_cols
                     if row.get(sc, 0) >= dominance_threshold], axis=1)

    def make_hover(row):
        lbl   = row['ClusterLabels']
        lines = [f"<b>{row['Country']}</b>", f"Cluster: {lbl}"]
        vals  = [(c, row.get(c, 0)) for c in prod_cols if row.get(c, 0) > 0]
        vals.sort(key=lambda x: x[1], reverse=True)
        if vals:
            lines.append('<br>Top Resources:')
            for res, v in vals[:3]:
                if v > 1e9:   lines.append(f'  {res}: ${v/1e9:.1f}B')
                elif v > 1e6: lines.append(f'  {res}: ${v/1e6:.0f}M')
                else:         lines.append(f'  {res}: ${v:,.0f}')
        return '<br>'.join(lines)

    df_map['hover_text'] = df_map.apply(make_hover, axis=1)

    fig = go.Figure()
    for cid in sorted(df_map['Cluster'].unique()):
        lbl   = cluster_names_map.get(cid, f'Cluster {cid}')
        color = label_colors.get(lbl, '#aaa')

        sub = df_map[(df_map['Cluster'] == cid) & (~df_map['Is_Dominant'])]
        if len(sub) > 0:
            fig.add_trace(go.Choropleth(
                locations=sub['Country Code'], z=[cid]*len(sub),
                colorscale=[[0, color], [1, color]], showscale=False,
                showlegend=True, name=lbl,
                customdata=sub['hover_text'].values,
                hovertemplate='%{customdata}<extra></extra>',
                marker=dict(line=dict(color='white', width=0.6)),
            ))

        sub_d = df_map[(df_map['Cluster'] == cid) & (df_map['Is_Dominant'])]
        if len(sub_d) > 0:
            fig.add_trace(go.Choropleth(
                locations=sub_d['Country Code'], z=[cid]*len(sub_d),
                colorscale=[[0, color], [1, color]], showscale=False,
                showlegend=False, name=f'{lbl} ★ major producer',
                customdata=sub_d['hover_text'].values,
                hovertemplate='%{customdata}<extra></extra>',
                marker=dict(line=dict(color='#111', width=2.2)),
            ))

    fig.add_trace(go.Choropleth(
        locations=['ZZZ'], z=[0],
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
        showscale=False, showlegend=True,
        name='★ >15% of global output',
        marker=dict(line=dict(color='#111', width=2.2)),
    ))

    fig.update_geos(
        projection_type='natural earth',
        showcountries=True, countrycolor='#ccc',
        showcoastlines=True, coastlinecolor='#ccc',
        showland=True, landcolor='#f0f0f0',
        showocean=True, oceancolor='#dde8f0',
        showframe=False,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=70),
        legend=dict(
            orientation='h', x=0.5, y=-0.08, xanchor='center', yanchor='top',
            font=dict(size=11, family=FONT),
            bgcolor='rgba(250,250,250,0.9)', bordercolor='#d0d0d0', borderwidth=1,
        ),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT),
    )
    return fig


# ── Pre-run: clustering data ─────────────────────────────────────────────────
print('Loading NaturalResource.csv and running clustering pipeline...')
nr_full   = load_nr()
nr_sample = nr_full[nr_full['Country Code'].isin(INCLUDE_LIST)]

pca_1995, pca_model_1995, feat_1995 = run_clustering(nr_sample, year_filter=1995)
pca_agg,  pca_model_agg,  feat_agg  = run_clustering(nr_sample, agg_years=[1995, 1999, 2005])
print('  Clustering done.\n')


# =============================================================================
#
#   SECTION 3.2 — DATA
#
# =============================================================================

# ── CHART 00 — Figure 1: Data sources by macro-indicator ─────────────────────
print('=== CHART 00 (Figure 1: Data sources overview) ===')

_variables_00 = [
    ('Total NR rents (% GDP)',         'Resource Rents',    'World Bank', '1995-2019'),
    ('Mineral rents (% GDP)',           'Resource Rents',    'World Bank', '1995-2019'),
    ('Natural gas rents (% GDP)',       'Resource Rents',    'World Bank', '1995-2019'),
    ('Oil rents (% GDP)',               'Resource Rents',    'World Bank', '1995-2019'),
    ('Manufacturing (% GDP)',           'GDP Structure',     'World Bank', '1995-2019'),
    ('Industry (% GDP)',                'GDP Structure',     'World Bank', '1995-2019'),
    ('High-tech exports (%)',           'GDP Structure',     'World Bank', '1995-2019'),
    ('Agriculture (% GDP)',             'GDP Structure',     'World Bank', '1995-2019'),
    ('Services (% GDP)',                'GDP Structure',     'World Bank', '1995-2019'),
    ('Gross savings (% GNI)',           'Finance',           'World Bank', '1995-2019'),
    ('NR depletion (% GNI)',            'Finance',           'World Bank', '1995-2019'),
    ('Domestic credit (% GDP)',         'Finance',           'World Bank', '1995-2019'),
    ('IMF credit (USD)',                'Finance',           'World Bank', '1995-2019'),
    ('Real interest rate (%)',          'Macro',             'World Bank', '1995-2019'),
    ('Lending interest rate (%)',       'Macro',             'World Bank', '1995-2019'),
    ('Inflation (%)',                   'Macro',             'World Bank', '1995-2019'),
    ('Trade (% GDP)',                   'Macro',             'World Bank', '1995-2019'),
    ('Employment in industry (%)',      'GDP Structure',     'World Bank', '1995-2019'),
    ('Employment in services (%)',      'GDP Structure',     'World Bank', '1995-2019'),
    ('Employment in agriculture (%)',   'GDP Structure',     'World Bank', '1995-2019'),
    ('Electricity access (%)',          'Infrastructure',    'World Bank', '1995-2019'),
    ('Mobile subscriptions (per 100)',  'Infrastructure',    'World Bank', '1995-2019'),
    ('Urban population (%)',            'Demographics',      'World Bank', '1995-2019'),
    ('Life expectancy (years)',         'Demographics',      'World Bank', '1995-2019'),
    ('Death rate (per 1000)',           'Demographics',      'World Bank', '1995-2019'),
    ('GDP per capita PPP',              'Macro',             'IMF',        '1995-2019'),
    ('Govt revenue (% GDP)',            'Macro',             'IMF',        '1995-2019'),
    ('Govt net debt (% GDP)',           'Finance',           'IMF',        '1995-2019'),
    ('Structural fiscal balance',       'Finance',           'IMF',        '1995-2019'),
    ('GFCF, all sectors (% GDP)',       'Finance',           'IMF',        '1995-2019'),
    ('Primary net lending (% GDP)',     'Finance',           'IMF',        '1995-2019'),
    ('Economic Complexity Index',       'Dependent Variable','Atlas / ECI', '1995-2019'),
    ('Electoral democracy index',       'Governance',        'V-Dem',      '1995-2019'),
    ('Liberal democracy index',         'Governance',        'V-Dem',      '1995-2019'),
    ('Participatory dem. index',        'Governance',        'V-Dem',      '1995-2019'),
    ('Deliberative dem. index',         'Governance',        'V-Dem',      '1995-2019'),
    ('Egalitarian dem. index',          'Governance',        'V-Dem',      '1995-2019'),
    ('Clientelism index',               'Governance',        'V-Dem',      '1995-2019'),
    ('Political corruption index',      'Governance',        'V-Dem',      '1995-2019'),
    ('Rule of law index',               'Governance',        'V-Dem',      '1995-2019'),
    ('Accountability index',            'Governance',        'V-Dem',      '1995-2019'),
    ('Property rights',                 'Governance',        'V-Dem',      '1995-2019'),
    ('Political stability (WGI)',       'Governance',        'V-Dem',      '1995-2019'),
    ('Civil war indicator',             'Governance',        'V-Dem',      '1995-2019'),
    ('Human capital index',             'Human Capital',     'PWT 11.0',   '1995-2019'),
    ('Capital stock (nat. acc.)',       'Finance',           'PWT 11.0',   '1995-2019'),
    ('TFP level',                       'Macro',             'PWT 11.0',   '1995-2019'),
    ('Welfare-relevant TFP',            'Macro',             'PWT 11.0',   '1995-2019'),
    ('Share of consumption in GDP',     'GDP Structure',     'PWT 11.0',   '1995-2019'),
    ('Share of investment in GDP',      'GDP Structure',     'PWT 11.0',   '1995-2019'),
    ('Share of govt spending in GDP',   'GDP Structure',     'PWT 11.0',   '1995-2019'),
    ('Capital depreciation rate',       'Finance',           'PWT 11.0',   '1995-2019'),
    ('Landlocked dummy',                'Geography',         'CEPII',      'time-invariant'),
] + [
    (f'{r} production (volume)',        'NR Production',     'EI / OWID',  '1995-2019')
    for r in ['Oil', 'Natural Gas', 'Coal', 'Copper', 'Nickel', 'Cobalt', 'Lithium',
              'Bauxite', 'Aluminium', 'Zinc', 'Tin', 'Manganese', 'Rare Earth',
              'Platinum Group', 'Vanadium', 'Natural Graphite']
] + [
    (f'{r} price ($/t)',                'NR Prices',         'EI / USGS',  '1995-2019')
    for r in ['Oil', 'Natural Gas', 'Coal', 'Copper', 'Nickel', 'Cobalt', 'Lithium',
              'Bauxite', 'Aluminium', 'Zinc', 'Tin', 'Manganese', 'Rare Earth',
              'Platinum Group', 'Vanadium', 'Natural Graphite']
]

_df00 = pd.DataFrame(_variables_00, columns=['Variable', 'Category', 'Source', 'Coverage'])
_summary00 = _df00.groupby(['Category', 'Source']).size().reset_index(name='N')

_src_order00 = ['World Bank', 'IMF', 'V-Dem', 'PWT 11.0', 'CEPII', 'EI / OWID', 'EI / USGS', 'Atlas / ECI']
_cat_order00 = ['Dependent Variable', 'Resource Rents', 'GDP Structure', 'Macro', 'Finance',
                'Governance', 'Human Capital', 'Infrastructure', 'Demographics', 'Geography',
                'NR Production', 'NR Prices']

_src_cmap00 = {'World Bank': '#4a6fa5', 'IMF': '#c23a3a', 'V-Dem': '#2e7d4a',
               'PWT 11.0': '#d4853b', 'CEPII': '#7a5c9e', 'EI / OWID': '#3a8fa5',
               'EI / USGS': '#d4a017', 'Atlas / ECI': '#999'}

fig00 = go.Figure()
for _, row in _summary00.iterrows():
    xi = _src_order00.index(row['Source']) if row['Source'] in _src_order00 else 0
    yi = _cat_order00.index(row['Category']) if row['Category'] in _cat_order00 else 0
    bsize = int(_math.log1p(row['N']) * 14 + 10)
    fig00.add_trace(go.Scatter(
        x=[xi], y=[yi], mode='markers+text',
        marker=dict(size=bsize, color=_src_cmap00.get(row['Source'], '#aaa'),
                    opacity=0.88, line=dict(width=1.5, color='white')),
        text=[str(row['N'])], textfont=dict(size=10, color='white'),
        textposition='middle center',
        hovertemplate=f"{row['Category']} / {row['Source']}: {row['N']} variable(s)<extra></extra>",
        showlegend=False,
    ))

fig00.update_xaxes(tickvals=list(range(len(_src_order00))), ticktext=_src_order00,
                   tickangle=-30, tickfont=dict(size=12, family=FONT), showgrid=False)
fig00.update_yaxes(tickvals=list(range(len(_cat_order00))), ticktext=_cat_order00,
                   tickfont=dict(size=12, family=FONT), showgrid=False)
fig00.update_layout(**base_layout(height=620, margin=dict(l=165, r=40, t=50, b=110), showlegend=False))
save(fig00, '00_intro__data_sources_overview', OUT_DESC, w=1100, h=620)


# ── CHART 02 — Table 2: Variable correlations with ECI ───────────────────────
print('\n=== CHART 02 (Table 2: Correlations with ECI) ===')

master = load_master()
panel  = build_sample(master)

FEAT_COLS = [
    'Human capital index',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP',
    'Rule of law index', 'Political stability \u2014 estimate',
    'Domestic credit to private sector (% of GDP)', 'Trade (% of GDP)',
    'Access to electricity (% of population)',
    'Urban population (% of total population)',
    'Total natural resources rents (% of GDP)',
    'Oil rents (% of GDP)', 'Mineral rents (% of GDP)',
    'Natural gas rents (% of GDP)',
    'GDP per capita (constant prices, PPP)', 'prod_pc',
]

eci_col = 'Economic Complexity Index'
corr_rows = []
for col in FEAT_COLS:
    if col not in panel.columns:
        continue
    sub = panel[[eci_col, col]].dropna()
    if len(sub) < 20:
        continue
    r = sub[eci_col].corr(sub[col])
    corr_rows.append({'Feature': col, 'Correlation': r})

corr_df = (pd.DataFrame(corr_rows).sort_values('Correlation', ascending=True).reset_index(drop=True))
corr_df['Label'] = corr_df['Feature'].apply(shorten_feat)
corr_df['Color'] = corr_df['Correlation'].apply(lambda r: PALETTE['blue'] if r >= 0 else PALETTE['red'])

fig02 = go.Figure(go.Bar(
    x=corr_df['Correlation'], y=corr_df['Label'], orientation='h',
    marker=dict(color=corr_df['Color'], opacity=0.85, line=dict(color='white', width=0.5)),
    hovertemplate='%{y}: %{x:.3f}<extra></extra>',
))
fig02.add_vline(x=0, line=dict(color='#444', width=1.5))
fig02.update_layout(**base_layout(
    height=600, margin=dict(l=200, r=80, t=60, b=60),
    xaxis=dict(title='Pearson Correlation with ECI (panel 1995-2019)', gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(tickfont=dict(size=11)),
))
save(fig02, '02_sample__variable_correlations_with_eci', OUT_DESC, w=1100, h=600)


# =============================================================================
#
#   SECTION 3.3 — DESCRIPTIVE STATISTICS & CLUSTERS
#
# =============================================================================

# ── CHART 26 — Figure 3: PCA resource loadings heatmap ───────────────────────
print('\n=== CHART 26 (Figure 3: PCA loadings heatmap) ===')

nr26      = load_nr()
nr_s26    = nr26[nr26['Country Code'].isin(INCLUDE_LIST)]
nr_1995_26 = nr_s26[nr_s26['Year'] == 1995]

pivot26 = nr_1995_26.pivot_table(
    index=['Country', 'Country Code', 'Year', 'Population'],
    columns='Resource', values='Production_TotalValue',
).reset_index()

resource_cols26 = [c for c in pivot26.columns
                   if c not in ['Country', 'Country Code', 'Year', 'Population']]
pivot26[resource_cols26] = pivot26[resource_cols26].div(pivot26['Population'], axis=0)
pivot26 = pivot26.fillna(0)

X26  = np.log1p(pivot26[resource_cols26].fillna(0))
pca26 = PCA(n_components=2, random_state=42)
pca26.fit(X26)

var1_26 = pca26.explained_variance_ratio_[0] * 100
var2_26 = pca26.explained_variance_ratio_[1] * 100

loadings26 = pd.DataFrame(pca26.components_.T, columns=['PC1', 'PC2'], index=resource_cols26)
top20_26   = loadings26.abs().sum(axis=1).nlargest(20).index
plot_df26  = loadings26.loc[top20_26]
plot_df26  = (plot_df26.assign(_s=plot_df26['PC1'].abs() + plot_df26['PC2'].abs())
              .sort_values('_s', ascending=False).drop(columns='_s'))

pc_labels_ordered = [
    f'PC1 ({var1_26:.1f}%)<br><i>\u2191 Oil & Gas</i>',
    f'PC2 ({var2_26:.1f}%)<br><i>\u2191 Copper, Gold & Coal</i>',
]
y_labels26 = pc_labels_ordered[::-1]
z_values26 = plot_df26[['PC2', 'PC1']].T.values

fig26 = go.Figure(go.Heatmap(
    z=z_values26, x=plot_df26.index.tolist(), y=y_labels26,
    colorscale=[[0.0, '#1a4a8a'], [0.5, '#ffffff'], [1.0, '#c23a3a']],
    zmid=0, zmin=-1, zmax=1,
    hovertemplate='<b>%{x}</b><br>%{y}: %{z:.3f}<extra></extra>',
    colorbar=dict(title=dict(text='Loading', font=dict(size=12)),
                  thickness=20, len=1.0, tickvals=[-1, -0.5, 0, 0.5, 1], tickfont=dict(size=11)),
))
fig26.update_xaxes(title_text='Resource/Feature', tickangle=-40,
                   tickfont=dict(size=10, family=FONT), showgrid=False)
fig26.update_yaxes(title_text='Principal Component',
                   tickfont=dict(size=12, family=FONT), showgrid=False)
fig26.update_layout(**base_layout(height=420, margin=dict(l=260, r=120, t=60, b=160)))
save(fig26, '26_diag__pca_resource_loadings_heatmap', OUT_CLUS, w=1300, h=420)


# ── CHART 03 — Figure 4: PCA biplot (K-means in PCA space) ───────────────────
print('\n=== CHART 03 (Figure 4: PCA biplot) ===')

def chart_03_biplot(pca_df, pca_model, feature_cols):
    loadings_df = pd.DataFrame(
        pca_model.components_.T * np.sqrt(pca_model.explained_variance_),
        columns=['PC1', 'PC2'], index=feature_cols,
    )
    importance = loadings_df.abs().sum(axis=1)
    top5  = importance.nlargest(5).index
    top10 = importance.nlargest(10).index
    scale = 2.8

    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100

    fig = go.Figure()
    for cid in sorted(pca_df['Cluster'].unique()):
        sub   = pca_df[pca_df['Cluster'] == cid]
        lbl   = sub['ClusterLabels'].iloc[0]
        color = _LABEL_COLORS_4K.get(lbl, '#999')
        fig.add_trace(go.Scatter(
            x=sub['PC1'], y=sub['PC2'], mode='markers+text',
            marker=dict(size=10, color=color, opacity=0.82, line=dict(width=1.2, color='white')),
            text=sub['Country Code'], textposition='top center',
            textfont=dict(size=8, color='#333'), name=lbl,
            hovertemplate='<b>%{text}</b><br>PC1=%{x:.2f}, PC2=%{y:.2f}<extra></extra>',
        ))

    for feat_name in top10:
        if feat_name in top5:
            continue
        fig.add_annotation(
            x=loadings_df.loc[feat_name, 'PC1'] * scale,
            y=loadings_df.loc[feat_name, 'PC2'] * scale,
            ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.2,
            arrowcolor='rgba(150,150,150,0.5)',
        )

    for feat_name in top5:
        x1 = loadings_df.loc[feat_name, 'PC1'] * scale
        y1 = loadings_df.loc[feat_name, 'PC2'] * scale
        fig.add_annotation(x=x1, y=y1, ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y',
                           showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.2, arrowcolor='#222')
        fig.add_annotation(x=x1 * 1.18, y=y1 * 1.18, text=f'<b>{feat_name}</b>', showarrow=False,
                           font=dict(size=10, color='#111', family=FONT),
                           bgcolor='rgba(255,255,255,0.7)', borderpad=2)

    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    fig.add_vline(x=0, line=dict(color=GRID, width=1))

    fig.update_layout(**base_layout(
        height=680, margin=dict(l=60, r=60, t=50, b=60),
        xaxis=dict(title=f'PC1 ({var1:.1f}% variance explained)', gridcolor=GRID, gridwidth=0.5),
        yaxis=dict(title=f'PC2 ({var2:.1f}% variance explained)', gridcolor=GRID, gridwidth=0.5),
        legend=dict(title='Resource profile (1995)', font=dict(size=10),
                    bgcolor='rgba(250,250,250,0.85)', bordercolor=GRID, borderwidth=1),
    ))
    return fig

fig03 = chart_03_biplot(pca_1995, pca_model_1995, feat_1995)
save(fig03, '03_cluster__pca_biplot_country_resource_groups', OUT_CLUS, w=1100, h=680)


# ── CHART 04a — Figure 5: Cluster world map (1995) ───────────────────────────
print('\n=== CHART 04a (Figure 5: Cluster world map) ===')

nr_1995_sub   = nr_sample[nr_sample['Year'] == 1995]
cnames_1995   = dict(zip(pca_1995['Cluster'], pca_1995['ClusterLabels']))
fig04a = create_cluster_map(pca_1995, nr_1995_sub, cluster_names_map=cnames_1995)
fig04a.update_layout(height=520)
save(fig04a, '04a_cluster__world_map_1995_resource_profiles', OUT_CLUS, w=1200, h=520)


# ── CHART 05 — Figure 6: ECI vs GDP per capita (animated, 1995-2019) ─────────
print('\n=== CHART 05 (Figure 6: ECI vs GDP animated) ===')

def chart_05_rosling(master_df, pca_df_agg, cluster_colors, cluster_names,
                     arrow_opacity=0.5, arrow_width=2):
    """Animated ECI vs log(GDP pc) Rosling chart with trajectory arrows."""
    data = master_df.copy()
    data['Log GDP per capita'] = np.log(
        data['GDP per capita (constant prices, PPP)'].replace(0, np.nan))
    data['Production_Per_Capita'] = (data['Total_Production_Value']
                                     / data['Population'].replace(0, np.nan))

    c1995 = (data[data['Year'] == 1995][['Country Code', 'Cluster']]
             .rename(columns={'Cluster': 'Cluster_1995'}))
    data  = data.merge(c1995, on='Country Code', how='left')
    data  = data.dropna(subset=['Cluster_1995', 'Log GDP per capita',
                                 'Economic Complexity Index', 'Production_Per_Capita'])
    data['Cluster_1995'] = data['Cluster_1995'].astype(int)

    data['Bubble_Size'] = np.sqrt(data['Production_Per_Capita'])
    mn, mx = data['Bubble_Size'].min(), data['Bubble_Size'].max()
    data['Bubble_Size_Scaled'] = 8 + (data['Bubble_Size'] - mn) / (mx - mn) * 42

    data  = data.sort_values(['Year', 'Country Code'])
    years = sorted(data['Year'].unique())
    countries_list = data['Country Code'].unique()
    clusters_all   = sorted(data['Cluster_1995'].unique())

    cdata = {}
    for code in countries_list:
        cdf    = data[data['Country Code'] == code].sort_values('Year')
        origin = cdf[cdf['Year'] == 1995]
        if len(origin) == 0:
            continue
        cdata[code] = {
            'years': cdf['Year'].values,
            'x': cdf['Log GDP per capita'].values,
            'y': cdf['Economic Complexity Index'].values,
            'x0': origin['Log GDP per capita'].values[0],
            'y0': origin['Economic Complexity Index'].values[0],
            'size': cdf['Bubble_Size_Scaled'].values,
            'name': cdf['Country Name'].iloc[0],
            'cluster': int(cdf['Cluster_1995'].iloc[0]),
            'prod_pc': cdf['Production_Per_Capita'].values,
        }

    valid_countries = list(cdata.keys())
    first_year = years[0]

    fig = go.Figure()

    # Initial state: arrows + markers for first year
    for cl in clusters_all:
        cc    = [c for c in valid_countries if cdata[c]['cluster'] == cl]
        color = cluster_colors.get(cl, '#999999')

        for code in cc:
            cd = cdata[code]
            fig.add_trace(go.Scatter(
                x=[cd['x0'], cd['x0']], y=[cd['y0'], cd['y0']],
                mode='lines', line=dict(color=color, width=arrow_width),
                opacity=arrow_opacity, legendgroup=f'cl_{cl}', showlegend=False, hoverinfo='skip',
            ))

        for code in cc:
            cd = cdata[code]
            idx = np.where(cd['years'] == first_year)[0]
            if len(idx) > 0:
                i = idx[0]
                xv, yv, sv, pv = [cd['x'][i]], [cd['y'][i]], cd['size'][i], cd['prod_pc'][i]
            else:
                xv, yv, sv, pv = [cd['x0']], [cd['y0']], 15, 0
            fig.add_trace(go.Scatter(
                x=xv, y=yv, mode='markers+text',
                marker=dict(size=sv, color=color, opacity=0.85,
                            line=dict(width=1.5, color='white')),
                text=[code], textposition='top center',
                textfont=dict(size=8, color='black'),
                name=cluster_names.get(cl, f'Cluster {cl}'),
                legendgroup=f'cl_{cl}', showlegend=(code == cc[0]),
                customdata=[[cd['name'], pv, first_year]],
                hovertemplate='<b>%{customdata[0]}</b><br>Log GDP pc: %{x:.2f}<br>'
                              'ECI: %{y:.2f}<br>Prod/capita: $%{customdata[1]:,.0f}<br>'
                              'Year: %{customdata[2]}<extra></extra>',
            ))

        for code in cc:
            cd = cdata[code]
            fig.add_trace(go.Scatter(
                x=[cd['x0']], y=[cd['y0']], mode='markers',
                marker=dict(size=5, color=color, opacity=0.6, symbol='circle'),
                legendgroup=f'cl_{cl}', showlegend=False, hoverinfo='skip',
            ))

    # Animation frames
    frames = []
    for year in years:
        fd = []
        for cl in clusters_all:
            cc    = [c for c in valid_countries if cdata[c]['cluster'] == cl]
            color = cluster_colors.get(cl, '#999999')
            for code in cc:
                cd  = cdata[code]
                idx = np.where(cd['years'] == year)[0]
                if len(idx) > 0:
                    xc, yc = cd['x'][idx[0]], cd['y'][idx[0]]
                else:
                    mask = cd['years'] <= year
                    li   = np.where(mask)[0][-1] if mask.any() else 0
                    xc, yc = cd['x'][li], cd['y'][li]
                fd.append(go.Scatter(x=[cd['x0'], xc], y=[cd['y0'], yc],
                                     mode='lines', line=dict(color=color, width=arrow_width),
                                     opacity=arrow_opacity))
            for code in cc:
                cd  = cdata[code]
                idx = np.where(cd['years'] == year)[0]
                if len(idx) > 0:
                    i = idx[0]
                    xv, yv, sv, pv = [cd['x'][i]], [cd['y'][i]], cd['size'][i], cd['prod_pc'][i]
                else:
                    mask = cd['years'] <= year
                    if mask.any():
                        li = np.where(mask)[0][-1]
                        xv, yv, sv, pv = [cd['x'][li]], [cd['y'][li]], cd['size'][li], cd['prod_pc'][li]
                    else:
                        xv, yv, sv, pv = [cd['x0']], [cd['y0']], 15, 0
                fd.append(go.Scatter(
                    x=xv, y=yv, mode='markers+text',
                    marker=dict(size=sv, color=color, opacity=0.85,
                                line=dict(width=1.5, color='white')),
                    text=[code], textposition='top center', textfont=dict(size=8),
                    customdata=[[cd['name'], pv, year]],
                    hovertemplate='<b>%{customdata[0]}</b><br>Log GDP pc: %{x:.2f}<br>'
                                  'ECI: %{y:.2f}<br>Prod/capita: $%{customdata[1]:,.0f}<br>'
                                  'Year: %{customdata[2]}<extra></extra>',
                ))
            for code in cc:
                cd = cdata[code]
                fd.append(go.Scatter(x=[cd['x0']], y=[cd['y0']], mode='markers',
                                     marker=dict(size=5, color=color, opacity=0.6, symbol='circle')))
        frames.append(go.Frame(data=fd, name=str(year)))

    fig.frames = frames

    eci_vals = data['Economic Complexity Index']
    x_vals   = data['Log GDP per capita']
    fig.update_layout(
        xaxis=dict(range=[x_vals.min()-0.2, x_vals.max()+0.2],
                   title='Log GDP per capita (PPP, constant 2017 USD)'),
        yaxis=dict(range=[eci_vals.min()-0.5, eci_vals.max()+0.5],
                   title='Economic Complexity Index'),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family=FONT, color=NAVY),
        legend=dict(title='Resource profile (1995 cluster)', x=1.02, y=0.99),
        updatemenus=[dict(
            type='buttons', showactive=True, x=1.0, y=-0.02,
            buttons=[
                dict(label='Play', method='animate',
                     args=[None, dict(frame=dict(duration=500, redraw=True),
                                      transition=dict(duration=300))]),
                dict(label='Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0), mode='immediate')]),
            ],
        )],
        sliders=[dict(
            active=0, len=0.85, x=0.05, y=-0.12,
            currentvalue=dict(prefix='Year: ', font=dict(size=14)),
            steps=[dict(
                args=[[str(y)], dict(frame=dict(duration=300, redraw=True), mode='immediate')],
                method='animate', label=str(y),
            ) for y in years],
        )],
    )
    return fig

master_ros = load_master()
master_ros = master_ros[master_ros['Country Code'].isin(INCLUDE_LIST)].copy()
master_ros = master_ros.merge(
    pca_agg[['Country Code', 'Cluster', 'ClusterLabels']],
    on='Country Code', how='left',
)
CLUSTER_COLORS_AGG = {
    cid: _LABEL_COLORS_4K.get(
        pca_agg.loc[pca_agg['Cluster'] == cid, 'ClusterLabels'].iloc[0], '#aaa')
    for cid in sorted(pca_agg['Cluster'].unique())
}
CLUSTER_NAMES_AGG = dict(zip(pca_agg['Cluster'], pca_agg['ClusterLabels']))

fig05 = chart_05_rosling(master_ros, pca_agg, CLUSTER_COLORS_AGG, CLUSTER_NAMES_AGG)
save(fig05, '05_cluster__eci_vs_gdp_animated_1995_to_2019', OUT_CLUS, w=1200, h=700)


# ── CHART 14 — ECI distribution shift, 1995 vs 2019 ──────────────────────────
print('\n=== CHART 14 (ECI distribution shift) ===')

master_r = load_master()
df14     = build_sample(master_r)

yr_95 = df14[df14['Year'] == 1995]['Economic Complexity Index'].dropna().sort_values().values
yr_19 = df14[df14['Year'] == 2019]['Economic Complexity Index'].dropna().sort_values().values

fig14 = go.Figure()
for vals, yr, col in [(yr_95, 1995, PALETTE['blue']), (yr_19, 2019, PALETTE['red'])]:
    pcts = np.linspace(0, 100, len(vals))
    fig14.add_trace(go.Scatter(
        x=pcts, y=vals, mode='lines', name=str(yr),
        line=dict(color=col, width=2.5),
        hovertemplate=f'{yr} | P%{{x:.0f}}: %{{y:.3f}}<extra></extra>',
    ))
fig14.update_layout(**base_layout(
    height=500,
    xaxis=dict(title='Percentile', gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(title='Economic Complexity Index', gridcolor=GRID, gridwidth=0.5,
               zeroline=True, zerolinecolor='#ddd', zerolinewidth=1),
    legend=dict(font=dict(size=11)),
))
save(fig14, '14_reg__eci_distribution_shift_1995_vs_2019', OUT_REG, w=1100, h=500)


# ── CHART 15 — ECI median trajectory by cluster ──────────────────────────────
print('\n=== CHART 15 (ECI trajectory by cluster) ===')

master15   = load_master()
clusters15 = load_clusters('1995')[['Country Code', 'Cluster']].drop_duplicates()
df15       = master15[master15['Country Code'].isin(INCLUDE_LIST)].copy()
df15       = df15.merge(clusters15, on='Country Code', how='left')

traj15 = (df15.groupby(['Year', 'Cluster'])['Economic Complexity Index'].median().reset_index())

fig15 = go.Figure()
for cl in sorted(traj15['Cluster'].dropna().unique()):
    sub = traj15[traj15['Cluster'] == cl]
    fig15.add_trace(go.Scatter(
        x=sub['Year'], y=sub['Economic Complexity Index'], mode='lines+markers',
        name=CLUSTER_LABELS.get(int(cl), f'Cluster {int(cl)}'),
        line=dict(color=CLUSTER_COLORS.get(int(cl), '#999'), width=2.2),
        marker=dict(size=5),
        hovertemplate='%{x}: %{y:.3f}<extra>' + CLUSTER_LABELS.get(int(cl), '') + '</extra>',
    ))
fig15.update_layout(**base_layout(
    height=480,
    xaxis=dict(title='Year', gridcolor=GRID, gridwidth=0.5, dtick=5),
    yaxis=dict(title='Median ECI', gridcolor=GRID, gridwidth=0.5,
               zeroline=True, zerolinecolor='#ddd', zerolinewidth=1),
    legend=dict(font=dict(size=10), bgcolor='rgba(255,255,255,0.9)',
                bordercolor=GRID, borderwidth=1),
    hovermode='x unified',
))
save(fig15, '15_reg__eci_mean_trajectory_by_cluster', OUT_REG, w=1100, h=480)


# =============================================================================
#
#   SECTION 3.4 — ECONOMETRICS
#
# =============================================================================

# ── CHART 17 — HCI x Production interaction scatter ──────────────────────────
print('\n=== CHART 17 (HCI x Production interaction) ===')

master17 = load_master()
df17     = build_sample(master17)
df17['prod_pc']     = df17['Total_Production_Value'] / df17['Population'].replace(0, np.nan)
df17['log_HCI']     = np.log1p(df17['Human capital index'])
df17['log_prod_pc'] = np.log1p(df17['prod_pc'])

country_avg = (df17[['Country Code', 'log_HCI', 'Economic Complexity Index', 'log_prod_pc']]
               .dropna().groupby('Country Code').mean().reset_index())
country_avg['Prod_quartile'] = pd.qcut(
    country_avg['log_prod_pc'], q=4,
    labels=['Q1 \u2014 Low production', 'Q2', 'Q3', 'Q4 \u2014 High production'],
)

q_colors = [PALETTE['light_blue'], PALETTE['blue'], PALETTE['orange'], PALETTE['red']]

fig17 = go.Figure()
for q, col in zip(['Q1 \u2014 Low production', 'Q2', 'Q3', 'Q4 \u2014 High production'], q_colors):
    sub = country_avg[country_avg['Prod_quartile'] == q]
    fig17.add_trace(go.Scatter(
        x=sub['log_HCI'], y=sub['Economic Complexity Index'],
        mode='markers+text', text=sub['Country Code'],
        textposition='top center', textfont=dict(size=8, color='#555'),
        marker=dict(color=col, size=9, opacity=0.85, line=dict(width=0.8, color='white')),
        name=f'Production {q}',
        hovertemplate='<b>%{text}</b><br>log(HCI): %{x:.2f}<br>ECI: %{y:.2f}<extra></extra>',
    ))
fig17.update_layout(**base_layout(
    height=540,
    xaxis=dict(title='log(Human Capital Index)', gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(title='Economic Complexity Index', gridcolor=GRID, gridwidth=0.5),
    legend=dict(title=dict(text='Avg. Production p.c. Quartile'),
                font=dict(size=10), bgcolor='rgba(255,255,255,0.9)',
                bordercolor=GRID, borderwidth=1),
))
save(fig17, '17_reg__hci_production_interaction_effect_on_eci', OUT_REG, w=1100, h=540)


# ── CHART 16 — Coefficient comparison: Model 3a vs 3b ────────────────────────
print('\n=== CHART 16 (Regression coefficients 3a vs 3b) ===')

try:
    import statsmodels.api as sm

    master16 = load_master()
    df16     = build_sample(master16)
    df16['Total_Production_Value_Per_Capita'] = (
        df16['Total_Production_Value'] / df16['Population'].replace(0, np.nan))
    df16['log_HCI']              = np.log1p(df16['Human capital index'])
    df16['log_GFCF']             = np.log1p(
        df16['Gross fixed capital formation, all, Constant prices, Percent of GDP'])
    df16['log_Production_Value'] = np.log1p(df16['Total_Production_Value_Per_Capita'])

    for col in ['log_HCI', 'log_GFCF', 'log_Production_Value']:
        df16[f'{col}_c'] = df16[col] - df16[col].mean()

    df16['log_HCI_x_log_Production']  = df16['log_HCI_c']  * df16['log_Production_Value_c']
    df16['log_GFCF_x_log_Production'] = df16['log_GFCF_c'] * df16['log_Production_Value_c']
    df16 = df16.sort_values(['Country Code', 'Year'])
    df16['ECI_lag1'] = df16.groupby('Country Code')['Economic Complexity Index'].shift(1)

    reg3_input   = ['log_HCI', 'log_GFCF', 'Political stability \u2014 estimate',
                    'Rule of law index', 'log_Production_Value', 'Trade (% of GDP)']
    INTERACT_VARS = ['log_HCI_x_log_Production', 'log_GFCF_x_log_Production']

    def _fit_dk(y, X, time, groups):
        import types
        raw = sm.OLS(y, X).fit()
        robust = raw.get_robustcov_results(
            cov_type='HAC-Groupsum', time=time, groups=groups,
            maxlags=2, kernel='bartlett', use_correction=True,
        )
        return types.SimpleNamespace(
            params  = pd.Series(robust.params,  index=X.columns),
            bse     = pd.Series(robust.bse,     index=X.columns),
            pvalues = pd.Series(robust.pvalues, index=X.columns),
        )

    reg3_cols = reg3_input + INTERACT_VARS + ['Economic Complexity Index', 'Country Code', 'Year']
    reg3_df   = df16[reg3_cols].dropna()
    m3a = _fit_dk(reg3_df['Economic Complexity Index'],
                  sm.add_constant(reg3_df[reg3_input + INTERACT_VARS]),
                  reg3_df['Year'], reg3_df['Country Code'])

    reg3b_cols = reg3_cols + ['ECI_lag1']
    reg3b_df   = df16[reg3b_cols].dropna()
    m3b = _fit_dk(reg3b_df['Economic Complexity Index'],
                  sm.add_constant(reg3b_df[reg3_input + INTERACT_VARS + ['ECI_lag1']]),
                  reg3b_df['Year'], reg3b_df['Country Code'])

    plot_vars = [v for v in reg3_input + INTERACT_VARS if v != 'const']
    labels16  = [v.replace('log_', 'log(').replace('_x_', ') \u00d7 log(') + (')' if 'log_' in v else '')
                 for v in plot_vars]

    fig16 = go.Figure()
    for model, col, name in [
        (m3a, PALETTE['blue'],       'Model 3a (no lag)'),
        (m3b, PALETTE['light_blue'], 'Model 3b (with lag)'),
    ]:
        coefs  = [model.params.get(v, np.nan) for v in plot_vars]
        lowers = [model.params.get(v, np.nan) - 1.96 * model.bse.get(v, np.nan) for v in plot_vars]
        uppers = [model.params.get(v, np.nan) + 1.96 * model.bse.get(v, np.nan) for v in plot_vars]
        fig16.add_trace(go.Scatter(
            y=labels16, x=coefs,
            error_x=dict(type='data', symmetric=False,
                         array=[u - c for c, u in zip(coefs, uppers)],
                         arrayminus=[c - l for c, l in zip(coefs, lowers)],
                         color=col, thickness=1.5, width=5),
            mode='markers', marker=dict(color=col, size=8, symbol='circle'), name=name,
        ))

    fig16.add_vline(x=0, line=dict(color='#c9cfd6', width=1.5, dash='dash'))
    fig16.update_layout(**base_layout(
        height=700, margin=dict(l=220, r=100, t=10, b=50),
        xaxis=dict(title='Coefficient (95% CI)', gridcolor=GRID, gridwidth=0.5, zeroline=False),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(font=dict(size=11), orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
    ))
    save(fig16, '16_reg__coefficients_model3a_vs_model3b', OUT_REG, w=1100, h=700)

except ImportError:
    print('  SKIPPED: statsmodels not available')
except Exception as e:
    print(f'  SKIPPED (error): {e}')


# ── CHART 30 — Regression variable correlation heatmap ────────────────────────
print('\n=== CHART 30 (Regression correlation heatmap) ===')

master30 = load_master()
df30     = build_sample(master30)

REG_CORR_VARS = {
    'Economic Complexity Index': 'ECI',
    'Human capital index':       'HCI',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'GFCF',
    'Political stability \u2014 estimate': 'Pol. Stability',
    'Rule of law index':              'Rule of Law',
    'Total_Production_Value': 'Production Value',
    'Trade (% of GDP)':               'Trade',
    'Domestic credit to private sector (% of GDP)': 'Domestic Credit',
    'Access to electricity (% of population)':      'Electricity',
    'Urban population (% of total population)':    'Urban Pop.',
    'GDP per capita (constant prices, PPP)':        'GDP p.c.',
    'Total natural resources rents (% of GDP)':    'NR Rents',
}

corr_cols30 = [c for c in REG_CORR_VARS if c in df30.columns]
corr_df30   = df30[corr_cols30].rename(columns=REG_CORR_VARS).corr().round(2)
labels30 = list(corr_df30.columns)
z30      = corr_df30.values

fig30 = go.Figure(go.Heatmap(
    z=z30, x=labels30, y=labels30,
    colorscale=[[0.0, PALETTE['red']], [0.5, '#fafafa'], [1.0, PALETTE['blue']]],
    zmid=0, zmin=-1, zmax=1,
    text=z30.round(2), texttemplate='%{text}',
    textfont=dict(size=9, family=FONT),
    hovertemplate='%{x} \u00d7 %{y}: %{z:.2f}<extra></extra>',
    colorbar=dict(thickness=14, len=0.9, tickfont=dict(size=10, family=FONT)),
))
fig30.update_layout(**base_layout(
    height=640, margin=dict(l=160, r=80, t=30, b=180),
    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
    yaxis=dict(tickfont=dict(size=9)),
))
save(fig30, '30_reg__regression_variable_correlation_heatmap', OUT_REG, w=1000, h=700)


# =============================================================================
#
#   SECTION 3.5 — MACHINE LEARNING
#
# =============================================================================

# ── CHART 07 — Figure 7: ML Feature Importance Consensus ─────────────────────
print('\n=== CHART 07 (Figure 7: ML importance consensus) ===')

_imp_path = os.path.join('Graphics', 'NB5', 'all_importance.csv')
if os.path.exists(_imp_path):
    imp = pd.read_csv(_imp_path)
    imp = imp[~imp['Feature'].apply(lambda f: any(e in f for e in LABEL_EXCL))]

    lin_cols = [c for c in ['LASSO', 'Ridge', 'Elastic Net'] if c in imp.columns]
    sort_col = 'Elastic Net' if 'Elastic Net' in imp.columns else lin_cols[0]
    imp = imp.sort_values(sort_col, ascending=False).head(12).reset_index(drop=True)
    imp = imp.iloc[::-1].reset_index(drop=True)
    imp['Label'] = imp['Feature'].apply(shorten_feat)

    fig07 = go.Figure()
    for _, row in imp.iterrows():
        vals = [row[c] for c in lin_cols if not pd.isna(row[c])]
        if len(vals) >= 2:
            fig07.add_trace(go.Scatter(
                x=[min(vals), max(vals)], y=[row['Label'], row['Label']],
                mode='lines', line=dict(color='#c0c8d4', width=3),
                showlegend=False, hoverinfo='skip',
            ))

    for mname, sym, col in [
        ('LASSO', 'circle', PALETTE['lasso']),
        ('Ridge', 'square', PALETTE['ridge']),
        ('Elastic Net', 'triangle-up', PALETTE['en']),
    ]:
        if mname not in imp.columns:
            continue
        fig07.add_trace(go.Scatter(
            x=imp[mname], y=imp['Label'], mode='markers',
            marker=dict(symbol=sym, size=13, color=col, line=dict(color='white', width=1.5)),
            name=mname,
            hovertemplate=f'%{{y}}: %{{x:.3f}}<extra>{mname}</extra>',
        ))

    x_max = imp[lin_cols].max().max()
    fig07.update_layout(**base_layout(
        height=560, margin=dict(l=200, r=80, t=70, b=80),
        xaxis=dict(title='Normalised Feature Importance (min-max, 0-1)',
                   range=[-0.02, x_max + 0.1], gridcolor=GRID, gridwidth=0.5),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=11)),
    ))
    save(fig07, '07_ml__feature_importance_consensus_three_models', OUT_ML, w=1600, h=800)
else:
    print('  SKIPPED: all_importance.csv not found')


# ── CHART 08 — Figure 8: ML Standardised Coefficients ────────────────────────
print('\n=== CHART 08 (Figure 8: ML coefficients) ===')

_tbl_path = os.path.join('Graphics', 'NB5', 'coefficient_summary_table.csv')
if os.path.exists(_tbl_path):
    tbl = pd.read_csv(_tbl_path)
    tbl = tbl[~tbl['Feature'].apply(lambda f: any(e in f for e in LABEL_EXCL))]
    tbl['abs_en'] = tbl['Elastic Net'].abs()
    top = tbl.nlargest(12, 'abs_en').sort_values('abs_en', ascending=True).reset_index(drop=True)

    fig08 = go.Figure()
    fig08.add_vline(x=0, line=dict(color='#444', width=1.5))
    for mname, col in [('LASSO', PALETTE['lasso']), ('Ridge', PALETTE['ridge']), ('Elastic Net', PALETTE['en'])]:
        if mname not in top.columns:
            continue
        fig08.add_trace(go.Bar(
            y=top['Feature'], x=top[mname], orientation='h', name=mname,
            marker=dict(color=col, opacity=0.88, line=dict(color='white', width=0.5)),
            hovertemplate=f'%{{y}}: %{{x:+.3f}}<extra>{mname}</extra>',
        ))

    fig08.update_layout(**base_layout(
        barmode='group', height=620, margin=dict(l=200, r=80, t=70, b=60),
        xaxis=dict(title='Coefficient (standardised inputs)', gridcolor=GRID, gridwidth=0.5, zeroline=False),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=11)),
    ))
    save(fig08, '08_ml__standardised_coefficients_lasso_ridge_en', OUT_ML, w=1100, h=620)
else:
    print('  SKIPPED: coefficient_summary_table.csv not found')


# ── CHART 27 — Figure 11: Random Forest Feature Importance ───────────────────
print('\n=== CHART 27 (Figure 11: RF importance) ===')

if os.path.exists(_imp_path):
    imp27 = pd.read_csv(_imp_path)
    imp27 = imp27[imp27['Model'] == 'RandomForest'] if 'Model' in imp27.columns else imp27.copy()

    if 'Model' not in imp27.columns:
        rf_col = 'Random Forest' if 'Random Forest' in imp27.columns else None
        if rf_col:
            imp27 = imp27[['Feature', rf_col]].rename(columns={rf_col: 'Importance'}).dropna()
    else:
        imp27 = imp27.rename(columns={'Importance': 'Importance'}) if 'Importance' in imp27.columns else imp27

    if 'Importance' not in imp27.columns:
        rf_col = next((c for c in imp27.columns if 'Forest' in c or 'RF' in c or 'rf' in c.lower()), None)
        if rf_col:
            imp27 = imp27[['Feature', rf_col]].rename(columns={rf_col: 'Importance'}).dropna()

    if 'Importance' in imp27.columns and 'Feature' in imp27.columns:
        imp27 = imp27[~imp27['Feature'].apply(lambda f: any(e in f for e in LABEL_EXCL))].copy()
        imp27 = imp27.sort_values('Importance', ascending=False).head(15).iloc[::-1].reset_index(drop=True)
        imp27['Label'] = imp27['Feature'].apply(shorten_feat)

        fig27 = go.Figure(go.Bar(
            x=imp27['Importance'], y=imp27['Label'], orientation='h',
            marker=dict(color=PALETTE['rf'], opacity=0.88, line=dict(color='white', width=0.5)),
            hovertemplate='%{y}: %{x:.3f}<extra>Random Forest</extra>',
        ))
        fig27.update_layout(**base_layout(
            height=540, margin=dict(l=200, r=80, t=60, b=60),
            xaxis=dict(title='Feature Importance', gridcolor=GRID, gridwidth=0.5),
            yaxis=dict(tickfont=dict(size=11)),
        ))
        save(fig27, '27_ml__random_forest_feature_importance', OUT_ML, w=1100, h=540)
    else:
        print('  SKIPPED: could not extract RF importance column')
else:
    print('  SKIPPED: all_importance.csv not found')


# ── CHART 09 — Figure 9: Train vs Test R^2 ───────────────────────────────────
print('\n=== CHART 09 (Figure 9: Train vs Test R2) ===')

_perf_l_path = os.path.join('Graphics', 'NB5', 'model_performance_level.csv')
_perf_d_path = os.path.join('Graphics', 'NB5', 'model_performance_delta.csv')
if os.path.exists(_perf_l_path) and os.path.exists(_perf_d_path):
    perf_l = pd.read_csv(_perf_l_path)
    perf_d = pd.read_csv(_perf_d_path)
    perf_l = perf_l[perf_l['Model'] != 'XGBoost'].reset_index(drop=True)
    perf_d = perf_d[perf_d['Model'] != 'XGBoost'].reset_index(drop=True)

    fig09 = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                          subplot_titles=['ECI Level', '\u0394ECI'])

    for col_idx, perf in enumerate([perf_l, perf_d], 1):
        models   = perf['Model'].tolist()
        train_r2 = perf['Train R\u00b2'].tolist()
        test_r2  = perf['Test R\u00b2'].tolist()

        fig09.add_trace(go.Bar(
            x=models, y=train_r2, name='Train R\u00b2', legendgroup='train',
            showlegend=(col_idx == 1),
            marker=dict(color='#4a6fa5', opacity=0.88, line=dict(width=0)),
            text=[f'{v:.3f}' for v in train_r2],
            textposition='outside', textfont=dict(size=10, color='#4a6fa5'),
            hovertemplate='%{x} Train: %{y:.3f}<extra></extra>',
        ), row=1, col=col_idx)

        fig09.add_trace(go.Bar(
            x=models, y=test_r2, name='Test R\u00b2', legendgroup='test',
            showlegend=(col_idx == 1),
            marker=dict(color='#c23a3a', opacity=0.88, line=dict(width=0)),
            text=[f'{v:.3f}' for v in test_r2],
            textposition='outside', textfont=dict(size=10, color='#c23a3a'),
            hovertemplate='%{x} Test: %{y:.3f}<extra></extra>',
        ), row=1, col=col_idx)

        fig09.update_xaxes(tickangle=-30, tickfont=dict(size=11), row=1, col=col_idx)
        fig09.update_yaxes(title_text='R\u00b2', gridcolor=GRID, gridwidth=0.5,
                           tickfont=dict(size=11), row=1, col=col_idx)

    fig09.update_layout(**base_layout(
        barmode='group', height=480, margin=dict(l=60, r=40, t=80, b=80),
        legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='center', x=0.5, font=dict(size=12)),
    ))
    save(fig09, '09_ml__train_vs_test_r2_all_models', OUT_ML, w=1200, h=480)
else:
    print('  SKIPPED: model_performance CSVs not found')


# ── CHART 10 — Figure 10: Actual vs Predicted ECI (test set) ─────────────────
print('\n=== CHART 10 (Figure 10: Predicted vs Actual) ===')

_preds_path = os.path.join('Graphics', 'NB5', 'test_predictions.csv')
if os.path.exists(_preds_path):
    preds = pd.read_csv(_preds_path)

    fig10 = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)

    for col_idx, (actual_col, pred_col, lbl) in enumerate([
        ('Actual_ECI', 'Predicted_ECI', 'ECI'),
        ('Actual_Delta', 'Predicted_Delta', '\u0394ECI'),
    ], 1):
        if actual_col not in preds.columns or pred_col not in preds.columns:
            continue
        actual = preds[actual_col].dropna().values
        pred   = preds.loc[preds[actual_col].notna(), pred_col].values
        codes  = preds.loc[preds[actual_col].notna(), 'Country Code'].values
        names  = preds.loc[preds[actual_col].notna(), 'Country Name'].values

        lims = [min(actual.min(), pred.min()) - 0.1, max(actual.max(), pred.max()) + 0.1]
        mid  = 0.0

        for x0, x1, y0, y1, fc in [
            (lims[0], mid, lims[0], mid, 'rgba(46,125,74,0.07)'),
            (mid, lims[1], mid, lims[1], 'rgba(46,125,74,0.07)'),
            (lims[0], mid, mid, lims[1], 'rgba(194,58,58,0.07)'),
            (mid, lims[1], lims[0], mid, 'rgba(194,58,58,0.07)'),
        ]:
            fig10.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1,
                            fillcolor=fc, line=dict(width=0), layer='below', row=1, col=col_idx)

        fig10.add_trace(go.Scatter(
            x=[lims[0], lims[1]], y=[lims[0], lims[1]],
            mode='lines', line=dict(color=PALETTE['red'], width=1.5, dash='dash'),
            name='45\u00b0 line', showlegend=(col_idx == 1), legendgroup='line45',
        ), row=1, col=col_idx)

        resid   = np.abs(actual - pred)
        top_idx = set(np.argsort(resid)[::-1][:5])
        mask_n  = np.array([i not in top_idx for i in range(len(actual))])

        fig10.add_trace(go.Scatter(
            x=actual[mask_n], y=pred[mask_n], mode='markers',
            marker=dict(size=6, color=PALETTE['blue'], opacity=0.65, line=dict(color='white', width=0.5)),
            name='Test obs.', showlegend=(col_idx == 1), legendgroup='obs',
            customdata=np.stack([codes[mask_n], names[mask_n]], axis=1),
            hovertemplate='<b>%{customdata[1]}</b><br>Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>',
        ), row=1, col=col_idx)

        out_idx = list(top_idx)
        fig10.add_trace(go.Scatter(
            x=actual[out_idx], y=pred[out_idx], mode='markers+text',
            marker=dict(size=9, color=PALETTE['orange'], opacity=0.9, line=dict(color='white', width=1)),
            text=codes[out_idx], textposition='top center', textfont=dict(size=9),
            name='Largest residuals', showlegend=(col_idx == 1), legendgroup='outliers',
            customdata=np.stack([codes[out_idx], names[out_idx]], axis=1),
            hovertemplate='<b>%{customdata[1]}</b><br>Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>',
        ), row=1, col=col_idx)

        fig10.add_hline(y=0, line=dict(color=GRID, width=1), row=1, col=col_idx)
        fig10.add_vline(x=0, line=dict(color=GRID, width=1), row=1, col=col_idx)
        fig10.update_xaxes(title_text=f'Actual {lbl} (test set)', range=lims,
                           gridcolor=GRID, gridwidth=0.5, row=1, col=col_idx)
        fig10.update_yaxes(title_text=f'Predicted {lbl}', range=lims,
                           gridcolor=GRID, gridwidth=0.5, row=1, col=col_idx)

    fig10.update_layout(**base_layout(
        height=560, margin=dict(l=70, r=40, t=60, b=80),
        legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='center', x=0.5, font=dict(size=10)),
    ))
    save(fig10, '10_ml__actual_vs_predicted_eci_test_set', OUT_ML, w=1100, h=560)
else:
    print('  SKIPPED: test_predictions.csv not found')


# ── CHART 11 — Figure Z: ECI Forecast (top improvers / decliners) ────────────
print('\n=== CHART 11 (Figure Z: ECI forecast 2020-2030) ===')

_fc_path   = os.path.join('Graphics', 'NB5', 'ECI_Forecast_2020_2030.csv')
_rank_path = os.path.join('Graphics', 'NB5', 'Country_Ranking_2020_2030.csv')
if os.path.exists(_fc_path) and os.path.exists(_rank_path):
    fc   = pd.read_csv(_fc_path)
    rank = pd.read_csv(_rank_path)

    if 'Total_Change' in rank.columns:
        rank = rank.sort_values('Total_Change', ascending=False)
    elif 'Ens_Delta_2019_2030' in rank.columns:
        rank = rank.sort_values('Ens_Delta_2019_2030', ascending=False)
        top3    = rank.head(3)['Country Code'].tolist()
        bottom3 = rank.tail(3)['Country Code'].tolist()
    else:
        top3    = rank.head(3)['Country Code'].tolist()
        bottom3 = rank.tail(3)['Country Code'].tolist()

    CASE_STUDIES = ['COG', 'AZE', 'CHL']
    master_fc    = load_master()
    hist_fc      = (master_fc[master_fc['Country Code'].isin(INCLUDE_LIST)]
                    [['Country Code', 'Country Name', 'Year', 'Economic Complexity Index']]
                    .dropna())

    TOP_COL  = '#2e7d4a'
    BOT_COL  = '#c23a3a'
    CASE_COL = '#4a6fa5'
    GREY_FC  = '#c0c8d4'

    Y_RANGE = [-3.5, 2.5]

    def _add_country_traces_fc(fig, cc, cname, color, lw=2, opacity=1.0,
                               legendgroup='', row_n=1, col_n=1):
        h = hist_fc[hist_fc['Country Code'] == cc].sort_values('Year')
        f = fc[fc['Country Code'] == cc].sort_values('Year')

        if len(h) > 0:
            fig.add_trace(go.Scatter(
                x=h['Year'], y=h['Economic Complexity Index'], mode='lines',
                line=dict(color=color, width=lw), opacity=opacity,
                legendgroup=legendgroup, showlegend=False, hoverinfo='skip',
            ), row=row_n, col=col_n)

        ens_col = 'Ensemble' if 'Ensemble' in f.columns else f.columns[-1]
        if len(f) > 0:
            # Bridge from last historical point
            bridge_x, bridge_y = [], []
            if len(h) > 0:
                bridge_x.append(h['Year'].iloc[-1])
                bridge_y.append(h['Economic Complexity Index'].iloc[-1])
            bridge_x.extend(f['Year'].tolist())
            bridge_y.extend(f[ens_col].tolist())

            fig.add_trace(go.Scatter(
                x=bridge_x, y=bridge_y, mode='lines',
                line=dict(color=color, width=lw, dash='dash'), opacity=opacity,
                legendgroup=legendgroup, showlegend=False,
                customdata=[[cname]] * len(bridge_x),
                hovertemplate='<b>%{customdata[0]}</b><br>Year: %{x}<br>ECI: %{y:.3f}<extra></extra>',
            ), row=row_n, col=col_n)

        ens = bridge_y if len(f) > 0 else []
        return float(ens[-1]) if ens else None

    def _deconflict_lbl(label_info, min_gap=0.22):
        sorted_lbl = sorted(label_info.items(), key=lambda x: x[1][0])
        adjusted, floor_y = {}, None
        for cc, (y_act, _) in sorted_lbl:
            y_place = y_act if floor_y is None else max(y_act, floor_y + min_gap)
            adjusted[cc] = y_place
            floor_y = y_place
        return adjusted

    fig11 = make_subplots(rows=1, cols=2, horizontal_spacing=0.07)

    for panel_col, highlight_group, highlight_col, grp_name in [
        (1, top3, TOP_COL, 'top3'),
        (2, bottom3, BOT_COL, 'bottom3'),
    ]:
        fig11.add_vrect(x0=2019.5, x1=2030.5, fillcolor='rgba(200,210,225,0.18)',
                        line=dict(width=0), layer='below', row=1, col=panel_col)
        fig11.add_vline(x=2019.5, line=dict(color='#aaa', width=1.5, dash='dot'), row=1, col=panel_col)

        highlighted_here = set(CASE_STUDIES + highlight_group)

        for cc in INCLUDE_LIST:
            if cc in highlighted_here:
                continue
            row_ = rank[rank['Country Code'] == cc]
            cname_ = row_['Country'].values[0] if len(row_) else cc
            _add_country_traces_fc(fig11, cc, cname_, GREY_FC, lw=0.7, opacity=0.3,
                                   legendgroup='others', row_n=1, col_n=panel_col)

        label_info = {}
        for cc in highlight_group:
            row_ = rank[rank['Country Code'] == cc]
            cname_ = row_['Country'].values[0] if len(row_) else cc
            y_end = _add_country_traces_fc(fig11, cc, cname_, highlight_col, lw=2.2, opacity=1.0,
                                           legendgroup=grp_name, row_n=1, col_n=panel_col)
            if y_end is not None:
                label_info[cc] = (y_end, highlight_col)

        for cc in CASE_STUDIES:
            row_ = rank[rank['Country Code'] == cc]
            cname_ = row_['Country'].values[0] if len(row_) else cc
            y_end = _add_country_traces_fc(fig11, cc, cname_, CASE_COL, lw=2.5, opacity=1.0,
                                           legendgroup='cases', row_n=1, col_n=panel_col)
            if y_end is not None:
                label_info[cc] = (y_end, CASE_COL)

        adjusted_y = _deconflict_lbl(label_info)
        xref = 'x' if panel_col == 1 else 'x2'
        yref = 'y' if panel_col == 1 else 'y2'

        for cc, (y_act, col) in label_info.items():
            fig11.add_annotation(
                x=2030, y=y_act, ax=2031, ay=adjusted_y[cc],
                axref=xref, ayref=yref, xref=xref, yref=yref,
                text=f'<b>{cc}</b>', showarrow=True,
                arrowhead=2, arrowwidth=1, arrowsize=0.8, arrowcolor=col,
                font=dict(size=9.5, color=col, family=FONT), xanchor='left', yanchor='middle',
            )

        fig11.update_xaxes(title_text='Year', gridcolor=GRID, gridwidth=0.5,
                           dtick=5, range=[1994, 2033], row=1, col=panel_col)
        fig11.update_yaxes(
            title_text='Economic Complexity Index' if panel_col == 1 else '',
            range=Y_RANGE, gridcolor=GRID, gridwidth=0.5, row=1, col=panel_col)

    for lbl, col, rk, grp in [
        ('Case studies \u2014 COG \u00b7 AZE \u00b7 CHL', CASE_COL, 1, 'cases'),
        ('Top 3 improvers \u2014 GNQ \u00b7 MNG \u00b7 ECU', TOP_COL, 2, 'top3'),
        ('Bottom 3 decliners \u2014 ZWE \u00b7 SAU \u00b7 KAZ', BOT_COL, 3, 'bottom3'),
        ('Other countries', GREY_FC, 4, 'others'),
    ]:
        fig11.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                   line=dict(color=col, width=2.5), name=lbl,
                                   legendgroup=grp, showlegend=True, legendrank=rk))
    fig11.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                               line=dict(color='#888', width=1.5), name='\u2500\u2500 Historical',
                               showlegend=True, legendrank=10))
    fig11.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                               line=dict(color='#888', width=1.5, dash='dash'),
                               name='- - Forecast', showlegend=True, legendrank=11))

    fig11.update_layout(**base_layout(
        height=620, margin=dict(l=70, r=20, t=70, b=110),
        legend=dict(
            orientation='h', font=dict(size=9.5),
            bgcolor='rgba(255,255,255,0.92)', bordercolor=GRID, borderwidth=1,
            x=0.0, y=-0.15, xanchor='left', yanchor='top', tracegroupgap=0,
        ),
    ))
    save(fig11, '11_ml__eci_forecast_top_improvers_2020_2030', OUT_ML, w=1200, h=620)
else:
    print('  SKIPPED: ECI_Forecast or Country_Ranking CSV not found')


# =============================================================================
print('\n' + '=' * 60)
print('generate_report_charts.py complete.')
print('17 report-body charts written.')
print('=' * 60)
