"""
generate_charts.py — Single script that produces ALL Capstone charts (01–34).

Run from project root:
    python3 scripts/generate_charts.py

Or from any directory (the walk-up loop below sets cwd to the project root).
"""

# ---------------------------------------------------------------------------
# 0. Walk up to project root (contains 'intermediary/' and 'Final/')
# ---------------------------------------------------------------------------
import os, sys

def _find_root(marker='intermediary'):
    d = '/Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP'
    for _ in range(6):
        if os.path.isdir(os.path.join(d, marker)):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"Could not find project root (looking for '{marker}' dir).")

ROOT = _find_root()
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

# ---------------------------------------------------------------------------
# 1. Standard library + third-party imports
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# 2. Import shared utilities from viz_utils
# ---------------------------------------------------------------------------
from viz_utils import (
    PALETTE, FONT, BG, NAVY, GRID, WRITE_CONFIG,
    base_layout, save,
    load_master, load_master_wide, load_clusters, load_nr, load_nb5, load_bootstrap,
    INCLUDE_LIST, resource_rich_codes, build_sample, shorten_feat,
    CLUSTER_LABELS, CLUSTER_COLORS, LABEL_TO_COLOR,
    analyze_country_missingness,
)

# ---------------------------------------------------------------------------
# 3. Output directory
# ---------------------------------------------------------------------------
OUT_DESC = 'Final/charts/descriptive'
OUT_CLUS = 'Final/charts/clusters'
OUT_ML   = 'Final/charts/ml'
OUT_REG  = 'Final/charts/regression'
OUT_MISC = 'Final/charts/misc'
OUT_CASE = 'Final/charts/case_studies'
for _d in [OUT_DESC, OUT_CLUS, OUT_ML, OUT_REG, OUT_MISC, OUT_CASE]:
    os.makedirs(_d, exist_ok=True)
OUT = OUT_DESC  # default (unused after per-chart mapping below)

# ---------------------------------------------------------------------------
# 4. Helper: hex → rgb tuple (used in chart 11)
# ---------------------------------------------------------------------------
def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# 5. Clustering pipeline (re-runs PCA + KMeans; used for charts 03, 04a/b/c/d)
# ---------------------------------------------------------------------------
_LABEL_COLORS_4K = {
    'Petrostates':      '#d4853b',
    'Oil Exporters':    '#4a6fa5',
    'Major Producers':  '#2e7d4a',
    'Limited Resources':'#c23a3a',
}

_LABEL_COLORS_4F = {
    'Petrostates':       '#d4853b',
    'Oil Exporters':     '#c23a3a',
    'Oil & Minerals':    '#7a5c9e',
    'Mineral Exporters': '#2e7d4a',
    'Low Resource':      '#4a6fa5',
}


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


def run_clustering_4feat(nr_data, year_filter=None, n_clusters=5, random_state=42):
    """Cluster on Oil / Natural Gas / Coal / Minerals (aggregate), k=5."""
    df = nr_data.copy()
    if year_filter is not None:
        df = df[df['Year'] == year_filter]

    KEEP = ['Oil', 'Natural Gas', 'Coal']
    df['_Category'] = df['Resource'].apply(lambda r: r if r in KEEP else 'Minerals')
    df_agg = (df.groupby(['Country', 'Country Code', 'Year', 'Population', '_Category'])
              ['Production_TotalValue'].sum().reset_index())

    pivot = df_agg.pivot_table(
        index=['Country', 'Country Code', 'Year', 'Population'],
        columns='_Category', values='Production_TotalValue',
    ).reset_index().fillna(0)

    feat_cols = [c for c in ['Coal', 'Minerals', 'Natural Gas', 'Oil'] if c in pivot.columns]
    pivot[feat_cols] = pivot[feat_cols].div(pivot['Population'], axis=0)
    pivot = pivot.fillna(0)

    X = np.log1p(pivot[feat_cols])
    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(Xp)

    pca_df = pd.DataFrame({
        'Country':      pivot['Country'],
        'Country Code': pivot['Country Code'],
        'Year':         pivot['Year'],
        'PC1': Xp[:, 0], 'PC2': Xp[:, 1],
        'Cluster': labels,
    })

    centroids = km.cluster_centers_
    pc1_rank  = list(np.argsort(-centroids[:, 0]))
    pc2_rank  = list(np.argsort(-centroids[:, 1]))

    label_map, labeled = {}, set()
    label_map[pc1_rank[0]] = 'Petrostates'; labeled.add(pc1_rank[0])
    min_id = next(c for c in pc2_rank if c not in labeled)
    label_map[min_id] = 'Mineral Exporters'; labeled.add(min_id)
    remaining = [c for c in pc1_rank if c not in labeled]
    label_map[remaining[0]] = 'Oil & Minerals'
    label_map[remaining[1]] = 'Oil Exporters'
    label_map[remaining[2]] = 'Low Resource'

    pca_df['ClusterLabels'] = pca_df['Cluster'].map(label_map)
    return pca_df


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


# ============================================================================
# Pre-run: build cluster data (needed by charts 01, 03, 04a/b/c/d, 05, 26)
# ============================================================================
print('Loading NaturalResource.csv and running clustering pipeline...')
nr_full   = load_nr()
nr_sample = nr_full[nr_full['Country Code'].isin(INCLUDE_LIST)]

pca_1995, pca_model_1995, feat_1995 = run_clustering(nr_sample, year_filter=1995)
pca_2019, pca_model_2019, feat_2019 = run_clustering(nr_sample, year_filter=2019)
pca_agg,  pca_model_agg,  feat_agg  = run_clustering(nr_sample, agg_years=[1995, 1999, 2005])
print('  Clustering done.')


# =============================================================================
# CHART 00 — Data Sources Overview (intro)
# =============================================================================
print('\n=== CHART 00 ===')

import math as _math

_variables_00 = [
    # --- World Bank ---
    ('Total NR rents (% GDP)',         'Resource Rents',    'World Bank', '1995–2019'),
    ('Mineral rents (% GDP)',           'Resource Rents',    'World Bank', '1995–2019'),
    ('Natural gas rents (% GDP)',       'Resource Rents',    'World Bank', '1995–2019'),
    ('Oil rents (% GDP)',               'Resource Rents',    'World Bank', '1995–2019'),
    ('Manufacturing (% GDP)',           'GDP Structure',     'World Bank', '1995–2019'),
    ('Industry (% GDP)',                'GDP Structure',     'World Bank', '1995–2019'),
    ('High-tech exports (%)',           'GDP Structure',     'World Bank', '1995–2019'),
    ('Agriculture (% GDP)',             'GDP Structure',     'World Bank', '1995–2019'),
    ('Services (% GDP)',                'GDP Structure',     'World Bank', '1995–2019'),
    ('Gross savings (% GNI)',           'Finance',           'World Bank', '1995–2019'),
    ('NR depletion (% GNI)',            'Finance',           'World Bank', '1995–2019'),
    ('Domestic credit (% GDP)',         'Finance',           'World Bank', '1995–2019'),
    ('IMF credit (USD)',                'Finance',           'World Bank', '1995–2019'),
    ('Real interest rate (%)',          'Macro',             'World Bank', '1995–2019'),
    ('Lending interest rate (%)',       'Macro',             'World Bank', '1995–2019'),
    ('Inflation (%)',                   'Macro',             'World Bank', '1995–2019'),
    ('Trade (% GDP)',                   'Macro',             'World Bank', '1995–2019'),
    ('Employment in industry (%)',      'GDP Structure',     'World Bank', '1995–2019'),
    ('Employment in services (%)',      'GDP Structure',     'World Bank', '1995–2019'),
    ('Employment in agriculture (%)',   'GDP Structure',     'World Bank', '1995–2019'),
    ('Electricity access (%)',          'Infrastructure',    'World Bank', '1995–2019'),
    ('Mobile subscriptions (per 100)',  'Infrastructure',    'World Bank', '1995–2019'),
    ('Urban population (%)',            'Demographics',      'World Bank', '1995–2019'),
    ('Life expectancy (years)',         'Demographics',      'World Bank', '1995–2019'),
    ('Death rate (per 1000)',           'Demographics',      'World Bank', '1995–2019'),
    # --- IMF (WEO + ICSD) ---
    ('GDP per capita PPP',              'Macro',             'IMF',        '1995–2019'),
    ('Govt revenue (% GDP)',            'Macro',             'IMF',        '1995–2019'),
    ('Govt net debt (% GDP)',           'Finance',           'IMF',        '1995–2019'),
    ('Structural fiscal balance',       'Finance',           'IMF',        '1995–2019'),
    ('GFCF, all sectors (% GDP)',       'Finance',           'IMF',        '1995–2019'),
    ('Primary net lending (% GDP)',     'Finance',           'IMF',        '1995–2019'),
    # --- ECI ---
    ('Economic Complexity Index',       'Dependent Variable','Atlas / ECI', '1995–2019'),
    # --- V-Dem ---
    ('Electoral democracy index',       'Governance',        'V-Dem',      '1995–2019'),
    ('Liberal democracy index',         'Governance',        'V-Dem',      '1995–2019'),
    ('Participatory dem. index',        'Governance',        'V-Dem',      '1995–2019'),
    ('Deliberative dem. index',         'Governance',        'V-Dem',      '1995–2019'),
    ('Egalitarian dem. index',          'Governance',        'V-Dem',      '1995–2019'),
    ('Clientelism index',               'Governance',        'V-Dem',      '1995–2019'),
    ('Political corruption index',      'Governance',        'V-Dem',      '1995–2019'),
    ('Rule of law index',               'Governance',        'V-Dem',      '1995–2019'),
    ('Accountability index',            'Governance',        'V-Dem',      '1995–2019'),
    ('Property rights',                 'Governance',        'V-Dem',      '1995–2019'),
    ('Political stability (WGI)',       'Governance',        'V-Dem',      '1995–2019'),
    ('Civil war indicator',             'Governance',        'V-Dem',      '1995–2019'),
    # --- PWT ---
    ('Human capital index',             'Human Capital',     'PWT 11.0',   '1995–2019'),
    ('Capital stock (nat. acc.)',       'Finance',           'PWT 11.0',   '1995–2019'),
    ('TFP level',                       'Macro',             'PWT 11.0',   '1995–2019'),
    ('Welfare-relevant TFP',            'Macro',             'PWT 11.0',   '1995–2019'),
    ('Share of consumption in GDP',     'GDP Structure',     'PWT 11.0',   '1995–2019'),
    ('Share of investment in GDP',      'GDP Structure',     'PWT 11.0',   '1995–2019'),
    ('Share of govt spending in GDP',   'GDP Structure',     'PWT 11.0',   '1995–2019'),
    ('Capital depreciation rate',       'Finance',           'PWT 11.0',   '1995–2019'),
    # --- CEPII ---
    ('Landlocked dummy',                'Geography',         'CEPII',      'time-invariant'),
] + [
    (f'{r} production (volume)',        'NR Production',     'EI / OWID',  '1995–2019')
    for r in ['Oil', 'Natural Gas', 'Coal', 'Copper', 'Nickel', 'Cobalt', 'Lithium',
              'Bauxite', 'Aluminium', 'Zinc', 'Tin', 'Manganese', 'Rare Earth',
              'Platinum Group', 'Vanadium', 'Natural Graphite']
] + [
    (f'{r} price ($/t)',                'NR Prices',         'EI / USGS',  '1995–2019')
    for r in ['Oil', 'Natural Gas', 'Coal', 'Copper', 'Nickel', 'Cobalt', 'Lithium',
              'Bauxite', 'Aluminium', 'Zinc', 'Tin', 'Manganese', 'Rare Earth',
              'Platinum Group', 'Vanadium', 'Natural Graphite']
]

_df00 = pd.DataFrame(_variables_00, columns=['Variable', 'Category', 'Source', 'Coverage'])

# Remap sources with fewer than 2 variables to 'Other'
_src_counts00 = _df00['Source'].value_counts()
_small_srcs00 = _src_counts00[_src_counts00 < 2].index.tolist()
_df00.loc[_df00['Source'].isin(_small_srcs00), 'Source'] = 'Other'

_src_order00  = ['World Bank', 'V-Dem', 'PWT 11.0', 'IMF', 'EI / OWID', 'EI / USGS', 'Other']
_src_colors00 = ['#4a6fa5',   '#c23a3a', '#2e7d4a', '#d4853b', '#7a5c9e', '#3a8fa5', '#999999']
_src_cmap00   = dict(zip(_src_order00, _src_colors00))

_cat_order00 = [
    'Governance', 'Human Capital', 'Infrastructure', 'Demographics',
    'GDP Structure', 'Macro', 'Finance', 'Resource Rents',
    'NR Production', 'NR Prices', 'Dependent Variable', 'Geography',
]

_pivot00 = _df00.groupby(['Category', 'Source']).size().reset_index(name='N')
_pivot00['Category'] = pd.Categorical(_pivot00['Category'], categories=_cat_order00, ordered=True)
_pivot00 = _pivot00.sort_values('Category')

fig00 = go.Figure()

for i, cat in enumerate(_cat_order00):
    if i % 2 == 0:
        fig00.add_shape(
            type='rect',
            x0=-0.5, x1=len(_src_order00) - 0.5,
            y0=i - 0.48, y1=i + 0.48,
            fillcolor='rgba(200,210,230,0.18)',
            line=dict(width=0),
            layer='below',
        )

for _, row in _pivot00.iterrows():
    xi = _src_order00.index(row['Source']) if row['Source'] in _src_order00 else 0
    yi = _cat_order00.index(row['Category']) if row['Category'] in _cat_order00 else 0
    # Use log-scaled size to prevent overlap
    bsize = int(_math.log1p(row['N']) * 14 + 10)
    fig00.add_trace(go.Scatter(
        x=[xi], y=[yi],
        mode='markers+text',
        marker=dict(
            size=bsize,
            color=_src_cmap00.get(row['Source'], '#aaa'),
            opacity=0.88,
            line=dict(width=1.5, color='white'),
        ),
        text=[str(row['N'])],
        textfont=dict(size=10, color='white'),
        textposition='middle center',
        hovertemplate=f"{row['Category']} / {row['Source']}: {row['N']} variable(s)<extra></extra>",
        showlegend=False,
    ))

fig00.update_xaxes(
    tickvals=list(range(len(_src_order00))),
    ticktext=_src_order00,
    tickangle=-30,
    tickfont=dict(size=12, family=FONT),
    showgrid=False,
)
fig00.update_yaxes(
    tickvals=list(range(len(_cat_order00))),
    ticktext=_cat_order00,
    tickfont=dict(size=12, family=FONT),
    showgrid=False,
)
fig00.update_layout(**base_layout(
    height=620,
    margin=dict(l=165, r=40, t=50, b=110),
    showlegend=False,
))

save(fig00, '00_intro__data_sources_overview', OUT_DESC, w=1100, h=620)


# =============================================================================
print('\n=== CHART 01 ===')

master = load_master()
cl95   = load_clusters('1995')[['Country Code', 'Cluster', 'ClusterLabels']].drop_duplicates()

# WB rents (% GDP) — this is what determined sample membership (≥5% threshold)
wb_rents = (master[master['Year'] == 1995]
            [['Country Code', 'Country Name', 'Total natural resources rents (% of GDP)']]
            .drop_duplicates('Country Code')
            .rename(columns={'Total natural resources rents (% of GDP)': 'NR_rents_pct'}))

map_df = cl95.merge(wb_rents, on='Country Code', how='left')

fig01 = go.Figure()
for lbl in sorted(map_df['ClusterLabels'].dropna().unique()):
    sub   = map_df[map_df['ClusterLabels'] == lbl]
    color = LABEL_TO_COLOR.get(lbl, '#888888')
    fig01.add_trace(go.Choropleth(
        locations=sub['Country Code'],
        z=sub['NR_rents_pct'].fillna(0),
        colorscale=[[0, color], [1, color]],
        showscale=False, showlegend=True, name=lbl,
        customdata=sub[['Country Name', 'ClusterLabels', 'NR_rents_pct']].values,
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            '%{customdata[1]}<br>'
            'NR rents: %{customdata[2]:.1f}% of GDP (1995)'
            '<extra></extra>'
        ),
        marker=dict(line=dict(color='white', width=0.5)),
    ))

fig01.update_geos(
    projection_type='natural earth',
    showcountries=True, countrycolor='#d0d0d0',
    showcoastlines=True, coastlinecolor='#d0d0d0',
    showland=True, landcolor='#f5f5f5',
    showocean=True, oceancolor='#dde8f0',
    showframe=False,
)
fig01.update_layout(
    margin=dict(l=0, r=0, t=50, b=80),
    legend=dict(
        orientation='h', x=0.5, y=-0.06, xanchor='center', yanchor='top',
        font=dict(size=11, family=FONT),
        bgcolor='rgba(250,250,250,0.9)', bordercolor=GRID, borderwidth=1,
    ),
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family=FONT, color=NAVY),
    annotations=[dict(
        x=0.5, y=-0.12, xref='paper', yref='paper',
        text=(
            'Source: World Bank WDI (NR rents % GDP), clusters1995.csv. '
            'Sample: 54 countries with total NR rents ≥ 5% of GDP in 1995. '
            'Cluster colours: resource profile typology from PCA + K-Means on USGS/Energy Institute production data.'
        ),
        showarrow=False, font=dict(size=9, color='#888'), xanchor='center',
    )],
)
save(fig01, '01_sample__54_resource_dependent_countries_map', OUT_DESC, w=1200, h=540)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 02 ===')

master = load_master()
panel  = build_sample(master)

FEAT_COLS = [
    'Human capital index',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP',
    'Rule of law index',
    'Political stability — estimate',
    'Domestic credit to private sector (% of GDP)',
    'Trade (% of GDP)',
    'Access to electricity (% of population)',
    'Urban population (% of total population)',
    'Total natural resources rents (% of GDP)',
    'Oil rents (% of GDP)',
    'Mineral rents (% of GDP)',
    'Natural gas rents (% of GDP)',
    'GDP per capita (constant prices, PPP)',
    'prod_pc',
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

corr_df = (pd.DataFrame(corr_rows)
           .sort_values('Correlation', ascending=True)
           .reset_index(drop=True))
corr_df['Label']  = corr_df['Feature'].apply(shorten_feat)
corr_df['Color']  = corr_df['Correlation'].apply(
    lambda r: PALETTE['blue'] if r >= 0 else PALETTE['red'])

fig02 = go.Figure(go.Bar(
    x=corr_df['Correlation'], y=corr_df['Label'],
    orientation='h',
    marker=dict(color=corr_df['Color'], opacity=0.85,
                line=dict(color='white', width=0.5)),
    hovertemplate='%{y}: %{x:.3f}<extra></extra>',
))
fig02.add_vline(x=0, line=dict(color='#444', width=1.5))
fig02.update_layout(**base_layout(
    height=600,
    margin=dict(l=200, r=80, t=60, b=60),
    xaxis=dict(title='Pearson Correlation with ECI (panel 1995–2019)',
               gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(tickfont=dict(size=11)),
))
save(fig02, '02_sample__variable_correlations_with_eci', OUT_DESC, w=1100, h=600)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 03 ===')

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
            x=sub['PC1'], y=sub['PC2'],
            mode='markers+text',
            marker=dict(size=10, color=color, opacity=0.82,
                        line=dict(width=1.2, color='white')),
            text=sub['Country Code'],
            textposition='top center',
            textfont=dict(size=8, color='#333'),
            name=lbl,
            hovertemplate='<b>%{text}</b><br>PC1=%{x:.2f}, PC2=%{y:.2f}<extra></extra>',
        ))

    for feat_name in top10:
        if feat_name in top5:
            continue
        fig.add_annotation(
            x=loadings_df.loc[feat_name, 'PC1'] * scale,
            y=loadings_df.loc[feat_name, 'PC2'] * scale,
            ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowwidth=1.2, arrowcolor='rgba(150,150,150,0.5)',
        )

    for feat_name in top5:
        x1 = loadings_df.loc[feat_name, 'PC1'] * scale
        y1 = loadings_df.loc[feat_name, 'PC2'] * scale
        fig.add_annotation(
            x=x1, y=y1, ax=0, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=2.2, arrowcolor='#222',
        )
        fig.add_annotation(
            x=x1 * 1.18, y=y1 * 1.18,
            text=f'<b>{feat_name}</b>', showarrow=False,
            font=dict(size=10, color='#111', family=FONT),
            bgcolor='rgba(255,255,255,0.7)', borderpad=2,
        )

    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    fig.add_vline(x=0, line=dict(color=GRID, width=1))

    fig.update_layout(**base_layout(
        height=680,
        margin=dict(l=60, r=60, t=50, b=60),
        xaxis=dict(title=f'PC1 ({var1:.1f}% variance explained)',
                   gridcolor=GRID, gridwidth=0.5),
        yaxis=dict(title=f'PC2 ({var2:.1f}% variance explained)',
                   gridcolor=GRID, gridwidth=0.5),
        legend=dict(title='Resource profile (1995)', font=dict(size=10),
                    bgcolor='rgba(250,250,250,0.85)',
                    bordercolor=GRID, borderwidth=1),
    ))
    return fig

fig03 = chart_03_biplot(pca_1995, pca_model_1995, feat_1995)
save(fig03, '03_cluster__pca_biplot_country_resource_groups', OUT_CLUS, w=1100, h=680)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 04a ===')

nr_1995_sub   = nr_sample[nr_sample['Year'] == 1995]
cnames_1995   = dict(zip(pca_1995['Cluster'], pca_1995['ClusterLabels']))
fig04a = create_cluster_map(pca_1995, nr_1995_sub, cluster_names_map=cnames_1995)
fig04a.update_layout(height=520)
save(fig04a, '04a_cluster__world_map_1995_resource_profiles', OUT_CLUS, w=1200, h=520)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 04b ===')

nr_2019_sub = nr_sample[nr_sample['Year'] == 2019]
cnames_2019 = dict(zip(pca_2019['Cluster'], pca_2019['ClusterLabels']))
fig04b = create_cluster_map(pca_2019, nr_2019_sub, cluster_names_map=cnames_2019)
fig04b.update_layout(height=520)
save(fig04b, '04b_cluster__world_map_2019_resource_profiles', OUT_CLUS, w=1200, h=520)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 04c ===')

nr_agg_sub = nr_sample[nr_sample['Year'].isin([1995, 1999, 2005])]
cnames_agg = dict(zip(pca_agg['Cluster'], pca_agg['ClusterLabels']))
fig04c = create_cluster_map(pca_agg, nr_agg_sub, cluster_names_map=cnames_agg)
fig04c.update_layout(height=520)
save(fig04c, '04c_cluster__world_map_agg_resource_profiles', OUT_CLUS, w=1200, h=520)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 04d ===')

pca_4f   = run_clustering_4feat(nr_sample, year_filter=1995)
cnames_4f = dict(zip(pca_4f['Cluster'], pca_4f['ClusterLabels']))
fig04d = create_cluster_map(pca_4f, nr_1995_sub, cluster_names_map=cnames_4f,
                            label_colors=_LABEL_COLORS_4F)
fig04d.update_layout(height=520)
save(fig04d, '04d_cluster__world_map_4feat_oil_gas_coal_minerals', OUT_CLUS, w=1200, h=520)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 05 ===')

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

    for cl in clusters_all:
        cc    = [c for c in valid_countries if cdata[c]['cluster'] == cl]
        color = cluster_colors.get(cl, '#999999')

        for code in cc:
            cd  = cdata[code]
            idx = np.where(cd['years'] == first_year)[0]
            xc  = cd['x'][idx[0]] if len(idx) > 0 else cd['x0']
            yc  = cd['y'][idx[0]] if len(idx) > 0 else cd['y0']
            fig.add_trace(go.Scatter(
                x=[cd['x0'], xc], y=[cd['y0'], yc],
                mode='lines', line=dict(color=color, width=arrow_width),
                opacity=arrow_opacity, legendgroup=f'cl_{cl}',
                showlegend=False, hoverinfo='skip',
            ))

        for code in cc:
            cd  = cdata[code]
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
                    text=[code], textposition='top center',
                    textfont=dict(size=8),
                    customdata=[[cd['name'], pv, year]],
                    hovertemplate='<b>%{customdata[0]}</b><br>Log GDP pc: %{x:.2f}<br>'
                                  'ECI: %{y:.2f}<br>Prod/capita: $%{customdata[1]:,.0f}<br>'
                                  'Year: %{customdata[2]}<extra></extra>',
                ))
            for code in cc:
                cd = cdata[code]
                fd.append(go.Scatter(x=[cd['x0']], y=[cd['y0']], mode='markers',
                                     marker=dict(size=5, color=color, opacity=0.6,
                                                 symbol='circle')))
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


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 07 ===')

_imp_path = os.path.join('Final', 'NB5', 'all_importance.csv')
if os.path.exists(_imp_path):
    LABEL_EXCL = ['L1_ECI', 'Inflation_roll5', 'RealRate_roll5', 'Resource_HHI']
    imp = pd.read_csv(_imp_path)
    imp = imp[~imp['Feature'].apply(lambda f: any(e in f for e in LABEL_EXCL))]

    lin_cols = [c for c in ['LASSO', 'Ridge', 'Elastic Net'] if c in imp.columns]
    sort_col = 'Elastic Net' if 'Elastic Net' in imp.columns else lin_cols[0]
    imp = (imp.sort_values(sort_col, ascending=False)
             .head(12).reset_index(drop=True))
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

    model_cfg07 = [
        ('LASSO',       'circle',      PALETTE['lasso']),
        ('Ridge',       'square',      PALETTE['ridge']),
        ('Elastic Net', 'triangle-up', PALETTE['en']),
    ]
    for mname, sym, col in model_cfg07:
        if mname not in imp.columns:
            continue
        fig07.add_trace(go.Scatter(
            x=imp[mname], y=imp['Label'],
            mode='markers',
            marker=dict(symbol=sym, size=13, color=col,
                        line=dict(color='white', width=1.5)),
            name=mname,
            hovertemplate=f'%{{y}}: %{{x:.3f}}<extra>{mname}</extra>',
        ))

    x_max = imp[lin_cols].max().max()
    fig07.update_layout(**base_layout(
        height=560,
        margin=dict(l=200, r=80, t=70, b=80),
        xaxis=dict(title='Normalised Feature Importance (min-max, 0–1)',
                   range=[-0.02, x_max + 0.1], gridcolor=GRID, gridwidth=0.5),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='center', x=0.5, font=dict(size=11)),
    ))
    save(fig07, '07_ml__feature_importance_consensus_three_models', OUT_ML, w=1600, h=800)
else:
    print('  SKIPPED: all_importance.csv not found')


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 08 ===')

_tbl_path = os.path.join('Final', 'NB5', 'coefficient_summary_table.csv')
if os.path.exists(_tbl_path):
    tbl = pd.read_csv(_tbl_path)
    tbl = tbl[~tbl['Feature'].apply(lambda f: any(e in f for e in LABEL_EXCL))]
    tbl['abs_en'] = tbl['Elastic Net'].abs()
    top = tbl.nlargest(12, 'abs_en').sort_values('abs_en', ascending=True).reset_index(drop=True)

    fig08 = go.Figure()
    fig08.add_vline(x=0, line=dict(color='#444', width=1.5))

    model_cfg08 = [
        ('LASSO',       PALETTE['lasso']),
        ('Ridge',       PALETTE['ridge']),
        ('Elastic Net', PALETTE['en']),
    ]
    for mname, col in model_cfg08:
        if mname not in top.columns:
            continue
        fig08.add_trace(go.Bar(
            y=top['Feature'], x=top[mname], orientation='h',
            name=mname,
            marker=dict(color=col, opacity=0.88, line=dict(color='white', width=0.5)),
            hovertemplate=f'%{{y}}: %{{x:+.3f}}<extra>{mname}</extra>',
        ))

    fig08.update_layout(**base_layout(
        barmode='group', height=620,
        margin=dict(l=200, r=80, t=70, b=60),
        xaxis=dict(title='Coefficient (standardised inputs)',
                   gridcolor=GRID, gridwidth=0.5, zeroline=False),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='center', x=0.5, font=dict(size=11)),
    ))
    save(fig08, '08_ml__standardised_coefficients_lasso_ridge_en', OUT_ML, w=1100, h=620)
else:
    print('  SKIPPED: coefficient_summary_table.csv not found')


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 09 ===')

_perf_l_path = os.path.join('Final', 'NB5', 'model_performance_level.csv')
_perf_d_path = os.path.join('Final', 'NB5', 'model_performance_delta.csv')
if os.path.exists(_perf_l_path) and os.path.exists(_perf_d_path):
    perf_l = pd.read_csv(_perf_l_path)
    perf_d = pd.read_csv(_perf_d_path)
    perf_l = perf_l[perf_l['Model'] != 'XGBoost'].reset_index(drop=True)
    perf_d = perf_d[perf_d['Model'] != 'XGBoost'].reset_index(drop=True)

    fig09 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.12,
        subplot_titles=['ECI Level', 'ΔECI'],
    )

    _TRAIN_COL = '#4a6fa5'
    _TEST_COL  = '#c23a3a'

    for col_idx, perf in enumerate([perf_l, perf_d], 1):
        models   = perf['Model'].tolist()
        train_r2 = perf['Train R²'].tolist()
        test_r2  = perf['Test R²'].tolist()

        fig09.add_trace(go.Bar(
            x=models, y=train_r2,
            name='Train R²', legendgroup='train',
            showlegend=(col_idx == 1),
            marker=dict(color=_TRAIN_COL, opacity=0.88, line=dict(width=0)),
            text=[f'{v:.3f}' for v in train_r2],
            textposition='outside', textfont=dict(size=10, color=_TRAIN_COL),
            hovertemplate='%{x} Train: %{y:.3f}<extra></extra>',
        ), row=1, col=col_idx)

        fig09.add_trace(go.Bar(
            x=models, y=test_r2,
            name='Test R²', legendgroup='test',
            showlegend=(col_idx == 1),
            marker=dict(color=_TEST_COL, opacity=0.88, line=dict(width=0)),
            text=[f'{v:.3f}' for v in test_r2],
            textposition='outside', textfont=dict(size=10, color=_TEST_COL),
            hovertemplate='%{x} Test: %{y:.3f}<extra></extra>',
        ), row=1, col=col_idx)

        fig09.update_xaxes(tickangle=-30, tickfont=dict(size=11), row=1, col=col_idx)
        fig09.update_yaxes(title_text='R²', gridcolor=GRID, gridwidth=0.5,
                           tickfont=dict(size=11), row=1, col=col_idx)

    fig09.update_layout(**base_layout(
        barmode='group',
        height=480, margin=dict(l=60, r=40, t=80, b=80),
        legend=dict(orientation='h', yanchor='bottom', y=1.06,
                    xanchor='center', x=0.5, font=dict(size=12)),
    ))
    save(fig09, '09_ml__train_vs_test_r2_all_models', OUT_ML, w=1200, h=480)
else:
    print('  SKIPPED: model_performance CSVs not found')


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 10 ===')

_preds_path = os.path.join('Final', 'NB5', 'test_predictions.csv')
if os.path.exists(_preds_path):
    preds = pd.read_csv(_preds_path)

    fig10 = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)

    for col_idx, (actual_col, pred_col, lbl) in enumerate([
        ('Actual_ECI',   'Predicted_ECI',   'ECI'),
        ('Actual_Delta', 'Predicted_Delta', 'ΔECI'),
    ], 1):
        if actual_col not in preds.columns or pred_col not in preds.columns:
            continue
        actual = preds[actual_col].dropna().values
        pred   = preds.loc[preds[actual_col].notna(), pred_col].values
        codes  = preds.loc[preds[actual_col].notna(), 'Country Code'].values
        names  = preds.loc[preds[actual_col].notna(), 'Country Name'].values

        lims = [min(actual.min(), pred.min()) - 0.1,
                max(actual.max(), pred.max()) + 0.1]
        mid  = 0.0

        for x0, x1, y0, y1, fc in [
            (lims[0], mid,     lims[0], mid,     'rgba(46,125,74,0.07)'),
            (mid,     lims[1], mid,     lims[1], 'rgba(46,125,74,0.07)'),
            (lims[0], mid,     mid,     lims[1], 'rgba(194,58,58,0.07)'),
            (mid,     lims[1], lims[0], mid,     'rgba(194,58,58,0.07)'),
        ]:
            fig10.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1,
                            fillcolor=fc, line=dict(width=0), layer='below',
                            row=1, col=col_idx)

        fig10.add_trace(go.Scatter(
            x=[lims[0], lims[1]], y=[lims[0], lims[1]],
            mode='lines', line=dict(color=PALETTE['red'], width=1.5, dash='dash'),
            name='45° line', showlegend=(col_idx == 1), legendgroup='line45',
        ), row=1, col=col_idx)

        resid   = np.abs(actual - pred)
        top_idx = set(np.argsort(resid)[::-1][:5])
        mask_n  = np.array([i not in top_idx for i in range(len(actual))])

        fig10.add_trace(go.Scatter(
            x=actual[mask_n], y=pred[mask_n], mode='markers',
            marker=dict(size=6, color=PALETTE['blue'], opacity=0.65,
                        line=dict(color='white', width=0.5)),
            name='Test obs.', showlegend=(col_idx == 1), legendgroup='obs',
            customdata=np.stack([codes[mask_n], names[mask_n]], axis=1),
            hovertemplate='<b>%{customdata[1]}</b><br>'
                          'Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>',
        ), row=1, col=col_idx)

        out_idx = list(top_idx)
        fig10.add_trace(go.Scatter(
            x=actual[out_idx], y=pred[out_idx], mode='markers+text',
            marker=dict(size=9, color=PALETTE['orange'], opacity=0.9,
                        line=dict(color='white', width=1)),
            text=codes[out_idx], textposition='top center', textfont=dict(size=9),
            name='Largest residuals', showlegend=(col_idx == 1), legendgroup='outliers',
            customdata=np.stack([codes[out_idx], names[out_idx]], axis=1),
            hovertemplate='<b>%{customdata[1]}</b><br>'
                          'Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>',
        ), row=1, col=col_idx)

        fig10.add_hline(y=0, line=dict(color=GRID, width=1), row=1, col=col_idx)
        fig10.add_vline(x=0, line=dict(color=GRID, width=1), row=1, col=col_idx)
        fig10.update_xaxes(title_text=f'Actual {lbl} (test set)', range=lims,
                           gridcolor=GRID, gridwidth=0.5, row=1, col=col_idx)
        fig10.update_yaxes(title_text=f'Predicted {lbl}', range=lims,
                           gridcolor=GRID, gridwidth=0.5, row=1, col=col_idx)

    fig10.update_layout(**base_layout(
        height=560, margin=dict(l=70, r=50, t=70, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.04,
                    xanchor='center', x=0.5, font=dict(size=10)),
    ))
    save(fig10, '10_ml__actual_vs_predicted_eci_test_set', OUT_ML, w=1100, h=560)
else:
    print('  SKIPPED: test_predictions.csv not found')


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 11 ===')

_fc_path   = os.path.join('Final', 'NB5', 'ECI_Forecast_2020_2030.csv')
_rank_path  = os.path.join('Final', 'NB5', 'Country_Ranking_2020_2030.csv')
if os.path.exists(_fc_path) and os.path.exists(_rank_path):
    fc   = pd.read_csv(_fc_path)
    rank = pd.read_csv(_rank_path)

    if os.path.exists(_perf_l_path):
        perf_tmp = pd.read_csv(_perf_l_path)
        perf_tmp = perf_tmp[perf_tmp['Model'] != 'XGBoost']
        best_rmse = perf_tmp.iloc[0]['Test RMSE'] if 'Test RMSE' in perf_tmp.columns else 0.08
    else:
        best_rmse = 0.08

    master_tmp = load_master()
    hist = master_tmp[['Country Code', 'Country Name', 'Year',
                        'Economic Complexity Index']].dropna()

    rank_sorted  = rank.sort_values('Total_Change', ascending=False).reset_index(drop=True)
    CASE_STUDIES = ['COG', 'AZE', 'CHL']
    top3    = [cc for cc in rank_sorted['Country Code'].tolist() if cc not in CASE_STUDIES][:3]
    bottom3 = [cc for cc in rank_sorted['Country Code'].tolist()[::-1] if cc not in CASE_STUDIES][:3]

    CASE_COL = '#4a6fa5'
    TOP_COL  = '#2e7d4a'
    BOT_COL  = '#c23a3a'
    GREY_FC  = '#b0b8c4'

    all_eci = pd.concat([
        hist[hist['Country Code'].isin(INCLUDE_LIST)]['Economic Complexity Index'],
        fc[fc['Country Code'].isin(INCLUDE_LIST)]['Ensemble'],
    ]).dropna()
    Y_RANGE = [all_eci.min() - 0.15, all_eci.max() + 0.15]

    def _add_country_traces_fc(fig, cc, cname, col, lw, opacity, legendgroup,
                                row_n, col_n):
        h = hist[hist['Country Code'] == cc].sort_values('Year')
        f = fc[fc['Country Code'] == cc].sort_values('Year')
        if h.empty or f.empty:
            return None
        ens = f['Ensemble'].values
        yrs = f['Year'].values

        fig.add_trace(go.Scatter(
            x=h['Year'], y=h['Economic Complexity Index'],
            mode='lines', line=dict(color=col, width=lw), opacity=opacity,
            legendgroup=legendgroup, showlegend=False,
            hovertemplate=f'<b>{cname} ({cc})</b><br>%{{x}}: %{{y:.3f}}<extra>Historical</extra>',
        ), row=row_n, col=col_n)

        if col != GREY_FC:
            rgb = ','.join(str(v) for v in _hex_to_rgb(col))
            fig.add_trace(go.Scatter(
                x=np.concatenate([yrs, yrs[::-1]]).tolist(),
                y=np.concatenate([ens + best_rmse, (ens - best_rmse)[::-1]]).tolist(),
                fill='toself', fillcolor=f'rgba({rgb},0.08)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False, hoverinfo='skip', legendgroup=legendgroup,
            ), row=row_n, col=col_n)

        last_yr  = int(h['Year'].iloc[-1])
        last_eci = float(h['Economic Complexity Index'].iloc[-1])
        fig.add_trace(go.Scatter(
            x=[last_yr] + yrs.tolist(), y=[last_eci] + ens.tolist(),
            mode='lines', line=dict(color=col, width=lw, dash='dash'),
            opacity=opacity, legendgroup=legendgroup, showlegend=False,
            hovertemplate=f'<b>{cname} ({cc})</b><br>%{{x}}: %{{y:.3f}}<extra>Forecast</extra>',
        ), row=row_n, col=col_n)
        return float(ens[-1])

    def _deconflict_lbl(label_info, min_gap=0.22):
        sorted_lbl = sorted(label_info.items(), key=lambda x: x[1][0])
        adjusted   = {}
        floor_y    = None
        for cc, (y_act, _) in sorted_lbl:
            y_place = y_act if floor_y is None else max(y_act, floor_y + min_gap)
            adjusted[cc] = y_place
            floor_y = y_place
        return adjusted

    fig11 = make_subplots(rows=1, cols=2, horizontal_spacing=0.07)

    for panel_col, highlight_group, highlight_col, grp_name in [
        (1, top3,    TOP_COL, 'top3'),
        (2, bottom3, BOT_COL, 'bottom3'),
    ]:
        fig11.add_vrect(x0=2019.5, x1=2030.5, fillcolor='rgba(200,210,225,0.18)',
                        line=dict(width=0), layer='below', row=1, col=panel_col)
        fig11.add_vline(x=2019.5, line=dict(color='#aaa', width=1.5, dash='dot'),
                        row=1, col=panel_col)

        highlighted_here = set(CASE_STUDIES + highlight_group)

        for cc in INCLUDE_LIST:
            if cc in highlighted_here:
                continue
            row_   = rank[rank['Country Code'] == cc]
            cname_ = row_['Country'].values[0] if len(row_) else cc
            _add_country_traces_fc(fig11, cc, cname_, GREY_FC, lw=0.7, opacity=0.3,
                                   legendgroup='others', row_n=1, col_n=panel_col)

        label_info = {}

        for cc in highlight_group:
            row_   = rank[rank['Country Code'] == cc]
            cname_ = row_['Country'].values[0] if len(row_) else cc
            y_end  = _add_country_traces_fc(fig11, cc, cname_, highlight_col,
                                            lw=2.2, opacity=1.0,
                                            legendgroup=grp_name, row_n=1, col_n=panel_col)
            if y_end is not None:
                label_info[cc] = (y_end, highlight_col)

        for cc in CASE_STUDIES:
            row_   = rank[rank['Country Code'] == cc]
            cname_ = row_['Country'].values[0] if len(row_) else cc
            y_end  = _add_country_traces_fc(fig11, cc, cname_, CASE_COL,
                                            lw=2.5, opacity=1.0,
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
                font=dict(size=9.5, color=col, family=FONT),
                xanchor='left', yanchor='middle',
            )

        fig11.update_xaxes(title_text='Year', gridcolor=GRID, gridwidth=0.5,
                           dtick=5, range=[1994, 2033], row=1, col=panel_col)
        fig11.update_yaxes(
            title_text='Economic Complexity Index' if panel_col == 1 else '',
            range=Y_RANGE, gridcolor=GRID, gridwidth=0.5, row=1, col=panel_col)

    for lbl, col, rk, grp in [
        ('Case studies — COG · AZE · CHL',       CASE_COL, 1, 'cases'),
        ('Top 3 improvers — GNQ · MNG · ECU',    TOP_COL,  2, 'top3'),
        ('Bottom 3 decliners — ZWE · SAU · KAZ', BOT_COL,  3, 'bottom3'),
        ('Other countries',                       GREY_FC,  4, 'others'),
    ]:
        fig11.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                   line=dict(color=col, width=2.5), name=lbl,
                                   legendgroup=grp, showlegend=True, legendrank=rk))
    fig11.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                               line=dict(color='#888', width=1.5), name='── Historical',
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

# ----

# =============================================================================
print('\n=== CHART 11b ===')

if os.path.exists(_fc_path):
    fc_hm = pd.read_csv(_fc_path)
    master_hm = load_master()
    hist_hm   = (master_hm[master_hm['Country Code'].isin(INCLUDE_LIST)]
                 [['Country Code', 'Country Name', 'Year', 'Economic Complexity Index']]
                 .dropna())

    # Combine historical and forecast
    fc_long = fc_hm[fc_hm['Country Code'].isin(INCLUDE_LIST)][['Country Code', 'Year', 'Ensemble']].copy()
    fc_long = fc_long.rename(columns={'Ensemble': 'ECI'})
    hist_long = hist_hm.rename(columns={'Economic Complexity Index': 'ECI'})
    hist_long = hist_long[['Country Code', 'Year', 'ECI']]

    combined = pd.concat([hist_long, fc_long], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Country Code', 'Year'])

    pivot_hm = combined.pivot(index='Country Code', columns='Year', values='ECI')
    pivot_hm = pivot_hm.reindex(sorted(pivot_hm.index))

    # Sort countries by their 2019 ECI
    sort_col = 2019 if 2019 in pivot_hm.columns else pivot_hm.columns.max()
    pivot_hm = pivot_hm.loc[pivot_hm[sort_col].sort_values().index]

    forecast_start = fc_hm['Year'].min() if 'Year' in fc_hm.columns else 2020

    fig11b = go.Figure(go.Heatmap(
        z=pivot_hm.values,
        x=[str(c) for c in pivot_hm.columns],
        y=pivot_hm.index.tolist(),
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title='ECI', thickness=16, len=0.9),
        hovertemplate='%{y} | %{x}: %{z:.3f}<extra></extra>',
    ))

    forecast_col_idx = list(pivot_hm.columns).index(forecast_start) if forecast_start in pivot_hm.columns else None
    if forecast_col_idx is not None:
        fig11b.add_shape(type='line',
                         x0=forecast_col_idx - 0.5, x1=forecast_col_idx - 0.5,
                         y0=-0.5, y1=len(pivot_hm) - 0.5,
                         line=dict(color='#333', width=2, dash='dot'),
                         xref='x', yref='y')
        fig11b.add_annotation(
            x=forecast_col_idx - 0.5, y=len(pivot_hm) + 0.2,
            xref='x', yref='y',
            text='<b>← Historical | Forecast →</b>',
            showarrow=False, font=dict(size=10, color='#444'),
            xanchor='center',
        )

    fig11b.update_layout(**base_layout(
        height=max(500, len(pivot_hm) * 12),
        margin=dict(l=80, r=100, t=60, b=80),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
        yaxis=dict(tickfont=dict(size=9), showgrid=False),
    ))
    save(fig11b, '11b_ml__eci_forecast_heatmap_all_countries', OUT_ML,
         w=1200, h=max(500, len(pivot_hm) * 12))
else:
    print('  SKIPPED: ECI_Forecast_2020_2030.csv not found')


# =============================================================================

# ----

#            (kept as alias in case chart numbering differs in presentation)
# =============================================================================
print('\n=== CHART 13 ===')

if os.path.exists(_fc_path):
    # Re-use the same figure built for 11b
    save(fig11b, '13_ml__eci_forecast_heatmap_historical_and_projected', OUT_ML,
         w=1200, h=max(500, len(pivot_hm) * 12))
else:
    print('  SKIPPED: ECI_Forecast_2020_2030.csv not found')


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 14 ===')

master_r = load_master()
df14     = build_sample(master_r)

yr_95 = df14[df14['Year'] == 1995]['Economic Complexity Index'].dropna().sort_values().values
yr_19 = df14[df14['Year'] == 2019]['Economic Complexity Index'].dropna().sort_values().values

fig14 = go.Figure()
for vals, yr, col in [(yr_95, 1995, PALETTE['blue']), (yr_19, 2019, PALETTE['red'])]:
    pcts = np.linspace(0, 100, len(vals))
    fig14.add_trace(go.Scatter(
        x=pcts, y=vals,
        mode='lines', name=str(yr),
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


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 15 ===')

master15  = load_master()
clusters15 = load_clusters('1995')[['Country Code', 'Cluster']].drop_duplicates()
df15      = master15[master15['Country Code'].isin(INCLUDE_LIST)].copy()
df15      = df15.merge(clusters15, on='Country Code', how='left')

traj15 = (df15.groupby(['Year', 'Cluster'])['Economic Complexity Index']
          .median().reset_index())

fig15 = go.Figure()
for cl in sorted(traj15['Cluster'].dropna().unique()):
    sub = traj15[traj15['Cluster'] == cl]
    fig15.add_trace(go.Scatter(
        x=sub['Year'], y=sub['Economic Complexity Index'],
        mode='lines+markers',
        name=CLUSTER_LABELS.get(int(cl), f'Cluster {int(cl)}'),
        line=dict(color=CLUSTER_COLORS.get(int(cl), '#999'), width=2.2),
        marker=dict(size=5),
        hovertemplate='%{x}: %{y:.3f}<extra>'
                      + CLUSTER_LABELS.get(int(cl), '') + '</extra>',
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

# ----

#   Re-estimates regressions inline (no statsmodels dependency warning needed)
# =============================================================================
print('\n=== CHART 16 ===')

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

    reg3_input   = ['log_HCI', 'log_GFCF', 'Political stability — estimate',
                    'Rule of law index', 'log_Production_Value', 'Trade (% of GDP)']
    INTERACT_VARS = ['log_HCI_x_log_Production', 'log_GFCF_x_log_Production']

    def _fit_dk(y, X, time, groups):
        raw = sm.OLS(y, X).fit()
        robust = raw.get_robustcov_results(
            cov_type='HAC-Groupsum', time=time, groups=groups,
            maxlags=2, kernel='bartlett', use_correction=True,
        )
        import types
        ns = types.SimpleNamespace(
            params   = pd.Series(robust.params,  index=X.columns),
            bse      = pd.Series(robust.bse,     index=X.columns),
            pvalues  = pd.Series(robust.pvalues, index=X.columns),
        )
        return ns

    reg3_cols = reg3_input + INTERACT_VARS + ['Economic Complexity Index', 'Country Code', 'Year']
    reg3_df   = df16[reg3_cols].dropna()
    m3a = _fit_dk(
        reg3_df['Economic Complexity Index'],
        sm.add_constant(reg3_df[reg3_input + INTERACT_VARS]),
        reg3_df['Year'], reg3_df['Country Code'],
    )

    reg3b_cols = reg3_cols + ['ECI_lag1']
    reg3b_df   = df16[reg3b_cols].dropna()
    m3b = _fit_dk(
        reg3b_df['Economic Complexity Index'],
        sm.add_constant(reg3b_df[reg3_input + INTERACT_VARS + ['ECI_lag1']]),
        reg3b_df['Year'], reg3b_df['Country Code'],
    )

    plot_vars = [v for v in reg3_input + INTERACT_VARS if v != 'const']
    labels16  = [v.replace('log_', 'log(').replace('_x_', ') × log(') + (')' if 'log_' in v else '')
                 for v in plot_vars]

    fig16 = go.Figure()
    for model, col, name in [
        (m3a, PALETTE['blue'],       'Model 3a (no lag)'),
        (m3b, PALETTE['light_blue'], 'Model 3b (with lag)'),
    ]:
        coefs  = [model.params.get(v, np.nan) for v in plot_vars]
        lowers = [model.params.get(v, np.nan) - 1.96 * model.bse.get(v, np.nan)
                  for v in plot_vars]
        uppers = [model.params.get(v, np.nan) + 1.96 * model.bse.get(v, np.nan)
                  for v in plot_vars]
        fig16.add_trace(go.Scatter(
            y=labels16, x=coefs,
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

    fig16.add_vline(x=0, line=dict(color='#c9cfd6', width=1.5, dash='dash'))
    fig16.update_layout(**base_layout(
        height=700,
        margin=dict(l=220, r=100, t=10, b=50),
        xaxis=dict(title='Coefficient (95% CI)', gridcolor=GRID, gridwidth=0.5,
                   zeroline=False),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(font=dict(size=11), orientation='h',
                    yanchor='bottom', y=1.01, xanchor='right', x=1),
    ))
    save(fig16, '16_reg__coefficients_model3a_vs_model3b', OUT_REG, w=1100, h=700)

except ImportError:
    print('  SKIPPED: statsmodels not available')
except Exception as e:
    print(f'  SKIPPED (error in chart 16): {e}')


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 17 ===')

master17 = load_master()
df17     = build_sample(master17)
df17['prod_pc']     = df17['Total_Production_Value'] / df17['Population'].replace(0, np.nan)
df17['log_HCI']     = np.log1p(df17['Human capital index'])
df17['log_prod_pc'] = np.log1p(df17['prod_pc'])

country_avg = (df17[['Country Code', 'log_HCI', 'Economic Complexity Index', 'log_prod_pc']]
               .dropna().groupby('Country Code').mean().reset_index())

country_avg['Prod_quartile'] = pd.qcut(
    country_avg['log_prod_pc'], q=4,
    labels=['Q1 — Low production', 'Q2', 'Q3', 'Q4 — High production'],
)

q_colors = [PALETTE['light_blue'], PALETTE['blue'], PALETTE['orange'], PALETTE['red']]

fig17 = go.Figure()
for q, col in zip(['Q1 — Low production', 'Q2', 'Q3', 'Q4 — High production'], q_colors):
    sub = country_avg[country_avg['Prod_quartile'] == q]
    fig17.add_trace(go.Scatter(
        x=sub['log_HCI'], y=sub['Economic Complexity Index'],
        mode='markers+text',
        text=sub['Country Code'],
        textposition='top center',
        textfont=dict(size=8, color='#555'),
        marker=dict(color=col, size=9, opacity=0.85,
                    line=dict(width=0.8, color='white')),
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


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 26 ===')

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

loadings26 = pd.DataFrame(pca26.components_.T, columns=['PC1', 'PC2'],
                          index=resource_cols26)
top20_26   = loadings26.abs().sum(axis=1).nlargest(20).index
plot_df26  = loadings26.loc[top20_26]
plot_df26  = (plot_df26.assign(_s=plot_df26['PC1'].abs() + plot_df26['PC2'].abs())
              .sort_values('_s', ascending=False).drop(columns='_s'))

pc_labels_ordered = [
    f'PC1 ({var1_26:.1f}%)<br><i>↑ Oil & Gas</i>',
    f'PC2 ({var2_26:.1f}%)<br><i>↑ Copper, Gold & Coal</i>',
]
y_labels26 = pc_labels_ordered[::-1]
z_values26 = plot_df26[['PC2', 'PC1']].T.values

fig26 = go.Figure(go.Heatmap(
    z=z_values26,
    x=plot_df26.index.tolist(),
    y=y_labels26,
    colorscale=[[0.0, '#1a4a8a'], [0.5, '#ffffff'], [1.0, '#c23a3a']],
    zmid=0, zmin=-1, zmax=1,
    hovertemplate='<b>%{x}</b><br>%{y}: %{z:.3f}<extra></extra>',
    colorbar=dict(
        title=dict(text='Loading', font=dict(size=12)),
        thickness=20, len=1.0,
        tickvals=[-1, -0.5, 0, 0.5, 1],
        tickfont=dict(size=11),
    ),
))

fig26.update_xaxes(title_text='Resource/Feature', tickangle=-40,
                   tickfont=dict(size=10, family=FONT), showgrid=False)
fig26.update_yaxes(title_text='Principal Component',
                   tickfont=dict(size=12, family=FONT), showgrid=False)
fig26.update_layout(**base_layout(
    height=420, margin=dict(l=260, r=120, t=60, b=160),
))
save(fig26, '26_diag__pca_resource_loadings_heatmap', OUT_MISC, w=1300, h=420)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 27 ===')

if os.path.exists(_imp_path):
    imp27 = pd.read_csv(_imp_path)
    imp27 = imp27[imp27['Model'] == 'RandomForest'] if 'Model' in imp27.columns else imp27.copy()

    if 'Model' not in imp27.columns:
        # all_importance.csv has one row per feature, RF column exists
        rf_col = 'Random Forest' if 'Random Forest' in imp27.columns else None
        if rf_col:
            imp27 = (imp27[['Feature', rf_col]]
                     .rename(columns={rf_col: 'Importance'})
                     .dropna())
    else:
        imp27 = imp27.rename(columns={'Importance': 'Importance'}) if 'Importance' in imp27.columns else imp27

    # Fallback: use the RF column from all_importance.csv
    if 'Importance' not in imp27.columns:
        rf_col = next((c for c in imp27.columns if 'Forest' in c or 'RF' in c or 'rf' in c.lower()), None)
        if rf_col:
            imp27 = imp27[['Feature', rf_col]].rename(columns={rf_col: 'Importance'}).dropna()

    if 'Importance' in imp27.columns and 'Feature' in imp27.columns:
        imp27 = imp27[~imp27['Feature'].apply(
            lambda f: any(e in f for e in LABEL_EXCL))].copy()
        imp27 = (imp27.sort_values('Importance', ascending=False)
                 .head(15).iloc[::-1].reset_index(drop=True))
        imp27['Label'] = imp27['Feature'].apply(shorten_feat)

        fig27 = go.Figure(go.Bar(
            x=imp27['Importance'], y=imp27['Label'], orientation='h',
            marker=dict(color=PALETTE['rf'], opacity=0.88,
                        line=dict(color='white', width=0.5)),
            hovertemplate='%{y}: %{x:.3f}<extra>Random Forest</extra>',
        ))
        fig27.update_layout(**base_layout(
            height=540,
            margin=dict(l=200, r=80, t=60, b=60),
            xaxis=dict(title='Feature Importance', gridcolor=GRID, gridwidth=0.5),
            yaxis=dict(tickfont=dict(size=11)),
        ))
        save(fig27, '27_ml__random_forest_feature_importance', OUT_ML, w=1100, h=540)
    else:
        print('  SKIPPED: could not extract RF importance column')
else:
    print('  SKIPPED: all_importance.csv not found')


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 28 ===')

_vif_path = os.path.join('Final', 'NB5', 'vif_table.csv')
if os.path.exists(_vif_path):
    vif_df = pd.read_csv(_vif_path)
    feat_col = next((c for c in ['Feature', 'Variable', 'feature'] if c in vif_df.columns), None)
    vif_col  = next((c for c in ['VIF', 'vif'] if c in vif_df.columns), None)
    if feat_col and vif_col:
        vif_df['Label'] = vif_df[feat_col].apply(shorten_feat)
        vif_df = vif_df.sort_values(vif_col, ascending=True)
        vif_df['Color'] = vif_df[vif_col].apply(
            lambda v: PALETTE['red'] if v > 10 else (PALETTE['orange'] if v > 5 else PALETTE['blue']))

        fig28 = go.Figure(go.Bar(
            x=vif_df[vif_col], y=vif_df['Label'], orientation='h',
            marker=dict(color=vif_df['Color'], opacity=0.85,
                        line=dict(color='white', width=0.5)),
            hovertemplate='%{y}: VIF = %{x:.2f}<extra></extra>',
        ))
        fig28.add_vline(x=5,  line=dict(color=PALETTE['orange'], width=1.5, dash='dash'))
        fig28.add_vline(x=10, line=dict(color=PALETTE['red'],    width=1.5, dash='dash'))
        fig28.update_layout(**base_layout(
            height=500,
            margin=dict(l=200, r=80, t=60, b=60),
            xaxis=dict(title='Variance Inflation Factor (VIF)',
                       gridcolor=GRID, gridwidth=0.5),
            yaxis=dict(tickfont=dict(size=11)),
        ))
        save(fig28, '28_diag__vif_multicollinearity', OUT_MISC, w=1100, h=500)
    else:
        print('  SKIPPED: VIF CSV missing expected columns')
else:
    # Compute VIF from Master.csv
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        master_vif = load_master()
        vif_vars   = [
            'Human capital index',
            'Gross fixed capital formation, all, Constant prices, Percent of GDP',
            'Political stability — estimate',
            'Rule of law index',
            'Trade (% of GDP)',
            'Total natural resources rents (% of GDP)',
            'Domestic credit to private sector (% of GDP)',
        ]
        vif_data = build_sample(master_vif)[vif_vars].dropna()
        vif_vals = [variance_inflation_factor(vif_data.values, i)
                    for i in range(vif_data.shape[1])]
        vif_df28 = pd.DataFrame({'Feature': vif_vars, 'VIF': vif_vals})
        vif_df28 = vif_df28.sort_values('VIF', ascending=True)
        vif_df28['Label'] = vif_df28['Feature'].apply(shorten_feat)
        vif_df28['Color'] = vif_df28['VIF'].apply(
            lambda v: PALETTE['red'] if v > 10 else (PALETTE['orange'] if v > 5 else PALETTE['blue']))

        fig28 = go.Figure(go.Bar(
            x=vif_df28['VIF'], y=vif_df28['Label'], orientation='h',
            marker=dict(color=vif_df28['Color'], opacity=0.85,
                        line=dict(color='white', width=0.5)),
            hovertemplate='%{y}: VIF = %{x:.2f}<extra></extra>',
        ))
        fig28.add_vline(x=5,  line=dict(color=PALETTE['orange'], width=1.5, dash='dash'))
        fig28.add_vline(x=10, line=dict(color=PALETTE['red'],    width=1.5, dash='dash'))
        fig28.update_layout(**base_layout(
            height=440,
            margin=dict(l=200, r=80, t=60, b=60),
            xaxis=dict(title='Variance Inflation Factor (VIF)',
                       gridcolor=GRID, gridwidth=0.5),
            yaxis=dict(tickfont=dict(size=11)),
        ))
        save(fig28, '28_diag__vif_multicollinearity', OUT_MISC, w=1100, h=440)
    except ImportError:
        print('  SKIPPED: statsmodels not available for VIF computation')
    except Exception as e:
        print(f'  SKIPPED (VIF error): {e}')


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 29 ===')

_boot_metrics = os.path.join('intermediary', 'bootstrap', 'nb5_boot_metrics.csv')
if os.path.exists(_boot_metrics):
    boot = pd.read_csv(_boot_metrics)

    model_cols = [c for c in boot.columns if c.endswith('_test_r2')]
    if model_cols:
        fig29 = go.Figure()
        colors29 = [PALETTE['lasso'], PALETTE['ridge'], PALETTE['en'], PALETTE['rf'],
                    PALETTE['teal'], PALETTE['purple']]

        for i, col in enumerate(model_cols):
            model_name = col.replace('_test_r2', '').replace('_', ' ')
            if 'XGBoost' in model_name:
                continue
            vals = boot[col].dropna()
            fig29.add_trace(go.Box(
                y=vals, name=model_name,
                marker=dict(color=colors29[i % len(colors29)], opacity=0.8),
                line=dict(color=colors29[i % len(colors29)]),
                boxmean='sd',
                hovertemplate=f'{model_name}<br>R²: %{{y:.3f}}<extra></extra>',
            ))

        fig29.update_layout(**base_layout(
            height=500,
            margin=dict(l=80, r=60, t=60, b=80),
            xaxis=dict(title='Model', gridcolor=GRID),
            yaxis=dict(title='Bootstrap Test R²', gridcolor=GRID, gridwidth=0.5),
        ))
        save(fig29, '29_diag__bootstrap_r2_stability', OUT_MISC, w=1100, h=500)
    else:
        print('  SKIPPED: no *_test_r2 columns in bootstrap metrics')
else:
    print('  SKIPPED: nb5_boot_metrics.csv not found')


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 30 ===')

master30 = load_master()
df30     = build_sample(master30)

REG_CORR_VARS = {
    'Economic Complexity Index': 'ECI',
    'Human capital index':       'HCI',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'GFCF',
    'Political stability — estimate': 'Pol. Stability',
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
    hovertemplate='%{x} × %{y}: %{z:.2f}<extra></extra>',
    colorbar=dict(thickness=14, len=0.9, tickfont=dict(size=10, family=FONT)),
))

fig30.update_layout(**base_layout(
    height=640,
    margin=dict(l=160, r=80, t=30, b=180),
    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
    yaxis=dict(tickfont=dict(size=9)),
))
save(fig30, '30_reg__regression_variable_correlation_heatmap', OUT_REG, w=1000, h=700)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 31 ===')

if os.path.exists(_preds_path):
    preds31 = pd.read_csv(_preds_path)

    country_stats = (
        preds31.groupby(['Country Code', 'Country Name'])
        .agg(
            Actual_mean    = ('Actual_ECI', 'mean'),
            Predicted_mean = ('Predicted_ECI', 'mean'),
            Actual_std     = ('Actual_ECI', 'std'),
            n_years        = ('Year', 'count'),
        )
        .reset_index()
        .sort_values('Actual_mean')
        .reset_index(drop=True)
    )
    country_stats['Actual_std'] = country_stats['Actual_std'].fillna(0)
    country_stats['In_band'] = (
        (country_stats['Actual_mean'] - country_stats['Predicted_mean']).abs()
        < country_stats['Actual_std']
    )

    fig31 = go.Figure()

    fig31.add_trace(go.Scatter(
        x=list(range(len(country_stats))) * 2
          + list(range(len(country_stats)))[::-1] * 2,
        y=(country_stats['Actual_mean'] + country_stats['Actual_std']).tolist()
          + (country_stats['Actual_mean'] - country_stats['Actual_std']).iloc[::-1].tolist(),
        fill='toself', fillcolor='rgba(74,111,165,0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip', name='±1 SD (actual ECI in test years)',
    ))

    fig31.add_trace(go.Scatter(
        x=list(range(len(country_stats))),
        y=country_stats['Actual_mean'],
        mode='lines', line=dict(color=PALETTE['blue'], width=2),
        name='Mean Actual ECI',
    ))

    for in_band, color, sym, lbl in [
        (True,  PALETTE['green'], 'circle',  'Predicted ≈ Actual (within ±1 SD)'),
        (False, PALETTE['red'],   'diamond', 'Predicted outside ±1 SD'),
    ]:
        mask = country_stats['In_band'] == in_band
        sub  = country_stats[mask]
        fig31.add_trace(go.Scatter(
            x=sub.index.tolist(),
            y=sub['Predicted_mean'],
            mode='markers',
            marker=dict(color=color, size=8 if in_band else 10,
                        symbol=sym, opacity=0.85),
            name=lbl,
            customdata=sub[['Country Code', 'Country Name']].values,
            hovertemplate='<b>%{customdata[1]}</b> (%{customdata[0]})<br>'
                          'Avg Actual: %{text}<br>Avg Predicted: %{y:.3f}<extra></extra>',
            text=[f'{v:.3f}' for v in sub['Actual_mean']],
        ))

    fig31.update_layout(**base_layout(
        height=500,
        xaxis=dict(
            title='Countries (sorted by mean actual ECI)',
            tickvals=list(range(len(country_stats))),
            ticktext=country_stats['Country Code'].tolist(),
            tickangle=-60, tickfont=dict(size=8),
            gridcolor=GRID,
        ),
        yaxis=dict(title='ECI (test set mean)', gridcolor=GRID, gridwidth=0.5),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='center', x=0.5, font=dict(size=10)),
    ))
    save(fig31, '31_diag__ml_prediction_intervals', OUT_MISC, w=1200, h=500)
else:
    print('  SKIPPED: test_predictions.csv not found')


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 32 ===')

raw_wide = load_master_wide()
sample_wide = raw_wide[raw_wide['Country Code'].isin(INCLUDE_LIST)].copy()
country_missing = analyze_country_missingness(sample_wide)

LABEL_THRESHOLD = 20.0

fig32 = go.Figure()
for above in [True, False]:
    mask  = (country_missing['% Missing'] >= LABEL_THRESHOLD) if above else \
            (country_missing['% Missing'] < LABEL_THRESHOLD)
    sub   = country_missing[mask]
    color = PALETTE['red'] if above else PALETTE['blue']
    size  = 12 if above else 8
    mode  = 'markers+text' if above else 'markers'

    fig32.add_trace(go.Scatter(
        x=sub['Vars with Data'],
        y=sub['% Missing'],
        mode=mode,
        text=sub['Code'] if above else None,
        textposition='top center',
        textfont=dict(size=9, color=PALETTE['red']),
        marker=dict(color=color, size=size,
                    opacity=0.75 if above else 0.55,
                    line=dict(color='white', width=0.8)),
        name=f'>= {LABEL_THRESHOLD}% missing' if above else f'< {LABEL_THRESHOLD}% missing',
        customdata=sub[['Code', 'Country', 'Complete Vars', 'Years Covered', 'Rows']].values,
        hovertemplate=(
            '<b>%{customdata[1]}</b> (%{customdata[0]})<br>'
            'Vars with data: %{x}<br>% Missing: %{y:.1f}%<br>'
            'Complete vars: %{customdata[2]}<br>'
            'Years covered: %{customdata[3]}<extra></extra>'
        ),
    ))

med_vars    = country_missing['Vars with Data'].median()
med_missing = country_missing['% Missing'].median()
fig32.add_hline(y=med_missing, line_dash='dash', line_color='#aaa', opacity=0.6,
                annotation_text=f'Median {med_missing:.1f}%',
                annotation_position='right')
fig32.add_vline(x=med_vars, line_dash='dash', line_color='#aaa', opacity=0.6,
                annotation_text=f'Median {med_vars:.0f} vars',
                annotation_position='top')

fig32.update_layout(**base_layout(
    height=520,
    xaxis=dict(title='Variables with Any Data', gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(title='% Missing Data Overall', gridcolor=GRID, gridwidth=0.5),
    legend=dict(font=dict(size=10), bgcolor='rgba(255,255,255,0.9)',
                bordercolor=GRID, borderwidth=1),
))
save(fig32, '32_diag__country_data_coverage_scatter', OUT_MISC, w=1100, h=520)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 33 ===')

master33 = load_master()
df33     = build_sample(master33)

ID_COLS = {'Country Code', 'Country Name', 'Year'}
data_cols33 = [c for c in df33.columns if c not in ID_COLS
               and df33[c].dtype in [np.float64, np.int64, float, int]]
data_cols33 = [c for c in data_cols33 if df33[c].notna().sum() > 100]

corr33 = df33[data_cols33].corr().round(2)
short_labels33 = [shorten_feat(c) for c in corr33.columns]

fig33 = go.Figure(go.Heatmap(
    z=corr33.values,
    x=short_labels33,
    y=short_labels33,
    colorscale=[[0.0, PALETTE['red']], [0.5, '#ffffff'], [1.0, PALETTE['blue']]],
    zmid=0, zmin=-1, zmax=1,
    hovertemplate='%{x} × %{y}: %{z:.2f}<extra></extra>',
    colorbar=dict(title='r', thickness=14, len=0.9, tickfont=dict(size=10)),
))

n33 = len(short_labels33)
h33 = max(600, n33 * 22)
w33 = max(700, n33 * 22)

fig33.update_layout(**base_layout(
    height=h33,
    margin=dict(l=180, r=80, t=40, b=200),
    xaxis=dict(tickangle=-55, tickfont=dict(size=8), showgrid=False),
    yaxis=dict(tickfont=dict(size=8), showgrid=False),
))
save(fig33, '33_diag__full_variable_correlation_matrix', OUT_MISC, w=w33, h=h33)


# =============================================================================

# ----

# =============================================================================
print('\n=== CHART 34 ===')

_hrv_candidates = [
    os.path.join('Final', 'NB5', 'hrv_diagnostics.csv'),
    os.path.join('intermediary', 'hrv_growth.csv'),
]
_hrv_path = next((p for p in _hrv_candidates if os.path.exists(p)), None)

if _hrv_path:
    hrv = pd.read_csv(_hrv_path)
    # Plot whatever numeric columns exist
    num_cols = [c for c in hrv.columns if hrv[c].dtype in [np.float64, float]
                and c not in {'Year', 'year'}]
    year_col = next((c for c in ['Year', 'year'] if c in hrv.columns), None)

    if year_col and num_cols:
        fig34 = go.Figure()
        for col in num_cols[:6]:
            fig34.add_trace(go.Scatter(
                x=hrv[year_col], y=hrv[col],
                mode='lines+markers', name=shorten_feat(col),
                line=dict(width=2),
                hovertemplate=f'{shorten_feat(col)}: %{{y:.3f}}<extra></extra>',
            ))
        fig34.update_layout(**base_layout(
            height=480,
            xaxis=dict(title='Year', gridcolor=GRID),
            yaxis=dict(title='Value', gridcolor=GRID, gridwidth=0.5),
        ))
        save(fig34, '34_diag__hrv_growth_diagnostics', OUT_MISC, w=1100, h=480)
    else:
        print('  SKIPPED: HRV file lacks year or numeric columns')
else:
    # Placeholder chart
    fig34 = go.Figure()
    fig34.add_trace(go.Scatter(
        x=[0], y=[0], mode='text',
        text=['HRV Growth Diagnostics — data not yet available'],
        textfont=dict(size=14, color='#888'),
    ))
    fig34.update_layout(**base_layout(
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=60, r=60, t=60, b=60),
    ))
    save(fig34, '34_diag__hrv_growth_diagnostics_placeholder', OUT_MISC, w=900, h=300)
    print('  Saved placeholder (no HRV source data found)')


# =============================================================================
# Done
# =============================================================================
print('\n' + '=' * 60)
print('generate_charts.py complete.')
print(f'All outputs written to: {os.path.abspath(OUT)}')
print('=' * 60)