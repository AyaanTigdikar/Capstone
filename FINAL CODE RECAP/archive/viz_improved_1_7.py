"""
Improved charts 1–7 based on document review feedback.
Run from: /Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP/
Output:   Final/improved_1_7/
"""
import matplotlib; matplotlib.use('Agg')
import os, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = 'Final/improved_1_7'
os.makedirs(OUT, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
FONT   = 'IBM Plex Sans, -apple-system, BlinkMacSystemFont, sans-serif'
BG     = '#fafafa'
NAVY   = '#1a2744'
GRID   = '#e5e7eb'

# FIXED cluster palette — used identically in biplot, choropleth, trajectories
CLUSTER_COLORS = {
    0: '#4a6fa5',   # blue   — No Oil, No Minerals
    1: '#c23a3a',   # red    — Some Oil, No Minerals
    2: '#2e7d4a',   # green  — Minerals, No Oil
    3: '#d4853b',   # orange — Oil, Few Minerals
}
CLUSTER_LABELS = {
    0: 'No Oil, No Minerals',
    1: 'Some Oil, No Minerals',
    2: 'Minerals, No Oil',
    3: 'Oil, Few Minerals',
}

PALETTE = dict(blue='#4a6fa5', red='#c23a3a', green='#2e7d4a',
               orange='#d4853b', grey='#999999',
               cat0='#4a6fa5', cat1='#c23a3a', cat2='#2e7d4a',
               cat3='#d4853b', cat4='#7a5c9e', cat5='#3a8fa5')

WRITE_CONFIG = {'displayModeBar': False, 'responsive': True}


def save(fig, name, w=1100, h=600):
    path = os.path.join(OUT, name)
    fig.write_html(f"{path}.html", config=WRITE_CONFIG)
    print(f"  Saved: {path}.html")


def base_layout(**kw):
    d = dict(template='plotly_white', plot_bgcolor=BG, paper_bgcolor=BG,
             font=dict(family=FONT, size=11, color=NAVY),
             margin=dict(l=60, r=40, t=40, b=50))
    d.update(kw)
    return d


# ── Clustering helper (shared by charts 3–6) ─────────────────────────────────
INCLUDE_LIST = [
    'AGO','ARE','AZE','BFA','BHR','BOL','CHL','CIV','CMR','COD','COG','DZA',
    'ECU','EGY','ETH','GAB','GHA','GIN','GNQ','IDN','IRN','IRQ','KAZ','KEN',
    'KWT','LAO','LBR','LBY','MDG','MLI','MMR','MNG','MOZ','MWI','MYS','NER',
    'NGA','OMN','PNG','QAT','RUS','RWA','SAU','TCD','TGO','TTO','TZA','UGA',
    'UZB','VEN','VNM','YEM','ZMB','ZWE',
]

def run_clustering(nr_df, year_filter=None, agg_years=None, k=4, random_state=42):
    """Return (pca_df, pca_model, feature_cols) with fixed cluster order."""
    if year_filter is not None:
        df = nr_df[nr_df['Year'] == year_filter].copy()
    elif agg_years is not None:
        df = nr_df[nr_df['Year'].isin(agg_years)].copy()
    else:
        df = nr_df.copy()

    pivot = df.pivot_table(
        index=['Country', 'Country Code', 'Year', 'Population'],
        columns='Resource', values='Production_TotalValue',
    ).reset_index()

    resource_cols = [c for c in pivot.columns
                     if c not in ['Country', 'Country Code', 'Year', 'Population']]
    pivot[resource_cols] = pivot[resource_cols].div(pivot['Population'], axis=0)
    pivot = pivot.fillna(0)

    feat = [c for c in pivot.columns
            if c not in ['Country', 'Country Code', 'Year', 'Population']]
    X = np.log1p(pivot[feat].fillna(0))

    pca = PCA(n_components=2, random_state=random_state)
    Xp  = pca.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    raw_labels = km.fit_predict(Xp)

    # Align raw cluster IDs to the fixed label schema by matching cluster
    # centroids to known resource profiles (sort by mean PC1 then PC2)
    centroids = np.array([Xp[raw_labels == i].mean(axis=0) for i in range(k)])
    order = np.argsort(centroids[:, 0])        # sort by PC1 ascending
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in raw_labels])

    out = pivot[['Country', 'Country Code', 'Year']].copy()
    out['PC1']         = Xp[:, 0]
    out['PC2']         = Xp[:, 1]
    out['Cluster']     = labels
    out['ClusterLabels'] = [CLUSTER_LABELS.get(l, f'Cluster {l}') for l in labels]
    return out, pca, feat


# =============================================================================
# CHART 1 — Variable Overview: source breakdown + grouped variable table
# =============================================================================
def chart1_variable_overview():
    """
    Two-panel figure:
      Top: bar chart — number of variables per data source
      Bottom: grouped table — variable, category, source, time coverage
    """
    # Variable metadata
    variables = [
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
        ('Real interest rate (%)',          'Macro',             'World Bank', '1995–2019'),
        ('Lending interest rate (%)',       'Macro',             'World Bank', '1995–2019'),
        ('Inflation (%)',                   'Macro',             'World Bank', '1995–2019'),
        ('IMF credit (USD)',                'Finance',           'World Bank', '1995–2019'),
        ('Employment in industry (%)',      'GDP Structure',     'World Bank', '1995–2019'),
        ('Employment in services (%)',      'GDP Structure',     'World Bank', '1995–2019'),
        ('Employment in agriculture (%)',   'GDP Structure',     'World Bank', '1995–2019'),
        ('Electricity access (%)',          'Infrastructure',    'World Bank', '1995–2019'),
        ('Mobile subscriptions (per 100)',  'Infrastructure',    'World Bank', '1995–2019'),
        ('Urban population (%)',            'Demographics',      'World Bank', '1995–2019'),
        ('Life expectancy (years)',         'Demographics',      'World Bank', '1995–2019'),
        ('Death rate (per 1000)',           'Demographics',      'World Bank', '1995–2019'),
        ('Trade (% GDP)',                   'Macro',             'World Bank', '1995–2019'),
        # --- IMF (WEO + ICSD) ---
        ('GDP per capita PPP',              'Macro',             'IMF',       '1995–2019'),
        ('Govt revenue (% GDP)',            'Macro',             'IMF',       '1995–2019'),
        ('Govt net debt (% GDP)',           'Finance',           'IMF',       '1995–2019'),
        ('Structural fiscal balance',       'Finance',           'IMF',       '1995–2019'),
        ('GFCF, all sectors (% GDP)',       'Finance',           'IMF',       '1995–2019'),
        ('Primary net lending (% GDP)',     'Finance',           'IMF',       '1995–2019'),
        # --- ECI ---
        ('Economic Complexity Index',       'Dependent Variable','Atlas / ECI','1995–2019'),
        # --- V-Dem ---
        ('Electoral democracy index',       'Governance',        'V-Dem',     '1995–2019'),
        ('Liberal democracy index',         'Governance',        'V-Dem',     '1995–2019'),
        ('Participatory dem. index',        'Governance',        'V-Dem',     '1995–2019'),
        ('Deliberative dem. index',         'Governance',        'V-Dem',     '1995–2019'),
        ('Egalitarian dem. index',          'Governance',        'V-Dem',     '1995–2019'),
        ('Clientelism index',               'Governance',        'V-Dem',     '1995–2019'),
        ('Political corruption index',      'Governance',        'V-Dem',     '1995–2019'),
        ('Rule of law index',               'Governance',        'V-Dem',     '1995–2019'),
        ('Accountability index',            'Governance',        'V-Dem',     '1995–2019'),
        ('Property rights',                 'Governance',        'V-Dem',     '1995–2019'),
        ('Political stability (WGI)',       'Governance',        'V-Dem',     '1995–2019'),
        ('Civil war indicator',             'Governance',        'V-Dem',     '1995–2019'),
        # --- PWT ---
        ('Human capital index',             'Human Capital',     'PWT 11.0',  '1995–2019'),
        ('Capital stock (nat. acc.)',       'Finance',           'PWT 11.0',  '1995–2019'),
        ('TFP level',                       'Macro',             'PWT 11.0',  '1995–2019'),
        ('Welfare-relevant TFP',            'Macro',             'PWT 11.0',  '1995–2019'),
        ('Share of consumption in GDP',     'GDP Structure',     'PWT 11.0',  '1995–2019'),
        ('Share of investment in GDP',      'GDP Structure',     'PWT 11.0',  '1995–2019'),
        ('Share of govt spending in GDP',   'GDP Structure',     'PWT 11.0',  '1995–2019'),
        ('Capital depreciation rate',       'Finance',           'PWT 11.0',  '1995–2019'),
        # --- CEPII ---
        ('Landlocked dummy',                'Geography',         'CEPII',     'time-invariant'),
        # --- Chile case study: Sernageomin ---
        ('Mineral facility inventory',      'Chile Supply Chain','Sernageomin','cross-section'),
        ('Facility type & operator',        'Chile Supply Chain','Sernageomin','cross-section'),
        ('Resource & reserve tonnages',     'Chile Supply Chain','Sernageomin','cross-section'),
        # --- Chile case study: USGS MRDS ---
        ('Facility records (supplement)',   'Chile Supply Chain','USGS MRDS', 'cross-section'),
        # --- Chile case study: COCHILCO ---
        ('Copper production (co./region)',  'Chile Production',  'COCHILCO',  '2000–2023'),
        ('Molybdenum production (mine)',    'Chile Production',  'COCHILCO',  '2000–2023'),
        ('Non-metallic minerals (25+)',     'Chile Production',  'COCHILCO',  '2000–2023'),
        ('Commodity prices (LME/other)',    'Chile Production',  'COCHILCO',  '2000–2023'),
        # --- Chile case study: Aduanas (Chilean Customs) ---
        ('Shipment-level export records',   'Chile Trade Flows', 'Aduanas',   '2000–2023'),
        # --- Chile case study: UN Comtrade ---
        ('Bilateral trade HS6',             'Chile Trade Flows', 'UN Comtrade','1995–2023'),
    ]

    df = pd.DataFrame(variables, columns=['Variable', 'Category', 'Source', 'Coverage'])

    # Remap any source with fewer than 2 variables to 'Other'
    src_counts = df['Source'].value_counts()
    small_srcs = src_counts[src_counts < 2].index.tolist()
    df.loc[df['Source'].isin(small_srcs), 'Source'] = 'Other'

    src_order  = ['World Bank', 'V-Dem', 'PWT 11.0', 'IMF',
                  'Sernageomin', 'COCHILCO', 'Other']
    src_colors = ['#4a6fa5', '#c23a3a', '#2e7d4a', '#d4853b',
                  '#c67c48', '#e6a817', '#999999']
    src_color_map = dict(zip(src_order, src_colors))

    cat_order = ['Governance', 'Human Capital', 'Infrastructure', 'Demographics',
                 'GDP Structure', 'Macro', 'Finance', 'Resource Rents',
                 'Dependent Variable', 'Geography',
                 'Chile Supply Chain', 'Chile Production', 'Chile Trade Flows']

    pivot = df.groupby(['Category', 'Source']).size().reset_index(name='N')
    pivot['Category'] = pd.Categorical(pivot['Category'], categories=cat_order, ordered=True)
    pivot = pivot.sort_values('Category')

    fig = go.Figure()

    # Alternating row shading
    for i, cat in enumerate(cat_order):
        if i % 2 == 0:
            fig.add_shape(
                type='rect',
                x0=-0.5, x1=len(src_order) - 0.5,
                y0=i - 0.48, y1=i + 0.48,
                fillcolor='rgba(200,210,230,0.18)',
                line=dict(width=0),
                layer='below',
            )

    for _, row in pivot.iterrows():
        xi = src_order.index(row['Source']) if row['Source'] in src_order else 0
        yi = cat_order.index(row['Category']) if row['Category'] in cat_order else 0
        fig.add_trace(go.Scatter(
            x=[xi], y=[yi],
            mode='markers+text',
            marker=dict(
                size=row['N'] * 8 + 10,
                color=src_color_map.get(row['Source'], '#aaa'),
                opacity=0.88,
                line=dict(width=1.5, color='white'),
            ),
            text=[str(row['N'])],
            textfont=dict(size=10, color='white'),
            textposition='middle center',
            hovertemplate=f"{row['Category']} / {row['Source']}: {row['N']} variable(s)<extra></extra>",
            showlegend=False,
        ))

    fig.update_xaxes(
        tickvals=list(range(len(src_order))),
        ticktext=src_order,
        tickangle=-30,
        tickfont=dict(size=12, family=FONT),
        showgrid=False,
    )
    fig.update_yaxes(
        tickvals=list(range(len(cat_order))),
        ticktext=cat_order,
        tickfont=dict(size=12, family=FONT),
        showgrid=False,
    )

    fig.update_layout(**base_layout(
        height=640,
        margin=dict(l=160, r=40, t=50, b=120),
        showlegend=False,
    ))

    save(fig, 'chart1_variable_overview', w=1200, h=640)


# =============================================================================
# CHART 1b — Country Selection Map (§4.2.1)
# =============================================================================
def chart_country_selection():
    """
    World map highlighting the 51-country resource-rich sample.
    In-sample countries: NAVY fill. Rest of world: light grey.
    """
    import pycountry

    # Build a dataframe: 1 = in sample, 0 = not
    all_iso3 = [c.alpha_3 for c in pycountry.countries]
    in_sample = set(INCLUDE_LIST)

    rows = []
    for code in all_iso3:
        rows.append({'iso3': code, 'in_sample': 1 if code in in_sample else 0})
    df_map = pd.DataFrame(rows)

    fig = go.Figure(go.Choropleth(
        locations=df_map['iso3'],
        z=df_map['in_sample'],
        locationmode='ISO-3',
        colorscale=[
            [0.0, '#d9dde4'],   # not in sample — light grey
            [0.5, '#d9dde4'],
            [0.5, '#1a2744'],   # in sample — navy
            [1.0, '#1a2744'],
        ],
        zmin=0, zmax=1,
        showscale=False,
        marker_line_color='white',
        marker_line_width=0.5,
        hovertemplate='%{location}<extra></extra>',
    ))

    # Annotate the count
    fig.add_annotation(
        x=0.01, y=0.05,
        xref='paper', yref='paper',
        text=f'<b>{len(in_sample)} countries</b> in sample',
        showarrow=False,
        font=dict(size=12, family=FONT, color=NAVY),
        xanchor='left', yanchor='bottom',
        bgcolor='rgba(250,250,250,0.85)',
        borderpad=4,
    )

    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        showcountries=False,
        showland=True,   landcolor='#d9dde4',
        showocean=True,  oceancolor='#f0f3f7',
        showlakes=False,
        projection_type='natural earth',
        bgcolor=BG,
    )

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, size=11, color=NAVY),
        geo=dict(bgcolor=BG),
        margin=dict(l=0, r=0, t=10, b=10),
        height=480,
    )

    save(fig, 'chart_country_selection', w=1000, h=480)


# =============================================================================
# CHART 2 — Correlation with ECI: diverging bar, grouped by category
# =============================================================================
def chart2_correlation_with_eci():
    """
    Horizontal diverging bar chart: each variable's Pearson correlation with ECI.
    Bars are grouped by category with alternating background shading.
    Color encodes category (not sign), so both direction and category are readable at once.
    """
    master = pd.read_csv('intermediary/Master.csv')

    VARS_BY_CATEGORY = {
        'Governance': [
            'Rule of law index', 'Property rights', 'Accountability index',
            'Political stability — estimate', 'Political corruption index',
        ],
        'Human Capital & Infrastructure': [
            'Human capital index', 'Access to electricity (% of population)',
            'Life expectancy at birth, total (years)',
            'Mobile cellular subscriptions (per 100 people)',
            'Urban population (% of total population)',
        ],
        'Finance & Investment': [
            'Gross fixed capital formation, all, Constant prices, Percent of GDP',
            'Adjusted savings: gross savings (% of GNI)',
            'Domestic credit to private sector (% of GDP)',
            'Capital depreciation rate',
        ],
        'Macro & Structure': [
            'GDP per capita (constant prices, PPP)', 'Government revenue',
            'Trade (% of GDP)', 'Manufacturing', 'Industry', 'Services', 'Agriculture',
            'Inflation, consumer prices (annual %)', 'Lending interest rate (%)',
        ],
        'Resource Rents': [
            'Total natural resources rents (% of GDP)', 'Oil rents (% of GDP)',
            'Natural gas rents (% of GDP)', 'Mineral rents (% of GDP)',
        ],
    }

    SHORT = {
        'Rule of law index':                                          'Rule of law',
        'Property rights':                                            'Property rights',
        'Accountability index':                                       'Accountability',
        'Political stability — estimate':                             'Political stability',
        'Political corruption index':                                 'Political corruption',
        'Human capital index':                                        'Human capital index',
        'Access to electricity (% of population)':                    'Electricity access',
        'Life expectancy at birth, total (years)':                    'Life expectancy',
        'Mobile cellular subscriptions (per 100 people)':             'Mobile subscriptions',
        'Urban population (% of total population)':                   'Urban population',
        'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'GFCF (% GDP)',
        'Adjusted savings: gross savings (% of GNI)':                 'Gross savings (% GNI)',
        'Domestic credit to private sector (% of GDP)':               'Domestic credit',
        'Capital depreciation rate':                                  'Capital depreciation',
        'GDP per capita (constant prices, PPP)':                      'GDP per capita (PPP)',
        'Government revenue':                                         'Govt revenue (% GDP)',
        'Trade (% of GDP)':                                           'Trade openness',
        'Manufacturing':                                              'Manufacturing (% GDP)',
        'Industry':                                                   'Industry (% GDP)',
        'Services':                                                   'Services (% GDP)',
        'Agriculture':                                                'Agriculture (% GDP)',
        'Inflation, consumer prices (annual %)':                      'Inflation',
        'Lending interest rate (%)':                                  'Lending interest rate',
        'Total natural resources rents (% of GDP)':                   'Total NR rents',
        'Oil rents (% of GDP)':                                       'Oil rents',
        'Natural gas rents (% of GDP)':                               'Gas rents',
        'Mineral rents (% of GDP)':                                   'Mineral rents',
    }

    CAT_COLORS = {
        'Governance':                 PALETTE['blue'],
        'Human Capital & Infrastructure': PALETTE['green'],
        'Finance & Investment':       PALETTE['orange'],
        'Macro & Structure':          PALETTE['cat4'],
        'Resource Rents':             PALETTE['red'],
    }

    eci = master['Economic Complexity Index']

    # Collect rows + check column availability
    all_vars = [v for vlist in VARS_BY_CATEGORY.values() for v in vlist]
    cat_lookup = {v: cat for cat, vlist in VARS_BY_CATEGORY.items() for v in vlist}

    rows = []
    for v in all_vars:
        if v not in master.columns:
            continue
        r = master[v].corr(eci)
        rows.append({'variable': v, 'short': SHORT.get(v, v),
                     'category': cat_lookup[v], 'r': round(r, 3)})
    df = pd.DataFrame(rows)

    # ── Run PCA on the variable set to get PC1 loadings ──────────────────────
    # Use a cross-sectional (year==2010) snapshot to avoid repeated-obs inflation,
    # then standardise before PCA so all variables are on comparable scale.
    snap = master[master['Year'] == 2010].copy()
    pca_vars = [v for v in df['variable'].tolist() if v in snap.columns]
    X_snap = snap[pca_vars].dropna()

    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X_snap)
    pca_tmp = PCA(n_components=1)
    pca_tmp.fit(Xs)

    pc1_var_explained = pca_tmp.explained_variance_ratio_[0] * 100
    loadings_pc1 = pd.Series(
        np.abs(pca_tmp.components_[0]),
        index=pca_vars,
    )

    # Attach PC1 loading to df, sort globally by loading descending
    df['pc1_loading'] = df['variable'].map(loadings_pc1).fillna(0)
    df = df.sort_values('pc1_loading', ascending=True).reset_index(drop=True)

    fig = go.Figure()

    # Bars — one trace per category so legend works
    for cat, var_list in VARS_BY_CATEGORY.items():
        sub = df[df['category'] == cat]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            y=sub['short'],
            x=sub['r'],
            orientation='h',
            marker_color=CAT_COLORS[cat],
            marker_opacity=0.85,
            name=cat,
            hovertemplate='%{y}: r = %{x:.3f}<extra></extra>',
            width=0.65,
        ))

    # invisible trace to force y-axis to use string labels
    fig.add_trace(go.Scatter(
        y=df['short'], x=[0] * len(df),
        mode='markers', marker=dict(size=0, opacity=0),
        showlegend=False, hoverinfo='skip',
    ))

    fig.add_vline(x=0, line=dict(color='#555', width=1.5))
    fig.add_vline(x=0.3,  line=dict(color=GRID, width=1, dash='dot'))
    fig.add_vline(x=-0.3, line=dict(color=GRID, width=1, dash='dot'))
    fig.add_vline(x=0.6,  line=dict(color=GRID, width=1, dash='dot'))
    fig.add_vline(x=-0.6, line=dict(color=GRID, width=1, dash='dot'))

    fig.update_layout(**base_layout(
        height=860,
        margin=dict(l=210, r=60, t=80, b=60),
        font=dict(family=FONT, size=13, color=NAVY),
        xaxis=dict(
            title=dict(text='Pearson r with ECI', font=dict(size=15)),
            range=[-1.05, 1.05],
            tickvals=[-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9],
            tickfont=dict(size=13),
            gridcolor=GRID, gridwidth=0.5,
            zeroline=False,
        ),
        yaxis=dict(tickfont=dict(size=13)),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='center', x=0.5,
            font=dict(size=13),
            itemsizing='constant',
        ),
        barmode='relative',
    ))

    save(fig, 'chart2_correlation_with_eci', w=980, h=860)


# =============================================================================
# CHART 3 — PCA Factor Loadings Heatmap (improved labels + interpretation)
# =============================================================================
def chart3_pca_loadings():
    """
    PCA loadings heatmap with:
    - Full readable resource labels (no cutoff)
    - Variance explained annotated on component axis
    - Annotation boxes marking PC1 = Hydrocarbons, PC2 = Metals/Coal
    """
    nr       = pd.read_csv('intermediary/NaturalResource.csv')
    nr_sample = nr[nr['Country Code'].isin(INCLUDE_LIST)]

    pca_df, pca_model, feat = run_clustering(nr_sample, year_filter=1995)

    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=['PC1', 'PC2'],
        index=feat,
    )

    top20 = loadings.abs().sum(axis=1).nlargest(20).index
    plot_df = loadings.loc[top20]

    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100

    # Sort by combined absolute loading (|PC1| + |PC2|) so most informative
    # resources appear at the top regardless of which component they drive
    plot_df = plot_df.assign(_s=plot_df['PC1'].abs() + plot_df['PC2'].abs()).sort_values('_s', ascending=False).drop(columns='_s')

    pc_labels = [
        f'PC1 ({var1:.1f}%)<br><i>↑ Hydrocarbons</i>',
        f'PC2 ({var2:.1f}%)<br><i>↑ Metals / Coal</i>',
    ]

    fig = go.Figure(go.Heatmap(
        z=plot_df.T.values,
        x=plot_df.index.tolist(),
        y=pc_labels,
        colorscale=[
            [0.0, '#e8c06b'], [0.5, '#fafafa'], [1.0, '#1a4a8a']
        ],
        zmid=0, zmin=-1, zmax=1,
        text=plot_df.T.round(2).values,
        texttemplate='%{text:.2f}',
        textfont=dict(size=8, family=FONT),
        hovertemplate='%{x} / %{y}: %{z:.3f}<extra></extra>',
        colorbar=dict(
            title=dict(text='Loading', font=dict(size=11)),
            thickness=14, len=0.8,
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
    ))

    fig.update_xaxes(
        tickangle=-40,
        tickfont=dict(size=10, family=FONT),
        showgrid=False,
    )
    fig.update_yaxes(
        tickfont=dict(size=11, family=FONT),
        showgrid=False,
    )
    fig.update_layout(**base_layout(
        height=320,
        margin=dict(l=220, r=100, t=70, b=140),
    ))

    save(fig, 'chart3_pca_loadings', w=1200, h=320)


# =============================================================================
# CHART 4 — PCA Biplot with cluster colours (improved arrow labeling)
# =============================================================================
def chart4_pca_biplot():
    """
    Biplot improvements:
    - Top 5 loading arrows labeled (bold, larger font)
    - Remaining top-10 arrows shown as thin grey lines (no label)
    - Cluster colors from fixed CLUSTER_COLORS palette
    """
    nr       = pd.read_csv('intermediary/NaturalResource.csv')
    nr_sample = nr[nr['Country Code'].isin(INCLUDE_LIST)]

    pca_df, pca_model, feat = run_clustering(nr_sample, year_filter=1995)

    # Loadings scaled for plotting
    loadings_raw = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)
    loadings_df  = pd.DataFrame(loadings_raw[:, :2], columns=['PC1', 'PC2'], index=feat)
    importance   = loadings_df.abs().sum(axis=1)

    scale = 2.8
    top5  = importance.nlargest(5).index
    top10 = importance.nlargest(10).index

    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100

    cluster_ids = sorted(pca_df['Cluster'].unique())
    fig = go.Figure()

    # Scatter points — one trace per cluster with fixed colors
    for cl in cluster_ids:
        sub   = pca_df[pca_df['Cluster'] == cl]
        color = CLUSTER_COLORS.get(cl, '#999')
        label = CLUSTER_LABELS.get(cl, f'Cluster {cl}')
        fig.add_trace(go.Scatter(
            x=sub['PC1'], y=sub['PC2'],
            mode='markers+text',
            marker=dict(size=10, color=color, opacity=0.82,
                        line=dict(width=1.2, color='white')),
            text=sub['Country Code'],
            textposition='top center',
            textfont=dict(size=8, color='#333'),
            name=label,
            hovertemplate='<b>%{text}</b><br>PC1=%{x:.2f}, PC2=%{y:.2f}<extra></extra>',
        ))

    # Background (thin grey) arrows for loading ranks 6–10
    for feat_name in top10:
        if feat_name in top5:
            continue
        x1 = loadings_df.loc[feat_name, 'PC1'] * scale
        y1 = loadings_df.loc[feat_name, 'PC2'] * scale
        fig.add_annotation(
            x=x1, y=y1, ax=0, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowwidth=1.2, arrowcolor='rgba(150,150,150,0.5)',
        )

    # Foreground (bold) arrows + labels for top 5
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
            text=f'<b>{feat_name}</b>',
            showarrow=False,
            font=dict(size=10, color='#111', family=FONT),
            bgcolor='rgba(255,255,255,0.7)',
            borderpad=2,
        )

    # Zero lines
    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    fig.add_vline(x=0, line=dict(color=GRID, width=1))

    fig.update_layout(**base_layout(
        height=680,
        margin=dict(l=60, r=60, t=50, b=60),
        xaxis=dict(
            title=f'PC1 ({var1:.1f}% variance explained)',
            gridcolor=GRID, gridwidth=0.5,
        ),
        yaxis=dict(
            title=f'PC2 ({var2:.1f}% variance explained)',
            gridcolor=GRID, gridwidth=0.5,
        ),
        legend=dict(
            font=dict(size=10),
            bgcolor='rgba(250,250,250,0.85)',
            bordercolor=GRID, borderwidth=1,
        ),
    ))

    save(fig, 'chart4_pca_biplot', w=1000, h=680)


# =============================================================================
# CHART 5 — Cluster Choropleth (consistent colors + improved dominance marker)
# =============================================================================
def chart5_cluster_map():
    """
    Choropleth improvements:
    - Colors match exactly the fixed CLUSTER_COLORS palette used in the biplot
    - Major producers: thick black border instead of red (more visible on all fills)
    - Hover shows dominant resource(s)
    """
    nr        = pd.read_csv('intermediary/NaturalResource.csv')
    nr_sample = nr[nr['Country Code'].isin(INCLUDE_LIST)]

    pca_df, _, _ = run_clustering(nr_sample, year_filter=1995)

    # Compute global production share per resource
    df_total = nr_sample.pivot_table(
        index=['Country', 'Country Code'],
        columns='Resource',
        values='Production_TotalValue',
        aggfunc='sum',
    ).reset_index().fillna(0)

    prod_cols = [c for c in df_total.columns if c not in ['Country', 'Country Code']]
    share_cols = []
    for col in prod_cols:
        total = df_total[col].sum()
        if total > 0:
            df_total[f'{col}_Share'] = df_total[col] / total * 100
            share_cols.append(f'{col}_Share')

    df_map = pca_df.merge(df_total[['Country Code'] + share_cols],
                          on='Country Code', how='left')

    THRESHOLD = 15.0
    df_map['Is_Dominant']        = (df_map[share_cols] >= THRESHOLD).any(axis=1)
    df_map['Dominant_Resources'] = df_map.apply(
        lambda r: ', '.join(
            sc.replace('_Share', '')
            for sc in share_cols if r.get(sc, 0) >= THRESHOLD
        ), axis=1,
    )

    fig = go.Figure()

    for cl in sorted(df_map['Cluster'].unique()):
        color = CLUSTER_COLORS.get(cl, '#aaa')
        label = CLUSTER_LABELS.get(cl, f'Cluster {cl}')
        sub   = df_map[df_map['Cluster'] == cl]

        # Normal countries (no dominance)
        s = sub[~sub['Is_Dominant']]
        if len(s):
            fig.add_trace(go.Choropleth(
                locations=s['Country Code'],
                z=[cl] * len(s),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                customdata=s[['ClusterLabels', 'Dominant_Resources']].values,
                hovertemplate='<b>%{location}</b><br>%{customdata[0]}<extra></extra>',
                name=f'{label}',
                marker=dict(line=dict(color='white', width=0.6)),
            ))

        # Dominant producers — thick black border
        s = sub[sub['Is_Dominant']]
        if len(s):
            fig.add_trace(go.Choropleth(
                locations=s['Country Code'],
                z=[cl] * len(s),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                customdata=s[['ClusterLabels', 'Dominant_Resources']].values,
                hovertemplate=(
                    '<b>%{location}</b><br>%{customdata[0]}<br>'
                    '<i>Major producer: %{customdata[1]}</i><extra></extra>'
                ),
                name=f'{label} ★ major producer',
                marker=dict(line=dict(color='#111', width=2.2)),
            ))

    # Add a legend-only trace for the black-border symbol
    fig.add_trace(go.Choropleth(
        locations=['ZZZ'],   # dummy — won't match any country
        z=[0],
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
        showscale=False,
        name='★ >15% of global output in ≥1 resource',
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
    fig.update_layout(**base_layout(
        height=500,
        margin=dict(l=0, r=180, t=50, b=10),
        legend=dict(
            x=1.01, y=0.5, xanchor='left', yanchor='middle',
            font=dict(size=10),
            bgcolor='rgba(250,250,250,0.9)',
            bordercolor=GRID, borderwidth=1,
        ),
    ))
    save(fig, 'chart5_cluster_map', w=1200, h=500)


# =============================================================================
# CHART 6 — ECI vs Log GDP animated (Rosling) — label outliers only
# =============================================================================
def chart6_rosling():
    """
    Rosling improvements:
    - Label only top/bottom 2 ECI outliers per cluster + extreme GDP outliers
    - All other country codes hidden (visible on hover only)
    - Subtitle corrects year reference (1995–2019)
    - Vertical annotation band marking forecast boundary if present
    """
    nr       = pd.read_csv('intermediary/NaturalResource.csv')
    nr_sample = nr[nr['Country Code'].isin(INCLUDE_LIST)]

    pca_agg, _, _ = run_clustering(nr_sample, agg_years=[1995, 1999, 2005])

    master = pd.read_csv('intermediary/Master.csv')
    master = master[master['Country Code'].isin(INCLUDE_LIST)]
    master = master.merge(
        pca_agg[['Country Code', 'Cluster', 'ClusterLabels']],
        on='Country Code', how='left',
    )

    data = master.copy()
    data['Log_GDP'] = np.log(data['GDP per capita (constant prices, PPP)'].clip(lower=1))
    data['Prod_PC'] = (data['Total_Production_Value'] / data['Population']).fillna(0).clip(lower=0)
    data = data.dropna(subset=['Cluster', 'Log_GDP', 'Economic Complexity Index'])
    data['Cluster'] = data['Cluster'].astype(int)

    # Determine which countries to label (outliers per cluster in 2019)
    d19 = data[data['Year'] == 2019].copy()
    label_set = set()
    for cl in data['Cluster'].unique():
        sub = d19[d19['Cluster'] == cl]
        if len(sub) == 0:
            continue
        label_set.update(sub.nlargest(2, 'Economic Complexity Index')['Country Code'])
        label_set.update(sub.nsmallest(2, 'Economic Complexity Index')['Country Code'])
        label_set.update(sub.nlargest(1, 'Log_GDP')['Country Code'])
        label_set.update(sub.nsmallest(1, 'Log_GDP')['Country Code'])

    # Build per-country data cache
    countries = data['Country Code'].unique()
    cdata = {}
    for code in countries:
        cdf = data[data['Country Code'] == code].sort_values('Year')
        origin = cdf[cdf['Year'] == 1995]
        if len(origin) == 0:
            continue
        mn, mx = data['Prod_PC'].min(), data['Prod_PC'].max()
        sizes = 8 + (np.sqrt(cdf['Prod_PC'].clip(0)) - np.sqrt(mn).clip(0)) / (
            max(np.sqrt(mx) - np.sqrt(mn), 1)
        ) * 38
        cdata[code] = {
            'years':   cdf['Year'].values,
            'x':       cdf['Log_GDP'].values,
            'y':       cdf['Economic Complexity Index'].values,
            'x0':      origin['Log_GDP'].values[0],
            'y0':      origin['Economic Complexity Index'].values[0],
            'size':    sizes.values,
            'name':    cdf['Country Name'].iloc[0],
            'cluster': int(cdf['Cluster'].iloc[0]),
            'prod_pc': cdf['Prod_PC'].values,
            'label':   code if code in label_set else '',
        }

    years = sorted(data['Year'].unique())
    clusters = sorted({cd['cluster'] for cd in cdata.values()})

    def build_traces(year):
        traces = []
        for cl in clusters:
            cc      = [c for c in cdata if cdata[c]['cluster'] == cl]
            color   = CLUSTER_COLORS.get(cl, '#aaa')
            cl_lbl  = CLUSTER_LABELS.get(cl, f'Cluster {cl}')

            # Trajectory lines from 1995 origin
            for code in cc:
                cd = cdata[code]
                idx = np.where(cd['years'] <= year)[0]
                xc = cd['x'][idx[-1]] if len(idx) else cd['x0']
                yc = cd['y'][idx[-1]] if len(idx) else cd['y0']
                traces.append(go.Scatter(
                    x=[cd['x0'], xc], y=[cd['y0'], yc],
                    mode='lines',
                    line=dict(color=color, width=1.2),
                    opacity=0.35,
                    showlegend=False, hoverinfo='skip',
                    legendgroup=f'cl_{cl}',
                ))

            # Bubble markers
            for first, code in enumerate(cc):
                cd  = cdata[code]
                idx = np.where(cd['years'] == year)[0]
                if len(idx):
                    xv, yv, sv = cd['x'][idx[0]], cd['y'][idx[0]], cd['size'][idx[0]]
                    pv = cd['prod_pc'][idx[0]]
                else:
                    mask = cd['years'] <= year
                    li   = np.where(mask)[0][-1] if mask.any() else 0
                    xv, yv, sv, pv = cd['x'][li], cd['y'][li], cd['size'][li], cd['prod_pc'][li]

                traces.append(go.Scatter(
                    x=[xv], y=[yv],
                    mode='markers+text',
                    marker=dict(size=sv, color=color, opacity=0.82,
                                line=dict(width=1, color='white')),
                    text=[cd['label']],
                    textposition='top center',
                    textfont=dict(size=8, color='#222'),
                    name=cl_lbl,
                    legendgroup=f'cl_{cl}',
                    showlegend=(first == 0),
                    customdata=[[cd['name'], pv, year]],
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        'Log GDP pc: %{x:.2f}<br>'
                        'ECI: %{y:.2f}<br>'
                        'Prod/capita: $%{customdata[1]:,.0f}<br>'
                        'Year: %{customdata[2]}<extra></extra>'
                    ),
                ))
        return traces

    first_traces = build_traces(years[0])
    fig = go.Figure(data=first_traces)

    frames = [go.Frame(data=build_traces(y), name=str(y)) for y in years]
    fig.frames = frames

    x_all = data['Log_GDP'].dropna()
    y_all = data['Economic Complexity Index'].dropna()
    fig.update_layout(**base_layout(
        height=660,
        margin=dict(l=60, r=60, t=70, b=80),
        xaxis=dict(
            title='Log GDP per capita (PPP, constant 2017 USD)',
            range=[x_all.min() - 0.2, x_all.max() + 0.2],
            gridcolor=GRID, gridwidth=0.5,
        ),
        yaxis=dict(
            title='Economic Complexity Index',
            range=[y_all.min() - 0.3, y_all.max() + 0.3],
            gridcolor=GRID, gridwidth=0.5,
            zeroline=True, zerolinecolor=GRID,
        ),
        legend=dict(
            font=dict(size=10),
            bgcolor='rgba(250,250,250,0.85)',
            bordercolor=GRID, borderwidth=1,
        ),
        updatemenus=[dict(
            type='buttons', showactive=True, x=1.0, y=-0.06,
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, dict(frame=dict(duration=450, redraw=True),
                                      transition=dict(duration=200))]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0), mode='immediate')]),
            ],
        )],
        sliders=[dict(
            active=0, len=0.88, x=0.04, y=-0.14,
            currentvalue=dict(prefix='Year: ', font=dict(size=13)),
            steps=[dict(
                args=[[str(y)], dict(frame=dict(duration=300, redraw=True), mode='immediate')],
                method='animate', label=str(y),
            ) for y in years],
        )],
    ))
    save(fig, 'chart6_rosling_eci_gdp', w=950, h=660)


# =============================================================================
# CHART 7 — Regression forest plot: Models 1, 3a, 3b side by side
# =============================================================================
def chart7_regression_forest_plot():
    """
    Forest plot for the parsimonious variable set (+ interactions) across
    Models 1 (kitchen-sink), 3a (no lag), 3b (with lag).
    Point estimates + 95% CIs. Zero line dashed.
    """
    master   = pd.read_csv('intermediary/Master.csv')
    clusters = pd.read_csv('intermediary/clusters1995.csv')
    cluster_map = clusters[['Country Code', 'ClusterLabels']].drop_duplicates('Country Code')

    df = master.merge(cluster_map, on='Country Code', how='left')
    df = df.sort_values(['Country Code', 'Year'])

    # Compute per-capita production value (not pre-computed in Master.csv)
    df['Total_Production_Value_Per_Capita'] = (
        df['Total_Production_Value'] / df['Population'].replace(0, np.nan)
    )

    # Feature engineering (mirrors NB6)
    df['log_HCI']              = np.log1p(df['Human capital index'])
    df['log_GFCF']             = np.log1p(df['Gross fixed capital formation, all, Constant prices, Percent of GDP'])
    df['log_Production_Value'] = np.log1p(df['Total_Production_Value_Per_Capita'])
    for col in ['log_HCI', 'log_GFCF', 'log_Production_Value']:
        df[f'{col}_c'] = df[col] - df[col].mean()
    df['log_HCI_x_log_Production']  = df['log_HCI_c']  * df['log_Production_Value_c']
    df['log_GFCF_x_log_Production'] = df['log_GFCF_c'] * df['log_Production_Value_c']
    df['ECI_lag1'] = df.groupby('Country Code')['Economic Complexity Index'].shift(1)

    PARSIMONIOUS = [
        'log_HCI', 'log_GFCF',
        'Political stability — estimate', 'Rule of law index',
        'log_Production_Value', 'Trade (% of GDP)',
    ]
    INTERACT = ['log_HCI_x_log_Production', 'log_GFCF_x_log_Production']

    def driscoll_kraay(y, X, time, groups):
        from types import SimpleNamespace
        raw = sm.OLS(y, X).fit()
        # HAC-Groupsum requires 0-based integer time index, NOT actual year values.
        # Passing years (1995–2019) directly inflates the bandwidth by ~2000x.
        time_idx = (time - time.min()).astype(int)
        rob = raw.get_robustcov_results(
            cov_type='HAC-Groupsum', time=time_idx, groups=groups, maxlags=2,
        )
        names = X.columns.tolist() if hasattr(X, 'columns') else raw.model.exog_names
        ns = SimpleNamespace(
            params=pd.Series(np.asarray(rob.params), index=names),
            bse   =pd.Series(np.asarray(rob.bse),    index=names),
        )
        return ns

    # Model 3a
    v3a = PARSIMONIOUS + INTERACT
    d3a = df[['Economic Complexity Index', 'Country Code', 'Year'] + v3a].dropna()
    m3a = driscoll_kraay(
        d3a['Economic Complexity Index'],
        sm.add_constant(d3a[v3a]),
        d3a['Year'].values, d3a['Country Code'].values,
    )

    # Model 3b (with lagged ECI)
    v3b = PARSIMONIOUS + INTERACT + ['ECI_lag1']
    d3b = df[['Economic Complexity Index', 'Country Code', 'Year'] + v3b].dropna()
    m3b = driscoll_kraay(
        d3b['Economic Complexity Index'],
        sm.add_constant(d3b[v3b]),
        d3b['Year'].values, d3b['Country Code'].values,
    )

    # ── Build display variable list + readable labels ─────────────────────────
    DISPLAY_VARS = PARSIMONIOUS + INTERACT    # 8 rows

    SHORT = {
        'log_HCI':                     'log(Human Capital Index)',
        'log_GFCF':                    'log(GFCF % GDP)',
        'Political stability — estimate': 'Political stability',
        'Rule of law index':           'Rule of law',
        'log_Production_Value':        'log(Prod. value p.c.)',
        'Trade (% of GDP)':            'Trade openness',
        'log_HCI_x_log_Production':    'log(HCI) × log(Prod.)',
        'log_GFCF_x_log_Production':   'log(GFCF) × log(Prod.)',
    }

    # ── Numeric y-axis layout (proper offsets, no string-jitter hacks) ──────────
    y_labels   = [SHORT[v] for v in DISPLAY_VARS]
    n          = len(DISPLAY_VARS)
    y_base     = list(range(n))          # 0 … 7

    # Models 3a and 3b only — Model 1 uses raw (unstandardised) variables so its
    # coefficients are on a 100–1000x different scale and would crush the x-axis.
    M_CFG = [
        (m3a, -0.18, '#4a6fa5', 'circle', 'Model 3a  no lag (DK SEs)'),
        (m3b, +0.18, '#2e7d4a', 'square', 'Model 3b  with lag (DK SEs)'),
    ]

    fig = go.Figure()

    for model_obj, y_off, col, sym, label in M_CFG:
        xs, ys, err_plus, err_minus, htexts = [], [], [], [], []
        for i, v in enumerate(DISPLAY_VARS):
            if v not in model_obj.params:
                continue
            c = float(model_obj.params[v])
            s = float(model_obj.bse[v])
            lo, hi = c - 1.96 * s, c + 1.96 * s
            xs.append(c)
            ys.append(y_base[i] + y_off)
            err_plus.append(hi - c)
            err_minus.append(c - lo)
            htexts.append(y_labels[i])

        if not xs:
            continue

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            error_x=dict(
                type='data', symmetric=False,
                array=err_plus, arrayminus=err_minus,
                color=col, thickness=1.8, width=5,
            ),
            mode='markers',
            marker=dict(color=col, size=9, symbol=sym,
                        line=dict(width=1.5, color='white')),
            name=label,
            customdata=htexts,
            hovertemplate='<b>%{customdata}</b><br>coef = %{x:.3f}<extra>' + label + '</extra>',
        ))

    # Alternating row shading — xref='paper' so shapes don't affect x autorange
    for i in range(n):
        if i % 2 == 0:
            fig.add_shape(
                type='rect',
                xref='paper', yref='y',
                x0=0, x1=1,
                y0=i - 0.48, y1=i + 0.48,
                fillcolor='rgba(200,210,230,0.12)',
                line=dict(width=0), layer='below',
            )

    # Separator between main effects (0-5) and interactions (6-7)
    sep = len(PARSIMONIOUS) - 0.5   # 5.5
    fig.add_shape(
        type='line',
        xref='paper', yref='y',
        x0=0, x1=1, y0=sep, y1=sep,
        line=dict(color=GRID, width=1.5, dash='dot'),
    )
    # Zero line
    fig.add_vline(x=0, line=dict(color='#888', width=1.5, dash='dash'))

    # Compute x-range from data (clamp to keep the chart readable)
    all_xs_lo = [float(model_obj.params[v]) - 1.96 * float(model_obj.bse[v])
                 for model_obj, _, _, _, _ in M_CFG for v in DISPLAY_VARS if v in model_obj.params]
    all_xs_hi = [float(model_obj.params[v]) + 1.96 * float(model_obj.bse[v])
                 for model_obj, _, _, _, _ in M_CFG for v in DISPLAY_VARS if v in model_obj.params]
    x_lo = min(all_xs_lo) - 0.05
    x_hi = max(all_xs_hi) + 0.05

    fig.update_layout(**base_layout(
        height=640,
        margin=dict(l=240, r=80, t=80, b=60),
        font=dict(family=FONT, size=13, color=NAVY),
        xaxis=dict(
            title=dict(text='Coefficient (95% CI)', font=dict(size=15)),
            tickfont=dict(size=13),
            gridcolor=GRID, gridwidth=0.5,
            zeroline=False,
            range=[x_lo, x_hi],
        ),
        yaxis=dict(
            tickvals=y_base,
            ticktext=y_labels,
            tickfont=dict(size=14),
            range=[-0.6, n - 0.4],
            showgrid=False,
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='center', x=0.5,
            font=dict(size=14),
            itemsizing='constant',
        ),
    ))
    save(fig, 'chart7_regression_forest', w=1050, h=640)


# =============================================================================
# RUNNER
# =============================================================================
if __name__ == '__main__':
    tasks = [
        ('Chart 1 — Variable overview',          chart1_variable_overview),
        ('Chart 1b — Country selection map',     chart_country_selection),
        ('Chart 2 — Correlation with ECI bar',   chart2_correlation_with_eci),
        ('Chart 3 — PCA loadings heatmap',        chart3_pca_loadings),
        ('Chart 4 — PCA biplot',                  chart4_pca_biplot),
        ('Chart 5 — Cluster choropleth',          chart5_cluster_map),
        ('Chart 6 — Rosling animated',            chart6_rosling),
        ('Chart 7 — Regression forest plot',      chart7_regression_forest_plot),
    ]

    results = {}
    for name, fn in tasks:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print('='*60)
        try:
            fn()
            results[name] = 'OK'
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[name] = f'FAIL: {e}'

    print(f"\n{'='*60}")
    print('SUMMARY')
    print('='*60)
    for name, status in results.items():
        icon = '✓' if status == 'OK' else '✗'
        print(f"  {icon}  {name}: {status}")
    print(f"\nOutputs in: {OUT}/")
