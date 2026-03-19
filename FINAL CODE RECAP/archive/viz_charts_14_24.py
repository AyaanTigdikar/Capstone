"""
Charts 14–24: Case studies, robustness, summary figures.
Run from: /Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP/
Output:   Final/improved_14_24/

Congo = COG (Congo, Rep.) — Cluster 0 "Some Oil, No Minerals"
"""
import matplotlib; matplotlib.use('Agg')
import os, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NB5  = 'Final/NB5'
INTR = 'intermediary'
BOOT = 'intermediary/bootstrap'
OUT  = 'Final/improved_14_24'
os.makedirs(OUT, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
FONT  = 'IBM Plex Sans, -apple-system, BlinkMacSystemFont, sans-serif'
BG    = '#fafafa'
NAVY  = '#1a2744'
GRID  = '#e5e7eb'

PALETTE = dict(
    blue   ='#4a6fa5',
    red    ='#c23a3a',
    green  ='#2e7d4a',
    orange ='#d4853b',
    grey   ='#999999',
    lasso  ='#c23a3a',
    ridge  ='#4a6fa5',
    en     ='#2e7d4a',
    rf     ='#d4853b',
)
CLUSTER_COLORS = {0: '#4a6fa5', 1: '#c23a3a', 2: '#2e7d4a', 3: '#d4853b'}
CLUSTER_LABELS = {
    0: 'Some Oil, No Minerals',
    1: 'No Oil, No Minerals',
    2: 'Minerals, No Oil',
    3: 'Oil & Minerals',
}
WRITE_CONFIG = {'displayModeBar': False, 'responsive': True}

# Case study countries
CASE_CODES  = ['COG', 'AZE', 'CHL']
CASE_NAMES  = {'COG': 'Congo, Rep.', 'AZE': 'Azerbaijan', 'CHL': 'Chile'}
CASE_COLORS = {'COG': '#c23a3a', 'AZE': '#4a6fa5', 'CHL': '#2e7d4a'}

# Chart 16 Congo peers (all cluster 0)
CONGO_PEERS = ['COG', 'AGO', 'MYS', 'AZE']
# Chart 17 Azerbaijan peers (cluster 0 selection)
AZE_PEERS   = ['AZE', 'NGA', 'IRN', 'BOL', 'EGY']


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.write_html(f"{path}.html", config=WRITE_CONFIG)
    print(f"  Saved: {path}.html")


def base_layout(**kw):
    d = dict(
        template='plotly_white',
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family=FONT, size=11, color=NAVY),
        margin=dict(l=60, r=40, t=40, b=50),
    )
    d.update(kw)
    return d


def load_master():
    df = pd.read_csv(os.path.join(INTR, 'Master.csv'))
    # Short alias column
    df['prod_pc'] = df['Total_Production_Value'] / df['Population'].replace(0, np.nan)
    return df


def load_clusters():
    return pd.read_csv(os.path.join(INTR, 'clusters1995.csv'))


def resource_rich_codes():
    """Return country codes in the resource-rich sample (from clusters1995.csv)."""
    cl = load_clusters()
    return cl['Country Code'].unique().tolist()


# =============================================================================
# CHART 14 — Case Study Comparison Table (enhanced)
# =============================================================================
def chart14_case_study_table():
    df  = load_master()
    cl  = load_clusters()[['Country Code', 'Cluster', 'ClusterLabels']].drop_duplicates()
    rr  = resource_rich_codes()
    df2 = df.merge(cl, on='Country Code', how='left')

    # Recent snapshot (2017-2019 avg) for level variables
    snap = df2[df2['Year'].between(2017, 2019)].groupby('Country Code').mean(numeric_only=True)
    snap_cl = (df2[df2['Year'] == 2019]
               [['Country Code', 'Country Name', 'Cluster', 'ClusterLabels']]
               .drop_duplicates().set_index('Country Code'))
    snap = snap.drop(columns=[c for c in ['Cluster'] if c in snap.columns])
    snap = snap.join(snap_cl[['Country Name', 'Cluster', 'ClusterLabels']])

    # ECI change: 2019 minus 1995 (per country, no averaging needed)
    eci95  = df2[df2['Year'] == 1995][['Country Code', 'Economic Complexity Index']].set_index('Country Code')
    eci19  = df2[df2['Year'] == 2019][['Country Code', 'Economic Complexity Index']].set_index('Country Code')
    eci_delta = (eci19['Economic Complexity Index'] - eci95['Economic Complexity Index']).rename('ECI_delta')
    snap = snap.join(eci_delta)

    # Columns to display: (source_col, display_label, fmt_type)
    # fmt_type: 'f2'=2dp, 'f1'=1dp, 'pct'=%, 'Mpc'=$M/cap, 'delta'=+/- 2dp
    VARS = [
        ('Economic Complexity Index',                                                'ECI (avg)',      'f2'),
        ('ECI_delta',                                                                'ECI Δ 95→19',    'delta'),
        ('Human capital index',                                                      'HCI',            'f2'),
        ('Access to electricity (% of population)',                                  'Electricity (%)', 'f1'),
        ('Rule of law index',                                                        'Rule of Law',    'f2'),
        ('Total natural resources rents (% of GDP)',                                 'NR Rents (%)',   'f1'),
        ('Mineral rents (% of GDP)',                                                 'Mineral Rents (%)','f1'),
        ('Gross fixed capital formation, all, Constant prices, Percent of GDP',     'GFCF (%)',       'f1'),
        ('Trade (% of GDP)',                                                         'Trade (%)',      'f1'),
    ]
    var_cols   = [v[0] for v in VARS]
    var_labels = [v[1] for v in VARS]
    var_fmts   = [v[2] for v in VARS]

    def fmt_val(val, ftype):
        if pd.isna(val):
            return '—'
        if ftype == 'f2':    return f"{val:.2f}"
        if ftype == 'f1':    return f"{val:.1f}"
        if ftype == 'delta': return f"{val:+.2f}"
        if ftype == 'Mpc':   return f"${val/1e6:.1f}M"
        return f"{val:.2f}"

    # Resource-rich sample for percentile coloring
    rr_snap = snap.loc[[c for c in rr if c in snap.index]]

    def pct_color(val, col, base_hex):
        """Blend white→base_hex based on percentile within sample."""
        if pd.isna(val) or col not in rr_snap.columns:
            return '#f7f8fa'
        series = rr_snap[col].dropna()
        if len(series) < 2:
            return '#f7f8fa'
        pct = float((series <= val).mean())   # empirical CDF
        r0, g0, b0 = 247, 248, 250           # near-white base
        r1 = int(base_hex[1:3], 16)
        g1 = int(base_hex[3:5], 16)
        b1 = int(base_hex[5:7], 16)
        r  = int(r0 + (r1 - r0) * pct * 0.55)
        g  = int(g0 + (g1 - g0) * pct * 0.55)
        b  = int(b0 + (b1 - b0) * pct * 0.55)
        return f'rgb({r},{g},{b})'

    # Build row data
    case_rows = []
    for code in CASE_CODES:
        if code not in snap.index:
            continue
        base_hex = CASE_COLORS[code]
        vals = {}
        for col in var_cols:
            vals[col] = snap.loc[code, col] if col in snap.columns else np.nan
        case_rows.append({'_code': code, '_type': 'case', '_base': base_hex,
                          'Country': CASE_NAMES[code],
                          'Cluster': snap.loc[code, 'ClusterLabels'] if 'ClusterLabels' in snap.columns else '—',
                          **vals})

    agg_rows = []
    for agg_label, agg_fn in [('Sample Mean', 'mean'), ('Sample Median', 'median'),
                                ('Sample P25',  lambda s: s.quantile(0.25)),
                                ('Sample P75',  lambda s: s.quantile(0.75))]:
        vals = {}
        for col in var_cols:
            if col in rr_snap.columns:
                vals[col] = rr_snap[col].dropna().agg(agg_fn) if isinstance(agg_fn, str) else agg_fn(rr_snap[col].dropna())
            else:
                vals[col] = np.nan
        agg_rows.append({'_code': None, '_type': 'agg', '_base': '#888888',
                         'Country': agg_label, 'Cluster': '—', **vals})

    all_rows = case_rows + agg_rows
    n = len(all_rows)

    # Build cell_vals columns: [Country, Cluster, var1, var2, ...]
    col_country = [r['Country'] for r in all_rows]
    col_cluster = [r['Cluster'] for r in all_rows]
    data_cols   = []
    for col, lbl, ftype in VARS:
        data_cols.append([fmt_val(r.get(col, np.nan), ftype) for r in all_rows])

    cell_vals   = [col_country, col_cluster] + data_cols
    header_vals = ['Country', 'Cluster'] + var_labels

    # Build fill_color: list of per-column color lists
    # Country + Cluster columns: fixed row color
    def row_bg(r):
        if r['_type'] == 'case':
            h = r['_base']
            rv, gv, bv = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
            return f'rgba({rv},{gv},{bv},0.10)'
        return 'rgba(235,235,235,0.5)'

    fill_country = [row_bg(r) for r in all_rows]
    fill_cluster = fill_country[:]   # same tint

    fill_data = []
    for col, lbl, ftype in VARS:
        col_fills = []
        for r in all_rows:
            val = r.get(col, np.nan)
            if r['_type'] == 'case':
                col_fills.append(pct_color(val, col, r['_base']))
            else:
                col_fills.append('rgba(235,235,235,0.4)')
        fill_data.append(col_fills)

    fill_colors = [fill_country, fill_cluster] + fill_data

    # Font colors: darker for country name, navy for rest
    font_country = [NAVY] * n
    # Make agg labels italic by using bold for case rows
    font_data    = [NAVY] * n

    # Build per-column font color lists (all NAVY, but case study Country cells bold)
    cell_font_color = [[NAVY] * n] * len(header_vals)

    fig = go.Figure(go.Table(
        columnwidth=[130, 120] + [75] * len(VARS),
        header=dict(
            values=[f'<b>{v}</b>' for v in header_vals],
            fill_color=[NAVY, NAVY] + [NAVY] * len(VARS),
            font=dict(color='white', size=10, family=FONT),
            align=['left', 'left'] + ['center'] * len(VARS),
            height=34,
        ),
        cells=dict(
            values=cell_vals,
            fill_color=fill_colors,
            font=dict(color=NAVY, size=11, family=FONT),
            align=['left', 'left'] + ['center'] * len(VARS),
            height=30,
            suffix=[''] * len(cell_vals),
        ),
    ))

    # Separator annotation between case countries and sample stats
    # (Plotly Tables don't have built-in separators — add an annotation outside)
    fig.update_layout(**base_layout(
        height=310,
        margin=dict(l=10, r=10, t=15, b=10),
    ))
    save(fig, 'chart14_case_study_table')


# =============================================================================
# CHART 15 — Congo Small Multiples (8 panels)
# =============================================================================
def chart15_congo_small_multiples():
    df = load_master()
    rr = resource_rich_codes()
    rr_df = df[df['Country Code'].isin(rr)]

    PANELS = [
        ('Human capital index',                                              'HCI'),
        ('Access to electricity (% of population)',                         'Electricity (%)'),
        ('Political stability — estimate',                                   'Political Stability'),
        ('Domestic credit to private sector (% of GDP)',                    'Credit (% GDP)'),
        ('prod_pc',                                                         'Production ($/cap)'),
        ('Oil rents (% of GDP)',                                            'Oil Rents (% GDP)'),
        ('Gross fixed capital formation, all, Constant prices, Percent of GDP', 'GFCF (% GDP)'),
        ('Trade (% of GDP)',                                                'Trade (% GDP)'),
    ]

    cog = df[df['Country Code'] == 'COG'].sort_values('Year')

    # Sample mean ± std per year
    means = rr_df.groupby('Year').mean(numeric_only=True)
    stds  = rr_df.groupby('Year').std(numeric_only=True)

    fig = make_subplots(rows=2, cols=4, shared_xaxes=True,
                        horizontal_spacing=0.06, vertical_spacing=0.12)

    for idx, (col, label) in enumerate(PANELS):
        row, c = divmod(idx, 4)
        row += 1; c += 1
        years = cog['Year'].values
        if col == 'prod_pc':
            cog_vals = (cog['Total_Production_Value'] / cog['Population'].replace(0, np.nan)).values
            m_vals = means.get('prod_pc', pd.Series(dtype=float)).reindex(years).values
            s_vals = stds.get('prod_pc', pd.Series(dtype=float)).reindex(years).values
        else:
            cog_vals = cog[col].values if col in cog.columns else np.full(len(years), np.nan)
            m_vals   = means[col].reindex(years).values if col in means.columns else np.full(len(years), np.nan)
            s_vals   = stds[col].reindex(years).values  if col in stds.columns  else np.full(len(years), np.nan)

        m_up = m_vals + s_vals
        m_lo = m_vals - s_vals

        # Band
        fig.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([m_up, m_lo[::-1]]),
            fill='toself', fillcolor='rgba(150,150,150,0.18)',
            line=dict(width=0), showlegend=(idx == 0), name='Sample ±1 SD',
            hoverinfo='skip',
        ), row=row, col=c)
        # Mean
        fig.add_trace(go.Scatter(
            x=years, y=m_vals,
            line=dict(color='#888888', width=1.5, dash='dash'),
            showlegend=(idx == 0), name='Sample Mean',
        ), row=row, col=c)
        # Congo
        fig.add_trace(go.Scatter(
            x=years, y=cog_vals,
            line=dict(color=CASE_COLORS['COG'], width=2.5),
            showlegend=(idx == 0), name='Congo, Rep.',
        ), row=row, col=c)

        fig.update_xaxes(showgrid=True, gridcolor=GRID, row=row, col=c)
        fig.update_yaxes(title_text=label, showgrid=True, gridcolor=GRID, row=row, col=c,
                         title_font=dict(size=10))

    fig.update_layout(**base_layout(height=520,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.03,
                    font=dict(size=10)),
    ))
    save(fig, 'chart15_congo_small_multiples')


# =============================================================================
# CHART 16 — ECI Trajectories: Congo vs Peers
# =============================================================================
def chart16_eci_congo_peers():
    df = load_master()
    cl = load_clusters()[['Country Code', 'Cluster', 'ClusterLabels']].drop_duplicates()

    # Cluster 0 average
    c0_codes = cl[cl['Cluster'] == 0]['Country Code'].tolist()
    c0_avg = (df[df['Country Code'].isin(c0_codes)]
              .groupby('Year')['Economic Complexity Index'].mean().reset_index())
    c0_avg['Country Code'] = '_C0_AVG'
    c0_avg['Country Name'] = 'Cluster 0 Avg'

    PEERS = {
        'COG': ('Congo, Rep.', CASE_COLORS['COG'], 'solid', 2.5),
        'AGO': ('Angola',      '#d4853b',           'dot',   1.8),
        'MYS': ('Malaysia',    '#2e7d4a',           'dot',   1.8),
        'AZE': ('Azerbaijan',  CASE_COLORS['AZE'],  'dot',   1.8),
    }

    fig = go.Figure()

    # Cluster average first (background)
    fig.add_trace(go.Scatter(
        x=c0_avg['Year'], y=c0_avg['Economic Complexity Index'],
        name='Cluster 0 Avg',
        line=dict(color='#cccccc', width=2, dash='dash'),
        mode='lines',
    ))

    for code, (name, color, dash, width) in PEERS.items():
        sub = df[df['Country Code'] == code].sort_values('Year')
        fig.add_trace(go.Scatter(
            x=sub['Year'], y=sub['Economic Complexity Index'],
            name=name,
            line=dict(color=color, width=width, dash=dash),
            mode='lines+markers',
            marker=dict(size=4 if dash == 'solid' else 3),
        ))

    fig.update_layout(**base_layout(height=440,
        xaxis=dict(title='Year', showgrid=True, gridcolor=GRID, dtick=5),
        yaxis=dict(title='Economic Complexity Index', showgrid=True, gridcolor=GRID),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.18, font=dict(size=10)),
        hovermode='x unified',
    ))
    save(fig, 'chart16_eci_congo_peers')


# =============================================================================
# CHART 17 — ECI Trajectories: Azerbaijan vs Peers
# =============================================================================
def chart17_eci_aze_peers():
    df = load_master()
    cl = load_clusters()[['Country Code', 'Cluster', 'ClusterLabels']].drop_duplicates()

    c0_codes = cl[cl['Cluster'] == 0]['Country Code'].tolist()
    c0_avg = (df[df['Country Code'].isin(c0_codes)]
              .groupby('Year')['Economic Complexity Index'].mean().reset_index())

    PEERS = {
        'AZE': ('Azerbaijan', CASE_COLORS['AZE'], 'solid', 2.5),
        'NGA': ('Nigeria',    '#d4853b',           'dot',   1.8),
        'IRN': ('Iran',       '#8b5e3c',           'dot',   1.8),
        'BOL': ('Bolivia',    '#2e7d4a',           'dot',   1.8),
        'EGY': ('Egypt',      '#c23a3a',           'dot',   1.8),
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=c0_avg['Year'], y=c0_avg['Economic Complexity Index'],
        name='Cluster 0 Avg',
        line=dict(color='#cccccc', width=2, dash='dash'),
        mode='lines',
    ))

    for code, (name, color, dash, width) in PEERS.items():
        sub = df[df['Country Code'] == code].sort_values('Year')
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['Year'], y=sub['Economic Complexity Index'],
            name=name,
            line=dict(color=color, width=width, dash=dash),
            mode='lines+markers',
            marker=dict(size=4 if dash == 'solid' else 3),
        ))

    fig.update_layout(**base_layout(height=440,
        xaxis=dict(title='Year', showgrid=True, gridcolor=GRID, dtick=5),
        yaxis=dict(title='Economic Complexity Index', showgrid=True, gridcolor=GRID),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.18, font=dict(size=10)),
        hovermode='x unified',
    ))
    save(fig, 'chart17_eci_aze_peers')


# =============================================================================
# CHART 18 — VIF Chart (restyled from coefficient_summary_table.csv)
# =============================================================================
def chart18_vif():
    coef = pd.read_csv(os.path.join(NB5, 'coefficient_summary_table.csv'))
    if 'VIF' not in coef.columns:
        print("  [chart18] VIF column not found in coefficient_summary_table.csv — skipping")
        return

    vif = coef[['Feature', 'VIF']].dropna().sort_values('VIF', ascending=True).reset_index(drop=True)

    def vif_color(v):
        if v < 5:   return PALETTE['green']
        if v < 10:  return PALETTE['orange']
        return PALETTE['red']

    colors = [vif_color(v) for v in vif['VIF']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=vif['Feature'], x=vif['VIF'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}" for v in vif['VIF']],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='%{y}<br>VIF: %{x:.2f}<extra></extra>',
    ))

    # Threshold lines
    for thresh, label in [(5, 'Concern (5)'), (10, 'Severe (10)')]:
        fig.add_vline(x=thresh, line=dict(color=PALETTE['grey'], width=1.5, dash='dash'))
        fig.add_annotation(x=thresh, y=1.02, yref='paper', text=label,
                           showarrow=False, font=dict(size=9, color=PALETTE['grey']),
                           xanchor='center')

    h = max(350, len(vif) * 26)
    fig.update_layout(**base_layout(height=h,
        xaxis=dict(title='Variance Inflation Factor', showgrid=True, gridcolor=GRID, range=[0, max(vif['VIF']) * 1.25]),
        yaxis=dict(autorange='reversed', showgrid=False),
        margin=dict(l=200, r=60, t=40, b=50),
    ))
    save(fig, 'chart18_vif')


# =============================================================================
# CHART 19 — Bootstrap Robustness (R² distributions + coefficient stability)
# =============================================================================
def chart19_bootstrap_robustness():
    metrics = pd.read_csv(os.path.join(BOOT, 'nb5_boot_metrics.csv'))
    coefs   = pd.read_csv(os.path.join(BOOT, 'nb5_boot_coefs.csv'))

    MODEL_COLS = {
        'LASSO':         ('LASSO_test_r2',        PALETTE['lasso']),
        'Ridge':         ('Ridge_test_r2',         PALETTE['ridge']),
        'Elastic Net':   ('Elastic Net_test_r2',   PALETTE['en']),
        'Random Forest': ('Random Forest_test_r2', PALETTE['rf']),
    }

    fig = make_subplots(rows=1, cols=2, subplot_titles=['', ''],
                        horizontal_spacing=0.12)

    # Left: R² violin per model
    def h2rgba(h, a=0.35):
        r, g, b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
        return f'rgba({r},{g},{b},{a})'

    for name, (col, color) in MODEL_COLS.items():
        if col not in metrics.columns:
            continue
        vals = metrics[col].dropna().values
        fig.add_trace(go.Violin(
            y=vals, name=name,
            fillcolor=h2rgba(color), line_color=color,
            box_visible=True, meanline_visible=True,
            points=False,
            showlegend=True,
        ), row=1, col=1)

    # Right: Coefficient stability for top LASSO features (sorted by mean |coef|)
    lasso_cols = [c for c in coefs.columns if c.startswith('LASSO__') and 'L1_ECI' not in c]
    if lasso_cols:
        feat_means = {c: coefs[c].abs().mean() for c in lasso_cols}
        top_feats = sorted(feat_means, key=feat_means.get, reverse=True)[:10]

        # Short labels
        def shorten(name):
            name = name.replace('LASSO__', '')
            replacements = {
                'Domestic credit to private sector (% of GDP)': 'Credit',
                'Human capital index': 'HCI',
                'Access to electricity (% of population)': 'Electricity',
                'Rule of law index': 'Rule of Law',
                'Trade (% of GDP)': 'Trade',
                'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'GFCF',
                'Political stability — estimate': 'Pol. Stability',
                'Total_Production_Value_Per_Capita': 'Production p.c.',
                'Urban population (% of total population)': 'Urban Pop.',
                'Government revenue': 'Gov. Revenue',
                'Use of IMF credit (DOD, current US$)': 'IMF Credit',
                'HCI_x_ProductionValue': 'HCI × Prod.',
                'RuleOfLaw_x_ProductionValue': 'RoL × Prod.',
                'Resource_HHI': 'Resource HHI',
                'Inflation_roll5': 'Inflation (5y)',
                'RealRate_roll5': 'Real Rate (5y)',
                'Capital depreciation rate': 'Cap. Deprec.',
                'Share of investment in GDP': 'Inv. Share',
                'Landlocked': 'Landlocked',
                'Adjusted savings: gross savings (% of GNI)': 'Savings',
                'Real interest rate (%)': 'Real Interest',
                'Inflation, consumer prices (annual %)': 'Inflation',
            }
            for k, v in replacements.items():
                name = name.replace(k, v)
            return name[:28]

        short_labels = [shorten(f) for f in top_feats]

        for i, (feat, label) in enumerate(zip(top_feats, short_labels)):
            vals = coefs[feat].dropna().values
            fig.add_trace(go.Box(
                x=vals, y=[label] * len(vals),
                orientation='h', name=label,
                marker_color=PALETTE['blue'],
                line_color=PALETTE['blue'],
                boxmean=True,
                showlegend=False,
                width=0.6,
            ), row=1, col=2)

        fig.add_vline(x=0, line=dict(color=PALETTE['grey'], width=1.5, dash='dash'), row=1, col=2)

    fig.update_yaxes(title_text='Test R²', row=1, col=1, showgrid=True, gridcolor=GRID)
    fig.update_xaxes(title_text='Model', row=1, col=1, showgrid=False)
    fig.update_xaxes(title_text='Standardised Coefficient', showgrid=True, gridcolor=GRID, row=1, col=2)
    fig.update_yaxes(showgrid=False, autorange='reversed', row=1, col=2)

    fig.update_layout(**base_layout(height=500, showlegend=True,
        legend=dict(orientation='h', x=0.25, xanchor='center', y=-0.15, font=dict(size=10)),
    ))
    save(fig, 'chart19_bootstrap_robustness')


# =============================================================================
# CHART 20 — Consensus Variable Importance Heatmap
# =============================================================================
def chart20_consensus_heatmap():
    imp = pd.read_csv(os.path.join(NB5, 'all_importance.csv'))
    imp = imp[~imp['Feature'].str.contains('L1_ECI', na=False)]

    MODEL_COLS = ['LASSO', 'Ridge', 'Elastic Net', 'Random Forest']
    available = [c for c in MODEL_COLS if c in imp.columns]

    imp['mean_imp'] = imp[available].mean(axis=1)
    imp = imp.sort_values('mean_imp', ascending=False).head(18).reset_index(drop=True)

    def shorten(name):
        m = {
            'Domestic credit to private sector (% of GDP)': 'Domestic Credit',
            'Human capital index': 'Human Capital Index',
            'Access to electricity (% of population)': 'Electricity Access',
            'Rule of law index': 'Rule of Law',
            'Trade (% of GDP)': 'Trade (% GDP)',
            'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'GFCF',
            'Political stability — estimate': 'Political Stability',
            'Total_Production_Value_Per_Capita': 'Production per Capita',
            'Urban population (% of total population)': 'Urban Population',
            'Government revenue': 'Government Revenue',
            'Use of IMF credit (DOD, current US$)': 'IMF Credit',
            'HCI_x_ProductionValue': 'HCI × Production',
            'RuleOfLaw_x_ProductionValue': 'Rule of Law × Production',
            'Resource_HHI': 'Resource HHI',
            'Inflation_roll5': 'Inflation (5yr Rolling)',
            'RealRate_roll5': 'Real Rate (5yr Rolling)',
            'Capital depreciation rate': 'Capital Depreciation',
            'Share of investment in GDP': 'Investment Share',
            'Landlocked': 'Landlocked',
            'Adjusted savings: gross savings (% of GNI)': 'Gross Savings',
            'Real interest rate (%)': 'Real Interest Rate',
            'Inflation, consumer prices (annual %)': 'Inflation',
        }
        return m.get(name, name[:32])

    labels = [shorten(f) for f in imp['Feature']]
    z = imp[available].values

    # Color annotation: rank within each column (1=most important)
    rank_z = np.zeros_like(z, dtype=float)
    for j in range(z.shape[1]):
        col_vals = z[:, j]
        ranks = len(col_vals) - col_vals.argsort().argsort()
        rank_z[:, j] = ranks

    text = [[f"#{int(rank_z[i,j])}<br>{z[i,j]:.2f}" for j in range(len(available))] for i in range(len(labels))]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=available,
        y=labels,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=9),
        colorscale=[
            [0.0,  '#f7f9fc'],
            [0.4,  '#a8c4e0'],
            [0.7,  '#4a6fa5'],
            [1.0,  '#1a2744'],
        ],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text='Norm. Imp.', side='right', font=dict(size=10)),
            thickness=14, len=0.8,
        ),
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(**base_layout(
        height=max(420, len(labels) * 28),
        xaxis=dict(side='top', showgrid=False),
        yaxis=dict(autorange='reversed', showgrid=False),
        margin=dict(l=180, r=80, t=50, b=30),
    ))
    save(fig, 'chart20_consensus_heatmap')


# =============================================================================
# CHART 21 — Spider / Radar Chart: Congo, Azerbaijan, Chile
# =============================================================================
def chart21_radar():
    df = load_master()
    rr = resource_rich_codes()

    # Use 2017-2019 averages
    snap = df[df['Year'].between(2017, 2019)].groupby('Country Code').mean(numeric_only=True)

    DIMS = {
        'Human capital index':                                              'HCI',
        'Access to electricity (% of population)':                        'Electricity',
        'Rule of law index':                                               'Rule of Law',
        'Economic Complexity Index':                                       'ECI',
        'prod_pc':                                                         'Production p.c.',
        'Political stability — estimate':                                  'Pol. Stability',
        'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'GFCF',
        'Trade (% of GDP)':                                                'Trade',
    }

    rr_snap = snap.loc[[c for c in rr if c in snap.index]]
    norm = {}
    for col in DIMS:
        s = rr_snap[col].dropna() if col in rr_snap.columns else pd.Series(dtype=float)
        vmin, vmax = s.min(), s.max()
        norm[col] = (vmin, vmax)

    def normalise(code, col):
        if col not in snap.columns or code not in snap.index:
            return 0.5
        val = snap.loc[code, col]
        vmin, vmax = norm[col]
        if vmax == vmin:
            return 0.5
        return float(np.clip((val - vmin) / (vmax - vmin), 0, 1))

    dim_labels = list(DIMS.values())
    dim_cols   = list(DIMS.keys())

    fig = go.Figure()
    for code in CASE_CODES:
        vals = [normalise(code, c) for c in dim_cols]
        vals_closed = vals + [vals[0]]
        labels_closed = dim_labels + [dim_labels[0]]
        c_hex = CASE_COLORS[code]
        r_,g_,b_ = int(c_hex[1:3],16), int(c_hex[3:5],16), int(c_hex[5:7],16)
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill='toself',
            fillcolor=f'rgba({r_},{g_},{b_},0.15)',
            line=dict(color=c_hex, width=2.5),
            name=CASE_NAMES[code],
            mode='lines+markers',
            marker=dict(size=5),
        ))

    fig.update_layout(**base_layout(height=480,
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75],
                ticktext=['25%', '50%', '75%'],
                tickfont=dict(size=9),
                gridcolor=GRID, linecolor=GRID,
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color=NAVY),
                gridcolor=GRID, linecolor=GRID,
            ),
        ),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.1, font=dict(size=11)),
        margin=dict(l=60, r=60, t=50, b=70),
    ))
    save(fig, 'chart21_radar')


# =============================================================================
# CHART 22 — ECI Quartile Transitions: Sankey (1995 → 2019)
# =============================================================================
def chart22_sankey():
    df  = load_master()
    cl  = load_clusters()[['Country Code', 'Cluster']].drop_duplicates()
    rr  = resource_rich_codes()

    eci95  = df[(df['Year'] == 1995) & (df['Country Code'].isin(rr))][['Country Code', 'Economic Complexity Index']]
    eci19  = df[(df['Year'] == 2019) & (df['Country Code'].isin(rr))][['Country Code', 'Economic Complexity Index']]
    merged = eci95.merge(eci19, on='Country Code', suffixes=('_1995', '_2019'))
    merged = merged.merge(cl, on='Country Code', how='left')
    merged = merged.dropna(subset=['Economic Complexity Index_1995', 'Economic Complexity Index_2019'])

    # Quartile bins from 1995 distribution
    edges95 = merged['Economic Complexity Index_1995'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    edges19 = merged['Economic Complexity Index_2019'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values

    def assign_q(val, edges):
        for i in range(3, 0, -1):
            if val >= edges[i]:
                return i + 1  # Q4, Q3, Q2
        return 1

    merged['Q95'] = merged['Economic Complexity Index_1995'].apply(lambda v: assign_q(v, edges95))
    merged['Q19'] = merged['Economic Complexity Index_2019'].apply(lambda v: assign_q(v, edges19))

    Q_LABELS = {1: 'Q1 (Low)', 2: 'Q2', 3: 'Q3', 4: 'Q4 (High)'}

    # Node order: Q1–Q4 for 1995 (nodes 0–3), Q1–Q4 for 2019 (nodes 4–7)
    node_labels = [f"1995 {Q_LABELS[q]}" for q in range(1, 5)] + \
                  [f"2019 {Q_LABELS[q]}" for q in range(1, 5)]
    node_colors = ['rgba(74,111,165,0.7)'] * 4 + ['rgba(46,125,74,0.7)'] * 4

    sources, targets, values, link_colors = [], [], [], []

    for (q95, q19, clust), grp in merged.groupby(['Q95', 'Q19', 'Cluster']):
        src = int(q95) - 1
        tgt = int(q19) - 1 + 4
        c   = int(clust) if not pd.isna(clust) else 0
        color_base = CLUSTER_COLORS.get(c, '#999999')
        r, g, b = int(color_base[1:3], 16), int(color_base[3:5], 16), int(color_base[5:7], 16)
        sources.append(src)
        targets.append(tgt)
        values.append(len(grp))
        link_colors.append(f'rgba({r},{g},{b},0.45)')

    fig = go.Figure(go.Sankey(
        node=dict(
            label=node_labels,
            color=node_colors,
            pad=18, thickness=22,
            line=dict(color='white', width=0.5),
            hovertemplate='%{label}: %{value} countries<extra></extra>',
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}: %{value} countries<extra></extra>',
        ),
        arrangement='snap',
    ))

    fig.update_layout(**base_layout(height=440))
    save(fig, 'chart22_sankey')


# =============================================================================
# CHART 23 — Growth Diagnostics Decision Tree (Hausmann-Rodrik-Velasco)
# =============================================================================
def chart23_growth_diagnostics():
    """Static conceptual diagram of the HRV growth diagnostics tree."""
    fig = go.Figure()

    # Node positions (x, y) in figure units [0,1]
    nodes = {
        'root':    (0.50, 0.92, 'Why is growth constrained?',                '#1a2744'),
        'fin':     (0.25, 0.74, 'High cost\nof finance?',                    '#4a6fa5'),
        'ret':     (0.75, 0.74, 'Low returns to\neconomic activity?',        '#4a6fa5'),
        'dom_sav': (0.12, 0.54, 'Low domestic\nsaving?',                    '#6e8fb5'),
        'int_fin': (0.38, 0.54, 'Low access to\ninternational finance?',    '#6e8fb5'),
        'soc_ret': (0.62, 0.54, 'Low social\nreturns?',                     '#6e8fb5'),
        'tax_reg': (0.88, 0.54, 'High taxes /\npoor regulation?',           '#6e8fb5'),
        'gov_sav': (0.06, 0.32, 'Low gov.\nsaving',                         '#2e7d4a'),
        'fin_int': (0.18, 0.32, 'Poor financial\nintermediation',           '#2e7d4a'),
        'macro':   (0.32, 0.32, 'Macro / FX\nrisk',                         '#2e7d4a'),
        'cap_ctrl':(0.44, 0.32, 'Capital\ncontrols',                        '#2e7d4a'),
        'hum_cap': (0.56, 0.32, 'Low human\ncapital',                       '#c23a3a'),
        'infra':   (0.68, 0.32, 'Poor infra-\nstructure',                   '#c23a3a'),
        'tax':     (0.80, 0.32, 'High taxes /\ncorruption',                  '#c23a3a'),
        'prop':    (0.92, 0.32, 'Weak contract\nenforcement',                '#c23a3a'),
    }

    edges = [
        ('root', 'fin'), ('root', 'ret'),
        ('fin', 'dom_sav'), ('fin', 'int_fin'),
        ('ret', 'soc_ret'), ('ret', 'tax_reg'),
        ('dom_sav', 'gov_sav'), ('dom_sav', 'fin_int'),
        ('int_fin', 'macro'), ('int_fin', 'cap_ctrl'),
        ('soc_ret', 'hum_cap'), ('soc_ret', 'infra'),
        ('tax_reg', 'tax'), ('tax_reg', 'prop'),
    ]

    # Draw edges
    for src, tgt in edges:
        x0, y0, *_ = nodes[src]
        x1, y1, *_ = nodes[tgt]
        fig.add_shape(type='line',
                      x0=x0, y0=y0 - 0.04, x1=x1, y1=y1 + 0.04,
                      line=dict(color='#cccccc', width=1.5),
                      xref='paper', yref='paper')

    # Draw nodes
    for key, (x, y, label, color) in nodes.items():
        depth = y
        fontsize = 12 if depth > 0.8 else (11 if depth > 0.6 else 10)
        fig.add_annotation(
            x=x, y=y, xref='paper', yref='paper',
            text=label.replace('\n', '<br>'),
            showarrow=False,
            font=dict(size=fontsize, color='white', family=FONT),
            bgcolor=color,
            bordercolor='white',
            borderwidth=1,
            borderpad=6,
            align='center',
            xanchor='center', yanchor='middle',
        )

    fig.update_layout(**base_layout(height=460,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0.18, 1.04]),
        margin=dict(l=20, r=20, t=30, b=20),
    ))
    save(fig, 'chart23_growth_diagnostics')


# =============================================================================
# CHART 24 — Section 4.7 Summary: Case Study Forecast vs Historical Trajectory
# =============================================================================
def chart24_case_study_forecast():
    df  = load_master()
    fdf = pd.read_csv(os.path.join(NB5, 'ECI_Forecast_2020_2030.csv'))

    fig = make_subplots(rows=1, cols=3,
                        shared_yaxes=True,
                        horizontal_spacing=0.04,
                        subplot_titles=[CASE_NAMES[c] for c in CASE_CODES])

    for col_idx, code in enumerate(CASE_CODES, start=1):
        color = CASE_COLORS[code]

        # Historical
        hist = df[df['Country Code'] == code].sort_values('Year')
        fig.add_trace(go.Scatter(
            x=hist['Year'], y=hist['Economic Complexity Index'],
            mode='lines+markers',
            line=dict(color=color, width=2.5),
            marker=dict(size=4),
            name=CASE_NAMES[code] + ' (hist.)',
            showlegend=(col_idx == 1),
            legendgroup='hist',
        ), row=1, col=col_idx)

        # Forecast
        fore = fdf[fdf['Country Code'] == code].sort_values('Year')
        if not fore.empty:
            ens  = fore['Ensemble'].values
            # CI: ±0.15 (approx 1 RMSE, conservative)
            ci   = 0.15
            fore_years = fore['Year'].values

            r_,g_,b_ = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            fig.add_trace(go.Scatter(
                x=np.concatenate([fore_years, fore_years[::-1]]),
                y=np.concatenate([ens + ci, (ens - ci)[::-1]]),
                fill='toself', fillcolor=f'rgba({r_},{g_},{b_},0.13)',
                line=dict(width=0), showlegend=False, hoverinfo='skip',
            ), row=1, col=col_idx)

            fig.add_trace(go.Scatter(
                x=fore_years, y=ens,
                mode='lines', line=dict(color=color, width=2.5, dash='dash'),
                name=CASE_NAMES[code] + ' (forecast)',
                showlegend=(col_idx == 1),
                legendgroup='fore',
            ), row=1, col=col_idx)

        # Vertical divider at 2019.5
        fig.add_vline(x=2019.5, line=dict(color='#aaaaaa', width=1, dash='dot'),
                      row=1, col=col_idx)
        # Grey band 2020-2030
        fig.add_vrect(x0=2020, x1=2030,
                      fillcolor='rgba(200,200,200,0.10)', line_width=0,
                      row=1, col=col_idx)

    fig.update_xaxes(showgrid=True, gridcolor=GRID, dtick=5)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, title_text='ECI', col=1)

    fig.update_layout(**base_layout(height=420,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.18, font=dict(size=10)),
        hovermode='x unified',
    ))
    save(fig, 'chart24_case_study_forecast')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("Running charts 14–24...")
    print("Chart 14: Case study comparison table")
    chart14_case_study_table()
    print("Chart 15: Congo small multiples")
    chart15_congo_small_multiples()
    print("Chart 16: ECI trajectories — Congo vs peers")
    chart16_eci_congo_peers()
    print("Chart 17: ECI trajectories — Azerbaijan vs peers")
    chart17_eci_aze_peers()
    print("Chart 18: VIF chart")
    chart18_vif()
    print("Chart 19: Bootstrap robustness")
    chart19_bootstrap_robustness()
    print("Chart 20: Consensus importance heatmap")
    chart20_consensus_heatmap()
    print("Chart 21: Radar chart")
    chart21_radar()
    print("Chart 22: Sankey — ECI quartile transitions")
    chart22_sankey()
    print("Chart 23: Growth diagnostics tree")
    chart23_growth_diagnostics()
    print("Chart 24: Case study forecast summary")
    chart24_case_study_forecast()
    print("\nAll done. Outputs in Final/improved_14_24/")
