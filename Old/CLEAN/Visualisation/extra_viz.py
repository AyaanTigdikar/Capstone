"""
extra_viz.py
============
Charts NOT in the main report body: diagnostics, appendix figures,
robustness checks, alternative cluster maps, NB2 missingness plots,
and the PDF export utility.

Run from project root (FINAL CODE RECAP/):
    python3 scripts/extra_viz.py

Contents
--------
  01   — Sample choropleth (Appendix Figure A1)
  04b  — Cluster world map, 2019 snapshot
  04c  — Cluster world map, aggregated (1995/1999/2005)
  04d  — 4-feature cluster map (Oil/Gas/Coal/Minerals, k=5)
  11b  — ECI forecast heatmap, all countries
  13   — Alias of 11b (presentation numbering)
  28   — VIF multicollinearity (Appendix)
  29   — Bootstrap R2 stability (Appendix)
  31   — ML prediction intervals (Appendix)
  32   — Country data coverage scatter (Appendix)
  33   — Full variable correlation matrix (Appendix)
  34   — HRV growth diagnostics (placeholder)

  NB2 missingness diagnostics (matplotlib):
    plot1  — Variable missingness bar
    plot2  — Country missingness bar (top 30)
    plot3  — Binary missingness matrix (top countries)
    plot5  — Heatmap problem areas
    plot6  — Missingness over time
    plot7  — Interactive country profile (Plotly)
    Correlation matrix (static + interactive)

  PDF export utility (requires playwright + Pillow)
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
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from viz_utils import (
    PALETTE, FONT, BG, NAVY, GRID, WRITE_CONFIG,
    base_layout, save,
    load_master, load_master_wide, load_clusters, load_nr, load_nb5, load_bootstrap,
    INCLUDE_LIST, resource_rich_codes, build_sample, shorten_feat,
    CLUSTER_LABELS, CLUSTER_COLORS, LABEL_TO_COLOR,
    analyze_country_missingness, analyze_variable_missingness,
)

# ── 2. Output directories ────────────────────────────────────────────────────
_VIZ_BASE = '/Users/leoss/Desktop/GitHub/Capstone/CLEAN/Visualisation'
OUT_CLUS = os.path.join(_VIZ_BASE, 'unc_charts', 'clusters')
OUT_ML   = os.path.join(_VIZ_BASE, 'unc_charts', 'ml')
OUT_MISC = os.path.join(_VIZ_BASE, 'unc_charts', 'misc')
OUT_NB2  = os.path.join(_VIZ_BASE, 'unc_charts', 'nb2')
for _d in [OUT_CLUS, OUT_ML, OUT_MISC, OUT_NB2]:
    os.makedirs(_d, exist_ok=True)

LABEL_EXCL = ['L1_ECI', 'Inflation_roll5', 'RealRate_roll5', 'Resource_HHI']

# ── 3. Clustering infrastructure (shared with generate_report_charts.py) ─────
_LABEL_COLORS_4K = {
    'Petrostates':       '#d4853b',
    'Oil Exporters':     '#4a6fa5',
    'Major Producers':   '#2e7d4a',
    'Limited Resources': '#c23a3a',
}
_LABEL_COLORS_4F = {
    'Petrostates':       '#d4853b',
    'Oil Exporters':     '#c23a3a',
    'Oil & Minerals':    '#7a5c9e',
    'Mineral Exporters': '#2e7d4a',
    'Low Resource':      '#4a6fa5',
}

# Clustering functions are duplicated here to keep this file self-contained.
# Canonical source is generate_report_charts.py; keep both in sync.
from generate_report_charts import run_clustering, create_cluster_map  # noqa: E402

# If you prefer zero cross-file dependencies, copy run_clustering() and
# create_cluster_map() from generate_report_charts.py into this file and
# remove the import above.

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
        'Country': pivot['Country'], 'Country Code': pivot['Country Code'],
        'Year': pivot['Year'], 'PC1': Xp[:, 0], 'PC2': Xp[:, 1], 'Cluster': labels,
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


# ── Pre-run clustering ───────────────────────────────────────────────────────
print('Loading NaturalResource.csv and running clustering pipeline...')
nr_full   = load_nr()
nr_sample = nr_full[nr_full['Country Code'].isin(INCLUDE_LIST)]

pca_1995, _, _ = run_clustering(nr_sample, year_filter=1995)
pca_2019, _, _ = run_clustering(nr_sample, year_filter=2019)
pca_agg,  _, _ = run_clustering(nr_sample, agg_years=[1995, 1999, 2005])
print('  Clustering done.\n')


# =============================================================================
#
#   APPENDIX / DIAGNOSTIC CHARTS
#
# =============================================================================

# ── CHART 01 — Sample choropleth (Appendix Figure A1) ────────────────────────
print('=== CHART 01 (Appendix: sample map) ===')

master01 = load_master()
cl95     = load_clusters('1995')[['Country Code', 'Cluster', 'ClusterLabels']].drop_duplicates()

wb_rents = (master01[master01['Year'] == 1995]
            [['Country Code', 'Country Name', 'Total natural resources rents (% of GDP)']]
            .drop_duplicates('Country Code')
            .rename(columns={'Total natural resources rents (% of GDP)': 'NR_rents_pct'}))

map_df = cl95.merge(wb_rents, on='Country Code', how='left')

fig01 = go.Figure()
for lbl in sorted(map_df['ClusterLabels'].dropna().unique()):
    sub   = map_df[map_df['ClusterLabels'] == lbl]
    color = LABEL_TO_COLOR.get(lbl, '#888888')
    fig01.add_trace(go.Choropleth(
        locations=sub['Country Code'], z=sub['NR_rents_pct'].fillna(0),
        colorscale=[[0, color], [1, color]], showscale=False, showlegend=True, name=lbl,
        customdata=sub[['Country Name', 'ClusterLabels', 'NR_rents_pct']].values,
        hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>'
                      'NR rents: %{customdata[2]:.1f}% of GDP (1995)<extra></extra>',
        marker=dict(line=dict(color='white', width=0.5)),
    ))
fig01.update_geos(projection_type='natural earth', showcountries=True, countrycolor='#d0d0d0',
                  showcoastlines=True, coastlinecolor='#d0d0d0', showland=True, landcolor='#f5f5f5',
                  showocean=True, oceancolor='#dde8f0', showframe=False)
fig01.update_layout(
    margin=dict(l=0, r=0, t=50, b=80),
    legend=dict(orientation='h', x=0.5, y=-0.06, xanchor='center', yanchor='top',
                font=dict(size=11, family=FONT), bgcolor='rgba(250,250,250,0.9)',
                bordercolor=GRID, borderwidth=1),
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family=FONT, color=NAVY),
)
save(fig01, '01_sample__54_resource_dependent_countries_map', OUT_MISC, w=1200, h=540)


# ── CHART 04b — Cluster world map, 2019 ──────────────────────────────────────
print('\n=== CHART 04b (Cluster map 2019) ===')
nr_2019_sub = nr_sample[nr_sample['Year'] == 2019]
cnames_2019 = dict(zip(pca_2019['Cluster'], pca_2019['ClusterLabels']))
fig04b = create_cluster_map(pca_2019, nr_2019_sub, cluster_names_map=cnames_2019)
fig04b.update_layout(height=520)
save(fig04b, '04b_cluster__world_map_2019_resource_profiles', OUT_CLUS, w=1200, h=520)


# ── CHART 04c — Cluster world map, aggregated ────────────────────────────────
print('\n=== CHART 04c (Cluster map aggregated) ===')
nr_agg_sub = nr_sample[nr_sample['Year'].isin([1995, 1999, 2005])]
cnames_agg = dict(zip(pca_agg['Cluster'], pca_agg['ClusterLabels']))
fig04c = create_cluster_map(pca_agg, nr_agg_sub, cluster_names_map=cnames_agg)
fig04c.update_layout(height=520)
save(fig04c, '04c_cluster__world_map_agg_resource_profiles', OUT_CLUS, w=1200, h=520)


# ── CHART 04d — 4-feature cluster map (k=5) ──────────────────────────────────
print('\n=== CHART 04d (4-feature cluster map) ===')
nr_1995_sub = nr_sample[nr_sample['Year'] == 1995]
pca_4f   = run_clustering_4feat(nr_sample, year_filter=1995)
cnames_4f = dict(zip(pca_4f['Cluster'], pca_4f['ClusterLabels']))
fig04d = create_cluster_map(pca_4f, nr_1995_sub, cluster_names_map=cnames_4f,
                            label_colors=_LABEL_COLORS_4F)
fig04d.update_layout(height=520)
save(fig04d, '04d_cluster__world_map_4feat_oil_gas_coal_minerals', OUT_CLUS, w=1200, h=520)


# ── CHART 11b — ECI forecast heatmap ─────────────────────────────────────────
print('\n=== CHART 11b (ECI forecast heatmap) ===')

_fc_path = os.path.join('Graphics', 'NB5', 'ECI_Forecast_2020_2030.csv')
if os.path.exists(_fc_path):
    fc_hm = pd.read_csv(_fc_path)
    master_hm = load_master()
    hist_hm = (master_hm[master_hm['Country Code'].isin(INCLUDE_LIST)]
               [['Country Code', 'Country Name', 'Year', 'Economic Complexity Index']].dropna())

    fc_long = fc_hm[fc_hm['Country Code'].isin(INCLUDE_LIST)][['Country Code', 'Year', 'Ensemble']].copy()
    fc_long = fc_long.rename(columns={'Ensemble': 'ECI'})
    hist_long = hist_hm.rename(columns={'Economic Complexity Index': 'ECI'})[['Country Code', 'Year', 'ECI']]

    combined = pd.concat([hist_long, fc_long], ignore_index=True).drop_duplicates(subset=['Country Code', 'Year'])
    pivot_hm = combined.pivot(index='Country Code', columns='Year', values='ECI')
    pivot_hm = pivot_hm.reindex(sorted(pivot_hm.index))

    sort_col = 2019 if 2019 in pivot_hm.columns else pivot_hm.columns.max()
    pivot_hm = pivot_hm.loc[pivot_hm[sort_col].sort_values().index]

    forecast_start = fc_hm['Year'].min() if 'Year' in fc_hm.columns else 2020

    fig11b = go.Figure(go.Heatmap(
        z=pivot_hm.values, x=[str(c) for c in pivot_hm.columns], y=pivot_hm.index.tolist(),
        colorscale='RdBu', zmid=0, colorbar=dict(title='ECI', thickness=16, len=0.9),
        hovertemplate='%{y} | %{x}: %{z:.3f}<extra></extra>',
    ))

    fc_idx = list(pivot_hm.columns).index(forecast_start) if forecast_start in pivot_hm.columns else None
    if fc_idx is not None:
        fig11b.add_shape(type='line', x0=fc_idx - 0.5, x1=fc_idx - 0.5,
                         y0=-0.5, y1=len(pivot_hm) - 0.5,
                         line=dict(color='#333', width=2, dash='dot'), xref='x', yref='y')
        fig11b.add_annotation(x=fc_idx - 0.5, y=len(pivot_hm) + 0.2, xref='x', yref='y',
                              text='<b>\u2190 Historical | Forecast \u2192</b>', showarrow=False,
                              font=dict(size=10, color='#444'), xanchor='center')

    h = max(500, len(pivot_hm) * 12)
    fig11b.update_layout(**base_layout(
        height=h, margin=dict(l=80, r=100, t=60, b=80),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
        yaxis=dict(tickfont=dict(size=9), showgrid=False),
    ))
    save(fig11b, '11b_ml__eci_forecast_heatmap_all_countries', OUT_ML, w=1200, h=h)
    # Chart 13 is an alias
    save(fig11b, '13_ml__eci_forecast_heatmap_historical_and_projected', OUT_ML, w=1200, h=h)
else:
    print('  SKIPPED: ECI_Forecast_2020_2030.csv not found')


# ── CHART 28 — VIF (Appendix) ────────────────────────────────────────────────
print('\n=== CHART 28 (VIF) ===')

_vif_path = os.path.join('Graphics', 'NB5', 'vif_table.csv')
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
            marker=dict(color=vif_df['Color'], opacity=0.85, line=dict(color='white', width=0.5)),
            hovertemplate='%{y}: VIF = %{x:.2f}<extra></extra>',
        ))
        fig28.add_vline(x=5,  line=dict(color=PALETTE['orange'], width=1.5, dash='dash'))
        fig28.add_vline(x=10, line=dict(color=PALETTE['red'],    width=1.5, dash='dash'))
        fig28.update_layout(**base_layout(
            height=500, margin=dict(l=200, r=80, t=60, b=60),
            xaxis=dict(title='Variance Inflation Factor (VIF)', gridcolor=GRID, gridwidth=0.5),
            yaxis=dict(tickfont=dict(size=11)),
        ))
        save(fig28, '28_diag__vif_multicollinearity', OUT_MISC, w=1100, h=500)
else:
    print('  SKIPPED: vif_table.csv not found (install statsmodels to compute inline)')


# ── CHART 29 — Bootstrap R2 stability (Appendix) ─────────────────────────────
print('\n=== CHART 29 (Bootstrap R2) ===')

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
                line=dict(color=colors29[i % len(colors29)]), boxmean='sd',
                hovertemplate=f'{model_name}<br>R\u00b2: %{{y:.3f}}<extra></extra>',
            ))
        fig29.update_layout(**base_layout(
            height=500, margin=dict(l=80, r=60, t=60, b=80),
            xaxis=dict(title='Model', gridcolor=GRID),
            yaxis=dict(title='Bootstrap Test R\u00b2', gridcolor=GRID, gridwidth=0.5),
        ))
        save(fig29, '29_diag__bootstrap_r2_stability', OUT_MISC, w=1100, h=500)
else:
    print('  SKIPPED: nb5_boot_metrics.csv not found')


# ── CHART 31 — ML prediction intervals (Appendix) ────────────────────────────
print('\n=== CHART 31 (ML prediction intervals) ===')

_test_pred_path = os.path.join('Graphics', 'NB5', 'test_predictions.csv')
if os.path.exists(_test_pred_path):
    tp = pd.read_csv(_test_pred_path)
    if 'Actual_ECI' in tp.columns and 'Predicted_ECI' in tp.columns:
        country_stats = (tp.groupby('Country Code').agg(
            Actual_mean=('Actual_ECI', 'mean'),
            Actual_std=('Actual_ECI', 'std'),
            Predicted_mean=('Predicted_ECI', 'mean'),
            Country_Name=('Country Name', 'first'),
        ).reset_index().sort_values('Actual_mean'))
        country_stats['In_band'] = (
            (country_stats['Predicted_mean'] >= country_stats['Actual_mean'] - country_stats['Actual_std']) &
            (country_stats['Predicted_mean'] <= country_stats['Actual_mean'] + country_stats['Actual_std'])
        )
        country_stats = country_stats.reset_index(drop=True)

        fig31 = go.Figure()
        fig31.add_trace(go.Scatter(
            x=list(range(len(country_stats))),
            y=country_stats['Actual_mean'] + country_stats['Actual_std'],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))
        fig31.add_trace(go.Scatter(
            x=list(range(len(country_stats))),
            y=country_stats['Actual_mean'] - country_stats['Actual_std'],
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(74,111,165,0.12)', showlegend=True,
            name='\u00b11 SD band', hoverinfo='skip',
        ))
        fig31.add_trace(go.Scatter(
            x=list(range(len(country_stats))),
            y=country_stats['Actual_mean'],
            mode='lines', line=dict(color=PALETTE['blue'], width=2), name='Mean Actual ECI',
        ))
        for in_band, color, sym, lbl in [
            (True,  PALETTE['green'], 'circle',  'Predicted \u2248 Actual (within \u00b11 SD)'),
            (False, PALETTE['red'],   'diamond', 'Predicted outside \u00b11 SD'),
        ]:
            mask = country_stats['In_band'] == in_band
            sub  = country_stats[mask]
            fig31.add_trace(go.Scatter(
                x=sub.index.tolist(), y=sub['Predicted_mean'],
                mode='markers' + ('+text' if not in_band else ''),
                marker=dict(color=color, size=8 if in_band else 10, symbol=sym, opacity=0.85),
                name=lbl,
                customdata=sub[['Country Code', 'Country_Name']].values,
                hovertemplate='<b>%{customdata[1]}</b> (%{customdata[0]})<br>'
                              'Avg Actual: %{text}<br>Avg Predicted: %{y:.3f}<extra></extra>',
                text=[f'{v:.3f}' for v in sub['Actual_mean']],
            ))
        fig31.update_layout(**base_layout(
            height=500,
            xaxis=dict(title='Countries (sorted by mean actual ECI)',
                       tickvals=list(range(len(country_stats))),
                       ticktext=country_stats['Country Code'].tolist(),
                       tickangle=-60, tickfont=dict(size=8), gridcolor=GRID),
            yaxis=dict(title='ECI (test set mean)', gridcolor=GRID, gridwidth=0.5),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=10)),
        ))
        save(fig31, '31_diag__ml_prediction_intervals', OUT_MISC, w=1200, h=500)
else:
    print('  SKIPPED: test_predictions.csv not found')


# ── CHART 32 — Country data coverage scatter (Appendix) ──────────────────────
print('\n=== CHART 32 (Country data coverage) ===')

raw_wide = load_master_wide()
sample_wide = raw_wide[raw_wide['Country Code'].isin(INCLUDE_LIST)].copy()
country_missing = analyze_country_missingness(sample_wide)

fig32 = go.Figure()
for above in [True, False]:
    mask  = (country_missing['% Missing'] >= 20.0) if above else (country_missing['% Missing'] < 20.0)
    sub   = country_missing[mask]
    color = PALETTE['red'] if above else PALETTE['blue']
    size  = 12 if above else 8
    mode  = 'markers+text' if above else 'markers'
    fig32.add_trace(go.Scatter(
        x=sub['Vars with Data'], y=sub['% Missing'], mode=mode,
        text=sub['Code'] if above else None, textposition='top center',
        textfont=dict(size=9, color=PALETTE['red']),
        marker=dict(color=color, size=size, opacity=0.75 if above else 0.55,
                    line=dict(color='white', width=0.8)),
        name=f'>= 20% missing' if above else f'< 20% missing',
        customdata=sub[['Code', 'Country', 'Complete Vars', 'Years Covered', 'Rows']].values,
        hovertemplate='<b>%{customdata[1]}</b> (%{customdata[0]})<br>'
                      'Vars with data: %{x}<br>% Missing: %{y:.1f}%<br>'
                      'Complete vars: %{customdata[2]}<br>Years covered: %{customdata[3]}<extra></extra>',
    ))
med_vars    = country_missing['Vars with Data'].median()
med_missing = country_missing['% Missing'].median()
fig32.add_hline(y=med_missing, line_dash='dash', line_color='#aaa', opacity=0.6,
                annotation_text=f'Median {med_missing:.1f}%', annotation_position='right')
fig32.add_vline(x=med_vars, line_dash='dash', line_color='#aaa', opacity=0.6,
                annotation_text=f'Median {med_vars:.0f} vars', annotation_position='top')
fig32.update_layout(**base_layout(
    height=520, xaxis=dict(title='Variables with Any Data', gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(title='% Missing Data Overall', gridcolor=GRID, gridwidth=0.5),
    legend=dict(font=dict(size=10), bgcolor='rgba(255,255,255,0.9)', bordercolor=GRID, borderwidth=1),
))
save(fig32, '32_diag__country_data_coverage_scatter', OUT_MISC, w=1100, h=520)


# ── CHART 33 — Full variable correlation matrix (Appendix) ───────────────────
print('\n=== CHART 33 (Full correlation matrix) ===')

master33 = load_master()
df33     = build_sample(master33)

ID_COLS = {'Country Code', 'Country Name', 'Year'}
data_cols33 = [c for c in df33.columns if c not in ID_COLS
               and df33[c].dtype in [np.float64, np.int64, float, int]]
data_cols33 = [c for c in data_cols33 if df33[c].notna().sum() > 100]

corr33 = df33[data_cols33].corr().round(2)
short_labels33 = [shorten_feat(c) for c in corr33.columns]

fig33 = go.Figure(go.Heatmap(
    z=corr33.values, x=short_labels33, y=short_labels33,
    colorscale=[[0.0, PALETTE['red']], [0.5, '#ffffff'], [1.0, PALETTE['blue']]],
    zmid=0, zmin=-1, zmax=1,
    hovertemplate='%{x} \u00d7 %{y}: %{z:.2f}<extra></extra>',
    colorbar=dict(title='r', thickness=14, len=0.9, tickfont=dict(size=10)),
))
n33 = len(short_labels33)
h33 = max(600, n33 * 22)
w33 = max(700, n33 * 22)
fig33.update_layout(**base_layout(
    height=h33, margin=dict(l=180, r=80, t=40, b=200),
    xaxis=dict(tickangle=-55, tickfont=dict(size=8), showgrid=False),
    yaxis=dict(tickfont=dict(size=8), showgrid=False),
))
save(fig33, '33_diag__full_variable_correlation_matrix', OUT_MISC, w=w33, h=h33)


# ── CHART 34 — HRV growth diagnostics (placeholder) ──────────────────────────
print('\n=== CHART 34 (HRV diagnostics) ===')

_hrv_candidates = [
    os.path.join('Graphics', 'NB5', 'hrv_diagnostics.csv'),
    os.path.join('intermediary', 'hrv_growth.csv'),
]
_hrv_path = next((p for p in _hrv_candidates if os.path.exists(p)), None)

if _hrv_path:
    hrv = pd.read_csv(_hrv_path)
    num_cols = [c for c in hrv.columns if hrv[c].dtype in [np.float64, float]
                and c not in {'Year', 'year'}]
    year_col = next((c for c in ['Year', 'year'] if c in hrv.columns), None)

    if year_col and num_cols:
        fig34 = go.Figure()
        for col in num_cols[:6]:
            fig34.add_trace(go.Scatter(
                x=hrv[year_col], y=hrv[col], mode='lines+markers', name=shorten_feat(col),
                line=dict(width=2),
                hovertemplate=f'{shorten_feat(col)}: %{{y:.3f}}<extra></extra>',
            ))
        fig34.update_layout(**base_layout(
            height=480, xaxis=dict(title='Year', gridcolor=GRID),
            yaxis=dict(title='Value', gridcolor=GRID, gridwidth=0.5),
        ))
        save(fig34, '34_diag__hrv_growth_diagnostics', OUT_MISC, w=1100, h=480)
    else:
        print('  SKIPPED: HRV file lacks year or numeric columns')
else:
    fig34 = go.Figure()
    fig34.add_trace(go.Scatter(x=[0], y=[0], mode='text',
                               text=['HRV Growth Diagnostics \u2014 data not yet available'],
                               textfont=dict(size=14, color='#888')))
    fig34.update_layout(**base_layout(height=300, xaxis=dict(visible=False), yaxis=dict(visible=False),
                                      margin=dict(l=60, r=60, t=60, b=60)))
    save(fig34, '34_diag__hrv_growth_diagnostics_placeholder', OUT_MISC, w=900, h=300)
    print('  Saved placeholder (no HRV source data found)')


# =============================================================================
#
#   NB2 MISSINGNESS DIAGNOSTICS (matplotlib)
#   Originally in nb2_charts.py. Requires: matplotlib, seaborn
#
# =============================================================================
print('\n' + '=' * 60)
print('NB2 MISSINGNESS DIAGNOSTICS')
print('=' * 60)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import LinearSegmentedColormap

    # Load pre-imputation wide data (NB2 style: NR rents > 1%, include Gulf)
    nb2_master = pd.read_csv('intermediary/master_data_wide.csv')
    if 'Unnamed: 0' in nb2_master.columns:
        nb2_master.drop(columns='Unnamed: 0', inplace=True)

    RENT_COL = 'Total natural resources rents (% of GDP)'
    FOREST_COL = 'Forest rents (% of GDP)'
    COAL_COL = 'Coal rents (% of GDP)'

    nb2_master['Extractive_NR_Rents'] = (
        nb2_master[RENT_COL].fillna(0) - nb2_master[FOREST_COL].fillna(0) - nb2_master[COAL_COL].fillna(0)
    ).clip(lower=0)

    rents_1995 = nb2_master[nb2_master['Year'] == 1995].dropna(subset=[RENT_COL])
    resource_countries = set(rents_1995[rents_1995['Extractive_NR_Rents'] >= 1.0]['Country Code'].tolist())
    GULF = {'ARE', 'BHR', 'KWT', 'OMN', 'QAT', 'SAU', 'IRQ', 'IRN', 'YEM'}
    resource_countries = resource_countries | GULF

    not_countries = ['HKG', 'MAC', 'PRI', 'VIR', 'GUM', 'ASM', 'CYM', 'BMU',
                     'GRL', 'MAF', 'SXM', 'CUW', 'ABW', 'FRO', 'MNP', 'PYF']

    cmaster = nb2_master[
        (nb2_master['Country Code'].isin(resource_countries)) &
        (~nb2_master['Country Code'].isin(not_countries))
    ]
    print(f'NB2 sample: {cmaster["Country Code"].nunique()} countries, {cmaster.shape[0]:,} rows')

    var_missing = analyze_variable_missingness(cmaster)
    country_missing_nb2 = analyze_country_missingness(cmaster)

    # Plot 1: Variable missingness
    fig1, ax1 = plt.subplots(figsize=(12, 16))
    var_sorted = var_missing.sort_values('% Missing', ascending=True)
    colors = ['#d73027' if x > 40 else '#fc8d59' if x > 20 else '#91cf60' for x in var_sorted['% Missing']]
    ax1.barh(range(len(var_sorted)), var_sorted['% Missing'], color=colors, height=0.7)
    ax1.set_xlabel('% Missing', fontsize=12)
    ax1.set_title('Missingness by Variable', fontsize=14, fontweight='bold', pad=20)
    ax1.set_yticks(range(len(var_sorted)))
    ax1.set_yticklabels(var_sorted['Variable'], fontsize=9)
    ax1.axvline(x=20, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=40, color='red', linestyle='--', alpha=0.7, linewidth=2)
    for i, (val, _) in enumerate(zip(var_sorted['% Missing'], var_sorted['Variable'])):
        ax1.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=8)
    legend_elements = [Patch(facecolor='#91cf60', label='< 20% missing'),
                       Patch(facecolor='#fc8d59', label='20-40% missing'),
                       Patch(facecolor='#d73027', label='> 40% missing')]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax1.set_xlim(0, max(var_sorted['% Missing']) + 10)
    plt.tight_layout()
    plt.savefig(f'{OUT_NB2}/plot1_variable_missingness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: plot1_variable_missingness.png')

    # Plot 6: Missingness over time
    data_cols_nb2 = [c for c in cmaster.columns if c not in ['Country Code', 'Country Name', 'Year']]
    yearly_missing = cmaster.groupby('Year')[data_cols_nb2].apply(lambda x: x.isna().mean() * 100).mean(axis=1).dropna()

    fig6, ax6 = plt.subplots(figsize=(14, 8))
    ax6.plot(yearly_missing.index, yearly_missing.values, marker='o', linewidth=2, markersize=8, color='#4575b4')
    ax6.fill_between(yearly_missing.index, yearly_missing.values, alpha=0.3, color='#4575b4')
    ax6.set_xlabel('Year', fontsize=12)
    ax6.set_ylabel('Average % Missing Across All Variables', fontsize=12)
    ax6.set_title('Missingness Over Time', fontsize=14, fontweight='bold', pad=20)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_NB2}/plot6_missingness_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: plot6_missingness_over_time.png')

    print('  NB2 matplotlib diagnostics complete.')

except ImportError as e:
    print(f'  SKIPPED NB2 matplotlib plots: {e}')
except FileNotFoundError as e:
    print(f'  SKIPPED NB2 plots (data not found): {e}')


# =============================================================================
#
#   PDF EXPORT UTILITY
#   Requires: playwright, Pillow
#   Run separately after all charts are generated.
#
# =============================================================================

def export_charts_to_pdf(charts_dir=None, out_pdf=None,
                         scale_factor=4, viewport_w=1400, viewport_h=800):
    """Screenshot all HTML charts at high DPI and combine into a single PDF.

    Requires:
        pip install playwright Pillow
        playwright install chromium
    """
    if charts_dir is None:
        charts_dir = os.path.join(_VIZ_BASE, 'charts')
    if out_pdf is None:
        out_pdf = os.path.join(_VIZ_BASE, 'charts_overview.pdf')
    try:
        from pathlib import Path
        from PIL import Image, ImageDraw
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('  PDF export requires: pip install playwright Pillow && playwright install chromium')
        return

    charts_path = Path(charts_dir)
    html_files = sorted(charts_path.rglob('*.html'))
    if not html_files:
        print(f'  No HTML files found in {charts_dir}')
        return

    shot_dir = Path(os.path.join(_VIZ_BASE, '_screenshot_tmp'))
    shot_dir.mkdir(exist_ok=True)

    print(f'\nExporting {len(html_files)} charts to PDF (scale={scale_factor}x)...')

    shot_paths = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={'width': viewport_w, 'height': viewport_h},
            device_scale_factor=scale_factor,
        )
        page = ctx.new_page()

        for i, html in enumerate(html_files, 1):
            url  = html.resolve().as_uri()
            out  = shot_dir / f'{html.stem}.png'
            print(f'  [{i:02d}/{len(html_files)}] {html.stem}')

            page.goto(url, wait_until='networkidle', timeout=45_000)
            page.wait_for_timeout(1000)

            chart_h = page.evaluate("""() => {
                const el = document.querySelector('.js-plotly-plot')
                         || document.querySelector('.plotly-graph-div');
                return el ? Math.ceil(el.getBoundingClientRect().height) : document.body.scrollHeight;
            }""")
            snap_h = max(chart_h, 200)
            page.set_viewport_size({'width': viewport_w, 'height': snap_h})
            page.wait_for_timeout(300)
            page.screenshot(path=str(out), clip={'x': 0, 'y': 0, 'width': viewport_w, 'height': snap_h})
            shot_paths.append(out)

        browser.close()

    # Compose PDF
    RENDER_DPI = 96 * scale_factor
    A4_W_PX = int(297 / 25.4 * RENDER_DPI)
    A4_H_PX = int(210 / 25.4 * RENDER_DPI)

    images = []
    for sp in shot_paths:
        img = Image.open(sp).convert('RGB')
        scale_fit = min(A4_W_PX / img.width, A4_H_PX / img.height, 1.0)
        if scale_fit < 1.0:
            img = img.resize((int(img.width * scale_fit), int(img.height * scale_fit)), Image.LANCZOS)
        canvas = Image.new('RGB', (A4_W_PX, A4_H_PX), 'white')
        canvas.paste(img, ((A4_W_PX - img.width) // 2, (A4_H_PX - img.height) // 2))
        images.append(canvas)

    images[0].save(str(out_pdf), save_all=True, append_images=images[1:], resolution=RENDER_DPI)
    print(f'  Saved: {out_pdf} ({len(images)} pages)')

    import shutil
    shutil.rmtree(shot_dir)


# Uncomment to run PDF export:
# export_charts_to_pdf()


# =============================================================================
print('\n' + '=' * 60)
print('extra_viz.py complete.')
print('=' * 60)
