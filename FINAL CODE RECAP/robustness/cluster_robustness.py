"""
cluster_robustness.py
---------------------
Three sensitivity checks on the k=5 PCA-KMeans clustering.

  rob_A  —  1995, minerals aggregated into one bucket (Oil / Gas / Coal / Minerals)
  rob_B  —  1995, PCA(3) instead of PCA(2); countries clustered in 3-D space
  rob_C  —  2019, standard full-feature pipeline (same as main analysis)

Each check produces one choropleth world map saved to:
  Final/charts/cluster_robustness/

Run from project root:
    python3 robustness/cluster_robustness.py
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ── locate project root ──────────────────────────────────────────────────────
def _find_root(marker='intermediary'):
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, marker)):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"Cannot find project root (looking for '{marker}/').")

ROOT = _find_root()
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from viz_utils import (
    load_nr, INCLUDE_LIST, FONT, BG, NAVY, GRID, WRITE_CONFIG, save,
)

OUT = os.path.join(ROOT, 'Final', 'charts', 'cluster_robustness')
os.makedirs(OUT, exist_ok=True)

HYDROCARBONS = {'Oil', 'Natural Gas', 'Coal'}

# ── colour palette (shared across all three maps) ────────────────────────────
_COLORS = {
    'Petrostates':           '#d4853b',
    'Oil Exporters':         '#4a6fa5',
    'Major Producers':       '#2e7d4a',   # k=4 only
    'Diversified Producers': '#2e7d4a',
    'Forestry Intensive':    '#c23a3a',
    'Mining Exporters':      '#7a5c9e',
    'Diversified Producers': '#d4a017',
    'Oil & Minerals':        '#3a8fa5',
}

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ── shared helpers ────────────────────────────────────────────────────────────
def _prep_pivot(nr_data, year, merge_minerals=False):
    """Filter to year, pivot to country × resource, normalise per capita."""
    df = nr_data[nr_data['Year'] == year].copy()
    if merge_minerals:
        df['Resource'] = df['Resource'].apply(
            lambda r: r if r in HYDROCARBONS else 'Minerals'
        )
        df = (df.groupby(['Country', 'Country Code', 'Year', 'Population', 'Resource'],
                         as_index=False)['Production_TotalValue'].sum())

    pivot = df.pivot_table(
        index=['Country', 'Country Code', 'Year', 'Population'],
        columns='Resource', values='Production_TotalValue',
    ).reset_index().fillna(0)

    feat_cols = [c for c in pivot.columns
                 if c not in ['Country', 'Country Code', 'Year', 'Population']]
    pivot[feat_cols] = pivot[feat_cols].div(pivot['Population'], axis=0)
    return pivot, feat_cols


def _run(pivot, feat_cols, n_pca=2, n_clusters=5, random_state=42):
    """Log-transform → PCA → KMeans. Returns pca_df and PCA model."""
    X = np.log1p(pivot[feat_cols].fillna(0))
    pca  = PCA(n_components=n_pca)
    Xp   = pca.fit_transform(X)
    km   = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    lbls = km.fit_predict(Xp)

    pca_df = pivot[['Country', 'Country Code']].copy()
    for i in range(n_pca):
        pca_df[f'PC{i+1}'] = Xp[:, i]
    pca_df['Cluster'] = lbls
    return pca_df, pca, km.cluster_centers_


def _assign_labels(pca_df, centroids, label_override=None):
    """Rank-based label assignment using PC1 / PC2 centroids."""
    if label_override is not None:
        pca_df['ClusterLabel'] = pca_df['Cluster'].map(label_override)
        return pca_df

    n_k = len(centroids)
    pc1_rank = list(np.argsort(-centroids[:, 0]))
    pc2_rank = list(np.argsort(-centroids[:, 1]))

    lmap, done = {}, set()
    lmap[pc1_rank[0]] = 'Petrostates';        done.add(pc1_rank[0])
    mid = next(c for c in pc2_rank if c not in done)
    lmap[mid] = 'Mining Exporters';           done.add(mid)
    remaining = [c for c in pc1_rank if c not in done]

    names = ['Oil Exporters', 'Major Producers', 'Forestry Intensive',
             'Diversified Producers', 'Oil & Minerals']
    for i, cid in enumerate(remaining):
        lmap[cid] = names[i] if i < len(names) else f'Cluster {cid}'

    pca_df['ClusterLabel'] = pca_df['Cluster'].map(lmap)
    return pca_df


def _world_map(pca_df, title, label_col='ClusterLabel'):
    """Simple choropleth coloured by cluster label."""
    fig = go.Figure()
    for lbl in sorted(pca_df[label_col].dropna().unique()):
        sub   = pca_df[pca_df[label_col] == lbl]
        color = _COLORS.get(lbl, '#aaa')
        fig.add_trace(go.Choropleth(
            locations=sub['Country Code'],
            z=[1] * len(sub),
            colorscale=[[0, color], [1, color]],
            showscale=False,
            showlegend=True,
            name=lbl,
            hovertemplate='<b>%{location}</b><br>' + lbl + '<extra></extra>',
            marker=dict(line=dict(color='white', width=0.6)),
        ))

    fig.update_geos(
        projection_type='natural earth',
        showcountries=True,  countrycolor='#ccc',
        showcoastlines=True, coastlinecolor='#ccc',
        showland=True,       landcolor='#f0f0f0',
        showocean=True,      oceancolor='#dde8f0',
        showframe=False,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT, size=13, color=NAVY), x=0.5),
        margin=dict(l=0, r=0, t=45, b=70),
        legend=dict(
            orientation='h', x=0.5, y=-0.08, xanchor='center', yanchor='top',
            font=dict(size=11, family=FONT),
            bgcolor='rgba(250,250,250,0.9)', bordercolor='#d0d0d0', borderwidth=1,
        ),
        paper_bgcolor=BG, font=dict(family=FONT),
    )
    return fig


# ── load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
nr_full   = load_nr()
nr_sample = nr_full[nr_full['Country Code'].isin(INCLUDE_LIST)]
print(f'  {len(nr_sample["Country Code"].unique())} countries in sample')


# ── rob_A: minerals aggregated, 1995, k=5, PCA(2) ────────────────────────────
print('\n[rob_A] 1995 | minerals merged | PCA(2) | k=5')
pivot_a, feat_a = _prep_pivot(nr_sample, year=1995, merge_minerals=True)
pca_a, pca_mod_a, centroids_a = _run(pivot_a, feat_a, n_pca=2, n_clusters=5)
pca_a = _assign_labels(pca_a, centroids_a)

print('  Cluster sizes:')
print(pca_a.groupby('ClusterLabel')['Country'].count().sort_values(ascending=False).to_string())

fig_a = _world_map(pca_a,
    title='Robustness A — 1995, minerals aggregated (Oil / Gas / Coal / Minerals), k=5')
save(fig_a, 'rob_A__1995_minerals_merged_k5', OUT, w=1200, h=520)


# ── rob_B: PCA(3) instead of PCA(2), 1995, k=5 ───────────────────────────────
print('\n[rob_B] 1995 | full features | PCA(3) | k=5')
pivot_b, feat_b = _prep_pivot(nr_sample, year=1995, merge_minerals=False)
pca_b, pca_mod_b, centroids_b = _run(pivot_b, feat_b, n_pca=3, n_clusters=5)

var_exp = pca_mod_b.explained_variance_ratio_ * 100
print(f'  Variance explained: PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%  PC3={var_exp[2]:.1f}%'
      f'  (total {var_exp[:3].sum():.1f}%)')

pca_b = _assign_labels(pca_b, centroids_b)
print('  Cluster sizes:')
print(pca_b.groupby('ClusterLabel')['Country'].count().sort_values(ascending=False).to_string())

fig_b = _world_map(pca_b,
    title='Robustness B — 1995, PCA(3), k=5')
save(fig_b, 'rob_B__1995_pca3_k5', OUT, w=1200, h=520)


# ── rob_C: 2019, standard pipeline, k=5, PCA(2) ──────────────────────────────
print('\n[rob_C] 2019 | full features | PCA(2) | k=5')
pivot_c, feat_c = _prep_pivot(nr_sample, year=2019, merge_minerals=False)
pca_c, pca_mod_c, centroids_c = _run(pivot_c, feat_c, n_pca=2, n_clusters=5)
pca_c = _assign_labels(pca_c, centroids_c)

print('  Cluster sizes:')
print(pca_c.groupby('ClusterLabel')['Country'].count().sort_values(ascending=False).to_string())

fig_c = _world_map(pca_c,
    title='Robustness C — 2019, standard pipeline, k=5')
save(fig_c, 'rob_C__2019_standard_k5', OUT, w=1200, h=520)


# ── summary ───────────────────────────────────────────────────────────────────
print('\nDone. Maps saved to:', OUT)
