#!/usr/local/bin/python3.10
"""
cluster_robustness.py
---------------------
Sensitivity checks on the k=5 PCA-KMeans clustering.

  Part 1 — Choice of k:
    Silhouette scores for k in {3, 4, 5, 6}, plus world maps for k=3, k=4, k=6.

  Part 2 — Alternative specifications (all k=5):
    rob_A  —  1995, minerals aggregated into one bucket (Oil / Gas / Coal / Minerals)
    rob_B  —  1995, PCA(3) instead of PCA(2); countries clustered in 3-D space
    rob_C  —  2019, standard full-feature pipeline (same as main analysis)

  Part 3 — ARI summary and country-level stability table.

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
from sklearn.metrics import silhouette_score, adjusted_rand_score

try:
    import kaleido  # noqa — needed by plotly.write_image
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False
    print('NOTE: kaleido not installed. PNGs will be skipped, HTML still saved.')
    print('      To enable PNG: open Terminal.app and run   pip install -U kaleido')

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

OUT = os.path.join(os.path.expanduser('~'), 'Desktop', 'GitHub', 'Capstone',
                   'FINAL CODE RECAP', 'Final', 'charts', 'cluster robustness')
os.makedirs(OUT, exist_ok=True)

HYDROCARBONS = {'Oil', 'Natural Gas', 'Coal'}

# ── colour palette (shared across all three maps) ────────────────────────────
_COLORS = {
    'Petrostates':           '#d4853b',
    'Oil Exporters':         '#4a6fa5',
    'Diversified Producers': '#d4a017',
    'Forestry Intensive':    '#c23a3a',
    'Mining Exporters':      '#7a5c9e',
    'Oil & Minerals':        '#3a8fa5',
    'Hydrocarbons':          '#d4853b',
    'Low Resource':          '#c23a3a',
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

    names = ['Oil Exporters', 'Diversified Producers', 'Forestry Intensive',
             'Oil & Minerals']
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
        title=dict(text=title, font=dict(family=FONT, size=14, color=NAVY), x=0.5),
        width=1400, height=700,
        margin=dict(l=0, r=0, t=50, b=80),
        legend=dict(
            orientation='h', x=0.5, y=-0.06, xanchor='center', yanchor='top',
            font=dict(size=12, family=FONT),
            bgcolor='rgba(250,250,250,0.9)', bordercolor='#d0d0d0', borderwidth=1,
        ),
        paper_bgcolor=BG, font=dict(family=FONT),
    )
    return fig

# PNG export settings
PNG_SCALE = 3   # 3x → 4200×2100 px

def _save_fig(fig, path_stem):
    """Save as PNG (requires kaleido)."""
    if HAS_KALEIDO:
        fig.write_image(path_stem + '.png', scale=PNG_SCALE)
        print(f'  Saved: {os.path.basename(path_stem)}.png')
    else:
        print(f'  SKIPPED: {os.path.basename(path_stem)}.png  (kaleido missing)')
        print('      To enable PNG: open Terminal.app and run   pip install -U kaleido')


# ── load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
nr_full   = load_nr()
nr_sample = nr_full[nr_full['Country Code'].isin(INCLUDE_LIST)]
print(f'  {len(nr_sample["Country Code"].unique())} countries in sample')


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: CHOICE OF k  —  silhouette scores + maps for k=3, 4, 6
# ══════════════════════════════════════════════════════════════════════════════

# Run baseline PCA(2) on 1995 once; reuse for all k values
pivot_base, feat_base = _prep_pivot(nr_sample, year=1995, merge_minerals=False)
X_base = np.log1p(pivot_base[feat_base].fillna(0))
pca_base = PCA(n_components=2, random_state=42)
Xp_base  = pca_base.fit_transform(X_base)

print('\n' + '='*70)
print('PART 1: Silhouette scores for k in {3, 4, 5, 6}')
print('='*70)

sil_scores = {}
km_results = {}   # store (labels, centroids) for each k
for k in [3, 4, 5, 6]:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    lbls = km.fit_predict(Xp_base)
    sil = silhouette_score(Xp_base, lbls)
    sil_scores[k] = sil
    km_results[k] = (lbls, km.cluster_centers_)
    print(f'  k={k}:  silhouette = {sil:.3f}')

# Label-name pools for different k values
_NAMES_BY_K = {
    3: ['Forestry Intensive'],                     # Petrostates + Mining assigned first, 1 left
    4: ['Oil Exporters', 'Forestry Intensive'],     # 2 remaining after Petrostates + Mining
    5: ['Oil Exporters', 'Diversified Producers', 'Forestry Intensive'],
    6: ['Oil Exporters', 'Diversified Producers', 'Forestry Intensive', 'Oil & Minerals'],
}

def _assign_labels_flex(pca_df, centroids, k):
    """Rank-based label assignment, with name pool adjusted per k."""
    pc1_rank = list(np.argsort(-centroids[:, 0]))
    pc2_rank = list(np.argsort(-centroids[:, 1]))

    lmap, done = {}, set()
    lmap[pc1_rank[0]] = 'Petrostates';  done.add(pc1_rank[0])
    mid = next(c for c in pc2_rank if c not in done)
    lmap[mid] = 'Mining Exporters';     done.add(mid)

    remaining = [c for c in pc1_rank if c not in done]
    names = _NAMES_BY_K.get(k, ['Oil Exporters', 'Diversified Producers',
                                 'Forestry Intensive', 'Oil & Minerals'])
    for i, cid in enumerate(remaining):
        lmap[cid] = names[i] if i < len(names) else f'Cluster {cid}'

    pca_df['ClusterLabel'] = pca_df['Cluster'].map(lmap)
    return pca_df


# Generate maps for k = 3, 4, 6  (k=5 is the baseline in the main viz)
for k in [3, 4, 6]:
    lbls, cents = km_results[k]
    pca_df = pivot_base[['Country', 'Country Code']].copy()
    pca_df['PC1'] = Xp_base[:, 0]
    pca_df['PC2'] = Xp_base[:, 1]
    pca_df['Cluster'] = lbls
    pca_df = _assign_labels_flex(pca_df, cents, k)

    sil = sil_scores[k]
    title = f'k={k} — 1995 cross-section (silhouette = {sil:.3f})'
    fig = _world_map(pca_df, title)
    _save_fig(fig, os.path.join(OUT, f'silhouette_k{k}__1995'))

    print(f'\n  k={k} cluster composition:')
    for lbl in sorted(pca_df['ClusterLabel'].unique()):
        codes = sorted(pca_df.loc[pca_df['ClusterLabel'] == lbl, 'Country Code'].tolist())
        print(f'    {lbl:25s} (n={len(codes):2d}): {", ".join(codes)}')
    print()

# Store k=5 baseline labels for ARI comparison later
lbls_k5, cents_k5 = km_results[5]
pca_baseline = pivot_base[['Country', 'Country Code']].copy()
pca_baseline['Cluster'] = lbls_k5


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: ALTERNATIVE SPECIFICATIONS  (all k=5)
# ══════════════════════════════════════════════════════════════════════════════


print('\n' + '='*70)
print('PART 2: Alternative specifications (all k=5)')
print('='*70)

# ── rob_A: minerals aggregated, 1995, k=5, PCA(2) ────────────────────────────
print('\n[rob_A] 1995 | minerals merged | PCA(2) | k=5')
pivot_a, feat_a = _prep_pivot(nr_sample, year=1995, merge_minerals=True)
pca_a, pca_mod_a, centroids_a = _run(pivot_a, feat_a, n_pca=2, n_clusters=5)
pca_a = _assign_labels(pca_a, centroids_a)

print('  Cluster sizes:')
print(pca_a.groupby('ClusterLabel')['Country'].count().sort_values(ascending=False).to_string())

fig_a = _world_map(pca_a,
    title='Robustness A — 1995, minerals aggregated (Oil / Gas / Coal / Minerals), k=5')
_save_fig(fig_a, os.path.join(OUT, 'rob_A__1995_minerals_merged_k5'))


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
_save_fig(fig_b, os.path.join(OUT, 'rob_B__1995_pca3_k5'))


# ── rob_C: 2019, standard pipeline, k=5, PCA(2) ──────────────────────────────
print('\n[rob_C] 2019 | full features | PCA(2) | k=5')
pivot_c, feat_c = _prep_pivot(nr_sample, year=2019, merge_minerals=False)
pca_c, pca_mod_c, centroids_c = _run(pivot_c, feat_c, n_pca=2, n_clusters=5)
pca_c = _assign_labels(pca_c, centroids_c)

print('  Cluster sizes:')
print(pca_c.groupby('ClusterLabel')['Country'].count().sort_values(ascending=False).to_string())

fig_c = _world_map(pca_c,
    title='Robustness C — 2019, standard pipeline, k=5')
_save_fig(fig_c, os.path.join(OUT, 'rob_C__2019_standard_k5'))


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: ARI SUMMARY + COUNTRY-LEVEL STABILITY
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 3: Adjusted Rand Index (vs baseline k=5)')
print('='*70)

ari_a = adjusted_rand_score(pca_baseline['Cluster'], pca_a['Cluster'])
ari_b = adjusted_rand_score(pca_baseline['Cluster'], pca_b['Cluster'])

# Rob C has a different year so align on Country Code
m_c = pca_baseline.merge(
    pca_c[['Country Code', 'Cluster']].rename(columns={'Cluster': 'Cluster_C'}),
    on='Country Code', how='inner')
ari_c = adjusted_rand_score(m_c['Cluster'], m_c['Cluster_C'])

print(f'  Rob A (minerals merged):   ARI = {ari_a:.3f}')
print(f'  Rob B (PCA-3):             ARI = {ari_b:.3f}')
print(f'  Rob C (2019 cross-sec):    ARI = {ari_c:.3f}')

# ── Country-level label comparison ───────────────────────────────────────────
# Assign labels to baseline k=5 using the same rank-based heuristic
pca_baseline_labelled = pca_baseline.copy()
pca_baseline_labelled['PC1'] = Xp_base[:, 0]
pca_baseline_labelled['PC2'] = Xp_base[:, 1]
pca_baseline_labelled = _assign_labels_flex(pca_baseline_labelled, cents_k5, k=5)

compare = pca_baseline_labelled[['Country Code', 'ClusterLabel']].copy()
compare = compare.rename(columns={'ClusterLabel': 'Main_k5'})
compare = compare.merge(pivot_base[['Country Code', 'Country']].drop_duplicates(), on='Country Code')
compare = compare.merge(pca_a[['Country Code', 'ClusterLabel']].rename(
    columns={'ClusterLabel': 'RobA'}), on='Country Code', how='left')
compare = compare.merge(pca_b[['Country Code', 'ClusterLabel']].rename(
    columns={'ClusterLabel': 'RobB'}), on='Country Code', how='left')
compare = compare.merge(pca_c[['Country Code', 'ClusterLabel']].rename(
    columns={'ClusterLabel': 'RobC'}), on='Country Code', how='left')

compare['n_diff'] = (
    (compare['RobA'] != compare['Main_k5']).astype(int) +
    (compare['RobB'] != compare['Main_k5']).astype(int) +
    (compare['RobC'] != compare['Main_k5']).astype(int)
)

compare.to_csv(os.path.join(OUT, 'country_label_comparison.csv'), index=False)

print('\n  Label stability summary:')
for nd in sorted(compare['n_diff'].unique()):
    n = (compare['n_diff'] == nd).sum()
    print(f'    Changed in {nd}/3 specs:  {n} countries')

print('\n  Countries unstable in 2+ specs:')
unstable = compare[compare['n_diff'] >= 2].sort_values('n_diff', ascending=False)
for _, r in unstable.iterrows():
    print(f'    {r["Country"]:30s}  Main={r["Main_k5"]:25s}  '
          f'A={r["RobA"]:25s}  B={r["RobB"]:25s}  C={r["RobC"]}')

print(f'\n  Saved: country_label_comparison.csv')


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: STABILITY VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('PART 4: Stability visualisations')
print('='*70)

# ── 4a. Stability world map ──────────────────────────────────────────────────
# Countries coloured by how many robustness checks changed their label (0–3)

STABILITY_COLORS = {
    0: '#1a7a3a',   # dark green  — stable in all 3
    1: '#a3c466',   # light green — shifted in 1
    2: '#e8a838',   # amber       — shifted in 2
    3: '#c23a3a',   # red         — shifted in all 3
}
STABILITY_LABELS = {
    0: 'Stable (0/3 changed)',
    1: 'Minor shift (1/3)',
    2: 'Moderate shift (2/3)',
    3: 'Unstable (3/3 changed)',
}

fig_stab = go.Figure()
for nd in [0, 1, 2, 3]:
    sub = compare[compare['n_diff'] == nd]
    if len(sub) == 0:
        continue
    color = STABILITY_COLORS[nd]
    fig_stab.add_trace(go.Choropleth(
        locations=sub['Country Code'],
        z=[1] * len(sub),
        colorscale=[[0, color], [1, color]],
        showscale=False,
        showlegend=True,
        name=STABILITY_LABELS[nd],
        hovertemplate=(
            '<b>%{location}</b><br>'
            + STABILITY_LABELS[nd]
            + '<extra></extra>'
        ),
        marker=dict(line=dict(color='white', width=0.6)),
    ))

fig_stab.update_geos(
    projection_type='natural earth',
    showcountries=True,  countrycolor='#ccc',
    showcoastlines=True, coastlinecolor='#ccc',
    showland=True,       landcolor='#f0f0f0',
    showocean=True,      oceancolor='#dde8f0',
    showframe=False,
)
fig_stab.update_layout(
    title=dict(
        text='Cluster Assignment Stability across Three Robustness Checks',
        font=dict(family=FONT, size=14, color=NAVY), x=0.5,
    ),
    width=1400, height=700,
    margin=dict(l=0, r=0, t=50, b=80),
    legend=dict(
        orientation='h', x=0.5, y=-0.06, xanchor='center', yanchor='top',
        font=dict(size=12, family=FONT),
        bgcolor='rgba(250,250,250,0.9)', bordercolor='#d0d0d0', borderwidth=1,
    ),
    paper_bgcolor=BG, font=dict(family=FONT),
)
_save_fig(fig_stab, os.path.join(OUT, 'stability_world_map'))


# ── 4b. Heatmap table: countries × specs, coloured by cluster label ──────────

# Sort: by n_diff (most stable first), then by baseline label, then country
compare_sorted = compare.sort_values(
    ['n_diff', 'Main_k5', 'Country'], ascending=[True, True, True]
).reset_index(drop=True)

# Assign numeric code per label for colouring
all_labels = sorted(set(
    compare_sorted['Main_k5'].tolist() +
    compare_sorted['RobA'].dropna().tolist() +
    compare_sorted['RobB'].dropna().tolist() +
    compare_sorted['RobC'].dropna().tolist()
))

# Colour scale: map each label to its colour
label_to_num = {lbl: i for i, lbl in enumerate(all_labels)}
n_labels = len(all_labels)

# Build the z-matrix (countries × 4 specs)
spec_cols = ['Main_k5', 'RobA', 'RobB', 'RobC']
spec_names = ['Baseline (k=5)', 'Rob A: Minerals merged', 'Rob B: PCA(3)', 'Rob C: 2019']
z_matrix = []
text_matrix = []
for _, row in compare_sorted.iterrows():
    z_row = []
    t_row = []
    for col in spec_cols:
        lbl = row[col] if pd.notna(row[col]) else 'N/A'
        z_row.append(label_to_num.get(lbl, -1))
        t_row.append(lbl)
    z_matrix.append(z_row)
    text_matrix.append(t_row)

# Build discrete colorscale from label colours
colorscale = []
for lbl in all_labels:
    idx = label_to_num[lbl]
    c = _COLORS.get(lbl, '#aaa')
    colorscale.append([idx / max(n_labels - 1, 1), c])
# Ensure 0 and 1 boundaries
if len(colorscale) > 0:
    discrete_cs = []
    for i, lbl in enumerate(all_labels):
        lo = i / n_labels
        hi = (i + 1) / n_labels
        c = _COLORS.get(lbl, '#aaa')
        discrete_cs.append([lo, c])
        discrete_cs.append([hi, c])

countries = compare_sorted['Country'].tolist()

fig_heat = go.Figure(data=go.Heatmap(
    z=z_matrix,
    x=spec_names,
    y=countries,
    text=text_matrix,
    texttemplate='%{text}',
    textfont=dict(size=9, family=FONT),
    colorscale=discrete_cs,
    showscale=False,
    hovertemplate='<b>%{y}</b><br>%{x}: %{text}<extra></extra>',
    xgap=2, ygap=1,
    zmin=0, zmax=n_labels,
))

fig_heat.update_layout(
    title=dict(
        text='Cluster Label Assignments: Baseline vs Robustness Checks',
        font=dict(family=FONT, size=14, color=NAVY), x=0.5,
    ),
    width=800,
    height=max(500, len(countries) * 22 + 120),
    margin=dict(l=180, r=30, t=60, b=40),
    xaxis=dict(
        side='top', tickfont=dict(size=11, family=FONT),
        tickangle=0,
    ),
    yaxis=dict(
        tickfont=dict(size=9, family=FONT),
        autorange='reversed',   # most stable at top
    ),
    paper_bgcolor=BG,
    font=dict(family=FONT),
)

# Add a horizontal line separating stability groups
cumulative = 0
for nd in [0, 1, 2]:
    count = (compare_sorted['n_diff'] == nd).sum()
    cumulative += count
    if cumulative < len(compare_sorted):
        fig_heat.add_hline(
            y=cumulative - 0.5,
            line=dict(color='#333', width=1.5, dash='dot'),
        )

_save_fig(fig_heat, os.path.join(OUT, 'stability_heatmap'))


print(f'\nDone. All outputs in: {OUT}')
