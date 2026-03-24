"""
Run clustering with k=4,5,6 on 1995 data and save only the choropleth maps.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px

OUT = 'temporary'
os.makedirs(OUT, exist_ok=True)

# ── Load data ──
nr = pd.read_csv('intermediary/NaturalResource.csv')
master = pd.read_csv('intermediary/Master.csv')
include_list = sorted(master['Country Code'].unique().tolist())
nr_sample = nr[nr['Country Code'].isin(include_list)]
print(f'Sample: {len(include_list)} countries')

# ── Clustering function (from NB4) ──
def run_clustering(nr_data, year_filter=None, n_clusters=4, random_state=42):
    df = nr_data.copy()
    if year_filter is not None:
        df = df[df['Year'] == year_filter]

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

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(pca_components)

    pca_df = pd.DataFrame({
        'Country': df_latest['Country'],
        'Country Code': df_latest['Country Code'],
        'Year': df_latest['Year'],
        'PC1': pca_components[:, 0],
        'PC2': pca_components[:, 1],
        'Cluster': clusters,
    })

    # Auto-label from centroids
    centroids = kmeans.cluster_centers_
    pc1_rank = list(np.argsort(-centroids[:, 0]))
    pc2_rank = list(np.argsort(-centroids[:, 1]))

    label_map = {}
    labeled = set()

    if n_clusters >= 2:
        oil_id = pc1_rank[0]
        label_map[oil_id] = 'Oil, Few Minerals'
        labeled.add(oil_id)

        mineral_id = next(c for c in pc2_rank if c not in labeled)
        label_map[mineral_id] = 'Minerals, No Oil'
        labeled.add(mineral_id)

    if n_clusters >= 3:
        remaining = [c for c in pc1_rank if c not in labeled]
        label_map[remaining[0]] = 'Some Oil, No Minerals'
        labeled.add(remaining[0])

    if n_clusters >= 4:
        remaining = [c for c in pc1_rank if c not in labeled]
        label_map[remaining[0]] = 'No Oil, No Minerals'
        labeled.add(remaining[0])

    # Any extras for k=5,6
    remaining = [c for c in range(n_clusters) if c not in labeled]
    for j, cid in enumerate(remaining):
        label_map[cid] = f'Cluster {cid}'

    pca_df['ClusterLabels'] = pca_df['Cluster'].map(label_map)

    sil = silhouette_score(pca_components, clusters)
    print(f'  k={n_clusters}  Silhouette: {sil:.3f}')
    for cid in sorted(label_map):
        n = (pca_df['Cluster'] == cid).sum()
        print(f'    {label_map[cid]}: {n}')

    return pca_df, pca, feature_cols


# ── Map function (from NB4) ──
def create_cluster_map(pca_df, nr_data, n_clusters, dominance_threshold=15.0):
    cluster_names_map = dict(zip(pca_df['Cluster'].unique(), pca_df['ClusterLabels'].unique()))

    df_total = nr_data.pivot_table(
        index=['Country', 'Country Code'], columns='Resource',
        values='Production_TotalValue', aggfunc='sum',
    ).reset_index().fillna(0)

    prod_cols = [c for c in df_total.columns if c not in ['Country', 'Country Code']]
    for col in prod_cols:
        total = df_total[col].sum()
        if total > 0:
            df_total[f'{col}_Share'] = (df_total[col] / total) * 100

    share_cols = [c for c in df_total.columns if c.endswith('_Share')]
    df_map = pca_df.merge(df_total[['Country Code'] + share_cols], on='Country Code', how='left')
    df_map['Is_Dominant'] = (df_map[share_cols] >= dominance_threshold).any(axis=1)

    def make_hover(row):
        lines = [f"<b>{row['Country']}</b>", f"Cluster: {row['ClusterLabels']}"]
        vals = [(c, row.get(c, 0)) for c in prod_cols if row.get(c, 0) > 0]
        vals.sort(key=lambda x: x[1], reverse=True)
        if vals:
            lines.append('<br>Top Resources:')
            for res, v in vals[:3]:
                if v > 1e9:
                    lines.append(f'  {res}: ${v/1e9:.1f}B')
                elif v > 1e6:
                    lines.append(f'  {res}: ${v/1e6:.0f}M')
                else:
                    lines.append(f'  {res}: ${v:,.0f}')
        return '<br>'.join(lines)

    df_map['hover_text'] = df_map.apply(make_hover, axis=1)

    colors = px.colors.qualitative.Bold
    fig = go.Figure()

    for cid in sorted(df_map['Cluster'].unique()):
        lbl = cluster_names_map.get(cid, f'Cluster {cid}')
        color = colors[cid % len(colors)]

        sub = df_map[(df_map['Cluster'] == cid) & (~df_map['Is_Dominant'])]
        if len(sub) > 0:
            fig.add_trace(go.Choropleth(
                locations=sub['Country Code'], z=[cid]*len(sub),
                colorscale=[[0, color], [1, color]], showscale=False,
                customdata=sub['hover_text'].values,
                hovertemplate='%{customdata}<extra></extra>',
                name=f'{lbl} ({len(sub)})',
                marker=dict(line=dict(color='white', width=0.5)),
            ))

        sub_d = df_map[(df_map['Cluster'] == cid) & (df_map['Is_Dominant'])]
        if len(sub_d) > 0:
            fig.add_trace(go.Choropleth(
                locations=sub_d['Country Code'], z=[cid]*len(sub_d),
                colorscale=[[0, color], [1, color]], showscale=False,
                customdata=sub_d['hover_text'].values,
                hovertemplate='%{customdata}<extra></extra>',
                name=f'{lbl} (major, {len(sub_d)})',
                marker=dict(line=dict(color='red', width=1.5)),
            ))

    fig.update_geos(
        projection_type='natural earth',
        showcountries=True, countrycolor='lightgray',
        showcoastlines=True, coastlinecolor='lightgray',
        showland=True, landcolor='whitesmoke',
        showocean=True, oceancolor='aliceblue',
    )
    fig.update_layout(
        title=dict(
            text=f'Resource Clusters k={n_clusters} (per capita production, 1995)<br>'
                 '<sup>Red border = >15% of global production</sup>',
            x=0.45, font=dict(size=14),
        ),
        width=1100, height=550,
        margin=dict(l=10, r=150, t=60, b=10),
        legend=dict(x=1.01, y=0.3, font=dict(size=10)),
    )
    return fig


# ── Run for k=4,5,6 ──
nr_1995 = nr_sample[nr_sample['Year'] == 1995]

for k in [4, 5, 6]:
    print(f'\n{"="*50}\nk = {k}\n{"="*50}')
    pca_df, pca_model, feat_cols = run_clustering(nr_sample, year_filter=1995, n_clusters=k)
    fig = create_cluster_map(pca_df, nr_1995, n_clusters=k)
    outpath = os.path.join(OUT, f'cluster_map_k{k}.html')
    fig.write_html(outpath, config={'displayModeBar': False, 'responsive': True})
    print(f'  Saved: {outpath}')

print('\nDone.')
