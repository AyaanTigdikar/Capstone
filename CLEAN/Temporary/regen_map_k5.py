"""
Regenerate Graphics/NB4/map_k5_1995.html with a clean legend.
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

K = 5
PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#A8DADC"]
OUT = "Graphics/NB4/map_k5_1995.html"

# ── Load data ──
nr = pd.read_csv("intermediary/NaturalResource.csv")
master = pd.read_csv("intermediary/Master.csv")
include_list = sorted(master["Country Code"].unique().tolist())
nr_sample = nr[nr["Country Code"].isin(include_list)]
nr_1995 = nr_sample[nr_sample["Year"] == 1995]
print(f"Sample: {len(include_list)} countries")

# ── Clustering (mirrors NB4 exactly) ──
df = nr_sample[nr_sample["Year"] == 1995].copy()

df_pivot = df.pivot_table(
    index=["Country", "Country Code", "Year", "Population"],
    columns="Resource", values="Production_TotalValue",
).reset_index()

resource_cols = df_pivot.columns.difference(["Country", "Country Code", "Year", "Population"])
df_pivot[resource_cols] = df_pivot[resource_cols].div(df_pivot["Population"], axis=0)
df_pivot.drop(columns="Population", inplace=True)
df_pivot = df_pivot.fillna(0)

df_latest = (df_pivot.sort_values("Year", ascending=True)
             .groupby(["Country", "Country Code"]).first().reset_index())

feature_cols = [c for c in df_latest.columns if c not in ["Country", "Country Code", "Year"]]
X_log = np.log1p(df_latest[feature_cols].fillna(0))

pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_log)

kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
clusters = kmeans.fit_predict(pca_components)

pca_df = pd.DataFrame({
    "Country": df_latest["Country"],
    "Country Code": df_latest["Country Code"],
    "Year": df_latest["Year"],
    "PC1": pca_components[:, 0],
    "PC2": pca_components[:, 1],
    "Cluster": clusters,
})

# Auto-label (mirrors NB4)
centroids = kmeans.cluster_centers_
pc1_rank = list(np.argsort(-centroids[:, 0]))
pc2_rank = list(np.argsort(-centroids[:, 1]))

label_map = {}
labeled = set()

oil_id = pc1_rank[0]
label_map[oil_id] = "Petrostates"
labeled.add(oil_id)

mineral_id = next(c for c in pc2_rank if c not in labeled)
label_map[mineral_id] = "Diversified Producers"
labeled.add(mineral_id)

remaining = [c for c in pc1_rank if c not in labeled]
label_map[remaining[0]] = "Oil Exporters"
mid, low = remaining[1], remaining[2]
if centroids[mid, 1] > centroids[low, 1]:
    label_map[mid] = "Mineral-Rich Developing"
    label_map[low] = "Hard Mineral Exporters"
else:
    label_map[mid] = "Low-Intensity Producers"
    label_map[low] = "Hard Mineral Exporters"

pca_df["ClusterLabels"] = pca_df["Cluster"].map(label_map)

sil = silhouette_score(pca_components, clusters)
print(f"k={K}, Silhouette: {sil:.3f}")
for cid in sorted(label_map):
    n = (pca_df["Cluster"] == cid).sum()
    print(f"  {label_map[cid]}: {n} countries — {sorted(pca_df[pca_df['Cluster']==cid]['Country Code'].tolist())}")

# ── Build map ──
DOMINANCE_THRESHOLD = 15.0

df_total = nr_1995.pivot_table(
    index=["Country", "Country Code"], columns="Resource",
    values="Production_TotalValue", aggfunc="sum",
).reset_index().fillna(0)

prod_cols = [c for c in df_total.columns if c not in ["Country", "Country Code"]]
for col in prod_cols:
    total = df_total[col].sum()
    if total > 0:
        df_total[f"{col}_Share"] = (df_total[col] / total) * 100

share_cols = [c for c in df_total.columns if c.endswith("_Share")]
df_map = pca_df.merge(df_total[["Country Code"] + share_cols], on="Country Code", how="left")
df_map["Is_Dominant"] = (df_map[share_cols] >= DOMINANCE_THRESHOLD).any(axis=1)

def make_hover(row):
    lines = [f"<b>{row['Country']}</b>", f"Cluster: {row['ClusterLabels']}"]
    vals = [(c, row.get(c, 0)) for c in prod_cols if row.get(c, 0) > 0]
    vals.sort(key=lambda x: x[1], reverse=True)
    if vals:
        lines.append("<br>Top Resources:")
        for res, v in vals[:3]:
            if v > 1e9:
                lines.append(f"  {res}: ${v/1e9:.1f}B")
            elif v > 1e6:
                lines.append(f"  {res}: ${v/1e6:.0f}M")
            else:
                lines.append(f"  {res}: ${v:,.0f}")
    return "<br>".join(lines)

df_map["hover_text"] = df_map.apply(make_hover, axis=1)

fig = go.Figure()

# Choropleth traces — legend suppressed on all
for cid in sorted(df_map["Cluster"].unique()):
    color = PALETTE[cid % len(PALETTE)]

    sub = df_map[(df_map["Cluster"] == cid) & (~df_map["Is_Dominant"])]
    if len(sub) > 0:
        fig.add_trace(go.Choropleth(
            locations=sub["Country Code"], z=[cid] * len(sub),
            colorscale=[[0, color], [1, color]], showscale=False,
            showlegend=False,
            customdata=sub["hover_text"].values,
            hovertemplate="%{customdata}<extra></extra>",
            marker=dict(line=dict(color="white", width=0.5)),
        ))

    sub_d = df_map[(df_map["Cluster"] == cid) & (df_map["Is_Dominant"])]
    if len(sub_d) > 0:
        fig.add_trace(go.Choropleth(
            locations=sub_d["Country Code"], z=[cid] * len(sub_d),
            colorscale=[[0, color], [1, color]], showscale=False,
            showlegend=False,
            customdata=sub_d["hover_text"].values,
            hovertemplate="%{customdata}<extra></extra>",
            marker=dict(line=dict(color="#c0392b", width=2.0)),
        ))

# Dummy Scattergeo traces — one per cluster, for legend only
cluster_order = sorted(label_map.keys(), key=lambda cid: [
    "Petrostates", "Oil Exporters", "Diversified Producers",
    "Mineral-Rich Developing", "Low-Intensity Producers", "Hard Mineral Exporters"
].index(label_map[cid]) if label_map[cid] in [
    "Petrostates", "Oil Exporters", "Diversified Producers",
    "Mineral-Rich Developing", "Low-Intensity Producers", "Hard Mineral Exporters"
] else 99)

for cid in cluster_order:
    lbl = label_map[cid]
    color = PALETTE[cid % len(PALETTE)]
    n = (pca_df["Cluster"] == cid).sum()
    fig.add_trace(go.Scattergeo(
        lat=[None], lon=[None],
        mode="markers",
        marker=dict(size=12, color=color, symbol="square", line=dict(width=0)),
        name=f"{lbl} (n={n})",
        showlegend=True,
    ))

# Red-border legend entry
fig.add_trace(go.Scattergeo(
    lat=[None], lon=[None],
    mode="markers",
    marker=dict(
        size=12, color="rgba(0,0,0,0)", symbol="square",
        line=dict(color="#c0392b", width=2),
    ),
    name=">15% global production",
    showlegend=True,
))

fig.update_geos(
    projection_type="natural earth",
    showcountries=True, countrycolor="lightgray",
    showcoastlines=True, coastlinecolor="lightgray",
    showland=True, landcolor="whitesmoke",
    showocean=True, oceancolor="aliceblue",
)
fig.update_layout(
    font=dict(family="IBM Plex Sans, Arial, sans-serif"),
    paper_bgcolor="#fafafa",
    title=dict(
        text="Natural Resource Clusters, k=5 — 1995 (per capita production value)<br>"
             "<sup>Red border = >15% of global production for any resource</sup>",
        x=0.45, font=dict(size=14, color="#1a1a2e"),
    ),
    width=1100, height=550,
    margin=dict(l=10, r=180, t=65, b=10),
    legend=dict(
        x=1.01, y=0.5, xanchor="left", yanchor="middle",
        font=dict(size=11, family="IBM Plex Sans, Arial, sans-serif"),
        bgcolor="rgba(250,250,250,0.9)",
        bordercolor="#cccccc", borderwidth=1,
        tracegroupgap=4,
    ),
)

fig.write_html(OUT, config={"displayModeBar": False, "responsive": True})
print(f"\nSaved: {OUT}")
