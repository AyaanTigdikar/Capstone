# Auto-extracted from 4_Clustering_FINAL.ipynb
import matplotlib; matplotlib.use('Agg')
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('Final/charts', exist_ok=True)
CHARTS = 'Final/charts'

_WRITE_CONFIG = {'displayModeBar': False, 'responsive': True}

def _save(fig, html_path: str, width: int = 1100, height: int = 600):
    """Write both HTML and PNG for a Plotly figure."""
    fig.write_html(html_path, config=_WRITE_CONFIG)
    print(f"  Saved: {html_path}")
    png_path = html_path.replace('.html', '.png')
    try:
        fig.write_image(png_path, width=width, height=height, scale=2)
        print(f"  Saved: {png_path}")
    except Exception as e:
        print(f"  PNG skipped ({e})")

# # Clustering: Natural Resource Profiles
# 
# **Capstone Project — Moody's Ratings**  
# *Pipeline step 4 of 5: Identifying resource-dependency typologies*
# 
# This notebook classifies countries into four natural resource profile groups using PCA and K-Means clustering on per-capita production values. The clustering is run for three time windows to capture both cross-sectional structure and temporal shifts.
# 
# **Methodology:**
# 1. Production values (quantity x price, in USD) are divided by population to obtain per-capita figures
# 2. A log(1+x) transformation compresses the extreme right skew typical of resource data
# 3. PCA reduces the feature space to 2 components (PC1 captures hydrocarbons, PC2 captures minerals)
# 4. K-Means (k=4) partitions countries in PCA space into four groups
# 
# **Why these choices:**
# - *Per capita* (not per GDP): avoids mixing resource profiles with economic structure, which is what we're trying to predict (ECI)
# - *log1p* (not z-scoring): appropriate for distributions with 1000:1 ratios between smallest and largest producers; StandardScaler would compress most countries into a narrow band
# - *PCA then K-Means* (not K-Means on raw features): removes correlated noise (oil and gas co-occur), reduces dimensionality for better Euclidean distance behaviour
# - *k=4*: validated with silhouette analysis; produces a clean typology matching the economic literature
# 
# **Three variants:**
# - **1995 snapshot:** Cluster assignments based on 1995 production profiles (baseline year)
# - **2019 snapshot:** Cluster assignments based on 2019 production profiles (end of panel)
# - **Aggregated:** Uses earliest available year per country across 1995-2005
# 
# **Inputs:**
# - `intermediary/NaturalResource.csv` (from Step 3)
# - `intermediary/Master.csv` (from Step 3, for the ECI evolution chart)
# 
# **Outputs:**
# - `intermediary/clusters1995.csv`, `intermediary/clusters2019.csv`, `intermediary/clustersagg.csv`
# 
# ---

# ## 0. Setup

import os
os.makedirs("intermediary", exist_ok=True)
os.makedirs("Graphics/NB4", exist_ok=True)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import sys; sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from viz_utils import base_layout, GRID, FONT

# Fixed colour palette keyed by cluster label (label-stable, not cluster-ID-stable)
_LABEL_COLORS = {
    'Petrostates':          '#d4853b',   # orange
    'Oil Exporters':        '#4a6fa5',   # blue
    'Major Producers':'#2e7d4a',   # green
    'Limited Resources':    '#c23a3a',   # red
}

# ## 1. Load Data and Define Sample
# 
# The 54-country sample corresponds to resource-dependent developing economies identified through the filtering criteria in Steps 1-3 (total natural resource rents > 5% of GDP in 1995, non-high-income).

nr = pd.read_csv("intermediary/NaturalResource.csv")

# Countries in the analysis sample
include_list = [
    'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
    'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
    'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
    'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
    'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
    'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
]

nr_sample = nr[nr["Country Code"].isin(include_list)]
print(f"NR data: {nr_sample.shape[0]:,} rows")
print(f"Countries: {nr_sample['Country Code'].nunique()}")
print(f"Years: {sorted(nr_sample['Year'].unique())}")
print(f"Resources: {nr_sample['Resource'].nunique()}")

# ## 2. Clustering Pipeline
# 
# The clustering function encapsulates the full pipeline: pivot to per-capita production values, log1p transform, PCA, K-Means. It returns the cluster assignments along with PC coordinates for plotting.
# 
# Cluster labels are assigned automatically based on the centroid position in PCA space rather than hardcoded to specific cluster numbers (since K-Means assigns arbitrary IDs that can change between runs). The labelling logic uses the sign and magnitude of PC1 and PC2 loadings: PC1 captures hydrocarbon dominance, PC2 captures mineral dominance.

def run_clustering(nr_data, year_filter=None, agg_years=None, n_clusters=4, random_state=42):
    """
    Full clustering pipeline: pivot -> per capita -> log1p -> PCA(2) -> KMeans(k).

    Parameters
    ----------
    nr_data : DataFrame
        NaturalResource data with Country, Country Code, Year, Resource,
        Production_TotalValue, Population columns.
    year_filter : int or None
        If set, restrict to a single year (e.g. 1995 or 2019).
    agg_years : list or None
        If set, restrict to these years and take earliest per country.
    n_clusters : int
        Number of K-Means clusters.
    random_state : int
        For KMeans reproducibility (PCA uses deterministic solver).

    Returns
    -------
    pca_df : DataFrame with Country, Country Code, Year, PC1, PC2, Cluster, ClusterLabels
    pca_model : fitted PCA object
    feature_cols : list of resource column names used
    """

    df = nr_data.copy()

    # ── Year selection ──
    if year_filter is not None:
        df = df[df["Year"] == year_filter]
    elif agg_years is not None:
        df = df[df["Year"].isin(agg_years)]

    # ── Pivot: one row per country, one column per resource ──
    df_pivot = df.pivot_table(
        index=["Country", "Country Code", "Year", "Population"],
        columns="Resource",
        values="Production_TotalValue",
    ).reset_index()

    resource_cols = df_pivot.columns.difference(
        ["Country", "Country Code", "Year", "Population"]
    )

    # ── Per-capita normalisation ──
    df_pivot[resource_cols] = df_pivot[resource_cols].div(
        df_pivot["Population"], axis=0
    )
    df_pivot.drop(columns="Population", inplace=True)
    df_pivot = df_pivot.fillna(0)

    # ── Take earliest year per country (for aggregated variant) ──
    df_latest = (
        df_pivot.sort_values("Year", ascending=True)
        .groupby(["Country", "Country Code"])
        .first()
        .reset_index()
    )

    feature_cols = [c for c in df_latest.columns if c not in ["Country", "Country Code", "Year"]]

    # ── log1p transform ──
    X = df_latest[feature_cols].fillna(0)
    X_log = np.log1p(X)

    # ── PCA ──
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_log)

    # ── K-Means ──
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    clusters = kmeans.fit_predict(pca_components)

    # ── Build results DataFrame ──
    pca_df = pd.DataFrame({
        "Country": df_latest["Country"],
        "Country Code": df_latest["Country Code"],
        "Year": df_latest["Year"],
        "PC1": pca_components[:, 0],
        "PC2": pca_components[:, 1],
        "Cluster": clusters,
    })

    # ── Auto-label clusters from centroids ──
    # FIX (Check 11): replaced hardcoded PC1/PC2 thresholds with rank-based labeling.
    # The old thresholds (pc1 > 1.5, etc.) were calibrated on data where oil production
    # was not annualised; after the unit fix in NB3 the scale changes and hardcoded
    # absolute values no longer generalise. Rank-based assignment is invariant to scale.
    #
    # Logic:
    #   1. Cluster with highest PC1 → Oil-dominant (hydrocarbons load on PC1)
    #   2. Cluster with highest PC2 (not already labeled) → Mineral-dominant
    #   3. Remaining cluster with higher PC1 → Some Oil
    #   4. Remaining cluster with lower PC1 → No Resources
    centroids = kmeans.cluster_centers_

    pc1_rank = list(np.argsort(-centroids[:, 0]))  # highest PC1 first
    pc2_rank = list(np.argsort(-centroids[:, 1]))  # highest PC2 first

    label_map = {}
    labeled = set()

    # 1. Highest PC1 → Oil-dominant (Gulf-style petrostates, high per-capita oil)
    oil_id = pc1_rank[0]
    label_map[oil_id] = "Petrostates"
    labeled.add(oil_id)

    # 2. Highest PC2 not yet labeled → Mineral/diversified (copper, coal, gold mix)
    mineral_id = next(c for c in pc2_rank if c not in labeled)
    label_map[mineral_id] = "Major Producers"
    labeled.add(mineral_id)

    # 3 & 4. Remaining two by PC1 rank
    remaining = [c for c in pc1_rank if c not in labeled]
    label_map[remaining[0]] = "Oil Exporters"
    label_map[remaining[1]] = "Limited Resources"

    pca_df["ClusterLabels"] = pca_df["Cluster"].map(label_map)

    # ── Metrics ──
    sil = silhouette_score(pca_components, clusters)
    print(f"Silhouette score: {sil:.3f}")
    print(f"Cluster distribution:")
    for cid in sorted(pca_df["Cluster"].unique()):
        n = (pca_df["Cluster"] == cid).sum()
        print(f"  {label_map[cid]}: {n} countries")

    return pca_df, pca, feature_cols

# ## 3. Validate k with Silhouette Analysis
# 
# Before running the final clustering, we check that k=4 is a reasonable choice by computing silhouette scores for k=2 through k=8. The silhouette score measures how similar each point is to its own cluster compared to neighbouring clusters (range: -1 to 1, higher is better).

# Run validation on 1995 data
nr_1995 = nr_sample[nr_sample["Year"] == 1995].copy()

df_pivot_val = nr_1995.pivot_table(
    index=["Country", "Country Code", "Year", "Population"],
    columns="Resource",
    values="Production_TotalValue",
).reset_index()

resource_cols_val = df_pivot_val.columns.difference(
    ["Country", "Country Code", "Year", "Population"]
)
df_pivot_val[resource_cols_val] = df_pivot_val[resource_cols_val].div(
    df_pivot_val["Population"], axis=0
)
df_pivot_val = df_pivot_val.fillna(0)

feat_cols_val = [c for c in df_pivot_val.columns
                 if c not in ["Country", "Country Code", "Year", "Population"]]
X_val = np.log1p(df_pivot_val[feat_cols_val].fillna(0))

pca_val = PCA(n_components=2)
X_pca_val = pca_val.fit_transform(X_val)

# Silhouette scores for k = 2..8
k_range = range(2, 9)
sil_scores = []
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_pca_val)
    sil_scores.append(silhouette_score(X_pca_val, labels))
    inertias.append(km.inertia_)

# Plot
fig_val = go.Figure()
fig_val.add_trace(go.Scatter(
    x=list(k_range), y=sil_scores,
    mode="lines+markers", name="Silhouette Score",
    marker=dict(size=10),
))
fig_val.add_vline(x=4, line_dash="dash", line_color="red",
                  annotation_text="k=4 (selected)", annotation_position="top right")
fig_val.update_layout(
    title="Silhouette Score by Number of Clusters (1995 data)",
    xaxis_title="k", yaxis_title="Silhouette Score",
    width=700, height=400,
)
# diagnostic only — not saved to Final/charts

print("\nSilhouette scores:")
for k, s in zip(k_range, sil_scores):
    marker = " <-- selected" if k == 4 else ""
    print(f"  k={k}: {s:.3f}{marker}")

# ## 4. Run Clustering for All Three Variants
# 
# The pipeline is run three times:
# - **1995:** Single-year snapshot at the start of the panel
# - **2019:** Single-year snapshot at the end of the panel
# - **Aggregated:** Uses data from 1995, 1999, and 2005, taking the earliest available year per country. This smooths out year-specific fluctuations while still capturing structural resource profiles

results = {}

# ── 1995 snapshot ──
print("=" * 60)
print("1995 SNAPSHOT")
print("=" * 60)
pca_1995, pca_model_1995, feat_1995 = run_clustering(
    nr_sample, year_filter=1995
)
results["1995"] = pca_1995
print()

# ── 2019 snapshot ──
print("=" * 60)
print("2019 SNAPSHOT")
print("=" * 60)
pca_2019, pca_model_2019, feat_2019 = run_clustering(
    nr_sample, year_filter=2019
)
results["2019"] = pca_2019
print()

# ── Aggregated (1995, 1999, 2005) ──
print("=" * 60)
print("AGGREGATED (1995, 1999, 2005)")
print("=" * 60)
pca_agg, pca_model_agg, feat_agg = run_clustering(
    nr_sample, agg_years=[1995, 1999, 2005]
)
results["agg"] = pca_agg

# ## 5. PCA Loadings Analysis
# 
# The loadings reveal which resources drive each principal component. PC1 is expected to capture hydrocarbon abundance (oil, natural gas), while PC2 should capture mineral production (copper, gold, zinc, etc.). This interpretation is central to the cluster labelling logic.

# Use 1995 model for loadings analysis (consistent with report)
loadings = pd.DataFrame(
    pca_model_1995.components_.T,
    columns=["PC1", "PC2"],
    index=feat_1995,
)

print("PCA Explained Variance (1995):")
for i, var in enumerate(pca_model_1995.explained_variance_ratio_):
    cum = pca_model_1995.explained_variance_ratio_[:i + 1].sum()
    print(f"  PC{i+1}: {var*100:.1f}% (cumulative: {cum*100:.1f}%)")

# Loadings heatmap (top 20 by overall importance)
top_features = loadings.abs().sum(axis=1).nlargest(20).index

fig_load = px.imshow(
    loadings.loc[top_features].T,
    labels=dict(x="Resource", y="Principal Component", color="Loading"),
    title="PCA Factor Loadings (Top 20 Resources, 1995)",
    color_continuous_scale="RdBu_r",
    aspect="auto", zmin=-1, zmax=1,
)
fig_load.update_layout(width=1100, height=350)
# loadings heatmap handled by viz_updates.py (chart 26)

print("\nTop loadings by component:")
for pc in ["PC1", "PC2"]:
    print(f"\n{pc}:")
    sorted_l = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
    for feat, val in sorted_l.head(8).items():
        print(f"  {feat:35s} {val:+.4f}")

# ## 6. Biplot: PCA Space with Cluster Assignments
# 
# The biplot overlays cluster assignments onto the PCA space, with arrows showing the direction and strength of each resource's contribution to the principal components. This is Figure 2 in the report.

def create_biplot(pca_df, pca_model, feature_cols, title_suffix=""):
    """PCA biplot — top-5 bold arrows, top-10 grey arrows, one trace per cluster."""

    loadings_df = pd.DataFrame(
        pca_model.components_.T * np.sqrt(pca_model.explained_variance_),
        columns=["PC1", "PC2"], index=feature_cols,
    )
    importance = loadings_df.abs().sum(axis=1)
    top5  = importance.nlargest(5).index
    top10 = importance.nlargest(10).index
    scale = 2.8

    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100

    fig = go.Figure()

    # One scatter trace per cluster
    for cid in sorted(pca_df["Cluster"].unique()):
        sub   = pca_df[pca_df["Cluster"] == cid]
        lbl   = sub["ClusterLabels"].iloc[0]
        color = _LABEL_COLORS.get(lbl, "#999")
        fig.add_trace(go.Scatter(
            x=sub["PC1"], y=sub["PC2"],
            mode="markers+text",
            marker=dict(size=10, color=color, opacity=0.82,
                        line=dict(width=1.2, color="white")),
            text=sub["Country Code"],
            textposition="top center",
            textfont=dict(size=8, color="#333"),
            name=lbl,
            hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f}, PC2=%{y:.2f}<extra></extra>",
        ))

    # Thin grey arrows — ranks 6–10
    for feat_name in top10:
        if feat_name in top5:
            continue
        fig.add_annotation(
            x=loadings_df.loc[feat_name, "PC1"] * scale,
            y=loadings_df.loc[feat_name, "PC2"] * scale,
            ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowwidth=1.2, arrowcolor="rgba(150,150,150,0.5)",
        )

    # Bold black arrows + labels — top 5
    for feat_name in top5:
        x1 = loadings_df.loc[feat_name, "PC1"] * scale
        y1 = loadings_df.loc[feat_name, "PC2"] * scale
        fig.add_annotation(
            x=x1, y=y1, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=2.2, arrowcolor="#222",
        )
        fig.add_annotation(
            x=x1 * 1.18, y=y1 * 1.18,
            text=f"<b>{feat_name}</b>", showarrow=False,
            font=dict(size=10, color="#111", family=FONT),
            bgcolor="rgba(255,255,255,0.7)", borderpad=2,
        )

    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    fig.add_vline(x=0, line=dict(color=GRID, width=1))

    fig.update_layout(**base_layout(
        height=680,
        margin=dict(l=60, r=60, t=50, b=60),
        xaxis=dict(title=f"PC1 ({var1:.1f}% variance explained)",
                   gridcolor=GRID, gridwidth=0.5),
        yaxis=dict(title=f"PC2 ({var2:.1f}% variance explained)",
                   gridcolor=GRID, gridwidth=0.5),
        legend=dict(title="Resource profile (1995)", font=dict(size=10),
                    bgcolor="rgba(250,250,250,0.85)",
                    bordercolor=GRID, borderwidth=1),
    ))
    return fig


fig_biplot = create_biplot(pca_1995, pca_model_1995, feat_1995, " — 1995")
_save(fig_biplot, os.path.join(CHARTS, '03_cluster__pca_biplot_country_resource_groups.html'))

# ## 7. Choropleth Map with Dominance Flags
# 
# Countries producing more than 15% of global output for any single resource are flagged with a red border. This highlights major producers (e.g. Chile for copper, Saudi Arabia for oil) whose resource profiles have global significance.

def create_cluster_map(pca_df, nr_data, cluster_names_map=None, dominance_threshold=15.0):
    """Choropleth map with red borders for major global producers."""

    if cluster_names_map is None:
        cluster_names_map = dict(
            zip(pca_df["Cluster"].unique(), pca_df["ClusterLabels"].unique())
        )

    # ── Global production shares ──
    df_total = nr_data.pivot_table(
        index=["Country", "Country Code"],
        columns="Resource",
        values="Production_TotalValue",
        aggfunc="sum",
    ).reset_index().fillna(0)

    prod_cols = [c for c in df_total.columns if c not in ["Country", "Country Code"]]
    for col in prod_cols:
        total = df_total[col].sum()
        if total > 0:
            df_total[f"{col}_Share"] = (df_total[col] / total) * 100

    share_cols = [c for c in df_total.columns if c.endswith("_Share")]

    # Merge shares into pca_df
    df_map = pca_df.merge(df_total[["Country Code"] + share_cols], on="Country Code", how="left")

    # Flag dominant producers (vectorised)
    df_map["Is_Dominant"] = (df_map[share_cols] >= dominance_threshold).any(axis=1)
    df_map["Dominant_Resources"] = df_map.apply(
        lambda row: [
            sc.replace("_Share", "")
            for sc in share_cols
            if row.get(sc, 0) >= dominance_threshold
        ],
        axis=1,
    )

    # ── Hover text ──
    def make_hover(row):
        lbl = row["ClusterLabels"]
        lines = [f"<b>{row['Country']}</b>", f"Cluster: {lbl}"]
        # Top resources
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

    # ── Build map ──
    fig = go.Figure()

    for cid in sorted(df_map["Cluster"].unique()):
        lbl   = cluster_names_map.get(cid, f"Cluster {cid}")
        color = _LABEL_COLORS.get(lbl, '#aaa')

        # Non-dominant countries
        sub = df_map[(df_map["Cluster"] == cid) & (~df_map["Is_Dominant"])]
        if len(sub) > 0:
            fig.add_trace(go.Choropleth(
                locations=sub["Country Code"], z=[cid]*len(sub),
                colorscale=[[0, color], [1, color]], showscale=False,
                showlegend=True,
                customdata=sub["hover_text"].values,
                hovertemplate="%{customdata}<extra></extra>",
                name=lbl,
                marker=dict(line=dict(color="white", width=0.6)),
            ))

        # Dominant producers — thick dark border
        sub_d = df_map[(df_map["Cluster"] == cid) & (df_map["Is_Dominant"])]
        if len(sub_d) > 0:
            fig.add_trace(go.Choropleth(
                locations=sub_d["Country Code"], z=[cid]*len(sub_d),
                colorscale=[[0, color], [1, color]], showscale=False,
                showlegend=False,           # same cluster — don't duplicate legend entry
                customdata=sub_d["hover_text"].values,
                hovertemplate="%{customdata}<extra></extra>",
                name=f"{lbl} ★ major producer",
                marker=dict(line=dict(color="#111", width=2.2)),
            ))

    # Dummy legend entry explaining the black border
    fig.add_trace(go.Choropleth(
        locations=["ZZZ"], z=[0],
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        showscale=False, showlegend=True,
        name="★ >15% of global output",
        marker=dict(line=dict(color="#111", width=2.2)),
    ))

    fig.update_geos(
        projection_type="natural earth",
        showcountries=True, countrycolor="#ccc",
        showcoastlines=True, coastlinecolor="#ccc",
        showland=True, landcolor="#f0f0f0",
        showocean=True, oceancolor="#dde8f0",
        showframe=False,
    )
    fig.update_layout(
        width=1200, height=520,
        margin=dict(l=0, r=0, t=50, b=70),
        legend=dict(
            orientation="h",
            x=0.5, y=-0.08, xanchor="center", yanchor="top",
            font=dict(size=11, family="IBM Plex Sans"),
            bgcolor="rgba(250,250,250,0.9)",
            bordercolor="#d0d0d0", borderwidth=1,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="IBM Plex Sans"),
    )
    return fig


# Generate maps for 1995, 2014, and aggregated
for map_label, pca_df_map, nr_year_filter, fname in [
    ("1995", pca_1995, 1995, '04a_cluster__world_map_1995_resource_profiles'),
    ("2019", pca_2019, 2019, '04b_cluster__world_map_2019_resource_profiles'),
    ("agg",  pca_agg,  None, '04c_cluster__world_map_agg_resource_profiles'),
]:
    nr_subset = nr_sample[nr_sample["Year"] == nr_year_filter] if nr_year_filter else nr_sample
    cluster_names_map = dict(zip(pca_df_map["Cluster"], pca_df_map["ClusterLabels"]))
    fig_map = create_cluster_map(pca_df_map, nr_subset, cluster_names_map=cluster_names_map)
    _save(fig_map, os.path.join(CHARTS, f'{fname}.html'))

# ## 8. ECI vs GDP Evolution (Rosling Chart)
# 
# This animated chart tracks how each country's Economic Complexity Index and GDP per capita evolved from 1995 to 2019. Countries are coloured by their 1995 cluster assignment (fixed throughout the animation), with bubble size proportional to production per capita. Arrows trace each country's trajectory from its 1995 starting position. This is Figure 4 in the report.

master = pd.read_csv("intermediary/Master.csv")
master = master[master["Country Code"].isin(include_list)]

# Merge cluster assignments (from aggregated clustering)
master = pd.merge(
    master,
    pca_agg[["Country Code", "Cluster", "ClusterLabels"]],
    on="Country Code",
    how="left",
)

CLUSTER_COLORS = {
    cid: _LABEL_COLORS.get(
        pca_agg.loc[pca_agg["Cluster"] == cid, "ClusterLabels"].iloc[0], '#aaa'
    )
    for cid in sorted(pca_agg["Cluster"].unique())
}

CLUSTER_NAMES = dict(zip(pca_agg["Cluster"], pca_agg["ClusterLabels"]))

def create_rosling_chart(df, cluster_colors, cluster_names, arrow_opacity=0.5, arrow_width=2):
    """Animated ECI vs log(GDP pc) chart with trajectory arrows from 1995."""

    data = df.copy()
    data["Log GDP per capita"] = np.log(data["GDP per capita (constant prices, PPP)"])
    data["Production_Per_Capita"] = data["Total_Production_Value"] / data["Population"]

    # Fix cluster to 1995 value
    c1995 = data[data["Year"] == 1995][["Country Code", "Cluster"]].copy()
    c1995 = c1995.rename(columns={"Cluster": "Cluster_1995"})
    data = data.merge(c1995, on="Country Code", how="left")
    data = data.dropna(subset=["Cluster_1995", "Log GDP per capita",
                                "Economic Complexity Index", "Production_Per_Capita"])
    data["Cluster_1995"] = data["Cluster_1995"].astype(int)

    # Bubble size
    data["Bubble_Size"] = np.sqrt(data["Production_Per_Capita"])
    mn, mx = data["Bubble_Size"].min(), data["Bubble_Size"].max()
    data["Bubble_Size_Scaled"] = 8 + (data["Bubble_Size"] - mn) / (mx - mn) * 42

    data = data.sort_values(["Year", "Country Code"])
    years = sorted(data["Year"].unique())
    countries_list = data["Country Code"].unique()
    clusters = sorted(data["Cluster_1995"].unique())

    # Build per-country data
    cdata = {}
    for code in countries_list:
        cdf = data[data["Country Code"] == code].sort_values("Year")
        origin = cdf[cdf["Year"] == 1995]
        if len(origin) == 0:
            continue
        cdata[code] = {
            "years": cdf["Year"].values,
            "x": cdf["Log GDP per capita"].values,
            "y": cdf["Economic Complexity Index"].values,
            "x0": origin["Log GDP per capita"].values[0],
            "y0": origin["Economic Complexity Index"].values[0],
            "size": cdf["Bubble_Size_Scaled"].values,
            "name": cdf["Country Name"].iloc[0],
            "cluster": cdf["Cluster_1995"].iloc[0],
            "prod_pc": cdf["Production_Per_Capita"].values,
        }

    valid_countries = list(cdata.keys())
    first_year = years[0]

    fig = go.Figure()

    for cl in clusters:
        cc = [c for c in valid_countries if cdata[c]["cluster"] == cl]
        color = cluster_colors.get(cl, "#999999")

        for code in cc:
            cd = cdata[code]
            idx = np.where(cd["years"] == first_year)[0]
            xc = cd["x"][idx[0]] if len(idx) > 0 else cd["x0"]
            yc = cd["y"][idx[0]] if len(idx) > 0 else cd["y0"]
            fig.add_trace(go.Scatter(
                x=[cd["x0"], xc], y=[cd["y0"], yc],
                mode="lines", line=dict(color=color, width=arrow_width),
                opacity=arrow_opacity, legendgroup=f"cl_{cl}", showlegend=False, hoverinfo="skip",
            ))

        for code in cc:
            cd = cdata[code]
            idx = np.where(cd["years"] == first_year)[0]
            if len(idx) > 0:
                i = idx[0]
                xv, yv, sv, pv = [cd["x"][i]], [cd["y"][i]], cd["size"][i], cd["prod_pc"][i]
            else:
                xv, yv, sv, pv = [cd["x0"]], [cd["y0"]], 15, 0
            fig.add_trace(go.Scatter(
                x=xv, y=yv, mode="markers+text",
                marker=dict(size=sv, color=color, opacity=0.85, line=dict(width=1.5, color="white")),
                text=[code], textposition="top center", textfont=dict(size=8, color="black"),
                name=cluster_names.get(cl, f"Cluster {cl}"),
                legendgroup=f"cl_{cl}", showlegend=(code == cc[0]),
                customdata=[[cd["name"], pv, first_year]],
                hovertemplate="<b>%{customdata[0]}</b><br>Log GDP pc: %{x:.2f}<br>"
                              "ECI: %{y:.2f}<br>Prod/capita: $%{customdata[1]:,.0f}<br>"
                              "Year: %{customdata[2]}<extra></extra>",
            ))

        for code in cc:
            cd = cdata[code]
            fig.add_trace(go.Scatter(
                x=[cd["x0"]], y=[cd["y0"]], mode="markers",
                marker=dict(size=5, color=color, opacity=0.6, symbol="circle"),
                legendgroup=f"cl_{cl}", showlegend=False, hoverinfo="skip",
            ))

    # Frames for animation
    frames = []
    for year in years:
        fd = []
        for cl in clusters:
            cc = [c for c in valid_countries if cdata[c]["cluster"] == cl]
            color = cluster_colors.get(cl, "#999999")
            for code in cc:
                cd = cdata[code]
                idx = np.where(cd["years"] == year)[0]
                if len(idx) > 0:
                    xc, yc = cd["x"][idx[0]], cd["y"][idx[0]]
                else:
                    mask = cd["years"] <= year
                    li = np.where(mask)[0][-1] if mask.any() else 0
                    xc, yc = cd["x"][li], cd["y"][li]
                fd.append(go.Scatter(x=[cd["x0"], xc], y=[cd["y0"], yc],
                                     mode="lines", line=dict(color=color, width=arrow_width), opacity=arrow_opacity))
            for code in cc:
                cd = cdata[code]
                idx = np.where(cd["years"] == year)[0]
                if len(idx) > 0:
                    i = idx[0]
                    xv, yv, sv, pv = [cd["x"][i]], [cd["y"][i]], cd["size"][i], cd["prod_pc"][i]
                else:
                    mask = cd["years"] <= year
                    if mask.any():
                        li = np.where(mask)[0][-1]
                        xv, yv, sv, pv = [cd["x"][li]], [cd["y"][li]], cd["size"][li], cd["prod_pc"][li]
                    else:
                        xv, yv, sv, pv = [cd["x0"]], [cd["y0"]], 15, 0
                fd.append(go.Scatter(
                    x=xv, y=yv, mode="markers+text",
                    marker=dict(size=sv, color=color, opacity=0.85, line=dict(width=1.5, color="white")),
                    text=[code], textposition="top center", textfont=dict(size=8),
                    customdata=[[cd["name"], pv, year]],
                    hovertemplate="<b>%{customdata[0]}</b><br>Log GDP pc: %{x:.2f}<br>"
                                  "ECI: %{y:.2f}<br>Prod/capita: $%{customdata[1]:,.0f}<br>"
                                  "Year: %{customdata[2]}<extra></extra>",
                ))
            for code in cc:
                cd = cdata[code]
                fd.append(go.Scatter(x=[cd["x0"]], y=[cd["y0"]], mode="markers",
                                     marker=dict(size=5, color=color, opacity=0.6, symbol="circle")))
        frames.append(go.Frame(data=fd, name=str(year)))

    fig.frames = frames

    eci_vals = data["Economic Complexity Index"]
    x_vals = data["Log GDP per capita"]
    fig.update_layout(
        xaxis=dict(range=[x_vals.min()-0.2, x_vals.max()+0.2], title="Log GDP per capita (PPP, constant 2017 USD)"),
        yaxis=dict(range=[eci_vals.min()-0.5, eci_vals.max()+0.5], title="Economic Complexity Index"),
        plot_bgcolor="white", width=850, height=650,
        legend=dict(title="Resource profile (1995 cluster)", x=1.02, y=0.99),
        updatemenus=[dict(
            type="buttons", showactive=True, x=1.0, y=-0.02,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=500, redraw=True), transition=dict(duration=300))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, len=0.85, x=0.05, y=-0.12,
            currentvalue=dict(prefix="Year: ", font=dict(size=14)),
            steps=[dict(args=[[str(y)], dict(frame=dict(duration=300, redraw=True), mode="immediate")],
                        method="animate", label=str(y)) for y in years],
        )],
    )
    return fig


fig_rosling = create_rosling_chart(master, CLUSTER_COLORS, CLUSTER_NAMES)
_save(fig_rosling, os.path.join(CHARTS, '05_cluster__eci_vs_gdp_animated_1995_to_2019.html'), width=1200, height=700)

# ## 9. Add Hover Text and Export
# 
# The cluster CSVs include hover text with top resources for each country, matching the format used in the original analysis files.

def add_hover_text(pca_df, nr_data):
    """Add hover text with top resources to cluster DataFrame."""
    # Get total production by country and resource
    totals = nr_data.pivot_table(
        index=["Country", "Country Code"],
        columns="Resource",
        values="Production_TotalValue",
        aggfunc="sum",
    ).reset_index().fillna(0)

    prod_cols = [c for c in totals.columns if c not in ["Country", "Country Code"]]

    df = pca_df.merge(totals, on="Country Code", how="left", suffixes=("", "_nr"))

    def make_ht(row):
        lines = [f"<b>{row['Country']}</b>",
                 f"Cluster: {row['Cluster']}"]
        lines.append("<br>Top Resources:")
        vals = [(c, row.get(c, 0)) for c in prod_cols if row.get(c, 0) > 0]
        vals.sort(key=lambda x: x[1], reverse=True)
        for res, v in vals[:3]:
            lines.append(f"  {res}: ${v:,.0f}")
        return "<br>".join(lines)

    df["hover_text"] = df.apply(make_ht, axis=1)
    return df[["Country", "Country Code", "Year", "PC1", "PC2",
               "Cluster", "ClusterLabels", "hover_text"]]


# Export each variant
for label, df, nr_subset in [
    ("1995", results["1995"], nr_sample[nr_sample["Year"] == 1995]),
    ("2019", results["2019"], nr_sample[nr_sample["Year"] == 2019]),
    ("agg",  results["agg"],  nr_sample[nr_sample["Year"].isin([1995, 1999, 2005])]),
]:
    out = add_hover_text(df, nr_subset)
    out_path = f"intermediary/clusters{label}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}: {len(out)} countries")

print("\nDone. All cluster CSVs exported.")

# ## 10. Simplified 4-Feature Clustering Map (Oil / Gas / Coal / Minerals)
#
# Re-clusters countries using only four aggregated features:
#   Oil, Natural Gas, Coal, Minerals (all other resources summed).
# This gives a simpler, more interpretable typology for the map.

def run_clustering_4feat(nr_data, year_filter=None, n_clusters=5, random_state=42):
    """Cluster on Oil / Natural Gas / Coal / Minerals (aggregate), k=5.

    k=5 is used because with aggregated features, k=4 can't separate
    'no oil, high minerals' (CHL, ZMB, MNG) from 'no oil, low minerals'.
    k=5 cleanly recovers this group (silhouette 0.522 vs 0.449 for k=4).
    """
    df = nr_data.copy()
    if year_filter is not None:
        df = df[df["Year"] == year_filter]

    KEEP = ["Oil", "Natural Gas", "Coal"]
    df["_Category"] = df["Resource"].apply(lambda r: r if r in KEEP else "Minerals")
    df_agg = (df.groupby(["Country", "Country Code", "Year", "Population", "_Category"])
              ["Production_TotalValue"].sum().reset_index())

    pivot = df_agg.pivot_table(
        index=["Country", "Country Code", "Year", "Population"],
        columns="_Category", values="Production_TotalValue",
    ).reset_index().fillna(0)

    feat_cols = ["Coal", "Minerals", "Natural Gas", "Oil"]
    feat_cols = [c for c in feat_cols if c in pivot.columns]

    pivot[feat_cols] = pivot[feat_cols].div(pivot["Population"], axis=0)
    pivot = pivot.fillna(0)

    X = np.log1p(pivot[feat_cols])
    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(Xp)

    pca_df = pd.DataFrame({
        "Country":      pivot["Country"],
        "Country Code": pivot["Country Code"],
        "Year":         pivot["Year"],
        "PC1": Xp[:, 0], "PC2": Xp[:, 1],
        "Cluster": labels,
    })

    # Auto-label from centroids:
    #   highest PC1          → Petrostates
    #   highest PC2 (remain) → Mineral Exporters  (CHL, ZMB, MNG etc.)
    #   2nd highest PC1      → Oil Exporters
    #   3rd highest PC1      → Oil & Minerals Mix
    #   lowest PC1           → Low Resource
    centroids = km.cluster_centers_
    pc1_rank  = list(np.argsort(-centroids[:, 0]))
    pc2_rank  = list(np.argsort(-centroids[:, 1]))

    label_map, labeled = {}, set()
    # 1. Highest PC1 → Petrostates
    label_map[pc1_rank[0]] = "Petrostates";        labeled.add(pc1_rank[0])
    # 2. Highest PC2 (not yet labeled) → Mineral Exporters (CHL, ZMB, MNG etc.)
    min_id = next(c for c in pc2_rank if c not in labeled)
    label_map[min_id] = "Mineral Exporters";        labeled.add(min_id)
    # 3. Remaining three sorted by PC1 (descending):
    #    highest PC1 → Oil & Minerals, middle → Oil Exporters, lowest → Low Resource
    remaining = [c for c in pc1_rank if c not in labeled]  # already in descending PC1 order
    label_map[remaining[0]] = "Oil & Minerals"
    label_map[remaining[1]] = "Oil Exporters"
    label_map[remaining[2]] = "Low Resource"

    pca_df["ClusterLabels"] = pca_df["Cluster"].map(label_map)

    sil = silhouette_score(Xp, labels)
    print(f"  4-feat silhouette (k={n_clusters}): {sil:.3f}")
    for cid in sorted(pca_df["Cluster"].unique()):
        sub = pca_df[pca_df["Cluster"] == cid]
        print(f"  {label_map[cid]} ({len(sub)}): {', '.join(sorted(sub['Country Code']))}")

    return pca_df

_LABEL_COLORS_4F = {
    "Petrostates":      "#d4853b",
    "Oil Exporters":    "#c23a3a",
    "Oil & Minerals":   "#7a5c9e",
    "Mineral Exporters":"#2e7d4a",
    "Low Resource":     "#4a6fa5",
}

print("\n" + "="*60)
print("4-FEATURE CLUSTERING (Oil / Gas / Coal / Minerals)")
print("="*60)
pca_4f = run_clustering_4feat(nr_sample, year_filter=1995)

nr_1995_full = nr_sample[nr_sample["Year"] == 1995]
cnames_4f    = dict(zip(pca_4f["Cluster"], pca_4f["ClusterLabels"]))

# Build map using same create_cluster_map but override palette via monkey-patch
_orig_label_colors = _LABEL_COLORS.copy()
_LABEL_COLORS.update(_LABEL_COLORS_4F)
fig_4f = create_cluster_map(pca_4f, nr_1995_full, cluster_names_map=cnames_4f)
_LABEL_COLORS.clear(); _LABEL_COLORS.update(_orig_label_colors)  # restore

_save(fig_4f, os.path.join(CHARTS, '04d_cluster__world_map_4feat_oil_gas_coal_minerals.html'))

# ---
# 
# ## Summary
# 
# | Component | Choice | Rationale |
# |-----------|--------|-----------|
# | **Normalisation** | Per capita | Separates resource abundance from economic structure |
# | **Transform** | log(1+x) | Compresses extreme skew; preserves zeros |
# | **Dim. reduction** | PCA, 2 components | Removes correlated noise; produces interpretable biplot |
# | **Clustering** | K-Means, k=4, random_state=42 | Validated with silhouette analysis; reproducible |
# | **Labelling** | Automatic from centroids | Robust to arbitrary cluster numbering |
# | **Dominance** | >15% of global production | Flags major producers (Chile/copper, Saudi/oil) |
# | **Variants** | 1995, 2019, aggregated | Captures both structure and temporal shifts |
# 
# **Pipeline position:** Step 4 of 5:  
# `1_cleaning` -> `2_MissingCheck` -> `3_Imputing` -> **`4_Clustering`** -> `5_ML_models`

# ── NB4 SUMMARY ──
print("=" * 70)
print("NB4: CLUSTERING SUMMARY")
print("=" * 70)

print(f"\nPCA variance explained (1995): PC1={pca_model_1995.explained_variance_ratio_[0]*100:.1f}%, PC2={pca_model_1995.explained_variance_ratio_[1]*100:.1f}%")

for label, df in results.items():
    print(f"\n--- {label} ({len(df)} countries) ---")
    for cid in sorted(df['Cluster'].unique()):
        sub = df[df['Cluster'] == cid]
        lbl = sub['ClusterLabels'].iloc[0]
        codes = ', '.join(sorted(sub['Country Code'].tolist()))
        print(f"  {lbl} (n={len(sub)}): {codes}")

print(f"\nSaved:")
for label in results:
    print(f"  intermediary/clusters{label}.csv")
