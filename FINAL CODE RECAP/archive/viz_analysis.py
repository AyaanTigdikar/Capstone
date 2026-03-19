# ============================================================
# ANALYSIS VISUALISATIONS
# Clustering, ML models, regressions
# Sources: 4_Clustering_FINAL, 5_ML_FINAL, 6_Regressions_Unified
# ============================================================

# --- SHARED SETUP ---
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as stats
from types import SimpleNamespace
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ── Shared style (NB5/NB6) ───────────────────────────────────────────────────
STYLE = {
    'font_family':       'IBM Plex Sans, -apple-system, BlinkMacSystemFont, sans-serif',
    'tick_size':         11,
    'axis_title_size':   13,
    'legend_size':       11,
    'annotation_size':   11,
    'title_color':       '#1a2744',
    'template':          'plotly_white',
    'plot_bg':           '#fafafa',
    'paper_bg':          '#fafafa',
    'chart_height':      550,
    'chart_height_small':420,
    'chart_height_tall': 700,
    'margin':            dict(l=60,  r=40,  t=10, b=50),
    'margin_bar':        dict(l=160, r=130, t=10, b=50),
    'grid_color':        '#e5e7eb',
    'grid_width':        0.5,
    'zero_line_color':   '#c9cfd6',
}
PALETTE = {
    'blue':        '#4a6fa5',
    'red':         '#c23a3a',
    'green':       '#2e7d4a',
    'orange':      '#d4853b',
    'light_blue':  '#7a9dc4',
    'light_red':   '#d46b6b',
    'light_green': '#5aa87a',
    'dark':        '#3d4f5f',
    'grey':        '#999999',
    'gold':        '#e6b980',
}
CLUSTER_COLORS = ['#4a6fa5', '#c23a3a', '#2e7d4a', '#d4853b']
WRITE_CONFIG = {'displayModeBar': False, 'responsive': True}

plt.rcParams.update({
    'font.family':       'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        120,
})


def base_layout(**kwargs) -> dict:
    layout = dict(
        template=STYLE['template'],
        plot_bgcolor=STYLE['plot_bg'],
        paper_bgcolor=STYLE['paper_bg'],
        font=dict(family=STYLE['font_family'], size=STYLE['tick_size'],
                  color=STYLE['title_color']),
        margin=STYLE['margin'],
        height=STYLE['chart_height'],
    )
    layout.update(kwargs)
    return layout


def save_chart(fig, path_no_ext: str, width: int = 1100, height: int = 700):
    """Show inline, write .html, attempt .png via kaleido/orca."""
    fig.show(config=WRITE_CONFIG)
    fig.write_html(f"{path_no_ext}.html", config=WRITE_CONFIG)
    print(f"  checkmark {path_no_ext}.html")
    for engine in ['kaleido', 'orca']:
        try:
            fig.write_image(f"{path_no_ext}.png", width=width, height=height,
                            scale=3, engine=engine)
            print(f"  checkmark {path_no_ext}.png ({engine})")
            return
        except Exception:
            pass
    print("  PNG skipped (kaleido not available).")


# ================================================================
# PART A: CLUSTERING (4_Clustering_FINAL.ipynb)
# ================================================================

os.makedirs("intermediary", exist_ok=True)
os.makedirs("Final/NB4", exist_ok=True)

INCLUDE_LIST = [
    'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
    'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
    'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
    'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
    'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
    'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
]


def run_clustering(nr_data, year_filter=None, agg_years=None, n_clusters=4, random_state=42):
    """
    Full clustering pipeline: pivot -> per capita -> log1p -> PCA(2) -> KMeans(k).
    """
    df = nr_data.copy()
    if year_filter is not None:
        df = df[df["Year"] == year_filter]
    elif agg_years is not None:
        df = df[df["Year"].isin(agg_years)]

    df_pivot = df.pivot_table(
        index=["Country", "Country Code", "Year", "Population"],
        columns="Resource",
        values="Production_TotalValue",
    ).reset_index()

    resource_cols = df_pivot.columns.difference(
        ["Country", "Country Code", "Year", "Population"]
    )

    df_pivot[resource_cols] = df_pivot[resource_cols].div(
        df_pivot["Population"], axis=0
    )
    df_pivot.drop(columns="Population", inplace=True)
    df_pivot = df_pivot.fillna(0)

    df_latest = (
        df_pivot.sort_values("Year", ascending=True)
        .groupby(["Country", "Country Code"])
        .first()
        .reset_index()
    )

    feature_cols = [c for c in df_latest.columns if c not in ["Country", "Country Code", "Year"]]

    X = df_latest[feature_cols].fillna(0)
    X_log = np.log1p(X)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_log)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    clusters = kmeans.fit_predict(pca_components)

    pca_df = pd.DataFrame({
        "Country": df_latest["Country"],
        "Country Code": df_latest["Country Code"],
        "Year": df_latest["Year"],
        "PC1": pca_components[:, 0],
        "PC2": pca_components[:, 1],
        "Cluster": clusters,
    })

    centroids = kmeans.cluster_centers_
    pc1_rank = list(np.argsort(-centroids[:, 0]))
    pc2_rank = list(np.argsort(-centroids[:, 1]))

    label_map = {}
    labeled = set()

    oil_id = pc1_rank[0]
    label_map[oil_id] = "Oil, Few Minerals"
    labeled.add(oil_id)

    mineral_id = next(c for c in pc2_rank if c not in labeled)
    label_map[mineral_id] = "Minerals, No Oil"
    labeled.add(mineral_id)

    remaining = [c for c in pc1_rank if c not in labeled]
    label_map[remaining[0]] = "Some Oil, No Minerals"
    label_map[remaining[1]] = "No Oil, No Minerals"

    pca_df["ClusterLabels"] = pca_df["Cluster"].map(label_map)

    sil = silhouette_score(pca_components, clusters)
    print(f"Silhouette score: {sil:.3f}")

    return pca_df, pca, feature_cols


# --- SECTION: Silhouette Analysis ---

# Silhouette score by number of clusters (k=2..8) to validate k=4
def plot_silhouette_validation():
    nr = pd.read_csv("intermediary/NaturalResource.csv")
    nr_sample = nr[nr["Country Code"].isin(INCLUDE_LIST)]

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

    k_range = range(2, 9)
    sil_scores = []
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_pca_val)
        sil_scores.append(silhouette_score(X_pca_val, labels))
        inertias.append(km.inertia_)

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
    fig_val.show()

    print("\nSilhouette scores:")
    for k, s in zip(k_range, sil_scores):
        marker = " <-- selected" if k == 4 else ""
        print(f"  k={k}: {s:.3f}{marker}")


# --- SECTION: PCA Loadings ---

# Interactive heatmap showing PCA factor loadings for top 20 resources (1995)
def plot_pca_loadings():
    nr = pd.read_csv("intermediary/NaturalResource.csv")
    nr_sample = nr[nr["Country Code"].isin(INCLUDE_LIST)]

    pca_1995, pca_model_1995, feat_1995 = run_clustering(nr_sample, year_filter=1995)

    loadings = pd.DataFrame(
        pca_model_1995.components_.T,
        columns=["PC1", "PC2"],
        index=feat_1995,
    )

    print("PCA Explained Variance (1995):")
    for i, var in enumerate(pca_model_1995.explained_variance_ratio_):
        cum = pca_model_1995.explained_variance_ratio_[:i + 1].sum()
        print(f"  PC{i+1}: {var*100:.1f}% (cumulative: {cum*100:.1f}%)")

    top_features = loadings.abs().sum(axis=1).nlargest(20).index

    fig_load = px.imshow(
        loadings.loc[top_features].T,
        labels=dict(x="Resource", y="Principal Component", color="Loading"),
        title="PCA Factor Loadings (Top 20 Resources, 1995)",
        color_continuous_scale="RdBu_r",
        aspect="auto", zmin=-1, zmax=1,
    )
    fig_load.update_layout(width=1100, height=350)
    fig_load.show()

    print("\nTop loadings by component:")
    for pc in ["PC1", "PC2"]:
        print(f"\n{pc}:")
        sorted_l = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
        for feat, val in sorted_l.head(8).items():
            print(f"  {feat:35s} {val:+.4f}")


# --- SECTION: PCA Biplot ---

# Biplot: PCA space with cluster assignments and loading arrows (1995)
def plot_pca_biplot():
    nr = pd.read_csv("intermediary/NaturalResource.csv")
    nr_sample = nr[nr["Country Code"].isin(INCLUDE_LIST)]

    pca_1995, pca_model_1995, feat_1995 = run_clustering(nr_sample, year_filter=1995)

    def create_biplot(pca_df, pca_model, feature_cols, title_suffix=""):
        """Create PCA biplot with cluster colours and loading arrows."""
        loadings_plot = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)
        loadings_df = pd.DataFrame(loadings_plot[:, :2], columns=["PC1", "PC2"], index=feature_cols)
        scale_factor = 2.5
        loadings_scaled = loadings_df * scale_factor

        importance = loadings_df.abs().sum(axis=1)
        top_n = min(15, len(feature_cols))
        top_feats = importance.nlargest(top_n).index

        fig = px.scatter(
            pca_df, x="PC1", y="PC2",
            color="ClusterLabels",
            hover_data=["Country", "Country Code", "Year"],
            title=f"PCA Biplot (k=4){title_suffix}",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )

        for feat in top_feats:
            fig.add_annotation(
                x=loadings_scaled.loc[feat, "PC1"],
                y=loadings_scaled.loc[feat, "PC2"],
                ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black",
            )
            fig.add_annotation(
                x=loadings_scaled.loc[feat, "PC1"] * 1.15,
                y=loadings_scaled.loc[feat, "PC2"] * 1.15,
                text=feat, showarrow=False,
                font=dict(size=9, color="black"),
            )

        var1 = pca_model.explained_variance_ratio_[0] * 100
        var2 = pca_model.explained_variance_ratio_[1] * 100
        fig.update_layout(
            width=1000, height=700,
            xaxis_title=f"PC1 ({var1:.1f}%)",
            yaxis_title=f"PC2 ({var2:.1f}%)",
        )
        return fig

    fig_biplot = create_biplot(pca_1995, pca_model_1995, feat_1995, " — 1995")
    fig_biplot.show()


# --- SECTION: Choropleth Cluster Map ---

# Choropleth map of cluster assignments with red borders for major global producers
def plot_cluster_choropleth():
    nr = pd.read_csv("intermediary/NaturalResource.csv")
    nr_sample = nr[nr["Country Code"].isin(INCLUDE_LIST)]

    pca_1995, pca_model_1995, feat_1995 = run_clustering(nr_sample, year_filter=1995)

    def create_cluster_map(pca_df, nr_data, cluster_names_map=None, dominance_threshold=15.0):
        """Choropleth map with red borders for major global producers."""
        if cluster_names_map is None:
            cluster_names_map = dict(
                zip(pca_df["Cluster"].unique(), pca_df["ClusterLabels"].unique())
            )

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

        df_map = pca_df.merge(df_total[["Country Code"] + share_cols], on="Country Code", how="left")
        df_map["Is_Dominant"] = (df_map[share_cols] >= dominance_threshold).any(axis=1)
        df_map["Dominant_Resources"] = df_map.apply(
            lambda row: [
                sc.replace("_Share", "")
                for sc in share_cols
                if row.get(sc, 0) >= dominance_threshold
            ],
            axis=1,
        )

        def make_hover(row):
            lbl = row["ClusterLabels"]
            lines = [f"<b>{row['Country']}</b>", f"Cluster: {lbl}"]
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

        colors = px.colors.qualitative.Bold
        fig = go.Figure()

        for cid in sorted(df_map["Cluster"].unique()):
            lbl = cluster_names_map.get(cid, f"Cluster {cid}")
            color = colors[cid % len(colors)]

            sub = df_map[(df_map["Cluster"] == cid) & (~df_map["Is_Dominant"])]
            if len(sub) > 0:
                fig.add_trace(go.Choropleth(
                    locations=sub["Country Code"], z=[cid]*len(sub),
                    colorscale=[[0, color], [1, color]], showscale=False,
                    customdata=sub["hover_text"].values,
                    hovertemplate="%{customdata}<extra></extra>",
                    name=f"{lbl} ({len(sub)})",
                    marker=dict(line=dict(color="white", width=0.5)),
                ))

            sub_d = df_map[(df_map["Cluster"] == cid) & (df_map["Is_Dominant"])]
            if len(sub_d) > 0:
                fig.add_trace(go.Choropleth(
                    locations=sub_d["Country Code"], z=[cid]*len(sub_d),
                    colorscale=[[0, color], [1, color]], showscale=False,
                    customdata=sub_d["hover_text"].values,
                    hovertemplate="%{customdata}<extra></extra>",
                    name=f"{lbl} (major producer, {len(sub_d)})",
                    marker=dict(line=dict(color="red", width=1.5)),
                ))

        fig.update_geos(
            projection_type="natural earth",
            showcountries=True, countrycolor="lightgray",
            showcoastlines=True, coastlinecolor="lightgray",
            showland=True, landcolor="whitesmoke",
            showocean=True, oceancolor="aliceblue",
        )
        fig.update_layout(
            title=dict(
                text="Natural Resource Clusters (per capita production value)<br>"
                     "<sup>Red border = >15% of a resource's global production</sup>",
                x=0.45, font=dict(size=14),
            ),
            width=1100, height=550,
            margin=dict(l=10, r=150, t=60, b=10),
            legend=dict(x=1.01, y=0.3, font=dict(size=10)),
        )
        return fig

    nr_1995_full = nr_sample[nr_sample["Year"] == 1995]
    fig_map = create_cluster_map(pca_1995, nr_1995_full)
    fig_map.show()


# --- SECTION: ECI vs GDP Animated Chart (Rosling) ---

# Animated Rosling chart: ECI vs log(GDP pc) coloured by 1995 cluster with trajectory arrows
def plot_rosling_eci_gdp():
    nr = pd.read_csv("intermediary/NaturalResource.csv")
    nr_sample = nr[nr["Country Code"].isin(INCLUDE_LIST)]

    pca_agg, pca_model_agg, feat_agg = run_clustering(
        nr_sample, agg_years=[1995, 1999, 2005]
    )

    master = pd.read_csv("intermediary/Master.csv")
    master = master[master["Country Code"].isin(INCLUDE_LIST)]

    master = pd.merge(
        master,
        pca_agg[["Country Code", "Cluster", "ClusterLabels"]],
        on="Country Code",
        how="left",
    )

    CLUSTER_COLORS_LOCAL = {}
    for cid in sorted(pca_agg["Cluster"].unique()):
        CLUSTER_COLORS_LOCAL[cid] = px.colors.qualitative.Bold[cid % len(px.colors.qualitative.Bold)]

    CLUSTER_NAMES = dict(zip(pca_agg["Cluster"], pca_agg["ClusterLabels"]))

    def create_rosling_chart(df, cluster_colors, cluster_names, arrow_opacity=0.5, arrow_width=2):
        """Animated ECI vs log(GDP pc) chart with trajectory arrows from 1995."""
        data = df.copy()
        data["Log GDP per capita"] = np.log(data["GDP per capita (constant prices, PPP)"])
        data["Production_Per_Capita"] = data["Total_Production_Value"] / data["Population"]

        c1995 = data[data["Year"] == 1995][["Country Code", "Cluster"]].copy()
        c1995 = c1995.rename(columns={"Cluster": "Cluster_1995"})
        data = data.merge(c1995, on="Country Code", how="left")
        data = data.dropna(subset=["Cluster_1995", "Log GDP per capita",
                                    "Economic Complexity Index", "Production_Per_Capita"])
        data["Cluster_1995"] = data["Cluster_1995"].astype(int)

        data["Bubble_Size"] = np.sqrt(data["Production_Per_Capita"])
        mn, mx = data["Bubble_Size"].min(), data["Bubble_Size"].max()
        data["Bubble_Size_Scaled"] = 8 + (data["Bubble_Size"] - mn) / (mx - mn) * 42

        data = data.sort_values(["Year", "Country Code"])
        years = sorted(data["Year"].unique())
        countries_list = data["Country Code"].unique()
        clusters = sorted(data["Cluster_1995"].unique())

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
            title=dict(text="Evolution of Economic Complexity vs Income<br>"
                            "<sup>Bubble size = Production per Capita</sup>", x=0.5),
            xaxis=dict(range=[x_vals.min()-0.2, x_vals.max()+0.2], title="Log GDP per capita (PPP)"),
            yaxis=dict(range=[eci_vals.min()-0.5, eci_vals.max()+0.5], title="Economic Complexity Index"),
            plot_bgcolor="white", width=850, height=650,
            legend=dict(title="Resource Profile (1995)", x=1.02, y=0.99),
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

    fig_rosling = create_rosling_chart(master, CLUSTER_COLORS_LOCAL, CLUSTER_NAMES)
    fig_rosling.show()


# ================================================================
# PART B: ML MODELS (5_ML_FINAL.ipynb)
# ================================================================

os.makedirs("Final/NB5", exist_ok=True)
OUT_NB5 = os.path.join('Final', 'NB5')

EXCLUDE = 'L1_ECI'


# --- SECTION: Static Visualisations ---

# Train vs Test R² and RMSE comparison bar charts (static, matplotlib)
def plot_oos_r2_rmse_static():
    # Requires: perf_level, perf_delta, PALETTE, OUT_NB5 from upstream pipeline cells
    # ── [1/8] Train vs Test R² ────────────────────────────────────────────────────
    print("[1/8] Train vs Test R² comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, perf, title in zip(axes,
                                [perf_level, perf_delta],
                                ['Target: ECI', 'Target: ΔECI']):
        model_names = perf['Model'].tolist()
        x = np.arange(len(model_names))
        w = 0.35
        ax.bar(x - w/2, perf['Train R²'], w, label='Train R²',
               color=PALETTE['blue'], alpha=0.85, edgecolor='black', linewidth=1)
        ax.bar(x + w/2, perf['Test R²'],  w, label='Test R²',
               color=PALETTE['red'],  alpha=0.85, edgecolor='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
        ax.set_ylabel('R²', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        for i, (tr, te) in enumerate(zip(perf['Train R²'], perf['Test R²'])):
            ax.text(i, max(tr, te) + 0.015, f'Δ={tr-te:+.3f}',
                    ha='center', fontsize=8.5, color='#555',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    plt.suptitle('Train vs Test R² — Overfitting Diagnostic', fontsize=15, fontweight='bold', y=1.02)
    plt.figtext(0.5, -0.02, 'Train: 1995–2014  |  Test: 2015–2019  |  Δ = train R² − test R²',
                ha='center', fontsize=10, color='#444')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_NB5, 'OOS_R2_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'OOS_R2_comparison.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved OOS_R2_comparison.{png,pdf}")

    # ── CHART 1b: Train vs Test RMSE ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, perf, title in zip(axes,
                                [perf_level, perf_delta],
                                ['Target: ECI', 'Target: ΔECI']):
        model_names = perf['Model'].tolist()
        x = np.arange(len(model_names))
        w = 0.35
        ax.bar(x - w/2, perf['Train RMSE'], w, label='Train RMSE',
               color=PALETTE['blue'], alpha=0.85, edgecolor='black', linewidth=1)
        ax.bar(x + w/2, perf['Test RMSE'],  w, label='Test RMSE',
               color=PALETTE['red'],  alpha=0.85, edgecolor='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        for i, (tr, te) in enumerate(zip(perf['Train RMSE'], perf['Test RMSE'])):
            ax.text(i, max(tr, te) + 0.002, f'Δ={te-tr:+.4f}',
                    ha='center', fontsize=8.5, color='#555',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    plt.suptitle('Train vs Test RMSE — Overfitting Diagnostic', fontsize=15, fontweight='bold', y=1.02)
    plt.figtext(0.5, -0.02, 'Train: 1995–2014  |  Test: 2015–2019  |  Δ = test RMSE − train RMSE',
                ha='center', fontsize=10, color='#444')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_NB5, 'OOS_RMSE_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'OOS_RMSE_comparison.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved OOS_RMSE_comparison.{png,pdf}")


# Predicted vs actual scatter for the best ECI and best ΔECI models (test set)
def plot_predicted_vs_actual():
    # Requires: perf_level, perf_delta, models_level, models_delta, X_test,
    #           y_test_level, y_test_delta, PALETTE, OUT_NB5
    print("\n[2/8] Predicted vs Actual (test set)...")

    best_level = perf_level.iloc[0]['Model']
    best_delta = perf_delta.iloc[0]['Model']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, actual, pred_vals, name, target in zip(
        axes,
        [y_test_level,  y_test_delta],
        [models_level[best_level].predict(X_test), models_delta[best_delta].predict(X_test)],
        [best_level,    best_delta],
        ['ECI',         'ΔECI'],
    ):
        r2_val   = r2_score(actual, pred_vals)
        rmse_val = np.sqrt(mean_squared_error(actual, pred_vals))
        ax.scatter(actual, pred_vals, alpha=0.45, s=30, color=PALETTE['blue'], edgecolor='none')
        lims = [min(actual.min(), pred_vals.min()) - 0.05,
                max(actual.max(), pred_vals.max()) + 0.05]
        ax.plot(lims, lims, '--', color=PALETTE['red'], linewidth=1.5, alpha=0.8, label='45° line')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f'Actual {target}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Predicted {target}', fontsize=12, fontweight='bold')
        ax.set_title(f'{name} — {target}  |  Test R²={r2_val:.3f}  RMSE={rmse_val:.4f}',
                     fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=10); ax.grid(alpha=0.2, linestyle='--')

    plt.suptitle('Predicted vs Actual — Test Set (2015–2019)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_NB5, 'PredVsActual_test.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'PredVsActual_test.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved PredVsActual_test.{png,pdf}")


# SHAP importance bar charts (Random Forest and optionally XGBoost)
def plot_shap_importance():
    # Requires: shap_results, all_features, short_names, EXCLUDE, PALETTE, OUT_NB5
    print("\n[3/8] SHAP importance...")
    if HAS_SHAP and shap_results:
        n_panels = len(shap_results)
        fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 9))
        if n_panels == 1: axes = [axes]
        for ax, (mname, sv) in zip(axes, shap_results.items()):
            mean_abs = np.abs(sv).mean(axis=0)
            feat_idx = [i for i, f in enumerate(all_features) if f != EXCLUDE]
            top15    = np.argsort(mean_abs[feat_idx])[-15:]
            vals_top = mean_abs[feat_idx][top15]
            names_top = [short_names[feat_idx[i]] for i in top15]
            cmap = plt.cm.YlOrRd
            colors = [cmap(0.3 + 0.65 * v / vals_top.max()) for v in vals_top]
            ax.barh(np.arange(len(top15)), vals_top, color=colors, edgecolor='black', linewidth=1.1, alpha=0.92)
            ax.set_yticks(np.arange(len(top15))); ax.set_yticklabels(names_top, fontsize=11, fontweight='bold')
            ax.set_xlabel('Mean |SHAP| (ECI units)', fontsize=11, fontweight='bold')
            ax.set_title(f'SHAP — {mname} (test set)', fontsize=12, fontweight='bold', pad=12)
            ax.grid(axis='x', alpha=0.25, linestyle='--')
        plt.suptitle('SHAP Feature Importance — Test Set', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_NB5, 'SHAP_importance.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(OUT_NB5, 'SHAP_importance.pdf'), bbox_inches='tight')
        plt.show()
        print("  Saved SHAP_importance.{png,pdf}")
    else:
        print("  Skipped — SHAP not available")


# 80% prediction interval band chart (sorted by actual ECI, test set)
def plot_prediction_intervals():
    # Requires: interval_df, coverage, avg_width, PALETTE, OUT_NB5
    print("\n[4/8] Prediction intervals...")
    plot_df = interval_df.sort_values('Actual').reset_index(drop=True)
    x_idx   = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(x_idx, plot_df['Q10'], plot_df['Q90'],
                    alpha=0.25, color=PALETTE['blue'], label='80% prediction band')
    ax.plot(x_idx, plot_df['Q50'], color=PALETTE['blue'], linewidth=2,
            label='Median (Q50)', alpha=0.9)
    ax.scatter(x_idx[plot_df['In_Band']],  plot_df.loc[plot_df['In_Band'],  'Actual'],
               color=PALETTE['green'], s=25, alpha=0.8, zorder=3, label='Actual (in band)')
    ax.scatter(x_idx[~plot_df['In_Band']], plot_df.loc[~plot_df['In_Band'], 'Actual'],
               color=PALETTE['red'],   s=35, alpha=0.9, zorder=4, marker='D', label='Actual (out of band)')
    ax.set_xlabel('Observations (sorted by actual ECI)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ECI', fontsize=12, fontweight='bold')
    ax.set_title(f'80% Prediction Intervals — Test Set (2015–2019)\n'
                 f'Coverage: {coverage:.1%}  |  Avg width: {avg_width:.4f}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_NB5, 'PredictionIntervals.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'PredictionIntervals.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved PredictionIntervals.{png,pdf}")


# VIF bar chart (colour-coded: red > 10, blue <= 10)
def plot_vif():
    # Requires: vif_data, shorten, PALETTE, OUT_NB5
    print("\n[5/8] VIF chart...")
    vif_plot = vif_data.copy()
    vif_plot['Short'] = vif_plot['Feature'].apply(shorten)
    vif_plot = vif_plot.sort_values('VIF', ascending=True).reset_index(drop=True)
    bar_colors = [PALETTE['red'] if v > 10 else PALETTE['blue'] for v in vif_plot['VIF']]

    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(len(vif_plot))
    ax.barh(y_pos, vif_plot['VIF'], color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.axvline(10, color=PALETTE['red'],    linewidth=2,   linestyle='--', alpha=0.8, label='VIF = 10')
    ax.axvline(5,  color=PALETTE['orange'], linewidth=1.5, linestyle=':',  alpha=0.7, label='VIF = 5')
    ax.set_yticks(y_pos); ax.set_yticklabels(vif_plot['Short'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12, fontweight='bold')
    ax.set_title('Multicollinearity Diagnostics — VIF by Feature', fontsize=14, fontweight='bold', pad=14)
    plt.figtext(0.5, 0.86, 'VIF > 10 flagged in red  |  computed on training set (1995–2014)',
                ha='center', fontsize=10, color='#444')
    for i, val in enumerate(vif_plot['VIF']):
        ax.text(min(val - 0.15, 10.8), i, f'{val:.1f}',
                va='center', ha='right', fontsize=9, fontweight='bold', color='black')
    ax.set_xlim(0, 11)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95)
    ax.grid(axis='x', alpha=0.25, linestyle='--')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(OUT_NB5, 'VIF_model2_resource_rich.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'VIF_model2_resource_rich.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved VIF_model2_resource_rich.{png,pdf}")


# 3-panel signed coefficient comparison: LASSO, Ridge, Elastic Net (top 15 each)
def plot_coefficient_comparison_3panel():
    # Requires: models_level, all_features, perf_level, shorten, EXCLUDE, PALETTE, OUT_NB5
    print("\n[6/8] 3-panel coefficient comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 12))
    for idx, mname in enumerate(['LASSO', 'Ridge', 'Elastic Net']):
        ax = axes[idx]
        coef_vals = models_level[mname].coef_.copy()
        abs_vals  = np.abs(coef_vals)
        excl_idx  = all_features.index(EXCLUDE) if EXCLUDE in all_features else None
        if excl_idx is not None: abs_vals[excl_idx] = -np.inf
        top15_idx  = np.argsort(abs_vals)[::-1][:15]
        top15_vals = coef_vals[top15_idx]
        colors     = [PALETTE['green'] if v > 0 else PALETTE['red'] for v in top15_vals]
        y_pos      = np.arange(len(top15_idx))
        ax.barh(y_pos, top15_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax.axvline(0, color='black', linewidth=2.5, alpha=0.8, zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([shorten(all_features[i]) for i in top15_idx], fontsize=11, fontweight='bold')
        ax.set_xlabel('Coefficient', fontsize=12, fontweight='bold')
        ax.invert_yaxis(); ax.grid(axis='x', alpha=0.25, linestyle='--')
        r2_tr = perf_level[perf_level['Model'] == mname]['Train R²'].values[0]
        r2_te = perf_level[perf_level['Model'] == mname]['Test R²'].values[0]
        n_sel = int(np.sum(models_level[mname].coef_ != 0))
        ax.set_title(f'{mname}\nTrain R²={r2_tr:.3f}  Test R²={r2_te:.3f}  |  {n_sel} features',
                     fontsize=13, fontweight='bold', pad=12)
        x_max = max(abs(top15_vals)) if len(top15_vals) else 1
        for i, val in enumerate(top15_vals):
            if abs(val) > 0.005:
                ax.text(abs(val) + x_max*0.03 if val > 0 else x_max*0.03, i, f'{val:+.3f}',
                        va='center', ha='left', fontsize=9.5, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=0.8))

    fig.legend(handles=[Patch(facecolor=PALETTE['green'], label='Positive', alpha=0.85, edgecolor='black', linewidth=1.5),
                        Patch(facecolor=PALETTE['red'],   label='Negative', alpha=0.85, edgecolor='black', linewidth=1.5)],
               loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=13, framealpha=0.98, edgecolor='black')
    plt.suptitle('Standardised Coefficients — LASSO, Ridge, Elastic Net', fontsize=18, fontweight='bold', y=0.98)
    plt.figtext(0.5, 0.94, 'Dep var: ECI  |  L1_ECI excluded  |  top 15 by absolute coefficient',
                ha='center', fontsize=12, color='#444')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(OUT_NB5, 'Coef_Comparison_model2_resource_rich.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'Coef_Comparison_model2_resource_rich.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved Coef_Comparison_model2_resource_rich.{png,pdf}")


# Model agreement dot-and-range chart: LASSO vs Ridge vs Elastic Net normalised importance
def plot_model_agreement():
    # Requires: all_importance, EXCLUDE, shorten, PALETTE, OUT_NB5
    print("\n[7/8] Model agreement dot-and-range chart...")
    imp_df = all_importance[all_importance['Feature'] != EXCLUDE].copy()
    top12  = imp_df.sort_values('Elastic Net', ascending=False).head(12).reset_index(drop=True)
    top12['Short'] = top12['Feature'].apply(shorten)
    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(top12))
    for i, row in top12.iterrows():
        vals = [row['LASSO'], row['Ridge'], row['Elastic Net']]
        ax.plot([min(vals), max(vals)], [i, i], color='#555555', alpha=0.45, linewidth=3, zorder=1)
        ax.scatter(row['LASSO'],       i, marker='o', s=160, alpha=0.92, color=PALETTE['red'],   edgecolor='black', linewidth=1.5, zorder=3, label='LASSO'       if i == 0 else '')
        ax.scatter(row['Ridge'],       i, marker='s', s=160, alpha=0.92, color=PALETTE['blue'],  edgecolor='black', linewidth=1.5, zorder=3, label='Ridge'       if i == 0 else '')
        ax.scatter(row['Elastic Net'], i, marker='^', s=160, alpha=0.92, color=PALETTE['green'], edgecolor='black', linewidth=1.5, zorder=3, label='Elastic Net' if i == 0 else '')
    ax.set_yticks(y_pos); ax.set_yticklabels(top12['Short'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Normalised Feature Importance', fontsize=13, fontweight='bold')
    ax.set_title('Model Agreement — LASSO, Ridge, Elastic Net', fontsize=15, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95, edgecolor='black')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    x_max_val = max(top12[['LASSO', 'Ridge', 'Elastic Net']].max())
    ax.set_xlim(-0.02, x_max_val + 0.12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(OUT_NB5, 'ModelAgreement_model2_resource_rich.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'ModelAgreement_model2_resource_rich.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved ModelAgreement_model2_resource_rich.{png,pdf}")


# Random Forest feature importance bar chart with YlOrRd colourmap and OOB R²
def plot_random_forest_importance():
    # Requires: rf, all_features, shorten, EXCLUDE, OUT_NB5
    print("\n[8/8] Random Forest importance...")
    rf_imp = pd.DataFrame({'Feature': all_features, 'Importance': rf.feature_importances_})
    rf_top = (rf_imp[rf_imp['Feature'] != EXCLUDE]
              .sort_values('Importance', ascending=False).head(15).reset_index(drop=True))
    norm_vals  = rf_top['Importance'] / rf_top['Importance'].max()
    cmap       = plt.cm.YlOrRd
    bar_colors = [cmap(0.35 + 0.60 * v) for v in norm_vals]
    fig, ax = plt.subplots(figsize=(13, 9))
    y_pos = np.arange(len(rf_top))
    ax.barh(y_pos, rf_top['Importance'], color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.92)
    ax.set_yticks(y_pos); ax.set_yticklabels([shorten(f) for f in rf_top['Feature']], fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Decrease in Impurity', fontsize=12, fontweight='bold')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=0.25, linestyle='--')
    ax.set_xlim(right=rf_top['Importance'].max() * 1.22)
    x_max_rf = rf_top['Importance'].max()
    for i, val in enumerate(rf_top['Importance']):
        ax.text(val + x_max_rf*0.012, i, f'{val:.4f}', va='center', ha='left', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.88, edgecolor='gray', linewidth=0.6))
    ax.set_title(f'Random Forest — Feature Importance\nOOB R²={rf.oob_score_:.3f}  |  200 trees  |  max_depth=4',
                 fontsize=13, fontweight='bold', pad=14)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=x_max_rf))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.025, pad=0.02)
    cbar.set_label('Importance', fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.97, 0.92])
    plt.savefig(os.path.join(OUT_NB5, 'RF_model2_resource_rich.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'RF_model2_resource_rich.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved RF_model2_resource_rich.{png,pdf}")


# --- SECTION: Interactive Visualisations ---

# Interactive train vs test R² grouped bar chart (Plotly)
def plot_oos_r2_interactive():
    # Requires: perf_level, perf_delta, PALETTE, STYLE, base_layout, save_chart, OUT_NB5
    print("[A] OOS R² comparison...")
    figA = make_subplots(rows=1, cols=2, subplot_titles=['Target: ECI', 'Target: ΔECI'],
                         horizontal_spacing=0.1)
    for col, perf in enumerate([perf_level, perf_delta], 1):
        figA.add_trace(go.Bar(name='Train R²', x=perf['Model'], y=perf['Train R²'],
                              marker_color=PALETTE['blue'], opacity=0.85,
                              text=[f'{v:.3f}' for v in perf['Train R²']], textposition='outside',
                              showlegend=(col==1)), row=1, col=col)
        figA.add_trace(go.Bar(name='Test R²',  x=perf['Model'], y=perf['Test R²'],
                              marker_color=PALETTE['red'],  opacity=0.85,
                              text=[f'{v:.3f}' for v in perf['Test R²']],  textposition='outside',
                              showlegend=(col==1)), row=1, col=col)
        figA.update_yaxes(title_text='R²' if col==1 else '',
                          gridcolor=STYLE['grid_color'], row=1, col=col)
    figA.update_layout(**base_layout(barmode='group', height=STYLE['chart_height'],
                                      legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center',
                                                  font=dict(size=STYLE['legend_size'])),
                                      margin=dict(l=60, r=40, t=80, b=80)))
    save_chart(figA, os.path.join(OUT_NB5, 'OOS_R2_interactive'))


# Interactive 80% prediction interval chart (Plotly)
def plot_prediction_intervals_interactive():
    # Requires: interval_df, coverage, avg_width, PALETTE, STYLE, base_layout, save_chart, OUT_NB5
    print("\n[B] Prediction intervals...")
    plot_s  = interval_df.sort_values('Actual').reset_index(drop=True)
    xlabels = plot_s.apply(lambda r: f"{r['Country Code']} {int(r['Year'])}", axis=1)

    figB = go.Figure()
    figB.add_trace(go.Scatter(
        x=list(range(len(plot_s))) + list(range(len(plot_s)))[::-1],
        y=plot_s['Q90'].tolist() + plot_s['Q10'].tolist()[::-1],
        fill='toself', fillcolor='rgba(74,111,165,0.2)',
        line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip', name='80% band'))
    figB.add_trace(go.Scatter(x=list(range(len(plot_s))), y=plot_s['Q50'],
        mode='lines', line=dict(color=PALETTE['blue'], width=2), name='Median (Q50)'))
    for mask, color, sym, lbl in [
        (plot_s['In_Band'],  PALETTE['green'], 'circle',  'In band'),
        (~plot_s['In_Band'], PALETTE['red'],   'diamond', 'Out of band'),
    ]:
        figB.add_trace(go.Scatter(
            x=plot_s.index[mask].tolist(), y=plot_s.loc[mask, 'Actual'],
            mode='markers', marker=dict(color=color, size=7 if lbl=='In band' else 9, symbol=sym, opacity=0.85),
            name=f'Actual ({lbl})',
            hovertemplate='%{text}<br>Actual: %{y:.3f}<extra></extra>',
            text=xlabels[mask].tolist()))
    figB.update_layout(**base_layout(
        xaxis=dict(title='Observations (sorted by actual)', showticklabels=False,
                   gridcolor=STYLE['grid_color']),
        yaxis=dict(title='ECI', gridcolor=STYLE['grid_color']),
        annotations=[dict(text=f"Coverage: {coverage:.1%}  |  Avg width: {avg_width:.4f}",
                          xref='paper', yref='paper', x=0.01, y=0.99, showarrow=False,
                          font=dict(size=11, color='#555'),
                          bgcolor='rgba(250,250,250,0.9)', bordercolor='#ddd', borderwidth=1, borderpad=6)],
        legend=dict(font=dict(size=STYLE['legend_size']), bgcolor='rgba(250,250,250,0.9)',
                    bordercolor='#e5e7eb', borderwidth=1)))
    save_chart(figB, os.path.join(OUT_NB5, 'PredictionIntervals_interactive'))


# Interactive VIF chart (Plotly)
def plot_vif_interactive():
    # Requires: vif_data, shorten, PALETTE, STYLE, base_layout, save_chart, OUT_NB5
    print("\n[C] VIF chart (interactive)...")
    vif_plot = vif_data.copy()
    vif_plot['Short'] = vif_plot['Feature'].apply(shorten)
    vif_plot = vif_plot.sort_values('VIF', ascending=True).reset_index(drop=True)
    colors_vif = [PALETTE['red'] if v > 10 else PALETTE['blue'] for v in vif_plot['VIF']]

    figC = go.Figure(go.Bar(
        y=vif_plot['Short'], x=vif_plot['VIF'], orientation='h',
        marker=dict(color=colors_vif, line=dict(color='#1a2744', width=0.5)),
        text=[f'{v:.1f}' for v in vif_plot['VIF']], textposition='outside',
        textfont=dict(size=STYLE['annotation_size'], color=STYLE['title_color']),
    ))
    figC.add_vline(x=10, line=dict(color=PALETTE['red'],    width=2,   dash='dash'),
                   annotation_text='VIF = 10', annotation_position='right',
                   annotation_font=dict(size=STYLE['annotation_size'], color=PALETTE['red']))
    figC.add_vline(x=5,  line=dict(color=PALETTE['orange'], width=1.5, dash='dot'),
                   annotation_text='VIF = 5',  annotation_position='right',
                   annotation_font=dict(size=STYLE['annotation_size'], color=PALETTE['orange']))
    figC.update_layout(**base_layout(height=STYLE['chart_height_tall'], margin=STYLE['margin_bar'],
                                      xaxis=dict(title=dict(text='Variance Inflation Factor',
                                                            font=dict(size=STYLE['axis_title_size'])),
                                                 gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
                                                 range=[0, max(vif_plot['VIF'].max()*1.15, 12)]),
                                      yaxis=dict(tickfont=dict(size=STYLE['tick_size'])),
                                      showlegend=False))
    save_chart(figC, os.path.join(OUT_NB5, 'VIF_model2_resource_rich_interactive'), width=1100, height=700)


# Interactive model agreement dot-and-range chart (Plotly)
def plot_model_agreement_interactive():
    # Requires: top12, PALETTE, STYLE, base_layout, save_chart, OUT_NB5
    print("\n[D] Model agreement (interactive)...")
    top12_r = top12.iloc[::-1].reset_index(drop=True)
    figD = go.Figure()
    for _, row in top12_r.iterrows():
        vals = [row['LASSO'], row['Ridge'], row['Elastic Net']]
        figD.add_trace(go.Scatter(x=[min(vals), max(vals)], y=[row['Short'], row['Short']],
                                   mode='lines', line=dict(color='#aab0b8', width=3),
                                   showlegend=False, hoverinfo='skip'))
    for mname, sym, col in [('LASSO','circle',PALETTE['red']),('Ridge','square',PALETTE['blue']),('Elastic Net','triangle-up',PALETTE['green'])]:
        figD.add_trace(go.Scatter(x=top12_r[mname], y=top12_r['Short'], mode='markers',
                                   marker=dict(symbol=sym, size=12, color=col, line=dict(color='#1a2744', width=1)),
                                   name=mname, hovertemplate='%{y}: %{x:.3f}<extra>' + mname + '</extra>'))
    figD.update_layout(**base_layout(height=STYLE['chart_height'], margin=STYLE['margin_bar'],
                                      xaxis=dict(title=dict(text='Normalised Feature Importance',
                                                            font=dict(size=STYLE['axis_title_size'])),
                                                 gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
                                                 range=[-0.02, top12[['LASSO','Ridge','Elastic Net']].max().max()+0.08]),
                                      yaxis=dict(tickfont=dict(size=STYLE['tick_size'])),
                                      legend=dict(font=dict(size=STYLE['legend_size']),
                                                  yanchor='bottom', y=0.02, xanchor='right', x=0.98,
                                                  bgcolor='rgba(250,250,250,0.95)', bordercolor='#e5e7eb', borderwidth=1)))
    save_chart(figD, os.path.join(OUT_NB5, 'ModelAgreement_model2_resource_rich_interactive'), width=1100, height=600)


# Interactive Random Forest importance chart (Plotly, lerp colour)
def plot_rf_importance_interactive():
    # Requires: rf, rf_top, shorten, STYLE, base_layout, save_chart, OUT_NB5
    print("\n[E] RF importance (interactive)...")
    def lerp_color(t, c_lo=(230,185,128), c_hi=(180,80,40)):
        return f'rgb({int(c_lo[0]+(c_hi[0]-c_lo[0])*t)},{int(c_lo[1]+(c_hi[1]-c_lo[1])*t)},{int(c_lo[2]+(c_hi[2]-c_lo[2])*t)})'

    rf_top_r       = rf_top.iloc[::-1].reset_index(drop=True)
    norm_vals_r    = (rf_top_r['Importance'] / rf_top_r['Importance'].max()).values
    bar_colors_rf  = [lerp_color(v) for v in norm_vals_r]

    figE = go.Figure(go.Bar(
        y=[shorten(f) for f in rf_top_r['Feature']], x=rf_top_r['Importance'], orientation='h',
        marker=dict(color=bar_colors_rf, line=dict(color='#1a2744', width=0.5)),
        text=[f'{v:.4f}' for v in rf_top_r['Importance']], textposition='outside',
        textfont=dict(size=STYLE['annotation_size'], color=STYLE['title_color']),
    ))
    figE.update_layout(**base_layout(height=STYLE['chart_height_tall'], margin=STYLE['margin_bar'],
                                      xaxis=dict(title=dict(text='Mean Decrease in Impurity',
                                                            font=dict(size=STYLE['axis_title_size'])),
                                                 gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
                                                 range=[0, rf_top['Importance'].max()*1.22]),
                                      yaxis=dict(tickfont=dict(size=STYLE['tick_size'])),
                                      showlegend=False,
                                      annotations=[dict(text=f"OOB R²={rf.oob_score_:.3f}  |  200 trees  |  max_depth=4",
                                                        xref='paper', yref='paper', x=0.98, y=0.02,
                                                        showarrow=False, font=dict(size=STYLE['annotation_size'], color='#666'),
                                                        bgcolor='rgba(250,250,250,0.95)', bordercolor='#e5e7eb',
                                                        borderwidth=1, borderpad=6)]))
    save_chart(figE, os.path.join(OUT_NB5, 'RF_model2_resource_rich_interactive'), width=1100, height=700)


# --- SECTION: Coefficient Summary Table ---

# Heatmap-styled matplotlib table: coefficients, importance, VIF, selection flags
def plot_coefficient_summary_table():
    # Requires: table, all_importance, vif_data, models_level, all_features, lin_models, shorten, EXCLUDE, mcolors, PALETTE, OUT_NB5
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    col_labels = ['Feature', 'LASSO\nCoef', 'Ridge\nCoef', 'E-Net\nCoef',
                  'LASSO\nImp', 'Ridge\nImp', 'E-Net\nImp', 'VIF', 'LASSO\nSel', 'EN\nSel']
    cell_vals = [[
        row['Feature'], f"{row['LASSO']:+.3f}", f"{row['Ridge']:+.3f}", f"{row['Elastic Net']:+.3f}",
        f"{row['LASSO_Imp']:.3f}", f"{row['Ridge_Imp']:.3f}", f"{row['Elastic Net_Imp']:.3f}",
        f"{row['VIF']:.2f}", '✓' if row['LASSO_Selected'] else '', '✓' if row['Elastic Net_Selected'] else '',
    ] for _, row in table.iterrows()]

    tbl = ax.table(cellText=cell_vals, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.6)
    for col_idx in range(len(col_labels)):
        tbl[0, col_idx].set_facecolor('#2C3E50')
        tbl[0, col_idx].set_text_props(color='white', fontweight='bold', fontsize=9)
    for row_idx, (_, row) in enumerate(table.iterrows(), start=1):
        tbl[row_idx, 0].set_facecolor('#ECF0F1' if row_idx % 2 else '#D5D8DC')
        tbl[row_idx, 0].set_text_props(fontweight='bold', ha='left')
        for ci, cn in [(1,'LASSO'),(2,'Ridge'),(3,'Elastic Net')]:
            val = row[cn]
            tbl[row_idx, ci].set_facecolor('#D5F5E3' if val > 0.005 else '#FADBD8' if val < -0.005 else '#FDFEFE')
        for ci, cn in [(4,'LASSO_Imp'),(5,'Ridge_Imp'),(6,'Elastic Net_Imp')]:
            alpha = 0.15 + 0.70 * row[cn]
            tbl[row_idx, ci].set_facecolor(mcolors.to_rgba(PALETTE['blue'], alpha=alpha))
        vif_val = row['VIF']
        if vif_val > 10:
            tbl[row_idx, 7].set_facecolor('#FADBD8'); tbl[row_idx, 7].set_text_props(color='#C0392B', fontweight='bold')
        elif vif_val > 5:
            tbl[row_idx, 7].set_facecolor('#FDEBD0'); tbl[row_idx, 7].set_text_props(color='#E67E22', fontweight='bold')
        for ci, sc in [(8,'LASSO_Selected'),(9,'Elastic Net_Selected')]:
            if row[sc]:
                tbl[row_idx, ci].set_facecolor('#D5F5E3'); tbl[row_idx, ci].set_text_props(color='#1E8449', fontweight='bold')

    ax.set_title('Summary Table — LASSO, Ridge, Elastic Net\nCoefficients, Importance, VIF, Selection',
                 fontsize=14, fontweight='bold', pad=16, y=0.98)
    plt.figtext(0.5, 0.01, 'Green=positive  |  Red=negative  |  Blue intensity=importance  |  Orange VIF=multicollinearity  |  checkmark=selected',
                ha='center', fontsize=9, color='#444')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_NB5, 'SummaryTable_model2_resource_rich.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'SummaryTable_model2_resource_rich.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved SummaryTable_model2_resource_rich.{png,pdf}")


# --- SECTION: Forecast Charts ---

# Country ranking bar chart: top 10 improvers and bottom 10 decliners 2020-2030
def plot_forecast_country_ranking():
    # Requires: country_summary, PALETTE, OUT_NB5
    print("[Forecast 1/4] Country ranking chart...")
    top10    = country_summary.head(10)
    bottom10 = country_summary.tail(10).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, sub, title in [(axes[0], top10, 'Top 10 Improvers (2020–2030)'),
                            (axes[1], bottom10, 'Bottom 10 Decliners (2020–2030)')]:
        col = [PALETTE['green'] if v > 0 else PALETTE['red'] for v in sub['Total_Change']]
        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub['Total_Change'], color=col, edgecolor='black', linewidth=1, alpha=0.88)
        ax.set_yticks(y_pos); ax.set_yticklabels(sub['Country'], fontsize=10, fontweight='bold')
        ax.axvline(0, color='black', linewidth=1.2, alpha=0.6)
        ax.set_xlabel('ΔECI 2019→2030 (ensemble)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.grid(axis='x', alpha=0.25, linestyle='--')
        for i, val in enumerate(sub['Total_Change']):
            ax.text(val + (0.005 if val > 0 else -0.005), i, f'{val:+.3f}',
                    va='center', ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')

    plt.suptitle('ECI Forecast 2020–2030 — Country Rankings', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_NB5, 'Forecast_CountryRanking.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'Forecast_CountryRanking.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved Forecast_CountryRanking.{png,pdf}")


# Line chart: historical + forecasted ECI trajectories for top 5 improvers
def plot_forecast_top_improvers_trajectory():
    # Requires: df, forecast_df, country_summary, OUT_NB5
    print("\n[Forecast 2/4] Top improvers trajectory...")
    top5_codes = country_summary.head(5)['Country Code'].tolist()
    hist_top5  = df[df['Country Code'].isin(top5_codes)][['Country Code','Country Name','Year','Economic Complexity Index']].copy()
    fc_top5    = forecast_df[forecast_df['Country Code'].isin(top5_codes)][['Country Code','Year','Ensemble']].rename(columns={'Ensemble':'Economic Complexity Index'})
    fc_top5['Country Name'] = fc_top5['Country Code'].map(dict(zip(hist_top5['Country Code'], hist_top5['Country Name'])))

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap_fc = plt.get_cmap('tab10')
    for i, cc in enumerate(top5_codes):
        col   = cmap_fc(i)
        cname = country_summary[country_summary['Country Code']==cc]['Country'].values[0]
        h = hist_top5[hist_top5['Country Code']==cc].sort_values('Year')
        f = fc_top5[fc_top5['Country Code']==cc].sort_values('Year')
        ax.plot(h['Year'], h['Economic Complexity Index'], color=col, linewidth=2, label=cname)
        ax.plot(f['Year'], f['Economic Complexity Index'], color=col, linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(2019.5, color='grey', linewidth=1.5, linestyle='--', alpha=0.8, label='Forecast start')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Economic Complexity Index', fontsize=12, fontweight='bold')
    ax.set_title('ECI Trajectories — Top 5 Forecast Improvers', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95); ax.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_NB5, 'Forecast_TopImprovers_Trajectory.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'Forecast_TopImprovers_Trajectory.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved Forecast_TopImprovers_Trajectory.{png,pdf}")


# Interactive heatmap of ECI for all countries across all years + forecast
def plot_forecast_heatmap_all_countries():
    # Requires: df, forecast_df, country_summary, base_layout, save_chart, OUT_NB5
    print("\n[Forecast 3/4] Heatmap (all countries)...")
    cnames_map = df[['Country Code','Country Name']].drop_duplicates()
    code_to_name = dict(zip(cnames_map['Country Code'], cnames_map['Country Name']))

    hist_pivot = df.pivot_table(index='Country Code', columns='Year',
                                 values='Economic Complexity Index', aggfunc='mean')
    fc_pivot   = forecast_df.pivot_table(index='Country Code', columns='Year',
                                          values='Ensemble', aggfunc='mean')
    full_pivot  = pd.concat([hist_pivot, fc_pivot], axis=1)
    full_pivot  = full_pivot.loc[:, ~full_pivot.columns.duplicated(keep='last')]
    full_pivot.columns = [int(c) for c in full_pivot.columns]
    full_pivot  = full_pivot[sorted(full_pivot.columns)]
    full_pivot  = full_pivot.loc[country_summary.sort_values('ECI_2030_Ensemble', ascending=True)['Country Code']]
    full_pivot.index = [code_to_name.get(c, c) for c in full_pivot.index]

    fig_heat = go.Figure(data=go.Heatmap(
        z=full_pivot.values, x=full_pivot.columns.tolist(), y=full_pivot.index.tolist(),
        colorscale='RdYlGn',
        colorbar=dict(title=dict(text='ECI', font=dict(size=11))),
        hovertemplate='%{y}<br>Year: %{x}<br>ECI: %{z:.3f}<extra></extra>',
    ))
    fig_heat.add_vline(x=2019.5, line=dict(color='white', width=3))
    fig_heat.update_layout(**base_layout(
        height=max(700, len(full_pivot)*16),
        margin=dict(l=140, r=40, t=20, b=60),
        xaxis=dict(title='Year', dtick=2, tickfont=dict(size=9), type='linear'),
        yaxis=dict(tickfont=dict(size=8)),
    ))
    save_chart(fig_heat, os.path.join(OUT_NB5, 'Forecast_Heatmap_AllCountries'),
               width=1200, height=max(700, len(full_pivot)*16))


# Scatter of actual vs predicted average ΔECI per country (2015-2019, test set)
def plot_avg_delta_eci_actual_vs_pred():
    # Requires: perf_delta, models_delta, test_df, X_test, df, PALETTE, OUT_NB5
    print("\n[Forecast 4/4] Avg ΔECI actual vs predicted by country...")

    best_delta_name = perf_delta.iloc[0]['Model']
    test_df_plot = test_df.copy().reset_index(drop=True)
    test_df_plot['Pred_Delta'] = models_delta[best_delta_name].predict(X_test)

    country_avg = test_df_plot.groupby('Country Code').agg(
        Country=('Country Name', 'first'),
        Actual_Avg_Delta=('ECI_delta', 'mean'),
        Pred_Avg_Delta=('Pred_Delta', 'mean'),
    ).reset_index()

    country_avg['Residual'] = country_avg['Actual_Avg_Delta'] - country_avg['Pred_Avg_Delta']
    country_avg['Abs_Residual'] = country_avg['Residual'].abs()
    mismatches = country_avg.sort_values('Abs_Residual', ascending=False).reset_index(drop=True)

    print("\nBiggest Mismatches — Actual vs Predicted Avg ΔECI (2015–2019):")
    print(mismatches[['Country Code', 'Country',
                       'Actual_Avg_Delta', 'Pred_Avg_Delta',
                       'Residual']].head(15).to_string(index=False))

    r2_val  = r2_score(country_avg['Actual_Avg_Delta'], country_avg['Pred_Avg_Delta'])
    rmse_val = np.sqrt(mean_squared_error(country_avg['Actual_Avg_Delta'], country_avg['Pred_Avg_Delta']))

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(country_avg['Actual_Avg_Delta'], country_avg['Pred_Avg_Delta'],
               s=50, color=PALETTE['blue'], alpha=0.75, edgecolor='black', linewidth=0.5)

    for _, row in country_avg.iterrows():
        ax.annotate(row['Country Code'], (row['Actual_Avg_Delta'], row['Pred_Avg_Delta']),
                    fontsize=7, ha='left', va='bottom', color='#333',
                    xytext=(4, 4), textcoords='offset points')

    lims = [min(country_avg['Actual_Avg_Delta'].min(), country_avg['Pred_Avg_Delta'].min()) - 0.02,
            max(country_avg['Actual_Avg_Delta'].max(), country_avg['Pred_Avg_Delta'].max()) + 0.02]
    ax.plot(lims, lims, '--', color=PALETTE['red'], linewidth=1.5, alpha=0.8, label='45° line')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    z = np.polyfit(country_avg['Actual_Avg_Delta'], country_avg['Pred_Avg_Delta'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_trend, p(x_trend), '-', color=PALETTE['green'], linewidth=2, alpha=0.8,
            label=f'OLS fit (slope={z[0]:.2f})')

    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('Actual Average ΔECI (2015–2019)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Average ΔECI (2015–2019)', fontsize=12, fontweight='bold')
    ax.set_title(f'{best_delta_name} — Country-Level Average ΔECI\n',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_NB5, 'AvgDeltaECI_ActualVsPred_Country.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUT_NB5, 'AvgDeltaECI_ActualVsPred_Country.pdf'), bbox_inches='tight')
    plt.show()
    print("  Saved AvgDeltaECI_ActualVsPred_Country.{png,pdf}")


# ================================================================
# PART C: REGRESSIONS (6_Regressions_Unified.ipynb)
# ================================================================

os.makedirs("Final/NB6", exist_ok=True)
OUT_NB6 = os.path.join('Final', 'NB6')


# --- SECTION: Descriptive Statistics ---

# CDF comparison of ECI distribution: 1995 vs 2019
def plot_eci_distribution_comparison():
    master = pd.read_csv('intermediary/Master.csv')
    cluster_1995_df = pd.read_csv('intermediary/clusters1995.csv')

    INCLUDE = [
        'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
        'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
        'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
        'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
        'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
        'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
    ]

    df = master[master['Country Code'].isin(INCLUDE)].copy()

    yr_95 = df[df['Year'] == 1995]['Economic Complexity Index'].dropna().sort_values().values
    yr_19 = df[df['Year'] == 2019]['Economic Complexity Index'].dropna().sort_values().values

    fig = go.Figure()
    for vals, yr, col in [(yr_95, 1995, PALETTE['blue']), (yr_19, 2019, PALETTE['red'])]:
        pcts = np.linspace(0, 100, len(vals))
        fig.add_trace(go.Scatter(
            x=pcts, y=vals,
            mode='lines', name=str(yr),
            line=dict(color=col, width=2.5),
        ))

    fig.update_layout(**base_layout(
        height=STYLE['chart_height'],
        xaxis=dict(
            title=dict(text='Percentile', font=dict(size=STYLE['axis_title_size'])),
            gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
        ),
        yaxis=dict(
            title=dict(text='Economic Complexity Index', font=dict(size=STYLE['axis_title_size'])),
            gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
            zeroline=True, zerolinecolor=STYLE['zero_line_color'], zerolinewidth=1,
        ),
        legend=dict(font=dict(size=STYLE['legend_size'])),
    ))

    save_chart(fig, os.path.join(OUT_NB6, 'eci_distribution_comparison'))


# Median ECI trajectory by resource-profile cluster, 1995-2019
def plot_eci_cluster_trajectories():
    master = pd.read_csv('intermediary/Master.csv')
    cluster_1995_df = pd.read_csv('intermediary/clusters1995.csv')

    INCLUDE = [
        'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
        'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
        'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
        'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
        'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
        'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
    ]

    df = master[master['Country Code'].isin(INCLUDE)].copy()
    cluster_1995_df = cluster_1995_df[['Country Code', 'Cluster']]
    df = df.merge(cluster_1995_df, on='Country Code', how='left')

    traj = (df.groupby(['Year', 'Cluster'])['Economic Complexity Index']
              .median().reset_index())

    clusters_present = sorted(traj['Cluster'].dropna().unique())

    fig = go.Figure()
    for i, cl in enumerate(clusters_present):
        sub = traj[traj['Cluster'] == cl]
        fig.add_trace(go.Scatter(
            x=sub['Year'], y=sub['Economic Complexity Index'],
            mode='lines+markers', name=str(cl),
            line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(**base_layout(
        xaxis=dict(
            title=dict(text='Year', font=dict(size=STYLE['axis_title_size'])),
            gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
            dtick=5,
        ),
        yaxis=dict(
            title=dict(text='Median ECI', font=dict(size=STYLE['axis_title_size'])),
            gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
            zeroline=True, zerolinecolor=STYLE['zero_line_color'], zerolinewidth=1,
        ),
        legend=dict(title=dict(text='Cluster'), font=dict(size=STYLE['legend_size'])),
    ))

    save_chart(fig, os.path.join(OUT_NB6, 'eci_cluster_trajectories'))


# Correlation heatmap of key regression variables
def plot_eci_correlation_heatmap():
    master = pd.read_csv('intermediary/Master.csv')

    INCLUDE = [
        'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
        'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
        'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
        'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
        'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
        'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
    ]

    df = master[master['Country Code'].isin(INCLUDE)].copy()
    df['Total_Production_Value_Per_Capita'] = df['Total_Production_Value'] / df['Population']

    DESC_VARS = {
        'Economic Complexity Index': 'ECI',
        'Human capital index':                          'Human capital index',
        'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'GFCF (% GDP)',
        'Domestic credit to private sector (% of GDP)': 'Domestic credit (% GDP)',
        'Access to electricity (% of population)':      'Electricity access (%)',
        'Rule of law index':                            'Rule of law',
        'Political stability — estimate':               'Political stability',
        'Total natural resources rents (% of GDP)':     'NR rents (% GDP)',
        'Total_Production_Value_Per_Capita':            'Prod. value per capita (USD)',
        'Trade (% of GDP)':                             'Trade (% GDP)',
    }

    corr_cols = list(DESC_VARS.keys())
    corr_df   = df[corr_cols].rename(columns=DESC_VARS).corr().round(2)

    labels = list(corr_df.columns)
    z      = corr_df.values

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[
            [0.0, PALETTE['red']], [0.5, '#fafafa'], [1.0, PALETTE['blue']]
        ],
        zmid=0, zmin=-1, zmax=1,
        text=z.round(2), texttemplate='%{text}',
        textfont=dict(size=9, family=STYLE['font_family']),
        hovertemplate='%{x} × %{y}: %{z:.2f}<extra></extra>',
        colorbar=dict(thickness=14, len=0.9,
                      tickfont=dict(size=STYLE['tick_size'],
                                    family=STYLE['font_family'])),
    ))

    fig.update_layout(**base_layout(
        height=620,
        margin=dict(l=180, r=60, t=10, b=180),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    ))

    save_chart(fig, os.path.join(OUT_NB6, 'eci_correlation_heatmap'), width=900, height=700)


# --- SECTION: Regression Visualisations ---

# Residual Q-Q plots for Models 3a and 3b
def plot_residual_qq():
    # Requires: m3a_raw, m3b_raw, PALETTE, OUT_NB6
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m_raw, label in [
        (axes[0], m3a_raw, 'Model 3a (no lag)'),
        (axes[1], m3b_raw, 'Model 3b (with lag)'),
    ]:
        resid = m_raw.resid
        (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist='norm')
        ax.scatter(osm, osr, s=18, alpha=0.6, color=PALETTE['blue'], edgecolor='none')
        ax.plot(
            [osm[0], osm[-1]],
            [osm[0] * slope + intercept, osm[-1] * slope + intercept],
            color=PALETTE['red'], linewidth=1.5, linestyle='--', label='Normal reference'
        )
        ax.set_xlabel('Theoretical quantiles', fontsize=11)
        ax.set_ylabel('Ordered residuals', fontsize=11)
        ax.set_title(f'Q-Q Plot — {label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_NB6, 'residual_qq.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print('  Saved residual_qq.png')

    print('\nBreusch-Pagan test (H0: homoskedastic):')
    for m_raw, X_reg, label in [
        (m3a_raw, X3a, 'Model 3a'),
        (m3b_raw, X3b, 'Model 3b'),
    ]:
        lm_stat, lm_pval, _, _ = het_breuschpagan(m_raw.resid, X_reg)
        verdict = 'reject H0 — heteroskedastic' if lm_pval < 0.05 else 'fail to reject H0'
        print(f'  {label}: LM={lm_stat:.3f}  p={lm_pval:.4f}  → {verdict}')
    print('  (Driscoll-Kraay SEs remain valid regardless of this result)')


# Coefficient dot-and-whisker chart: Model 3a vs 3b with 95% CI (Driscoll-Kraay)
def plot_coef_comparison_3a_3b():
    # Requires: m3a, m3b, reg3_input, INTERACT_VARS, PALETTE, STYLE, base_layout, save_chart, OUT_NB6
    plot_vars = [v for v in reg3_input + INTERACT_VARS if v != 'const']

    labels = [v.replace('log_', 'log(').replace('_x_', ') × log(') + (')' if 'log_' in v else '')
              for v in plot_vars]

    models_to_plot = [
        (m3a,  PALETTE['blue'],      'Model 3a (no lag)'),
        (m3b,  PALETTE['light_blue'],'Model 3b (with lag)'),
    ]

    fig = go.Figure()
    for model, col, name in models_to_plot:
        coefs  = [model.params.get(v, np.nan)  for v in plot_vars]
        lowers = [model.params.get(v, np.nan) - 1.96 * model.bse.get(v, np.nan) for v in plot_vars]
        uppers = [model.params.get(v, np.nan) + 1.96 * model.bse.get(v, np.nan) for v in plot_vars]
        fig.add_trace(go.Scatter(
            y=labels, x=coefs,
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

    fig.add_vline(x=0, line=dict(color=STYLE['zero_line_color'], width=1.5, dash='dash'))

    fig.update_layout(**base_layout(
        height=STYLE['chart_height_tall'],
        margin=STYLE['margin_bar'],
        xaxis=dict(
            title=dict(text='Coefficient (95 % CI, Driscoll-Kraay)', font=dict(size=STYLE['axis_title_size'])),
            gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
            zeroline=False,
        ),
        yaxis=dict(tickfont=dict(size=STYLE['tick_size'])),
        legend=dict(font=dict(size=STYLE['legend_size']),
                    orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
    ))

    save_chart(fig, os.path.join(OUT_NB6, 'coef_comparison_3a_3b'), width=1100, height=700)


# ECI vs log(HCI) scatter coloured by per-capita production value quartile
def plot_eci_hci_production_interaction():
    master = pd.read_csv('intermediary/Master.csv')

    INCLUDE = [
        'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
        'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
        'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
        'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
        'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
        'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
    ]

    df = master[master['Country Code'].isin(INCLUDE)].copy()
    df['Total_Production_Value_Per_Capita'] = df['Total_Production_Value'] / df['Population']
    df['log_HCI']              = np.log1p(df['Human capital index'])
    df['log_Production_Value'] = np.log1p(df['Total_Production_Value_Per_Capita'])

    plot_df = df[['log_HCI', 'Economic Complexity Index', 'log_Production_Value',
                  'Country Code', 'Year']].dropna()

    plot_df['Prod_quartile'] = pd.qcut(
        plot_df['log_Production_Value'], q=4,
        labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
    )

    q_colors = [PALETTE['light_blue'], PALETTE['blue'], PALETTE['orange'], PALETTE['red']]

    fig = go.Figure()
    for q, col in zip(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'], q_colors):
        sub = plot_df[plot_df['Prod_quartile'] == q]
        fig.add_trace(go.Scatter(
            x=sub['log_HCI'], y=sub['Economic Complexity Index'],
            mode='markers',
            marker=dict(color=col, size=5, opacity=0.65,
                        line=dict(width=0.3, color='white')),
            name=f'Production {q}',
            hovertemplate='%{customdata[0]} %{customdata[1]}<br>'
                          'log(HCI)=%{x:.2f}  ECI=%{y:.2f}<extra></extra>',
            customdata=sub[['Country Code', 'Year']].values,
        ))

    fig.update_layout(**base_layout(
        xaxis=dict(
            title=dict(text='log(Human Capital Index)', font=dict(size=STYLE['axis_title_size'])),
            gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
        ),
        yaxis=dict(
            title=dict(text='Economic Complexity Index', font=dict(size=STYLE['axis_title_size'])),
            gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
        ),
        legend=dict(title=dict(text='Prod. Value p.c. Quartile'),
                    font=dict(size=STYLE['legend_size'])),
    ))

    save_chart(fig, os.path.join(OUT_NB6, 'eci_hci_production_interaction'), width=1000, height=600)
