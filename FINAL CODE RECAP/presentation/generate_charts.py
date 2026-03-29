#!/usr/bin/env python3
"""
generate_charts.py — Presentation-quality interactive Plotly figures
Replicates (and improves on) all report charts for the Resource Curse / ECI Capstone.

Reads from:
  ../v1/intermediary/          — capstone panel data + clusters
  ../chile/intermediary/       — Chile pipeline state
  ml_cache.pkl                 — cached model coefficients (auto-created on first run)

Writes to: outputs/

Run:  /usr/local/bin/python3.10 generate_charts.py
"""

import os, sys, math, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)                          # FINAL CODE RECAP/
V1     = os.path.join(ROOT, "v1",    "intermediary")
CHILE  = os.path.join(ROOT, "chile", "intermediary")
OUT    = os.path.join(HERE, "outputs")
CACHE  = os.path.join(HERE, "ml_cache.pkl")
os.makedirs(OUT, exist_ok=True)

# ── Design system ─────────────────────────────────────────────────────────────
FONT  = "IBM Plex Sans, system-ui, -apple-system, sans-serif"
NAVY  = "#1a2744"
SUBTT = "#6b7280"
BG    = "#fafafa"
GRID  = "#e5e7eb"
CFG   = dict(displayModeBar=False, displaylogo=False, responsive=True)

CLUSTER_COLORS = {
    # k=4 labels
    "Oil, Few Minerals":   "#2A9D8F",
    "No Oil, No Minerals": "#E63946",
    "Some Oil, No Minerals":"#457B9D",
    "Minerals, No Oil":    "#E9C46A",
    # k=5 labels
    "Petrostates":        "#E63946",
    "Oil Exporters":      "#457B9D",
    "Major Producers":    "#2A9D8F",
    "Mining Exporters":   "#E9C46A",
    "Forestry Intensive": "#8B5CF6",
}

def base_layout(**kw):
    d = dict(
        template="plotly_white",
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family=FONT, size=12, color=NAVY),
        margin=dict(l=60, r=40, t=60, b=50),
        height=560,
    )
    d.update(kw)
    return d

def title(text, sub=None):
    t = dict(text=text if sub is None else
             f"{text}<br><sup style='font-size:11px;font-weight:normal;color:{SUBTT}'>{sub}</sup>",
             x=0.5, xanchor="center",
             font=dict(size=16, color=NAVY, family=FONT))
    return t

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.write_html(path, config=CFG, include_plotlyjs="cdn")
    print(f"  → {name}")

# ── Load capstone data ────────────────────────────────────────────────────────
print("\n=== Loading capstone data ===")

master      = pd.read_csv(os.path.join(V1, "Master.csv"))
include_lst = pd.read_csv(os.path.join(V1, "high_resource_countries.csv"))["Country Code"].unique().tolist()
cl1995      = pd.read_csv(os.path.join(V1, "clusters1995.csv"))
cl_agg      = pd.read_csv(os.path.join(V1, "clusters_k5_agg.csv"))

panel = master[(master["Year"].between(1995, 2019)) &
               (master["Country Code"].isin(include_lst))].copy()
panel = panel.sort_values(["Country Code", "Year"]).reset_index(drop=True)

cl_map = cl1995[["Country Code", "ClusterLabels"]].drop_duplicates("Country Code")
panel  = panel.merge(cl_map, on="Country Code", how="left")
panel["Log_GDP_pc"]   = np.log(panel["GDP per capita (constant prices, PPP)"].replace(0, np.nan))
panel["Prod_pc"]      = panel["Total_Production_Value"] / panel["Population"].replace(0, np.nan)
panel["Bubble"]       = np.sqrt(panel["Prod_pc"].fillna(0))
bmin, bmax            = panel["Bubble"].min(), panel["Bubble"].max()
panel["Bubble_Scaled"] = 8 + (panel["Bubble"] - bmin) / (bmax - bmin + 1e-9) * 42

print(f"  Panel: {panel['Country Code'].nunique()} countries, {len(panel):,} obs, 1995-2019")


# ══════════════════════════════════════════════════════════════════════════════
# 01  ECI VS LOG GDP — ANIMATED SCATTER WITH TRAILING PATHS
# ══════════════════════════════════════════════════════════════════════════════
print("\n01 ECI animated scatter …")

data05 = panel.dropna(subset=["Log_GDP_pc", "Economic Complexity Index", "ClusterLabels"]).copy()
years  = sorted(data05["Year"].unique())

cdata = {}
for cc, cdf in data05.groupby("Country Code"):
    cdf = cdf.sort_values("Year")
    if cdf["Year"].min() > 1995:
        continue
    cdata[cc] = dict(
        years  = cdf["Year"].values,
        x      = cdf["Log_GDP_pc"].values,
        y      = cdf["Economic Complexity Index"].values,
        size   = cdf["Bubble_Scaled"].values,
        name   = cdf["Country Name"].iloc[0],
        code   = cc,
        clust  = cdf["ClusterLabels"].iloc[0],
    )

valid_cc = list(cdata.keys())

fig01 = go.Figure()

# Initial state — dots + tail lines up to year[0]
for lbl in sorted(set(cdata[cc]["clust"] for cc in valid_cc)):
    color   = CLUSTER_COLORS.get(lbl, "#aaa")
    cc_list = [cc for cc in valid_cc if cdata[cc]["clust"] == lbl]
    first   = True
    for cc in cc_list:
        cd = cdata[cc]
        i0 = 0
        fig01.add_trace(go.Scatter(
            x=[cd["x"][i0]], y=[cd["y"][i0]], mode="markers+text",
            marker=dict(size=cd["size"][i0], color=color, opacity=0.85,
                        line=dict(width=1.5, color="white")),
            text=[cc], textposition="top center",
            textfont=dict(size=8, color="#333"),
            name=lbl, legendgroup=lbl, showlegend=first,
            customdata=[[cd["name"]]],
            hovertemplate="<b>%{customdata[0]}</b><br>Log GDP/cap: %{x:.2f}<br>ECI: %{y:.2f}<extra></extra>",
        ))
        first = False

# Frames
frames = []
for yr in years:
    fd = []
    for lbl in sorted(set(cdata[cc]["clust"] for cc in valid_cc)):
        color   = CLUSTER_COLORS.get(lbl, "#aaa")
        cc_list = [cc for cc in valid_cc if cdata[cc]["clust"] == lbl]
        for cc in cc_list:
            cd    = cdata[cc]
            mask  = cd["years"] <= yr
            idxs  = np.where(mask)[0]
            if len(idxs) == 0:
                xi, yi, si = cd["x"][0], cd["y"][0], cd["size"][0]
                tx, ty = [xi], [yi]
            else:
                i   = idxs[-1]
                xi, yi, si = cd["x"][i], cd["y"][i], cd["size"][i]
                tx  = cd["x"][idxs].tolist() + [None]
                ty  = cd["y"][idxs].tolist() + [None]

            fd.append(go.Scatter(
                x=[xi], y=[yi], mode="markers+text",
                marker=dict(size=si, color=color, opacity=0.85,
                            line=dict(width=1.5, color="white")),
                text=[cc], textposition="top center",
                textfont=dict(size=8),
                customdata=[[cd["name"]]],
                hovertemplate="<b>%{customdata[0]}</b><br>Log GDP/cap: %{x:.2f}<br>ECI: %{y:.2f}<extra></extra>",
            ))
    frames.append(go.Frame(data=fd, name=str(yr)))

fig01.frames = frames

x_rng = [data05["Log_GDP_pc"].min()-0.2, data05["Log_GDP_pc"].max()+0.2]
y_rng = [data05["Economic Complexity Index"].min()-0.3, data05["Economic Complexity Index"].max()+0.3]

fig01.update_layout(**base_layout(height=620, margin=dict(l=70,r=40,t=80,b=110)),
    title=title("Evolution of Economic Complexity vs Income",
                "Bubble size = Production per Capita · Resource Profile (k=5 clusters, 1995 assignment)"),
    xaxis=dict(title="Log GDP per capita (PPP)", range=x_rng, gridcolor=GRID),
    yaxis=dict(title="Economic Complexity Index", range=y_rng, gridcolor=GRID,
               zeroline=True, zerolinecolor="#ccc"),
    legend=dict(title="Resource Profile", x=1.01, y=0.99,
                font=dict(size=11), bgcolor="rgba(250,250,250,0.9)",
                bordercolor=GRID, borderwidth=1),
    updatemenus=[dict(
        type="buttons", showactive=True, x=0.98, y=-0.13, xanchor="right",
        buttons=[
            dict(label="▶  Play", method="animate",
                 args=[None, dict(frame=dict(duration=600, redraw=True),
                                  transition=dict(duration=300))]),
            dict(label="⏸  Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
        ],
    )],
    sliders=[dict(
        active=0, len=0.82, x=0.02, y=-0.10,
        currentvalue=dict(prefix="Year: ", font=dict(size=14, color=NAVY)),
        steps=[dict(
            args=[[str(y)], dict(frame=dict(duration=400, redraw=True), mode="immediate")],
            method="animate", label=str(y),
        ) for y in years],
        font=dict(size=10),
    )],
)
save(fig01, "01_eci_animated_scatter.html")


# ══════════════════════════════════════════════════════════════════════════════
# 02  DATASET VARIABLES TREEMAP
# ══════════════════════════════════════════════════════════════════════════════
print("02 Dataset variables treemap …")

# Each entry: (full_name, short_label)
VAR_GROUPS = {
    "Identifiers": [
        ("Country Code",              "Country Code"),
        ("Country Name",              "Country Name"),
        ("Year",                      "Year"),
        ("Economic Complexity Index", "ECI"),
    ],
    "Resource Dependence": [
        ("Oil rents (% of GDP)",                  "Oil Rents"),
        ("Natural gas rents (% of GDP)",           "Gas Rents"),
        ("Mineral rents (% of GDP)",               "Mineral Rents"),
        ("Forestry rents (% of GDP)",              "Forestry Rents"),
        ("Total natural resources rents (% of GDP)","Total NR Rents"),
        ("Total_Production",                       "Total Production"),
        ("Total_Reserves",                         "Total Reserves"),
        ("Total_Production_Value",                 "Production Value"),
        ("Total_Reserves_Value",                   "Reserves Value"),
        ("Hydrocarbons_Dominant",                  "Hydrocarbons Dom."),
        ("Subsoil_Metals_Dominant",                "Subsoil Metals Dom."),
        ("Precious_Metals_Dominant",               "Precious Metals Dom."),
    ],
    "Economic Structure": [
        ("Trade (% of GDP)", "Trade"),
        ("Industry",         "Industry"),
        ("Manufacturing",    "Manufacturing"),
        ("Agriculture",      "Agriculture"),
        ("Services",         "Services"),
    ],
    "Macroeconomic Indicators": [
        ("GDP per capita (constant prices, PPP)",                                        "GDP per Capita"),
        ("Share of investment in GDP",                                                   "Investment Share"),
        ("Share of government spending in GDP",                                          "Gov. Spending Share"),
        ("Share of consumption in GDP",                                                  "Consumption Share"),
        ("Inflation, consumer prices (annual %)",                                        "Inflation"),
        ("Gross fixed capital formation, all, Constant prices, Percent of GDP",          "Gross Fixed Cap."),
        ("Capital depreciation rate",                                                    "Depreciation Rate"),
    ],
    "Fiscal & Financial": [
        ("Government revenue",                                          "Gov. Revenue"),
        ("Primary net lending, General government, Percent of GDP",    "Primary Net Lending"),
        ("Adjusted savings: gross savings (% of GNI)",                 "Gross Savings"),
        ("Domestic credit to private sector (% of GDP)",               "Domestic Credit"),
        ("Lending interest rate (%)",                                  "Lending Rate"),
        ("Real interest rate (%)",                                     "Real Rate"),
        ("Use of IMF credit (DOD, current US$)",                       "IMF Credit"),
    ],
    "Governance": [
        ("Rule of law index",             "Rule of Law"),
        ("Political corruption index",    "Pol. Corruption"),
        ("Clientelism index",             "Clientelism"),
        ("Political stability — estimate","Pol. Stability"),
        ("Property rights",               "Property Rights"),
    ],
    "Human Capital": [
        ("Human capital index",                "Human Capital Index"),
        ("Death rates, crude per 1000 people", "Death Rate"),
        ("Life expectancy at birth, total (years)", "Life Expectancy"),
    ],
    "Infrastructure": [
        ("Access to electricity (% of population)",        "Electricity Access"),
        ("Mobile cellular subscriptions (per 100 people)", "Mobile Subs."),
    ],
    "Structural": [
        ("Landlocked",                              "Landlocked"),
        ("Urban population (% of total population)","Urban Population"),
        ("knn_reliance_pct",                        "NR Reliance (KNN)"),
    ],
}

GRP_COLORS = {
    "Identifiers":           "#546E7A",
    "Resource Dependence":   "#E67E22",
    "Economic Structure":    "#2980B9",
    "Macroeconomic Indicators":"#8E44AD",
    "Fiscal & Financial":    "#C0392B",
    "Governance":            "#16A085",
    "Human Capital":         "#27AE60",
    "Infrastructure":        "#1A9E8F",
    "Structural":            "#7D3C98",
}

# Short group labels shown on parent tiles
GRP_LABELS = {
    "Identifiers":             "Identifiers",
    "Resource Dependence":     "Resource Dependence",
    "Economic Structure":      "Economic Structure",
    "Macroeconomic Indicators":"Macro Indicators",
    "Fiscal & Financial":      "Fiscal & Financial",
    "Governance":              "Governance",
    "Human Capital":           "Human Capital",
    "Infrastructure":          "Infrastructure",
    "Structural":              "Structural",
}

t_ids, t_labels, t_parents, t_vals, t_cols, t_hover = [], [], [], [], [], []
for grp, vars_ in VAR_GROUPS.items():
    g_short = GRP_LABELS[grp]
    g_col   = GRP_COLORS[grp]
    t_ids.append(grp); t_labels.append(g_short); t_parents.append("")
    t_vals.append(len(vars_)); t_cols.append(g_col + "99"); t_hover.append(f"{grp}<br>{len(vars_)} variables")
    for full, short in vars_:
        t_ids.append(f"{grp}|{full}"); t_labels.append(short); t_parents.append(grp)
        t_vals.append(1); t_cols.append(g_col); t_hover.append(full)

n_vars = sum(len(v) for v in VAR_GROUPS.values())
fig02 = go.Figure(go.Treemap(
    ids=t_ids, labels=t_labels, parents=t_parents, values=t_vals,
    customdata=t_hover,
    textinfo="label",
    textfont=dict(size=12, family=FONT, color="white"),
    insidetextfont=dict(size=11, family=FONT, color="white"),
    outsidetextfont=dict(size=11, family=FONT, color=NAVY),
    marker=dict(colors=t_cols, cornerradius=6, line=dict(width=2, color="white")),
    hovertemplate="<b>%{customdata}</b><extra></extra>",
    branchvalues="total",
    tiling=dict(packing="squarify", pad=3),
))
fig02.update_layout(
    **base_layout(height=560, margin=dict(l=10, r=10, t=75, b=10)),
    title=title("Dataset Variables Map",
                f"{n_vars} variables · {len(VAR_GROUPS)} thematic groups · hover for full name"),
)
save(fig02, "02_variables_treemap.html")


# ══════════════════════════════════════════════════════════════════════════════
# 03  DATA SOURCES BUBBLE CHART
# ══════════════════════════════════════════════════════════════════════════════
print("03 Data sources bubble chart …")

DS_MATRIX = [
    # (indicator_group, source, count)
    ("Resource Rents",    "World Bank",  4),
    ("Finance",           "World Bank",  4),
    ("Macro",             "World Bank",  4),
    ("GDP Structure",     "World Bank",  8),
    ("Demographics",      "World Bank",  3),
    ("Infrastructure",    "World Bank",  2),
    ("Human Capital",     "PWT 11.0",    1),
    ("Finance",           "PWT 11.0",    2),
    ("Macro",             "PWT 11.0",    2),
    ("GDP Structure",     "PWT 11.0",    3),
    ("Finance",           "IMF",         4),
    ("Macro",             "IMF",         2),
    ("Governance",        "V-Dem",      12),
    ("NR Production",     "EI / OWID",  16),
    ("NR Prices",         "EI / USGS",  16),
    ("Geography",         "Other",       1),
    ("Dependent Variable","Other",       1),
]

SRC_ORDER = ["World Bank", "V-Dem", "PWT 11.0", "IMF", "EI / OWID", "EI / USGS", "Other"]
IND_ORDER = ["Geography","Dependent Variable","NR Prices","NR Production",
             "Resource Rents","Finance","Macro","GDP Structure",
             "Demographics","Infrastructure","Human Capital","Governance"]
SRC_COLORS = {
    "World Bank":"#2980B9","V-Dem":"#E74C3C","PWT 11.0":"#27AE60",
    "IMF":"#E67E22","EI / OWID":"#8E44AD","EI / USGS":"#1ABC9C","Other":"#95A5A6",
}

ds_df = pd.DataFrame(DS_MATRIX, columns=["indicator","source","count"])

fig03 = go.Figure()
for src in SRC_ORDER:
    sub = ds_df[ds_df["source"] == src]
    fig03.add_trace(go.Scatter(
        x=[src]*len(sub), y=sub["indicator"],
        mode="markers+text",
        marker=dict(
            size=sub["count"]*6+12,
            color=SRC_COLORS[src],
            opacity=0.88,
            line=dict(width=1.5, color="white"),
        ),
        text=sub["count"].astype(str),
        textfont=dict(color="white", size=11, family=FONT),
        textposition="middle center",
        name=src,
        customdata=list(zip(sub["indicator"], sub["count"])),
        hovertemplate="<b>%{customdata[0]}</b><br>Source: " + src +
                      "<br>Variables: <b>%{customdata[1]}</b><extra></extra>",
    ))

fig03.update_layout(
    **base_layout(height=520, margin=dict(l=140, r=40, t=70, b=60)),
    title=title("Data Sources by Macro-Indicator Group",
                "Bubble size proportional to number of variables from that source"),
    xaxis=dict(title="", categoryorder="array", categoryarray=SRC_ORDER,
               tickangle=-30, gridcolor=GRID, tickfont=dict(size=11)),
    yaxis=dict(title="", categoryorder="array", categoryarray=IND_ORDER[::-1],
               gridcolor=GRID, tickfont=dict(size=11)),
    showlegend=False,
)
save(fig03, "03_data_sources_bubble.html")


# ══════════════════════════════════════════════════════════════════════════════
# 04  CORRELATION WITH ECI — HORIZONTAL BAR
# ══════════════════════════════════════════════════════════════════════════════
print("04 Correlation bar chart …")

CAT_MAP = {
    "Total natural resources rents (% of GDP)": ("NR Rents",          "Resource Rents"),
    "Oil rents (% of GDP)":                      ("Oil Rents",         "Resource Rents"),
    "Mineral rents (% of GDP)":                  ("Mineral Rents",     "Resource Rents"),
    "Natural gas rents (% of GDP)":              ("Gas Rents",         "Resource Rents"),
    "GDP per capita (constant prices, PPP)":     ("GDP per Capita",    "Macro & Structure"),
    "Manufacturing":                             ("Manufacturing",     "Macro & Structure"),
    "Agriculture":                               ("Agriculture",       "Macro & Structure"),
    "Services":                                  ("Services",          "Macro & Structure"),
    "Industry":                                  ("Industry",          "Macro & Structure"),
    "Trade (% of GDP)":                          ("Trade",             "Macro & Structure"),
    "Urban population (% of total population)":  ("Urban Population",  "Macro & Structure"),
    "Domestic credit to private sector (% of GDP)": ("Domestic Credit","Finance & Investment"),
    "Adjusted savings: gross savings (% of GNI)":("Savings",          "Finance & Investment"),
    "Gross fixed capital formation, all, Constant prices, Percent of GDP":
                                                 ("Capital Formation", "Finance & Investment"),
    "Share of investment in GDP":                ("Investment Share",  "Finance & Investment"),
    "Real interest rate (%)":                    ("Interest Rate",     "Finance & Investment"),
    "Lending interest rate (%)":                 ("Lending interest rate (%)","Finance & Investment"),
    "Inflation, consumer prices (annual %)":     ("Inflation",         "Finance & Investment"),
    "Capital depreciation rate":                 ("Depreciation",      "Finance & Investment"),
    "Human capital index":                       ("Human Capital",     "Human Capital & Infra"),
    "Life expectancy at birth, total (years)":   ("Life Expectancy",   "Human Capital & Infra"),
    "Access to electricity (% of population)":   ("Electricity Access","Human Capital & Infra"),
    "Mobile cellular subscriptions (per 100 people)":("Mobile Subs",  "Human Capital & Infra"),
    "Death rates, crude per 1000 people":        ("Death Rates",       "Human Capital & Infra"),
    "Rule of law index":                         ("Rule of Law",       "Governance"),
    "Political stability — estimate":            ("Political Stability","Governance"),
    "Property rights":                           ("Property Rights",   "Governance"),
    "Political corruption index":                ("Pol. Corruption",   "Governance"),
    "Government revenue":                        ("Gov Revenue",       "Governance"),
    "Primary net lending, General government, Percent of GDP":
                                                 ("Primary Lending",   "Governance"),
    "Landlocked":                                ("Landlocked",        "Macro & Structure"),
}

CAT_COLORS = {
    "Resource Rents":         "#E74C3C",
    "Macro & Structure":      "#8B5CF6",
    "Finance & Investment":   "#E67E22",
    "Human Capital & Infra":  "#1ABC9C",
    "Governance":             "#3498DB",
}

eci_col = "Economic Complexity Index"
corr_rows = []
for col, (short, cat) in CAT_MAP.items():
    if col not in panel.columns:
        continue
    sub = panel[[col, eci_col]].dropna()
    if len(sub) < 20:
        continue
    r = float(np.corrcoef(sub[col], sub[eci_col])[0, 1])
    corr_rows.append(dict(feature=col, short=short, cat=cat, r=r))

corr_df = pd.DataFrame(corr_rows).sort_values("r").reset_index(drop=True)

fig04 = go.Figure()
fig04.add_vline(x=0, line=dict(color="#ccc", width=1.5))
for cat in CAT_COLORS:
    sub = corr_df[corr_df["cat"] == cat]
    if len(sub) == 0:
        continue
    fig04.add_trace(go.Bar(
        y=sub["short"], x=sub["r"], orientation="h",
        name=cat, legendgroup=cat,
        marker=dict(color=CAT_COLORS[cat], opacity=0.87,
                    line=dict(color="white", width=0.4)),
        customdata=list(zip(sub["feature"], sub["r"])),
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Pearson r = %{x:.3f}<extra></extra>",
    ))
fig04.update_layout(
    **base_layout(height=600, margin=dict(l=140,r=40,t=70,b=70)),
    title=title("Correlation of Features with ECI",
                "Pearson r · 54-country sample · pooled 1995–2019"),
    barmode="overlay",
    xaxis=dict(title="Pearson r with ECI", gridcolor=GRID,
               zeroline=False, range=[-0.5, 0.7]),
    yaxis=dict(gridcolor=GRID, tickfont=dict(size=11)),
    legend=dict(title="Category", x=1.01, y=0.99, font=dict(size=11),
                bgcolor="rgba(250,250,250,0.9)", bordercolor=GRID, borderwidth=1),
)
save(fig04, "04_correlation_bar.html")


# ══════════════════════════════════════════════════════════════════════════════
# 05  NATURAL RESOURCE CLUSTERS — WORLD CHOROPLETH (1995)
# ══════════════════════════════════════════════════════════════════════════════
print("05 Cluster world map …")

cl95_map = cl1995[["Country Code", "Country", "ClusterLabels"]].drop_duplicates("Country Code")

fig05 = go.Figure()
for lbl in sorted(cl95_map["ClusterLabels"].dropna().unique()):
    sub   = cl95_map[cl95_map["ClusterLabels"] == lbl]
    color = CLUSTER_COLORS.get(lbl, "#aaa")
    fig05.add_trace(go.Choropleth(
        locations=sub["Country Code"],
        z=[1]*len(sub),
        colorscale=[[0, color], [1, color]],
        showscale=False, showlegend=True, name=lbl,
        text=sub["Country"],
        hovertemplate="<b>%{text}</b><br>" + lbl + "<extra></extra>",
        marker=dict(line=dict(color="white", width=0.7)),
    ))

fig05.update_geos(
    projection_type="natural earth",
    showcountries=True, countrycolor="#d0d0d0",
    showcoastlines=False,
    showland=True, landcolor="#f2f4f6",
    showocean=True, oceancolor="#dce9f5",
    showframe=False,
)
fig05.update_layout(
    **base_layout(height=560, margin=dict(l=0, r=0, t=70, b=80)),
    title=title("Natural Resource Clusters by Country",
                "k=5 clustering based on 1995 resource profile · resource-dependent countries only"),
    legend=dict(
        orientation="h",
        xanchor="center", x=0.5,
        yanchor="top",    y=-0.04,
        font=dict(size=12, family=FONT),
        bgcolor="rgba(250,250,250,0.95)",
        bordercolor=GRID, borderwidth=1,
        tracegroupgap=0,
    ),
    geo=dict(bgcolor=BG),
)
save(fig05, "05_cluster_world_map.html")


# ══════════════════════════════════════════════════════════════════════════════
# 06  MEDIAN ECI BY CLUSTER OVER TIME
# ══════════════════════════════════════════════════════════════════════════════
print("06 Median ECI by cluster …")

traj = (panel.dropna(subset=["ClusterLabels", "Economic Complexity Index"])
              .groupby(["Year", "ClusterLabels"])["Economic Complexity Index"]
              .agg(["median", "std", "count"]).reset_index())
traj.columns = ["Year", "ClusterLabels", "median", "std", "n"]
traj["se"] = traj["std"] / np.sqrt(traj["n"])

fig06 = go.Figure()
for lbl in sorted(traj["ClusterLabels"].unique()):
    sub   = traj[traj["ClusterLabels"] == lbl].sort_values("Year")
    color = CLUSTER_COLORS.get(lbl, "#999")
    # CI band
    def hex_rgba(h, a):
        r_, g_, b_ = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
        return f"rgba({r_},{g_},{b_},{a})"
    fig06.add_trace(go.Scatter(
        x=sub["Year"].tolist() + sub["Year"].tolist()[::-1],
        y=(sub["median"]+sub["se"]).tolist() + (sub["median"]-sub["se"]).tolist()[::-1],
        fill="toself", fillcolor=hex_rgba(color, 0.12),
        line=dict(width=0), showlegend=False, hoverinfo="skip", legendgroup=lbl,
    ))
    fig06.add_trace(go.Scatter(
        x=sub["Year"], y=sub["median"],
        mode="lines+markers", name=lbl, legendgroup=lbl,
        line=dict(color=color, width=2.4),
        marker=dict(size=5, color=color),
        customdata=list(zip(sub["n"], sub["std"])),
        hovertemplate=(
            "<b>" + lbl + "</b> · %{x}<br>"
            "Median ECI: <b>%{y:.3f}</b><br>"
            "Countries: %{customdata[0]}<extra></extra>"
        ),
    ))

fig06.update_layout(
    **base_layout(height=480, margin=dict(l=70,r=40,t=70,b=60)),
    title=title("Median ECI by Resource Cluster", "Shaded band = ±1 SE · 1995–2019"),
    xaxis=dict(title="Year", gridcolor=GRID, dtick=5),
    yaxis=dict(title="Median ECI", gridcolor=GRID,
               zeroline=True, zerolinecolor="#ccc"),
    legend=dict(x=1.01, y=0.99, font=dict(size=11),
                bgcolor="rgba(250,250,250,0.9)", bordercolor=GRID, borderwidth=1),
)
save(fig06, "06_median_eci_by_cluster.html")


# ══════════════════════════════════════════════════════════════════════════════
# 07 + 08 + 09  ML MODELS — LASSO/RIDGE/EN + RF + FORECAST
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ML models ===")

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

ml_df = master[(master["Year"].between(1995, 2019)) &
               (master["Country Code"].isin(include_lst))].copy()
ml_df = ml_df.sort_values(["Country Code", "Year"]).reset_index(drop=True)
ml_df = ml_df.merge(cl_map, on="Country Code", how="left")

ml_df["Total_Production_Value_Per_Capita"] = (
    ml_df["Total_Production_Value"] / ml_df["Population"].replace(0, np.nan))
ml_df["L1_ECI"] = ml_df.groupby("Country Code")["Economic Complexity Index"].shift(1)
ml_df["ECI_delta"] = ml_df["Economic Complexity Index"] - ml_df["L1_ECI"]
ml_df = ml_df.dropna(subset=["L1_ECI", "Economic Complexity Index"])

LOG_COLS = [
    "Human capital index", "Total_Production_Value_Per_Capita",
    "Gross fixed capital formation, all, Constant prices, Percent of GDP",
    "Government revenue", "Use of IMF credit (DOD, current US$)",
]
ml_df[LOG_COLS] = np.log1p(ml_df[LOG_COLS]).replace([np.inf, -np.inf], np.nan)

hci_m  = ml_df["Human capital index"].mean()
prod_m = ml_df["Total_Production_Value_Per_Capita"].mean()
gfcf_m = ml_df["Gross fixed capital formation, all, Constant prices, Percent of GDP"].mean()
ml_df["HCI_x_ProductionValue"]  = ((ml_df["Human capital index"] - hci_m) *
                                    (ml_df["Total_Production_Value_Per_Capita"] - prod_m))
ml_df["GFCF_x_ProductionValue"] = ((ml_df["Gross fixed capital formation, all, Constant prices, Percent of GDP"] - gfcf_m) *
                                    (ml_df["Total_Production_Value_Per_Capita"] - prod_m))

BASE_FEATS = [
    "Total_Production_Value_Per_Capita", "Human capital index",
    "Rule of law index", "Political stability \u2014 estimate",
    "Trade (% of GDP)",
    "Gross fixed capital formation, all, Constant prices, Percent of GDP",
    "Share of investment in GDP", "Domestic credit to private sector (% of GDP)",
    "Landlocked", "Urban population (% of total population)",
    "Government revenue", "Capital depreciation rate",
    "Use of IMF credit (DOD, current US$)", "Real interest rate (%)",
    "Inflation, consumer prices (annual %)", "Access to electricity (% of population)",
    "Adjusted savings: gross savings (% of GNI)", "L1_ECI",
    "Forestry rents (% of GDP)",
]
ALL_FEATS = BASE_FEATS + ["HCI_x_ProductionValue", "GFCF_x_ProductionValue"]
NAME_MAP  = {
    "Total_Production_Value_Per_Capita": "Production Value",
    "Human capital index": "Human Capital",
    "Rule of law index": "Rule of Law",
    "Political stability \u2014 estimate": "Political Stability",
    "Trade (% of GDP)": "Trade",
    "Gross fixed capital formation, all, Constant prices, Percent of GDP": "Capital Formation",
    "Share of investment in GDP": "Investment Share",
    "Domestic credit to private sector (% of GDP)": "Domestic Credit",
    "Landlocked": "Landlocked",
    "Urban population (% of total population)": "Urban Population",
    "Government revenue": "Gov Revenue",
    "Capital depreciation rate": "Depreciation",
    "Use of IMF credit (DOD, current US$)": "IMF Credit",
    "Real interest rate (%)": "Real Rate",
    "Inflation, consumer prices (annual %)": "Inflation",
    "Access to electricity (% of population)": "Electricity",
    "Adjusted savings: gross savings (% of GNI)": "Gross Savings",
    "L1_ECI": "Lagged ECI",
    "Forestry rents (% of GDP)": "Forestry rents (% GDP)",
    "HCI_x_ProductionValue":  "HC \u00d7 Production",
    "GFCF_x_ProductionValue": "GFCF \u00d7 Production",
}
SHORT = [NAME_MAP.get(f, f) for f in ALL_FEATS]
EXCL  = "Lagged ECI"

ml_df = ml_df.dropna(subset=ALL_FEATS)
print(f"  ML sample: {ml_df['Country Code'].nunique()} countries, {len(ml_df):,} obs")

# Panel temporal CV
class PanelTemporalCV:
    def __init__(self, years, n_splits=5, gap=1, min_train_years=8):
        uy  = np.sort(np.unique(years))
        ec  = uy[0] + min_train_years - 1
        lc  = uy[-1] - gap - 1
        self.cutoffs  = np.unique(np.linspace(ec, lc, n_splits).astype(int))
        self.n_splits = len(self.cutoffs)
        self.years    = np.asarray(years)
        self.gap      = gap
    def split(self, X=None, y=None, groups=None):
        for c in self.cutoffs:
            ti = np.where(self.years <= c)[0]
            vi = np.where(self.years > c + self.gap)[0]
            if len(ti) and len(vi):
                yield ti, vi
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

train_df = ml_df[ml_df["Year"] <= 2014].copy()
test_df  = ml_df[ml_df["Year"] >= 2015].copy()

scaler  = StandardScaler()
X_train = scaler.fit_transform(train_df[ALL_FEATS].values)
X_test  = scaler.transform(test_df[ALL_FEATS].values)
y_train = train_df["Economic Complexity Index"].values
y_test  = test_df["Economic Complexity Index"].values

tscv = PanelTemporalCV(train_df["Year"].values, n_splits=5, gap=1, min_train_years=8)

# Load from cache or refit
if os.path.exists(CACHE):
    print("  Loading ML models from cache …")
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    lasso   = cache["lasso"]
    ridge   = cache["ridge"]
    elastic = cache["elastic"]
    rf      = cache["rf"]
    fc_lasso   = cache["fc_lasso"]
    fc_ridge   = cache["fc_ridge"]
    fc_elastic = cache["fc_elastic"]
    fc_rf      = cache["fc_rf"]
    scaler_full = cache["scaler_full"]
else:
    print("  Fitting LASSO …")
    lasso   = LassoCV(cv=tscv, random_state=42, max_iter=10000).fit(X_train, y_train)
    print("  Fitting Ridge …")
    ridge   = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=tscv).fit(X_train, y_train)
    print("  Fitting Elastic Net …")
    elastic = ElasticNetCV(l1_ratio=[0.5], cv=tscv, random_state=42, max_iter=10000).fit(X_train, y_train)
    print("  Fitting Random Forest …")
    rf      = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=10,
                                     random_state=42, n_jobs=-1, oob_score=True).fit(X_train, y_train)
    # Retrain on full sample for forecasting
    X_full_raw  = ml_df[ALL_FEATS].values
    y_full      = ml_df["Economic Complexity Index"].values
    scaler_full = StandardScaler()
    X_full      = scaler_full.fit_transform(X_full_raw)
    tscv_full   = PanelTemporalCV(ml_df["Year"].values, n_splits=5, gap=1, min_train_years=8)
    print("  Fitting full-sample models for forecasting …")
    fc_lasso   = LassoCV(cv=tscv_full, random_state=42, max_iter=10000).fit(X_full, y_full)
    fc_ridge   = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=tscv_full).fit(X_full, y_full)
    fc_elastic = ElasticNetCV(l1_ratio=[0.5], cv=tscv_full, random_state=42, max_iter=10000).fit(X_full, y_full)
    fc_rf      = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=10,
                                        random_state=42, n_jobs=-1).fit(X_full, y_full)
    with open(CACHE, "wb") as f:
        pickle.dump(dict(lasso=lasso, ridge=ridge, elastic=elastic, rf=rf,
                         fc_lasso=fc_lasso, fc_ridge=fc_ridge, fc_elastic=fc_elastic,
                         fc_rf=fc_rf, scaler_full=scaler_full), f)
    print("  Saved to cache.")

def minmax(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)

imp = pd.DataFrame({"Feature": ALL_FEATS, "Short": SHORT})
imp["LASSO"]       = minmax(np.abs(lasso.coef_))
imp["Ridge"]       = minmax(np.abs(ridge.coef_))
imp["Elastic Net"] = minmax(np.abs(elastic.coef_))
imp["RF"]          = minmax(rf.feature_importances_)
imp["avg"]         = imp[["LASSO","Ridge","Elastic Net","RF"]].mean(axis=1)
imp_show = imp[imp["Short"] != EXCL].sort_values("avg", ascending=False).reset_index(drop=True)


# ── CHART 07: LASSO/Ridge/EN Coefficients ─────────────────────────────────────
print("07 LASSO/Ridge/EN coefficients …")

coef_df = pd.DataFrame({"Feature": ALL_FEATS, "Short": SHORT})
coef_df["LASSO"]       = lasso.coef_
coef_df["Ridge"]       = ridge.coef_
coef_df["Elastic Net"] = elastic.coef_
coef_df = coef_df[coef_df["Short"] != EXCL].copy()
coef_df["abs_avg"] = coef_df[["LASSO","Ridge","Elastic Net"]].abs().mean(axis=1)
coef_df = coef_df.sort_values("abs_avg", ascending=True).reset_index(drop=True)

MC = {"LASSO": "#c23a3a", "Ridge": "#3498DB", "Elastic Net": "#2e7d4a"}
fig07 = go.Figure()
fig07.add_vline(x=0, line=dict(color="#c9cfd6", width=1.5))
for m, col in MC.items():
    fig07.add_trace(go.Bar(
        y=coef_df["Short"], x=coef_df[m], name=m, orientation="h",
        marker=dict(color=col, opacity=0.88, line=dict(color="white", width=0.5)),
        hovertemplate=f"%{{y}}: %{{x:+.4f}}<extra>{m}</extra>",
    ))
fig07.update_layout(
    **base_layout(height=560, margin=dict(l=165,r=40,t=70,b=70)),
    title=title("LASSO, Ridge and Elastic Net Coefficients",
                "Standardised inputs · Panel 1996–2014 training"),
    barmode="group",
    xaxis=dict(title="Coefficient (standardised inputs)", gridcolor=GRID),
    yaxis=dict(gridcolor=GRID, tickfont=dict(size=11)),
    legend=dict(x=1.01, y=0.99, font=dict(size=11),
                bgcolor="rgba(250,250,250,0.9)", bordercolor=GRID, borderwidth=1),
)
save(fig07, "07_lasso_ridge_en_coefficients.html")


# ── CHART 08: RF Feature Importance ──────────────────────────────────────────
print("08 RF feature importance …")

rf_show = imp_show.sort_values("RF", ascending=True).copy()
fig08 = go.Figure(go.Bar(
    y=rf_show["Short"], x=rf_show["RF"], orientation="h",
    marker=dict(color="#c97030", opacity=0.9, line=dict(color="white", width=0.5)),
    hovertemplate="%{y}: %{x:.4f}<extra>Random Forest (MDI)</extra>",
))
fig08.update_layout(
    **base_layout(height=500, margin=dict(l=165,r=40,t=70,b=70)),
    title=title("Random Forest Feature Importance",
                "Mean Decrease in Impurity (MDI) · 200 trees · normalised"),
    xaxis=dict(title="Feature Importance (MDI, normalised)", gridcolor=GRID),
    yaxis=dict(gridcolor=GRID, tickfont=dict(size=11)),
    showlegend=False,
)
save(fig08, "08_rf_feature_importance.html")


# ── CHART 09: ECI FORECAST 2020-2030 ─────────────────────────────────────────
print("09 ECI forecast trajectories …")

FORECAST_YEARS  = list(range(2020, 2031))
trend_feats     = [f for f in BASE_FEATS if f != "L1_ECI"]

def extrapolate_country(cdf, yrs):
    last5 = cdf.tail(5)
    rows  = []
    for yr in yrs:
        row = {"Year": yr, "Country Code": cdf["Country Code"].iloc[0],
               "Country Name": cdf["Country Name"].iloc[0]}
        for feat in trend_feats:
            vals = last5[feat].dropna().values
            if len(vals) >= 2:
                slope, intercept = np.polyfit(np.arange(len(vals)), vals, 1)
                steps = yr - int(last5["Year"].iloc[-1])
                row[feat] = float(intercept + slope * (len(vals) - 1 + steps))
            else:
                row[feat] = float(vals[-1]) if len(vals) else 0.0
        rows.append(row)
    return rows

future_rows = []
for cc, cdf in ml_df.groupby("Country Code"):
    future_rows.extend(extrapolate_country(cdf, FORECAST_YEARS))
future_df = pd.DataFrame(future_rows)

records = []
fc_models = {"LASSO": fc_lasso, "Ridge": fc_ridge, "Elastic Net": fc_elastic, "RF": fc_rf}
for cc, cdf in ml_df.groupby("Country Code"):
    last_eci = float(cdf.sort_values("Year")["Economic Complexity Index"].iloc[-1])
    fsub     = future_df[future_df["Country Code"] == cc].sort_values("Year")
    running  = {n: last_eci for n in fc_models}
    for _, frow in fsub.iterrows():
        preds = {}
        for name, model in fc_models.items():
            row2 = frow.copy()
            row2["L1_ECI"] = running[name]
            row2["HCI_x_ProductionValue"]  = ((row2["Human capital index"] - hci_m) *
                                               (row2["Total_Production_Value_Per_Capita"] - prod_m))
            row2["GFCF_x_ProductionValue"] = ((row2["Gross fixed capital formation, all, Constant prices, Percent of GDP"] - gfcf_m) *
                                               (row2["Total_Production_Value_Per_Capita"] - prod_m))
            x_vec = np.array([row2.get(f, 0) for f in ALL_FEATS]).reshape(1, -1)
            pred  = model.predict(scaler_full.transform(x_vec))[0]
            preds[name] = pred
            running[name] = pred
        rec = {"Country Code": cc, "Country Name": frow["Country Name"],
               "Year": int(frow["Year"]), "Last_Known_ECI": last_eci,
               "Ensemble": float(np.mean(list(preds.values())))}
        rec.update(preds)
        records.append(rec)

forecast_df = pd.DataFrame(records)
forecast_df["ECI_std"] = forecast_df[list(fc_models.keys())].std(axis=1)

ranking = (forecast_df.groupby("Country Code")
           .agg(Country=("Country Name","first"),
                ECI_2019=("Last_Known_ECI","first"),
                ECI_2030=("Ensemble","last"))
           .reset_index())
ranking["Total_Change"] = ranking["ECI_2030"] - ranking["ECI_2019"]
ranking = ranking.sort_values("Total_Change", ascending=False).reset_index(drop=True)

top3    = ranking.head(3)["Country Code"].tolist()
bottom3 = ranking.tail(3)["Country Code"].tolist()
print(f"  Top 3 improvers: {top3}")
print(f"  Bottom 3 decliners: {bottom3}")

hist = ml_df[["Country Code","Country Name","Year","Economic Complexity Index"]].dropna()
all_cc = ml_df["Country Code"].unique().tolist()
TOP_C  = "#2e7d4a"
BOT_C  = "#c23a3a"
GREY_L = "#9aafc4"

def hex_rgba(h, a):
    r_, g_, b_ = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return f"rgba({r_},{g_},{b_},{a})"

figZ = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, shared_yaxes=True,
                     subplot_titles=[
                         f"Top 3 improvers: {' · '.join(top3)}",
                         f"Bottom 3 decliners: {' · '.join(bottom3)}",
                     ])

for panel_i, (highlight, h_col) in enumerate([(top3, TOP_C), (bottom3, BOT_C)], 1):
    for cc in all_cc:
        if cc in highlight: continue
        h = hist[hist["Country Code"] == cc].sort_values("Year")
        if len(h) == 0: continue
        figZ.add_trace(go.Scatter(
            x=h["Year"], y=h["Economic Complexity Index"], mode="lines",
            line=dict(color=GREY_L, width=0.6), opacity=0.2,
            showlegend=False, hoverinfo="skip",
        ), row=1, col=panel_i)
    for cc in highlight:
        h = hist[hist["Country Code"] == cc].sort_values("Year")
        f = forecast_df[forecast_df["Country Code"] == cc].sort_values("Year")
        if len(h) == 0: continue
        cname = h["Country Name"].iloc[0]
        figZ.add_trace(go.Scatter(
            x=h["Year"], y=h["Economic Complexity Index"], mode="lines",
            line=dict(color=h_col, width=2.2), showlegend=False,
            name=cname,
            hovertemplate=f"<b>{cname}</b><br>%{{x}}: ECI=%{{y:.3f}}<extra></extra>",
        ), row=1, col=panel_i)
        if len(f) == 0: continue
        f = f.sort_values("Year")
        last_y = float(h["Economic Complexity Index"].iloc[-1])
        last_x = int(h["Year"].iloc[-1])
        bx = [last_x] + f["Year"].tolist()
        by = [last_y] + f["Ensemble"].tolist()
        bstd = [0.0]  + f["ECI_std"].tolist()
        upper = [y+s for y, s in zip(by, bstd)]
        lower = [y-s for y, s in zip(by, bstd)]
        figZ.add_trace(go.Scatter(
            x=bx+bx[::-1], y=upper+lower[::-1],
            fill="toself", fillcolor=hex_rgba(h_col, 0.10),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=1, col=panel_i)
        figZ.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(color=h_col, width=1.8, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=panel_i)
        # Inline label
        xref = "x" if panel_i == 1 else "x2"
        yref = "y" if panel_i == 1 else "y2"
        figZ.add_annotation(
            x=2030, y=by[-1], xref=xref, yref=yref,
            text=f"<b>{cc}</b>", showarrow=True, ax=14, ay=0,
            arrowwidth=1, arrowcolor=h_col,
            font=dict(size=9, color=h_col), xanchor="left",
        )
    figZ.add_vline(x=2019.5, line=dict(color="#aaa", width=1.2, dash="dot"),
                   row=1, col=panel_i)
    figZ.update_xaxes(title_text="Year", gridcolor=GRID, dtick=5,
                       range=[1994, 2033], row=1, col=panel_i)

figZ.update_yaxes(title_text="Economic Complexity Index", gridcolor=GRID,
                   row=1, col=1)
figZ.update_yaxes(gridcolor=GRID, row=1, col=2)
figZ.update_layout(
    template="plotly_white", plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family=FONT, size=11, color=NAVY),
    height=580, margin=dict(l=70,r=50,t=80,b=50),
    title=title("Projected ECI Trajectories, 2020–2030",
                "Solid = historical · Dashed = ensemble forecast · Band = model disagreement"),
    showlegend=False,
)
save(figZ, "09_eci_forecast_trajectories.html")


# ══════════════════════════════════════════════════════════════════════════════
# 10  COUNTRY ECI TRAJECTORY COMPARATORS (with dropdown)
# ══════════════════════════════════════════════════════════════════════════════
print("10 ECI trajectory comparators …")

colors = {"CHL": "#c23a3a", "AZE": "#4a6fa5", "COG": "#2e7d4a"}
CODE_TO_NAME = master.groupby("Country Code")["Country Name"].first().to_dict()

traj_panel = master[master["Year"].between(1995, 2019)].dropna(
    subset=["Economic Complexity Index"]).copy()

fig10 = go.Figure()
for cc, col in colors.items():
    sub = traj_panel[traj_panel["Country Code"] == cc].sort_values("Year")
    if len(sub) == 0:
        continue
    nm = CODE_TO_NAME.get(cc, cc)
    fig10.add_trace(go.Scatter(
        x=sub["Year"], y=sub["Economic Complexity Index"],
        mode="lines+markers", name=nm,
        line=dict(color=col, width=2.5),
        marker=dict(size=5, color=col),
        hovertemplate=f"<b>{nm}</b> · %{{x}}: ECI=%{{y:.3f}}<extra></extra>",
    ))

fig10.update_layout(
    **base_layout(height=500, margin=dict(l=70, r=40, t=100, b=60)),
    title=title("ECI Trajectory: Case Study Countries",
                "1995–2019 · Chile, Azerbaijan, Congo"),
    xaxis=dict(title="Year", gridcolor=GRID, dtick=5),
    yaxis=dict(title="Economic Complexity Index", gridcolor=GRID,
               zeroline=True, zerolinecolor="#ccc"),
    legend=dict(x=1.01, y=0.99, font=dict(size=11),
                bgcolor="rgba(250,250,250,0.9)", bordercolor=GRID, borderwidth=1),
)
save(fig10, "10_eci_trajectory_comparators.html")

# ══════════════════════════════════════════════════════════════════════════════
# 11–13  CHILE CHARTS  (from pipeline state)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Chile charts ===")

PKL = os.path.join(CHILE, "_pipeline_state_6.pkl")
if not os.path.exists(PKL):
    print(f"  WARNING: Chile pipeline state not found at {PKL}, skipping charts 11-13")
else:
    with open(PKL, "rb") as _f:
        state = pickle.load(_f)

    inv       = state["inv"].copy()
    edges     = state["edges"].copy()
    ports_df  = state["ports_df"].copy()
    export_df = state["export_df"].copy()

    inv["lat"] = inv["LATITUD"].astype(float)
    inv["lon"] = inv["LONGITUD"].astype(float)

    # ── Mineral labels ──────────────────────────────────────────────────────
    MINERAL_GROUPS_CH = {
        "USD_VALUE_CU":    ("Copper",     "Base metals"),
        "USD_VALUE_MO":    ("Molybdenum", "Base metals"),
        "USD_VALUE_FE":    ("Iron",       "Base metals"),
        "USD_VALUE_AU":    ("Gold",       "Precious metals"),
        "USD_VALUE_AG":    ("Silver",     "Precious metals"),
        "USD_VALUE_LICO3": ("Lithium",    "Battery/strategic"),
        "USD_VALUE_LIOH":  ("Lithium",    "Battery/strategic"),
        "USD_VALUE_LISO4": ("Lithium",    "Battery/strategic"),
        "USD_VALUE_IO":    ("Iodine",     "Battery/strategic"),
        "USD_VALUE_NO3":   ("Nitrates",   "Industrial minerals"),
        "USD_VALUE_KCL":   ("Potash",     "Industrial minerals"),
    }
    SIMPLE_COLORS_CH = {
        "Copper":     "#1d4e89",
        "Molybdenum": "#4a86c8",
        "Iron":       "#6baed6",
        "Gold":       "#d4853b",
        "Silver":     "#b0bec5",
        "Lithium":    "#1b7837",
        "Iodine":     "#006d2c",
        "Nitrates":   "#cb181d",
        "Potash":     "#ef3b2c",
        "Other":      "#9e9e9e",
    }
    def fmt_usd(val):
        if val >= 1e9:  return f"${val/1e9:.1f}B"
        if val >= 1e6:  return f"${val/1e6:.0f}M"
        return f"${val:.0f}"

    usd_cols = [c for c in inv.columns if c.startswith("USD_VALUE_") and c != "USD_VALUE_TOTAL"]
    usd_active = [c for c in usd_cols if inv[c].sum() > 0]

    def simple_mineral(row):
        vals = {c: row[c] for c in usd_active if pd.notna(row[c]) and row[c] > 0}
        if not vals: return "Other"
        top  = max(vals, key=vals.get)
        return MINERAL_GROUPS_CH.get(top, ("Other",))[0]

    inv["simple_mineral"] = inv.apply(simple_mineral, axis=1)
    valued = inv[inv["USD_VALUE_TOTAL"] > 0].copy()

    # ── Helper short region ─────────────────────────────────────────────────
    def short_region(name):
        m = {
            "Libertador General Bernardo O'Higgins": "O'Higgins",
            "Metropolitana de Santiago": "Metropolitana",
            "Magallanes y de la Antártica Chilena": "Magallanes",
            "Aysén del General Carlos Ibáñez del Campo": "Aysén (Ibáñez)",
        }
        return m.get(name, name)

    # ── CHART 11: Top facilities bar (interactive stacked + filter) ─────────
    print("11 Chile top facilities bar …")

    for smin in SIMPLE_COLORS_CH:
        cols_s = [c for c in usd_active if MINERAL_GROUPS_CH.get(c, ("Other",))[0] == smin]
        valued[f"S_{smin}"] = valued[cols_s].sum(axis=1) if cols_s else 0.0
    s_non_other_cols = [f"S_{m}" for m in SIMPLE_COLORS_CH if m != "Other"]
    valued["S_Other"] = (valued["USD_VALUE_TOTAL"] - valued[s_non_other_cols].sum(axis=1)).clip(lower=0)
    valued["S_TOTAL"] = valued[[f"S_{m}" for m in SIMPLE_COLORS_CH]].sum(axis=1)

    # De-duplicate nearly-identical allocated rows
    def _sig(r):
        tot = r["S_TOTAL"]
        if tot <= 0: return "zero"
        return "|".join(f"{r[f'S_{m}']/tot:.2f}" for m in sorted(SIMPLE_COLORS_CH))
    valued["_ck"] = valued["S_TOTAL"].round(-3).astype(str) + "|" + valued.apply(_sig, axis=1)
    seen, drop_idx = {}, []
    for idx, row in valued.iterrows():
        ck = row["_ck"]
        if ck in seen and ck != "zero":
            drop_idx.append(idx)
        else:
            seen[ck] = idx
    valued = valued.drop(drop_idx).copy()

    top_fac = valued.nlargest(10, "USD_VALUE_TOTAL").sort_values("S_TOTAL", ascending=True)
    top_fac["short_name"] = top_fac["FACILITY_NAME"].str[:28]

    traces11 = []
    for mineral in SIMPLE_COLORS_CH:
        traces11.append(go.Bar(
            y=top_fac["short_name"],
            x=top_fac[f"S_{mineral}"],
            name=mineral, orientation="h",
            marker_color=SIMPLE_COLORS_CH[mineral],
            hovertemplate=f"<b>%{{y}}</b><br>{mineral}: %{{x:$,.0f}}<extra></extra>",
        ))

    # Annotations for total labels
    anns11 = []
    for _, row in top_fac.iterrows():
        v = row["USD_VALUE_TOTAL"]
        if v > 0:
            anns11.append(dict(
                x=v, y=row["short_name"],
                text=f"  {fmt_usd(v)}", showarrow=False,
                xanchor="left", font=dict(size=10, color="#555", family=FONT),
            ))

    # Add filter buttons: All / Exclude Copper
    n_traces = len(traces11)
    vis_all  = [True] * n_traces
    vis_nocu = [m != "Copper" for m in SIMPLE_COLORS_CH]

    n_fac_total = len(valued[valued["USD_VALUE_TOTAL"] > 0])

    # Full pool sorted ascending for JS
    pool_sorted = valued[valued["USD_VALUE_TOTAL"] > 0].sort_values("S_TOTAL", ascending=True)
    all_names_pool = pool_sorted["short_name"].tolist() if "short_name" in pool_sorted.columns else \
                     pool_sorted["FACILITY_NAME"].str[:28].tolist()

    def _make_btn11(label, vis, sort_col, n=10):
        top_n   = pool_sorted.nlargest(n, sort_col) if sort_col in pool_sorted.columns else pool_sorted.head(n)
        ordered = top_n.sort_values(sort_col if sort_col in top_n.columns else "S_TOTAL",
                                    ascending=True)["short_name"].tolist() \
                  if "short_name" in top_n.columns else \
                  top_n["FACILITY_NAME"].str[:28].tolist()
        rest    = [nm for nm in all_names_pool if nm not in ordered]
        cat     = ordered + rest
        anns_btn = []
        for _, row in top_n.iterrows():
            sc = sort_col if sort_col in row.index else "S_TOTAL"
            v  = row[sc]
            nm = row.get("short_name", row["FACILITY_NAME"][:28])
            if v > 0:
                anns_btn.append(dict(
                    x=v, y=nm, text=f"  {fmt_usd(v)}",
                    showarrow=False, xanchor="left",
                    font=dict(size=10, color="#555", family=FONT),
                ))
        return dict(label=label, method="update", args=[
            {"visible": vis},
            {"annotations": anns_btn, "xaxis.autorange": True,
             "yaxis.categoryarray": cat, "yaxis.range": [-0.5, n - 0.5]},
        ])

    # Add short_name if missing
    if "short_name" not in pool_sorted.columns:
        pool_sorted["short_name"] = pool_sorted["FACILITY_NAME"].str[:28]

    vis_all11  = [True] * len(traces11)
    vis_nocu11 = [m != "Copper" for m in SIMPLE_COLORS_CH]

    btn_all  = _make_btn11("  All minerals  ",  vis_all11,  "S_TOTAL")
    btn_nocu = _make_btn11("  Exclude copper  ", vis_nocu11, "S_TOTAL")

    fig11 = go.Figure(traces11)
    fig11.update_layout(
        **base_layout(height=660, margin=dict(l=220,r=110,t=90,b=90)),
        title=title("Top Facilities by Production Value",
                    f"Estimated 2024 production value (USD) · {n_fac_total} facilities in memory · toggle legend to filter & re-rank"),
        barmode="stack",
        xaxis=dict(title="Estimated value (USD)", gridcolor=GRID,
                   tickformat="$.2s", autorange=True, zeroline=False),
        yaxis=dict(title="", gridcolor=GRID, tickfont=dict(size=11),
                   categoryorder="array",
                   categoryarray=btn_all["args"][1]["yaxis.categoryarray"],
                   range=[-0.5, 9.5]),
        annotations=btn_all["args"][1]["annotations"],
        legend=dict(orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5,
                    font=dict(size=11), bgcolor="rgba(250,250,250,0.9)",
                    bordercolor=GRID, borderwidth=1),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=1.0, xanchor="right", y=1.14, yanchor="top",
            bgcolor="#f0f2f5", bordercolor="#c9cfd6", borderwidth=1,
            font=dict(size=11, family=FONT),
            buttons=[btn_all, btn_nocu],
        )],
    )

    # Inject JS for live legend-click rescaling (mirrors original chile_visualisations.py)
    _js11 = """<script>
(function waitForPlotly() {
    var gd = document.querySelector('.js-plotly-plot');
    if (!gd || !gd._fullData) { setTimeout(waitForPlotly, 200); return; }

    function fmtUsd(val) {
        if (val >= 1e9) return '$' + (val/1e9).toFixed(1) + 'B';
        if (val >= 1e6) return '$' + Math.round(val/1e6) + 'M';
        if (val >= 1e3) return '$' + Math.round(val/1e3) + 'K';
        return '$' + Math.round(val);
    }

    function rescaleAndSort() {
        var allNames = {};
        gd._fullData.forEach(function(trace) {
            trace.y.forEach(function(name) { allNames[name] = 0; });
        });
        gd._fullData.forEach(function(trace) {
            if (trace.visible === 'legendonly' || trace.visible === false) return;
            trace.y.forEach(function(name, i) {
                allNames[name] += (trace.x[i] || 0);
            });
        });

        var sorted = Object.keys(allNames).sort(function(a, b) {
            return allNames[a] - allNames[b];
        });
        var top10 = sorted.filter(function(n) { return allNames[n] >= 40e6; }).slice(-10);

        var anns = top10.map(function(name) {
            return {
                x: allNames[name], y: name,
                text: '  ' + fmtUsd(allNames[name]),
                showarrow: false, xanchor: 'left',
                font: {size: 10, color: '#555', family: 'IBM Plex Sans, sans-serif'}
            };
        });

        Plotly.relayout(gd, {
            'xaxis.autorange': true,
            'yaxis.categoryarray': sorted,
            'yaxis.range': [sorted.length - 10.5, sorted.length - 0.5],
            'annotations': anns
        });
    }

    gd.on('plotly_legendclick', function(evtData) {
        var idx = evtData.curveNumber;
        var vis = gd.data[idx].visible;
        var newVis = (vis === 'legendonly') ? true : 'legendonly';
        Plotly.restyle(gd, {'visible': newVis}, [idx]).then(rescaleAndSort);
        return false;
    });
    gd.on('plotly_legenddoubleclick', function() { return false; });
    gd.on('plotly_buttonclicked',     function() { setTimeout(rescaleAndSort, 50); });
})();
</script>"""

    _html11 = fig11.to_html(config=CFG, include_plotlyjs="cdn", full_html=True)
    _html11 = _html11.replace("</body>", _js11 + "\n</body>")
    _path11 = os.path.join(OUT, "11_chile_top_facilities_bar.html")
    with open(_path11, "w") as _f:
        _f.write(_html11)
    print("  → 11_chile_top_facilities_bar.html")

    # ── CHART 12: Export choropleth with mineral dropdown ───────────────────
    print("12 Chile export choropleth …")

    def estimate_usd(row):
        val  = row.get("EXPORT_VALUE", 0)
        unit = str(row.get("EXPORT_UNIT", ""))
        pf   = str(row.get("PRODUCT_FORM", ""))
        comm = str(row.get("COMMODITIES", ""))
        if not val or pd.isna(val): return 0
        if unit in ("$FOB", "$USD"): return float(val)
        if unit == "$M_FOB": return float(val) * 1e6
        if comm == "Copper" and unit == "kMT":
            prices = {"cathode": 9200, "concentrate": 2800, "blister": 8800}
            return float(val) * 1000 * prices.get(pf, 5000)
        if unit == "MT": return float(val) * 46954
        return float(val)

    exp = export_df.copy()
    exp["USD_EST"] = exp.apply(estimate_usd, axis=1)
    all_comm = sorted(exp["COMMODITIES"].dropna().unique().tolist())

    def make_exp_map(comm_filter=None):
        sub = exp if comm_filter is None else exp[exp["COMMODITIES"] == comm_filter]
        ag  = sub.groupby("TO_NAME")["USD_EST"].sum().reset_index()
        ag.columns = ["Country", "Value"]
        return ag

    # "All Minerals" + per-commodity datasets
    datasets = {"All Minerals": make_exp_map()}
    for comm in all_comm:
        datasets[comm] = make_exp_map(comm)

    # Country → ISO-3 mapping (best effort)
    CTRY_ISO = {
        "China":"CHN","USA":"USA","Japan":"JPN","South Korea":"KOR",
        "Germany":"DEU","Spain":"ESP","France":"FRA","Italy":"ITA",
        "Netherlands":"NLD","Belgium":"BEL","Sweden":"SWE","Finland":"FIN",
        "Brazil":"BRA","Argentina":"ARG","Peru":"PER","Colombia":"COL",
        "Bolivia":"BOL","Ecuador":"ECU","Australia":"AUS","India":"IND",
        "Thailand":"THA","Malaysia":"MYS","Indonesia":"IDN","Philippines":"PHL",
        "Vietnam":"VNM","Singapore":"SGP","Taiwan":"TWN","Mexico":"MEX",
        "Canada":"CAN","UK":"GBR","United Kingdom":"GBR","Turkey":"TUR",
        "Bahrain":"BHR","UAE":"ARE","Saudi Arabia":"SAU","Kuwait":"KWT",
        "South Africa":"ZAF","Morocco":"MAR","Ghana":"GHA","Panama":"PAN",
        "Costa Rica":"CRI","Guatemala":"GTM","Paraguay":"PRY","Uruguay":"URY",
        "New Zealand":"NZL","Norway":"NOR","Pakistan":"PAK","Sri Lanka":"LKA",
        "Bangladesh":"BGD","Cambodia":"KHM","Hong Kong":"HKG",
        "Bulgaria":"BGR","Poland":"POL","Portugal":"PRT","Greece":"GRC",
        "Cyprus":"CYP","Austria":"AUT","Ireland":"IRL","Lithuania":"LTU",
        "Denmark":"DNK","Hungary":"HUN","Switzerland":"CHE","Dominican Rep.":"DOM",
        "Jamaica":"JAM","El Salvador":"SLV","Honduras":"HND","Nicaragua":"NIC",
        "Venezuela":"VEN","Namibia":"NAM","Nigeria":"NGA","Algeria":"DZA",
        "Mozambique":"MOZ","Congo":"COG","Israel":"ISR","Lebanon":"LBN",
        "Guyana":"GUY","Guatemala":"GTM","Suriname":"SUR","Kuwait":"KWT",
        "Rhenium":"", "Other":"",
    }

    first_ds = datasets["All Minerals"].copy()
    first_ds["iso"] = first_ds["Country"].map(CTRY_ISO)
    first_ds = first_ds.dropna(subset=["iso"])

    all_labels = list(datasets.keys())
    fig12 = go.Figure()

    # One choropleth trace per dataset
    for li, label in enumerate(all_labels):
        ds = datasets[label].copy()
        ds["iso"] = ds["Country"].map(CTRY_ISO)
        ds = ds.dropna(subset=["iso"])
        if len(ds) == 0:
            continue
        fig12.add_trace(go.Choropleth(
            locations=ds["iso"],
            z=ds["Value"],
            text=ds["Country"],
            colorscale="YlOrRd",
            zmin=0, zmax=ds["Value"].max() if len(ds) else 1,
            showscale=True, visible=(li == 0),
            hovertemplate="<b>%{text}</b><br>Value: $%{z:,.0f}<extra></extra>",
            colorbar=dict(
                title=dict(text="USD", side="right"),
                tickprefix="$", tickformat=",.0f",
                len=0.7, y=0.5,
            ),
        ))

    # Dropdown for commodity
    buttons12 = []
    for li, label in enumerate(all_labels):
        vis = [False]*len(all_labels)
        vis[li] = True
        buttons12.append(dict(
            label=label, method="update",
            args=[{"visible": vis}, {"title.text": f"Chilean Mineral Export Destinations — {label}"}],
        ))

    fig12.update_geos(
        projection_type="natural earth",
        showcountries=True, countrycolor="#d0d0d0",
        showland=True, landcolor="#f2f4f6",
        showocean=True, oceancolor="#dce9f5",
        showframe=False, showcoastlines=False,
    )
    fig12.update_layout(
        **base_layout(height=520, margin=dict(l=0,r=0,t=90,b=10)),
        title=title("Chilean Mineral Export Destinations — All Minerals",
                    "17 commodities · select from dropdown · no-data countries in grey"),
        geo=dict(bgcolor=BG),
        updatemenus=[dict(
            buttons=buttons12, direction="down",
            x=0.98, y=1.08, xanchor="right", showactive=True,
            bgcolor="white", bordercolor=GRID,
            font=dict(family=FONT, size=11),
        )],
    )
    save(fig12, "12_chile_export_choropleth.html")

    # ── CHART 13: Regional tile cartogram ───────────────────────────────────
    print("13 Chile regional tile cartogram …")

    # Regional production value
    region_vals = {}
    if "REGION" in inv.columns:
        for region, grp in inv.groupby("REGION"):
            region_vals[short_region(region)] = float(grp["USD_VALUE_TOTAL"].sum())

    REGION_AREA_KM2 = {
        "Arica y Parinacota":  16873, "Tarapacá":  42226, "Antofagasta": 126049,
        "Atacama":  75176, "Coquimbo":  40580, "Valparaíso":  16396,
        "Metropolitana":  15403, "O'Higgins":  16387, "Maule":  30296,
        "Biobío":  23720, "Aysén (Ibáñez)": 108494, "Magallanes": 132291,
    }
    REGION_POP = {
        "Arica y Parinacota":  261000, "Tarapacá":  428000, "Antofagasta":  718000,
        "Atacama":  334000, "Coquimbo":  880000, "Valparaíso": 2009000,
        "Metropolitana": 8125000, "O'Higgins": 1012000, "Maule": 1113000,
        "Biobío": 1564000, "Aysén (Ibáñez)":  111000, "Magallanes":  183000,
    }

    REGION_ORDER = [
        "Arica y Parinacota","Tarapacá","Antofagasta","Atacama","Coquimbo",
        "Valparaíso","Metropolitana","O'Higgins","Maule","Biobío",
        "Aysén (Ibáñez)","Magallanes",
    ]
    C_MINERAL = "#4a6fa5"
    C_AREA    = "#6a7a8a"
    C_POP     = "#9a7a5a"

    UNIT_V = 1e9         # $1B per tile
    UNIT_A = 10_000      # 10,000 km² per tile
    UNIT_P = 250_000     # 250k people per tile

    COL1, COL2, COL3 = 0, 16, 32
    GW = 8

    all_tiles, all_annotations = [], []
    y_cursor = 0

    max_rows_per_region = 1
    for region in REGION_ORDER:
        v_tiles = max(1, int(round(region_vals.get(region, 0) / UNIT_V)))
        a_tiles = max(1, int(round(REGION_AREA_KM2.get(region, 0) / UNIT_A)))
        p_tiles = max(1, int(round(REGION_POP.get(region, 0) / UNIT_P)))
        rows_v  = math.ceil(v_tiles / GW)
        rows_a  = math.ceil(a_tiles / GW)
        rows_p  = math.ceil(p_tiles / GW)
        max_rows_per_region = max(max_rows_per_region, rows_v, rows_a, rows_p)

    y_cursor = 0
    for region in REGION_ORDER:
        v_tiles = max(1, int(round(region_vals.get(region, 0) / UNIT_V)))
        a_tiles = max(1, int(round(REGION_AREA_KM2.get(region, 0) / UNIT_A)))
        p_tiles = max(1, int(round(REGION_POP.get(region, 0) / UNIT_P)))
        rows_v  = math.ceil(v_tiles / GW)
        rows_a  = math.ceil(a_tiles / GW)
        rows_p  = math.ceil(p_tiles / GW)
        max_rows = max(rows_v, rows_a, rows_p)

        # Add region label
        all_annotations.append(dict(
            x=COL1 - 1, y=y_cursor + max_rows / 2,
            text=f"<b>{region}</b>", showarrow=False,
            xanchor="right", yanchor="middle",
            font=dict(size=10, color=NAVY, family=FONT),
        ))

        # Add tiles for each column
        for col_start, n_tiles, color, hover_templ, unit_str in [
            (COL1, v_tiles, C_MINERAL,
             f"<b>{region}</b><br>Mineral Value: {fmt_usd(region_vals.get(region,0))}<br>1 tile ≈ $1B",
             "1 tile ≈ $1B"),
            (COL2, a_tiles, C_AREA,
             f"<b>{region}</b><br>Area: {REGION_AREA_KM2.get(region,0):,} km²<br>1 tile ≈ 10,000 km²",
             "1 tile ≈ 10,000 km²"),
            (COL3, p_tiles, C_POP,
             f"<b>{region}</b><br>Population: {REGION_POP.get(region,0):,}<br>1 tile ≈ 250k people",
             "1 tile ≈ 250k people"),
        ]:
            row_offset = (max_rows - math.ceil(n_tiles / GW)) / 2
            for i in range(n_tiles):
                col_i = i % GW
                row_i = i // GW
                all_tiles.append(dict(
                    x=col_start + col_i, y=y_cursor + row_offset + row_i,
                    color=color, hover=hover_templ,
                ))

        y_cursor += max_rows + 0.6

    tiles_df = pd.DataFrame(all_tiles)
    fig13 = go.Figure()

    # Column headers
    for x_c, label, sub_l, color in [
        (COL1+GW/2, "<b>Mineral Value</b>",  "1 tile ≈ $1B",         C_MINERAL),
        (COL2+GW/2, "<b>Surface Area</b>",   "1 tile ≈ 10,000 km²",  C_AREA),
        (COL3+GW/2, "<b>Population</b>",      "1 tile ≈ 250k people",  C_POP),
    ]:
        all_annotations.append(dict(
            x=x_c, y=y_cursor+0.6, text=label,
            showarrow=False, xanchor="center",
            font=dict(size=12, color=color, family=FONT),
        ))
        all_annotations.append(dict(
            x=x_c, y=y_cursor+0.1, text=sub_l,
            showarrow=False, xanchor="center",
            font=dict(size=9, color=SUBTT, family=FONT),
        ))

    if len(tiles_df) > 0:
        for c in [C_MINERAL, C_AREA, C_POP]:
            sub_t = tiles_df[tiles_df["color"] == c]
            fig13.add_trace(go.Scatter(
                x=sub_t["x"]+0.5, y=sub_t["y"]+0.5,
                mode="markers",
                marker=dict(symbol="square", size=11, color=c,
                            line=dict(color="white", width=1.5)),
                text=sub_t["hover"], hoverinfo="text",
                showlegend=False,
            ))

    fig13.update_layout(
        **base_layout(height=700, margin=dict(l=180,r=40,t=70,b=30)),
        title=title("Regional Distribution of Mineral Production Value",
                    "Each tile represents a fixed unit · Antofagasta dominates mineral output"),
        xaxis=dict(visible=False, range=[-2, COL3+GW+2]),
        yaxis=dict(visible=False, range=[-0.5, y_cursor+1.5]),
        annotations=all_annotations,
    )
    save(fig13, "13_chile_regional_tiles.html")

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n✓ All charts saved to: {OUT}")
print("  Open any .html file in a browser to view.")
