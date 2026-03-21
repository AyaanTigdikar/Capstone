#!/usr/bin/env python3
"""
chile_visualisations.py  –  Improved v2
Generates 8 interactive Plotly HTML figures + 1 priority supply-chain map.

Reads:  intermediary/_pipeline_state_6.pkl
Writes: outputs/*.html

Run from anywhere:  python3 scripts/chile_visualisations.py
"""

import sys, os, pickle, math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import sample_colorscale

# ── PATHS ─────────────────────────────────────────────────────────────────────
# Derive root from this file's location:  scripts/ → Chile/ (parent)

_script_dir  = os.path.dirname(os.path.abspath(__file__))
_chile_root  = os.path.dirname(_script_dir)
DIR_OUTPUT   = os.path.join(_chile_root, "outputs")
DIR_INTERMED = os.path.join(_chile_root, "intermediary")
PKL_PATH     = os.path.join(DIR_INTERMED, "_pipeline_state_6.pkl")
NEW_OUT_DIR  = DIR_OUTPUT

# ── DESIGN SYSTEM ─────────────────────────────────────────────────────────────

FONT = "IBM Plex Sans, system-ui, -apple-system, sans-serif"

STYLE = dict(
    font_family   = FONT,
    title_size    = 16,
    axis_size     = 12,
    ann_size      = 11,
    legend_size   = 11,
    title_color   = "#1a2744",
    subtitle_color= "#6b7280",
    plot_bg       = "#fafafa",
    paper_bg      = "#fafafa",
    grid_color    = "#e5e7eb",
    grid_width    = 0.5,
    template      = "plotly_white",
)

# Shared write config – no modebar, no Plotly logo
WRITE_CONFIG = dict(displayModeBar=False, displaylogo=False, responsive=True)

# ── MINERAL COLOUR MAP (fixed – same mineral = same colour everywhere) ────────

MINERAL_GROUPS = {
    "USD_VALUE_CU":        ("Copper",            "Base metals"),
    "USD_VALUE_MO":        ("Molybdenum",        "Base metals"),
    "USD_VALUE_FE":        ("Iron",              "Base metals"),
    "USD_VALUE_ZN":        ("Zinc",              "Base metals"),
    "USD_VALUE_PB":        ("Lead",              "Base metals"),
    "USD_VALUE_AU":        ("Gold",              "Precious metals"),
    "USD_VALUE_AG":        ("Silver",            "Precious metals"),
    "USD_VALUE_LICO3":     ("Lithium Carbonate", "Battery/strategic"),
    "USD_VALUE_LIOH":      ("Lithium Hydroxide", "Battery/strategic"),
    "USD_VALUE_LISO4":     ("Lithium Sulfate",   "Battery/strategic"),
    "USD_VALUE_IO":        ("Iodine",            "Battery/strategic"),
    "USD_VALUE_NO3":       ("Nitrates",          "Industrial minerals"),
    "USD_VALUE_ULEXITE":   ("Ulexite",           "Industrial minerals"),
    "USD_VALUE_BORICACID": ("Boric Acid",        "Industrial minerals"),
    "USD_VALUE_KCL":       ("Potash",            "Industrial minerals"),
    "USD_VALUE_SALT":      ("Salt",              "Industrial minerals"),
    "USD_VALUE_CUSO4":     ("Copper Sulfate",    "Industrial minerals"),
    "USD_VALUE_LIMESTONE": ("Limestone",         "Industrial minerals"),
    "USD_VALUE_COQUINA":   ("Coquina",           "Industrial minerals"),
    "USD_VALUE_WHITECACO3":("White CaCO3",       "Industrial minerals"),
    "USD_VALUE_GYPSUM":    ("Gypsum",            "Industrial minerals"),
    "USD_VALUE_PUMICITE":  ("Pumicite",          "Industrial minerals"),
    "USD_VALUE_QUARTZ":    ("Quartz",            "Industrial minerals"),
    "USD_VALUE_SILICASAND":("Silica Sand",       "Industrial minerals"),
    "USD_VALUE_BAUXCLAY":  ("Bauxitic Clay",     "Industrial minerals"),
    "USD_VALUE_KAOLIN":    ("Kaolin",            "Industrial minerals"),
    "USD_VALUE_BENTONITE": ("Bentonite",         "Industrial minerals"),
    "USD_VALUE_DIATOMITE": ("Diatomite",         "Industrial minerals"),
    "USD_VALUE_DOLOMITE":  ("Dolomite",          "Industrial minerals"),
    "USD_VALUE_TALC":      ("Talc",              "Industrial minerals"),
    "USD_VALUE_PERLITE":   ("Perlite",           "Industrial minerals"),
    "USD_VALUE_PEAT":      ("Peat",              "Industrial minerals"),
    "USD_VALUE_PHOSPHATE": ("Phosphate Rocks",   "Industrial minerals"),
    "USD_VALUE_ZEOLITE":   ("Zeolite",           "Industrial minerals"),
}

# Per-mineral colours
MINERAL_COLORS = {
    # Base metals – blue family
    "Copper":            "#1d4e89",
    "Molybdenum":        "#4a86c8",
    "Iron":              "#6baed6",
    "Zinc":              "#9ecae1",
    "Lead":              "#c6dbef",
    # Precious metals – gold/amber
    "Gold":              "#d4853b",
    "Silver":            "#b0bec5",
    # Battery/strategic – green family
    "Lithium Carbonate": "#1b7837",
    "Lithium Hydroxide": "#31a354",
    "Lithium Sulfate":   "#74c476",
    "Iodine":            "#006d2c",
    # Industrial – red/orange family
    "Nitrates":          "#cb181d",
    "Potash":            "#ef3b2c",
    "Boric Acid":        "#67000d",
    "Ulexite":           "#99000d",
    "Salt":              "#fc9272",
    "Copper Sulfate":    "#3182bd",
    "Limestone":         "#d9a56e",
    "Coquina":           "#e8c49a",
    "White CaCO3":       "#f0ddc0",
    "Gypsum":            "#c8b8a0",
    "Pumicite":          "#b8a898",
    "Quartz":            "#a8c8d8",
    "Silica Sand":       "#90b8cc",
    "Bauxitic Clay":     "#d4a870",
    "Kaolin":            "#e8d4b8",
    "Bentonite":         "#c4a87c",
    "Diatomite":         "#d8c4a0",
    "Dolomite":          "#b8a88c",
    "Talc":              "#c8d8c0",
    "Perlite":           "#b0c8b8",
    "Peat":              "#8c7060",
    "Phosphate Rocks":   "#c8b440",
    "Zeolite":           "#a0c0a8",
    "Other":             "#9e9e9e",
}

# Group-level (for parent tiles / legend groups) – slightly desaturated
GROUP_COLORS = {
    "Base metals":         "#4a6fa5",
    "Precious metals":     "#c4782c",
    "Battery/strategic":   "#2e7d4a",
    "Industrial minerals": "#b83030",
}

# Export commodity → colour (export_df uses plain commodity names, not column names)
COMMODITY_COLORS = {
    "Copper":         "#1d4e89",
    "Molybdenum":     "#4a86c8",
    "Iron":           "#6baed6",
    "Zinc":           "#9ecae1",
    "Lead":           "#c6dbef",
    "Gold":           "#d4853b",
    "Silver":         "#b0bec5",
    "Lithium":        "#1b7837",
    "Iodine":         "#006d2c",
    "Nitrate":        "#cb181d",
    "Boron":          "#67000d",
    "Salt":           "#fc9272",
    "Potash":         "#ef3b2c",
    "Rhenium":        "#8b5cf6",
    "Copper Sulfate": "#3182bd",
    "Sulfuric Acid":  "#b8860b",
    "Selenium":       "#ec4899",
}

# Facility-type colours for supply chain map (separate from mineral palette)
FACILITY_COLORS = {
    "mine":        "#1d4e89",
    "concentrator":"#4a86c8",
    "smelter":     "#e07b39",
    "plant":       "#1a9850",
    "mo_plant":    "#8b5cf6",
    "sx_ew":       "#16a085",
    "re_plant":    "#b83030",
    "port":        "#e74c3c",
    "country":     "#9e9e9e",
}

FACILITY_LABELS = {
    "mine":        "Mine",
    "concentrator":"Concentrator",
    "smelter":     "Smelter",
    "plant":       "Processing plant",
    "mo_plant":    "Mo plant",
    "sx_ew":       "SX-EW plant",
    "re_plant":    "Re plant",
    "port":        "Export port",
    "country":     "Export country",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def fmt_usd(val):
    if val >= 1e9:
        return f"${val/1e9:.1f}B"
    elif val >= 1e6:
        return f"${val/1e6:.0f}M"
    elif val >= 1e3:
        return f"${val/1e3:.0f}K"
    return f"${val:.0f}"


def short_region(name):
    mapping = {
        "Libertador General Bernardo O'Higgins": "O'Higgins",
        "Metropolitana de Santiago": "Metropolitana",
        "Magallanes y de la Antártica Chilena": "Magallanes",
        "Arica y Parinacota": "Arica y Parinacota",
        "Aysén del General Carlos Ibáñez del Campo": "Aysén (Ibáñez)",
    }
    return mapping.get(name, name)


def desaturate(hex_color, factor=0.45):
    """Blend hex_color toward white by factor (0=original, 1=white)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def subtitle_annotation(text, y=1.04):
    return dict(
        text=text, xref="paper", yref="paper",
        x=0.5, y=y, xanchor="center", yanchor="bottom",
        showarrow=False,
        font=dict(size=STYLE["ann_size"], color=STYLE["subtitle_color"],
                  family=FONT),
    )


def base_layout(**kw):
    d = dict(
        template=STYLE["template"],
        plot_bgcolor=STYLE["plot_bg"],
        paper_bgcolor=STYLE["paper_bg"],
        font=dict(family=FONT, size=STYLE["axis_size"],
                  color=STYLE["title_color"]),
        margin=dict(l=60, r=40, t=20, b=50),
        height=560,
    )
    d.update(kw)
    return d


def great_circle_arcs(rows_iter, n_pts=25):
    """
    Given an iterable of (lat1, lon1, lat2, lon2), return concatenated
    lat/lon lists with None separators for Scattergeo line traces.
    """
    all_lats, all_lons = [], []
    for (φ1d, λ1d, φ2d, λ2d) in rows_iter:
        φ1, λ1 = math.radians(φ1d), math.radians(λ1d)
        φ2, λ2 = math.radians(φ2d), math.radians(λ2d)
        d = 2 * math.asin(math.sqrt(
            math.sin((φ2 - φ1) / 2) ** 2
            + math.cos(φ1) * math.cos(φ2) * math.sin((λ2 - λ1) / 2) ** 2
        ))
        if d < 0.001:
            all_lats += [φ1d, φ2d, None]
            all_lons += [λ1d, λ2d, None]
            continue
        t = np.linspace(0, 1, n_pts)
        A = np.sin((1 - t) * d) / math.sin(d)
        B = np.sin(t * d) / math.sin(d)
        x = A * math.cos(φ1) * math.cos(λ1) + B * math.cos(φ2) * math.cos(λ2)
        y = A * math.cos(φ1) * math.sin(λ1) + B * math.cos(φ2) * math.sin(λ2)
        z = A * math.sin(φ1) + B * math.sin(φ2)
        lats = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        lons = np.degrees(np.arctan2(y, x))
        all_lats += lats.tolist() + [None]
        all_lons += lons.tolist() + [None]
    return all_lats, all_lons


PRODUCT_FORM_PRICES = {
    "cathode": 9_200, "concentrate": 2_800, "blister": 8_800,
}


def estimate_usd(row):
    val  = row.get("EXPORT_VALUE", 0)
    unit = str(row.get("EXPORT_UNIT", ""))
    pf   = str(row.get("PRODUCT_FORM", ""))
    comm = str(row.get("COMMODITIES", ""))
    if not val or pd.isna(val):
        return 0
    if unit in ("$FOB", "$USD"):
        return val
    if unit == "$M_FOB":
        return val * 1e6
    if comm == "Copper" and unit == "kMT":
        return val * 1_000 * PRODUCT_FORM_PRICES.get(pf, 5_000)
    if unit == "MT":
        return val * 46_954
    return val


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    if not os.path.exists(PKL_PATH):
        print(f"ERROR: pipeline state not found:\n  {PKL_PATH}")
        sys.exit(1)

    print(f"Loading {PKL_PATH}")
    with open(PKL_PATH, "rb") as _f:
        state = pickle.load(_f)

    inv       = state["inv"].copy()
    edges     = state["edges"].copy()
    ports_df  = state["ports_df"].copy()
    export_df = state["export_df"].copy()

    inv["lat"] = inv["LATITUD"].astype(float)
    inv["lon"] = inv["LONGITUD"].astype(float)

    usd_cols        = [c for c in inv.columns if c.startswith("USD_VALUE_") and c != "USD_VALUE_TOTAL"]
    usd_cols_active = [c for c in usd_cols if inv[c].sum() > 0]

    def dominant_mineral(row):
        vals = {c: row[c] for c in usd_cols_active if pd.notna(row[c]) and row[c] > 0}
        if not vals:
            return "Other", "Industrial minerals"
        top = max(vals, key=vals.get)
        return MINERAL_GROUPS.get(top, ("Other", "Industrial minerals"))

    inv[["dominant_mineral", "mineral_group"]] = pd.DataFrame(
        inv.apply(dominant_mineral, axis=1).tolist(), index=inv.index
    )
    valued = inv[inv["USD_VALUE_TOTAL"] > 0].copy()

    os.makedirs(NEW_OUT_DIR, exist_ok=True)
    print(f"Saving figures to: {NEW_OUT_DIR}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # 01  TREEMAP – production value by mineral
    # ══════════════════════════════════════════════════════════════════════════
    print("01 Treemap …")

    tree_data = []
    for col in usd_cols_active:
        val = inv[col].sum()
        if val > 0:
            mineral, group = MINERAL_GROUPS[col]
            tree_data.append(dict(mineral=mineral, group=group, value=val))
    tree_df = pd.DataFrame(tree_data).sort_values("value", ascending=False)

    # Build ids / labels / parents / values / colors manually for full control
    t_ids, t_labels, t_parents, t_values, t_colors, t_text = [], [], [], [], [], []

    # Group level
    for grp, gcol in GROUP_COLORS.items():
        gtotal = tree_df[tree_df["group"] == grp]["value"].sum()
        if gtotal <= 0:
            continue
        t_ids.append(grp); t_labels.append(grp); t_parents.append("")
        t_values.append(gtotal)
        t_colors.append(desaturate(gcol, 0.5))
        t_text.append(fmt_usd(gtotal))

    # Mineral level
    for _, row in tree_df.iterrows():
        mid = f"{row['group']}|{row['mineral']}"
        t_ids.append(mid); t_labels.append(row["mineral"]); t_parents.append(row["group"])
        t_values.append(row["value"])
        t_colors.append(MINERAL_COLORS.get(row["mineral"], GROUP_COLORS.get(row["group"], "#999")))
        t_text.append(fmt_usd(row["value"]))

    fig1 = go.Figure(go.Treemap(
        ids=t_ids, labels=t_labels, parents=t_parents, values=t_values,
        text=t_text, textinfo="label+text",
        textfont=dict(size=14, family=FONT, color="white"),
        insidetextfont=dict(size=14, family=FONT),
        marker=dict(colors=t_colors, cornerradius=4,
                    line=dict(width=1.5, color="white")),
        hovertemplate="<b>%{label}</b><br>Value: %{text}<extra></extra>",
        branchvalues="total",
    ))
    fig1.update_layout(
        **base_layout(height=560, margin=dict(l=10, r=10, t=65, b=10)),
        title=dict(
            text=(
                "Production Value by Mineral"
                "<br><sup style='font-size:11px;font-weight:normal;color:#6b7280'>"
                "Estimated 2024 production value (USD) · 31 minerals · 4 categories"
                "</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=STYLE["title_size"], color=STYLE["title_color"]),
        ),
        coloraxis_showscale=False,
    )
    fig1.write_html(f"{NEW_OUT_DIR}/01_treemap_mineral_value.html",
                    config=WRITE_CONFIG, include_plotlyjs="cdn")
    print("  → 01_treemap_mineral_value.html")

    # ══════════════════════════════════════════════════════════════════════════
    # 02  TOP FACILITIES BAR – full pool in memory, JS auto-sort/rescale
    # ══════════════════════════════════════════════════════════════════════════
    print("02 Top facilities bar …")

    # Simplified mineral buckets (fewer categories for legibility)
    SIMPLE_MINERALS = {
        "USD_VALUE_CU":    "Copper",
        "USD_VALUE_MO":    "Molybdenum",
        "USD_VALUE_FE":    "Iron",
        "USD_VALUE_AU":    "Gold",
        "USD_VALUE_AG":    "Silver",
        "USD_VALUE_LICO3": "Lithium",
        "USD_VALUE_LIOH":  "Lithium",
        "USD_VALUE_LISO4": "Lithium",
        "USD_VALUE_IO":    "Iodine",
        "USD_VALUE_NO3":   "Nitrates",
        "USD_VALUE_KCL":   "Potash",
    }
    for col in usd_cols_active:
        SIMPLE_MINERALS.setdefault(col, "Other")

    SIMPLE_COLORS = {
        "Copper":     MINERAL_COLORS["Copper"],
        "Molybdenum": MINERAL_COLORS["Molybdenum"],
        "Iron":       MINERAL_COLORS["Iron"],
        "Gold":       MINERAL_COLORS["Gold"],
        "Silver":     MINERAL_COLORS["Silver"],
        "Lithium":    MINERAL_COLORS["Lithium Carbonate"],
        "Iodine":     MINERAL_COLORS["Iodine"],
        "Nitrates":   MINERAL_COLORS["Nitrates"],
        "Potash":     MINERAL_COLORS["Potash"],
        "Other":      "#9e9e9e",
    }

    pool = valued.copy()
    for smin in SIMPLE_COLORS:
        src = [c for c in usd_cols_active if SIMPLE_MINERALS.get(c) == smin]
        pool[f"S_{smin}"] = pool[src].sum(axis=1) if src else 0.0
    s_cols = [f"S_{m}" for m in SIMPLE_COLORS]
    pool["S_TOTAL"] = pool[s_cols].sum(axis=1)
    pool["S_NONCU"] = pool["S_TOTAL"] - pool.get("S_Copper", 0)

    # Collapse equal-weight allocated clusters (identical rounded totals + mineral mix)
    def _sig(r):
        tot = r["S_TOTAL"]
        if tot <= 0: return "zero"
        return "|".join(f"{r[f'S_{m}']/tot:.2f}" for m in sorted(SIMPLE_COLORS))
    pool["_ck"] = pool["S_TOTAL"].round(-3).astype(str) + "|" + pool.apply(_sig, axis=1)
    for ck, grp in pool.groupby("_ck"):
        if len(grp) > 1:
            keep = grp.index[0]
            n    = len(grp)
            pool.loc[keep, "FACILITY_NAME"] = pool.loc[keep, "FACILITY_NAME"][:18] + f" +{n-1} others"
            pool = pool.drop(grp.index[1:])

    # Full pool in memory — no top-N filter
    pool["short_name"] = pool["FACILITY_NAME"].str[:32]
    n_fac_total = len(pool[pool["S_TOTAL"] > 0])

    # All names in ascending-total order (baseline for JS re-sort)
    all_sorted = pool[pool["S_TOTAL"] > 0].sort_values("S_TOTAL", ascending=True)
    all_names  = all_sorted["short_name"].tolist()

    mineral_display = [m for m in SIMPLE_COLORS if pool[f"S_{m}"].sum() > 0]
    trace_names = []
    traces2 = []
    for mineral in mineral_display:
        # Every trace carries ALL facilities so JS can sum across any subset
        traces2.append(go.Bar(
            y=all_names,
            x=all_sorted[f"S_{mineral}"].tolist(),
            name=mineral, orientation="h",
            marker_color=SIMPLE_COLORS[mineral],
            hovertemplate=f"<b>%{{y}}</b><br>{mineral}: %{{x:$.3s}}<extra></extra>",
        ))
        trace_names.append(mineral)

    def _make_btn(label, vis, sort_col, n=10):
        top_n   = pool.nlargest(n, sort_col)
        ordered = top_n.sort_values(sort_col, ascending=True)["short_name"].tolist()
        rest    = [nm for nm in all_names if nm not in ordered]
        cat     = ordered + rest   # ordered FIRST so range [-0.5, n-0.5] shows them
        anns = []
        for _, row in top_n.iterrows():
            v = row[sort_col]
            if v > 0:
                anns.append(dict(
                    x=v, y=row["short_name"],
                    text=f"  {fmt_usd(v)}",
                    showarrow=False, xanchor="left",
                    font=dict(size=10, color="#555", family=FONT),
                ))
        return dict(label=label, method="update", args=[
            {"visible": vis},
            {"annotations": anns, "xaxis.autorange": True,
             "yaxis.categoryarray": cat, "yaxis.range": [-0.5, n - 0.5]},
        ])

    vis_all  = [True] * len(traces2)
    vis_nocu = [tn != "Copper" for tn in trace_names]

    # Derive initial layout from the "All minerals" button state
    _init_btn = _make_btn("init", vis_all, "S_TOTAL")

    fig2 = go.Figure(data=traces2)
    fig2.update_layout(
        **base_layout(height=660, margin=dict(l=220, r=110, t=75, b=90)),
        barmode="stack",
        title=dict(
            text=(
                "Top Facilities by Production Value"
                "<br><sup style='font-size:11px;font-weight:normal;color:#6b7280'>"
                f"Estimated 2024 production value (USD) · {n_fac_total} facilities in memory"
                " · toggle legend to filter &amp; re-rank"
                "</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=STYLE["title_size"], color=STYLE["title_color"]),
        ),
        annotations=_init_btn["args"][1]["annotations"],
        xaxis=dict(
            title=dict(text="Estimated value (USD)", font=dict(size=STYLE["axis_size"])),
            tickformat="$.2s",
            gridcolor=STYLE["grid_color"], gridwidth=STYLE["grid_width"],
            showgrid=True, zeroline=False, autorange=True,
        ),
        yaxis=dict(
            title="", categoryorder="array",
            categoryarray=_init_btn["args"][1]["yaxis.categoryarray"],
            range=[-0.5, 9.5],
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5,
            font=dict(size=STYLE["legend_size"]),
        ),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=1.0, xanchor="right", y=1.16, yanchor="top",
            bgcolor="#f0f2f5", bordercolor="#c9cfd6", borderwidth=1,
            font=dict(size=11, family=FONT),
            buttons=[
                _make_btn("  All minerals  ", vis_all,  "S_TOTAL"),
                _make_btn("  Exclude copper  ", vis_nocu, "S_NONCU"),
            ],
        )],
    )

    # JS patch — mirrors notebook exactly: gd._fullData for data, gd.data for click state
    _js2 = """<script>
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

    _html2 = fig2.to_html(config=WRITE_CONFIG, include_plotlyjs="cdn", full_html=True)
    _html2 = _html2.replace("</body>", _js2 + "\n</body>")
    with open(f"{NEW_OUT_DIR}/02_top_facilities_bar.html", "w") as _f:
        _f.write(_html2)
    print(f"  → 02_top_facilities_bar.html  ({n_fac_total} facilities in memory)")

    # ══════════════════════════════════════════════════════════════════════════
    # 03  FACILITY MAP – light basemap, sized by value, coloured by mineral
    # ══════════════════════════════════════════════════════════════════════════
    print("03 Facility map …")

    # Fetch Natural Earth 110m countries GeoJSON once at script time,
    # strip all properties (geometry only), exclude Chile → embedded grey overlay
    import json, urllib.request as _ureq
    _NE_URL = (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
        "/master/geojson/ne_110m_admin_0_countries.geojson"
    )
    _non_chile_layer = None
    try:
        with _ureq.urlopen(_NE_URL, timeout=15) as _r:
            _world = json.loads(_r.read().decode())
        _non_chile_layer = {
            "sourcetype": "geojson",
            "source": {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": f["geometry"], "properties": {}}
                    for f in _world["features"]
                    if f.get("properties", {}).get("ADM0_A3") != "CHL"
                ],
            },
            "type": "fill",
            "color": "rgba(185,185,185,0.48)",
        }
        print("  Grey overlay loaded.")
    except Exception as _e:
        print(f"  Warning: grey overlay unavailable ({_e})")

    map_df = valued.copy()
    # Sqrt-scale with generous floor/cap so markers are visible but not overwhelming
    map_df["size_raw"] = np.sqrt(map_df["USD_VALUE_TOTAL"] / 1e6)
    map_df["size_px"]  = (map_df["size_raw"] * 1.5).clip(lower=4, upper=30)
    map_df["hover"] = map_df.apply(lambda r: (
        f"<b>{r['FACILITY_NAME']}</b><br>"
        f"Type: {r.get('FACILITY_TYPE', r.get('FACTYPE',''))}<br>"
        f"Mineral: {r['dominant_mineral']}<br>"
        f"Region: {short_region(r['REGION'])}<br>"
        f"Value: <b>{fmt_usd(r['USD_VALUE_TOTAL'])}</b>"
    ), axis=1)

    fig3 = go.Figure()

    # Add traces per mineral group (for legend)
    for grp in ["Base metals", "Precious metals", "Battery/strategic", "Industrial minerals"]:
        sub = map_df[map_df["mineral_group"] == grp]
        if len(sub) == 0:
            continue
        fig3.add_trace(go.Scattermap(
            lat=sub["lat"], lon=sub["lon"],
            mode="markers", name=grp,
            marker=dict(
                size=sub["size_px"], color=GROUP_COLORS[grp],
                opacity=0.75, sizemode="diameter",
            ),
            text=sub["hover"], hoverinfo="text",
            hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT),
        ))

    # Size legend markers (invisible traces, just for legend)
    for label, ref_val in [("$100M", 100e6), ("$1B", 1e9), ("$10B", 10e9)]:
        sz = float(np.clip(np.sqrt(ref_val / 1e6) * 1.5, 4, 30))
        fig3.add_trace(go.Scattermap(
            lat=[None], lon=[None], mode="markers", name=label,
            marker=dict(size=sz, color="#cccccc", opacity=0.9),
            legendgroup="size", legendgrouptitle_text="Size = value",
        ))

    fig3.update_layout(
        **base_layout(height=700, margin=dict(l=0, r=0, t=20, b=0)),
        annotations=[subtitle_annotation("Marker size ∝ √(production value); colour by mineral group", y=0.97)],
        map=dict(
            style="open-street-map",
            center=dict(lat=-26.5, lon=-69.5),
            zoom=4.2,
            layers=[_non_chile_layer] if _non_chile_layer else [],
        ),
        legend=dict(
            yanchor="bottom", y=0.03, xanchor="right", x=0.98,
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#dde1e7", borderwidth=1,
            font=dict(size=STYLE["legend_size"]),
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Reset view", method="relayout",
                          args=[{"map.center.lat": -26.5,
                                 "map.center.lon": -69.5,
                                 "map.zoom": 4.2}])],
            x=0.98, xanchor="right", y=0.97, yanchor="top",
            bgcolor="#f0f2f5", bordercolor="#c9cfd6", borderwidth=1,
            font=dict(size=11, family=FONT),
        )],
    )
    fig3.write_html(f"{NEW_OUT_DIR}/03_facility_map.html",
                    config=WRITE_CONFIG, include_plotlyjs="cdn")
    print("  → 03_facility_map.html")

    # ══════════════════════════════════════════════════════════════════════════
    # 05  EXPORT CHOROPLETH – sequential colour, grey for no-data, $B colorbar
    # ══════════════════════════════════════════════════════════════════════════
    print("05 Export choropleth …")

    COUNTRY_ISO = {
        "China":"CHN","Japan":"JPN","South Korea":"KOR","USA":"USA",
        "Brazil":"BRA","India":"IND","Germany":"DEU","Spain":"ESP",
        "France":"FRA","Italy":"ITA","Netherlands":"NLD","Belgium":"BEL",
        "Sweden":"SWE","Bulgaria":"BGR","Finland":"FIN","Canada":"CAN",
        "Mexico":"MEX","Taiwan":"TWN","Thailand":"THA","Philippines":"PHL",
        "Malaysia":"MYS","Indonesia":"IDN","Vietnam":"VNM","Peru":"PER",
        "Colombia":"COL","Argentina":"ARG","Turkey":"TUR",
        "United Kingdom":"GBR","Switzerland":"CHE","Singapore":"SGP",
        "Greece":"GRC","Portugal":"PRT","Panama":"PAN","Bahrain":"BHR",
        "UAE":"ARE","Hong Kong":"HKG","Poland":"POL","Norway":"NOR",
        "Costa Rica":"CRI","Cambodia":"KHM","South Africa":"ZAF",
        "Namibia":"NAM","Bangladesh":"BGD","Bolivia":"BOL","Ecuador":"ECU",
        "Dominican Rep.":"DOM","Paraguay":"PRY","Australia":"AUS",
        "Congo":"COG","Guatemala":"GTM","Denmark":"DNK","Uruguay":"URY",
        "Honduras":"HND","Cyprus":"CYP","Saudi Arabia":"SAU",
        "Austria":"AUT","New Zealand":"NZL","Ireland":"IRL",
        "Nigeria":"NGA","Ghana":"GHA","Pakistan":"PAK","Morocco":"MAR",
        "Jamaica":"JAM","Algeria":"DZA","Mozambique":"MOZ","Hungary":"HUN",
        "El Salvador":"SLV","Nicaragua":"NIC","Venezuela":"VEN",
        "Lebanon":"LBN","Kuwait":"KWT","Sri Lanka":"LKA","Israel":"ISR",
        "Lithuania":"LTU",
    }

    # Build per-commodity country aggregates
    comm_country_data = {}
    for comm in export_df["COMMODITIES"].unique():
        sub = export_df[export_df["COMMODITIES"] == comm].copy()
        sub["USD_EST"] = sub.apply(estimate_usd, axis=1)
        total = sub["USD_EST"].sum()
        if total < 1e6:
            continue
        by_c = sub.groupby("TO_NAME")["USD_EST"].sum().reset_index()
        by_c.columns = ["country", "usd"]
        by_c["iso"] = by_c["country"].map(COUNTRY_ISO)
        by_c = by_c.dropna(subset=["iso"])
        by_c = by_c[by_c["usd"] > 0]
        by_c["hover"] = by_c.apply(
            lambda r: f"<b>{r['country']}</b><br>Export value: {fmt_usd(r['usd'])}<br>Commodity: {comm}",
            axis=1)
        comm_country_data[comm] = (by_c, total)

    sorted_comms = sorted(comm_country_data, key=lambda c: -comm_country_data[c][1])

    # "All" aggregate (used for default view)
    all_frames = [comm_country_data[c][0][["country","usd","iso"]].copy() for c in sorted_comms]
    all_combined = pd.concat(all_frames).groupby(["country","iso"])["usd"].sum().reset_index()
    # Add top commodity per country for "All" hover
    top_comm_per_country = {}
    for comm in sorted_comms:
        cdf = comm_country_data[comm][0]
        for _, r in cdf.iterrows():
            if r["country"] not in top_comm_per_country or \
               r["usd"] > comm_country_data.get(top_comm_per_country[r["country"]], (None, 0))[1]:
                top_comm_per_country[r["country"]] = comm
    all_combined["hover"] = all_combined.apply(
        lambda r: (f"<b>{r['country']}</b><br>Total exports: {fmt_usd(r['usd'])}"
                   f"<br>Largest commodity: {top_comm_per_country.get(r['country'], '')}"),
        axis=1)

    top3 = sorted_comms[:3]
    rest = sorted_comms[3:]
    trace_labels = ["All Minerals"] + top3 + rest
    n_traces5    = len(trace_labels)

    COLORSCALES5 = {
        "All Minerals": "YlOrRd",
        "Copper":       "Blues",
        "Molybdenum":   "Oranges",
        "Lithium":      "Greens",
        "Iodine":       "Reds",
        "Iron":         "Greys",
        "Gold":         "YlOrBr",
    }
    _DEFAULT_CS5 = "Purples"

    fig5 = go.Figure()

    for idx, label in enumerate(trace_labels):
        if label == "All Minerals":
            cdf    = all_combined
            zvals  = cdf["usd"] / 1e9
            zmin5, zmax5 = 0, zvals.max()
            cs     = "YlOrRd"
            cbtitle= "USD (B)"
        else:
            cdf   = comm_country_data[label][0]
            zvals = cdf["usd"] / 1e9
            zmin5, zmax5 = 0, zvals.max() if len(zvals) else 1
            cs    = COLORSCALES5.get(label, _DEFAULT_CS5)
            cbtitle = "USD (B)"

        # Grey fill for all countries not in this dataset
        fig5.add_trace(go.Choropleth(
            locations=cdf["iso"], z=zvals,
            text=cdf["hover"], hoverinfo="text",
            colorscale=cs,
            zmin=zmin5, zmax=zmax5,
            marker_line_color="#c9cfd6", marker_line_width=0.4,
            colorbar=dict(
                title=dict(text=cbtitle, font=dict(size=11)),
                thickness=14, len=0.65,
                tickformat="$.1f",
                ticksuffix="B",
            ),
            visible=(idx == 0),
        ))

    def _vis5(i):
        v = [False] * n_traces5
        v[i] = True
        return v

    # Single dropdown on right — all commodities in one place, no title overlap
    all_dd_items5 = [dict(label=" All Minerals ", method="update",
                          args=[{"visible": _vis5(0)}])]
    for i, lb in enumerate(sorted_comms):
        total_m = comm_country_data[lb][1] / 1e6
        all_dd_items5.append(dict(
            label=f"{lb}  (${total_m:,.0f}M)",
            method="update", args=[{"visible": _vis5(i + 1)}],
        ))

    update_menus5 = [dict(
        type="dropdown", x=1.0, xanchor="right", y=1.06, yanchor="top",
        bgcolor="#f0f2f5", bordercolor="#c9cfd6", borderwidth=1,
        font=dict(size=11, family=FONT), buttons=all_dd_items5,
    )]

    fig5.update_layout(
        **base_layout(height=520, margin=dict(l=0, r=0, t=85, b=0)),
        title=dict(
            text=(
                "Chilean Mineral Export Destinations"
                "<br><sup style='font-size:11px;font-weight:normal;color:#6b7280'>"
                f"{len(sorted_comms)} commodities · select from dropdown (top right) · no-data countries in grey"
                "</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=STYLE["title_size"], color=STYLE["title_color"]),
        ),
        annotations=[],
        geo=dict(
            showframe=False, showcoastlines=True, coastlinecolor="#c9cfd6",
            projection_type="natural earth", bgcolor="rgba(0,0,0,0)",
            showland=True, landcolor="#f5f5f5", showcountries=True,
            countrycolor="#dde1e7", countrywidth=0.4,
            showocean=True, oceancolor="#eaf3fb",
        ),
        updatemenus=update_menus5,
    )
    fig5.write_html(f"{NEW_OUT_DIR}/05_export_choropleth.html",
                    config=WRITE_CONFIG, include_plotlyjs="cdn")
    print(f"  → 05_export_choropleth.html  ({n_traces5} commodity traces)")

    # ══════════════════════════════════════════════════════════════════════════
    # 06b  THREE-COLUMN TILE CARTOGRAM – Mineral Value vs Area vs Population
    # ══════════════════════════════════════════════════════════════════════════
    print("06b Regional tile cartogram …")

    region_vals = inv.groupby("REGION")["USD_VALUE_TOTAL"].sum()
    region_vals.index = region_vals.index.map(short_region)
    if "Santiago" in region_vals.index and "Metropolitana" in region_vals.index:
        region_vals["Metropolitana"] += region_vals.pop("Santiago")
    elif "Santiago" in region_vals.index:
        region_vals = region_vals.rename(index={"Santiago": "Metropolitana"})

    REGION_ORDER_NS = [
        "Arica y Parinacota", "Tarapacá", "Antofagasta", "Atacama",
        "Coquimbo", "Valparaíso", "Metropolitana", "O'Higgins",
        "Maule", "Biobío", "Aysén (Ibáñez)", "Magallanes",
    ]
    REGION_AREA_KM2 = {
        "Arica y Parinacota": 16873,  "Tarapacá": 42226,
        "Antofagasta": 126049,         "Atacama": 75176,
        "Coquimbo": 40580,             "Valparaíso": 16396,
        "Metropolitana": 15403,        "O'Higgins": 16387,
        "Maule": 30296,                "Biobío": 23890,
        "Aysén (Ibáñez)": 108494,      "Magallanes": 132291,
    }
    REGION_POP = {
        "Arica y Parinacota": 239126,  "Tarapacá": 336769,
        "Antofagasta": 622640,          "Atacama": 312486,
        "Coquimbo": 771085,             "Valparaíso": 1825757,
        "Metropolitana": 7314176,       "O'Higgins": 918751,
        "Maule": 1042989,               "Biobío": 2114286,
        "Aysén (Ibáñez)": 108328,       "Magallanes": 164661,
    }

    GW6        = 8          # tiles per row in each column (wider → fewer rows, better H stretch)
    TILE_PAD6  = 0.07
    GAP_ROWS6  = 0.6
    VALUE_TILE = 1e9
    AREA_TILE  = 10_000
    POP_TILE   = 250_000

    C6_MINERAL = "#4a6fa5"
    C6_AREA    = "#a8b8c8"
    C6_POP     = "#c4a882"

    COL1 = 0
    COL2 = GW6 + 2.5
    COL3 = COL2 + GW6 + 2.5
    LABEL_X = -0.3

    def _tiles6(items, unit):
        return [items[0]] * max(1, round(items[1] / unit)) if items[1] > 0 else []

    def _rows6(n): return 0 if n == 0 else (n - 1) // GW6 + 1

    all_tiles6, anns6 = [], []
    cur_y6 = 0

    for region in reversed(REGION_ORDER_NS):
        vt = _tiles6((f"<b>{region}</b><br>{fmt_usd(region_vals.get(region, 0))}",
                      region_vals.get(region, 0)), VALUE_TILE)
        at = _tiles6((f"<b>{region}</b><br>{REGION_AREA_KM2.get(region, 0):,.0f} km²",
                      REGION_AREA_KM2.get(region, 0)), AREA_TILE)
        pt = _tiles6((f"<b>{region}</b><br>{REGION_POP.get(region, 0):,.0f} people",
                      REGION_POP.get(region, 0)), POP_TILE)

        if not vt and not at and not pt:
            continue

        max_rows = max(_rows6(len(vt)), _rows6(len(at)), _rows6(len(pt)))

        for col_start, tiles, color in [
            (COL1, vt, C6_MINERAL),
            (COL2, at, C6_AREA),
            (COL3, pt, C6_POP),
        ]:
            row_offset = (max_rows - _rows6(len(tiles))) / 2
            for i, hover in enumerate(tiles):
                all_tiles6.append(dict(
                    x=col_start + i % GW6,
                    y=cur_y6 + row_offset + i // GW6,
                    color=color, hover=hover,
                ))

        anns6.append(dict(
            x=LABEL_X, y=cur_y6 + max_rows / 2,
            text=f"<b>{region}</b>",
            showarrow=False, xanchor="right",
            font=dict(size=11, color="#333", family=FONT),
        ))
        cur_y6 += max_rows + GAP_ROWS6

    tiles6_df = pd.DataFrame(all_tiles6)

    # Column headers (added after loop so cur_y6 = top of chart)
    for cx6, label, unit_note, color in [
        (COL1 + GW6 / 2, "Mineral Value",  "1 tile ≈ $1B",         C6_MINERAL),
        (COL2 + GW6 / 2, "Surface Area",   "1 tile ≈ 10,000 km²",  "#6a7a8a"),
        (COL3 + GW6 / 2, "Population",     "1 tile ≈ 250k people",  "#9a7a5a"),
    ]:
        anns6.append(dict(
            x=cx6, y=cur_y6 + 0.8,
            text=f"<b>{label}</b><br>"
                 f"<span style='font-size:10px;color:#777'>{unit_note}</span>",
            showarrow=False, xanchor="center",
            font=dict(size=14, color=color, family=FONT),
        ))

    fig6b = go.Figure()
    for _, t in tiles6_df.iterrows():
        fig6b.add_shape(
            type="rect",
            x0=t["x"] + TILE_PAD6, x1=t["x"] + 1 - TILE_PAD6,
            y0=t["y"] + TILE_PAD6, y1=t["y"] + 1 - TILE_PAD6,
            fillcolor=t["color"], opacity=0.85,
            line=dict(width=0), layer="below",
        )
    fig6b.add_trace(go.Scatter(
        x=tiles6_df["x"] + 0.5, y=tiles6_df["y"] + 0.5,
        mode="markers",
        marker=dict(size=20, color="rgba(0,0,0,0)"),
        text=tiles6_df["hover"], hoverinfo="text",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT),
        showlegend=False,
    ))

    total_w6 = COL3 + GW6 + 1
    fig6b.update_layout(
        plot_bgcolor=STYLE["plot_bg"], paper_bgcolor=STYLE["paper_bg"],
        font=dict(family=FONT, color=STYLE["title_color"]),
        width=980,
        height=max(660, int(cur_y6 * 15) + 80),
        margin=dict(l=130, r=30, t=20, b=20),
        annotations=anns6,
        xaxis=dict(visible=False, range=[LABEL_X - 1.5, total_w6 + 0.8]),
        yaxis=dict(visible=False, range=[-0.8, cur_y6 + 2.2]),
        showlegend=False,
    )
    fig6b.write_html(f"{NEW_OUT_DIR}/06b_regional_tile_cartogram.html",
                     config=WRITE_CONFIG, include_plotlyjs="cdn",
                     default_width="100%", default_height="100%")
    print("  → 06b_regional_tile_cartogram.html")

    # ══════════════════════════════════════════════════════════════════════════
    # 07  SUNBURST – region (inner) → mineral group → mineral (outer)
    # ══════════════════════════════════════════════════════════════════════════
    print("07 Sunburst …")

    sun_records = []
    for _, row in valued.iterrows():
        region = short_region(row["REGION"])
        for col in usd_cols_active:
            v = row[col]
            if pd.notna(v) and v > 0:
                mineral, group = MINERAL_GROUPS[col]
                sun_records.append(dict(region=region, group=group, mineral=mineral, value=v))
    sun_df  = pd.DataFrame(sun_records)
    sun_agg = sun_df.groupby(["region","group","mineral"])["value"].sum().reset_index()

    # Collapse small regions into "Other regions" (< 2% of total)
    reg_totals = sun_agg.groupby("region")["value"].sum()
    total_val  = reg_totals.sum()
    big_regs   = reg_totals[reg_totals / total_val >= 0.02].index.tolist()
    sun_agg["region"] = sun_agg["region"].apply(lambda r: r if r in big_regs else "Other regions")
    sun_agg = sun_agg.groupby(["region","group","mineral"], observed=True)["value"].sum().reset_index()

    # Collapse small minerals (< $300M) to "Other (group)"
    rows_out = []
    for (reg, grp), g in sun_agg.groupby(["region","group"], observed=True):
        big   = g[g["value"] >= 300e6]
        small = g[g["value"] < 300e6]
        rows_out.extend(big.to_dict("records"))
        if small["value"].sum() > 0:
            rows_out.append(dict(region=reg, group=grp,
                                 mineral=f"Other ({grp.split()[0].lower()})",
                                 value=small["value"].sum()))
    sun_agg = pd.DataFrame(rows_out)

    region_order7 = (sun_agg.groupby("region", observed=True)["value"]
                     .sum().sort_values(ascending=False).index.tolist())
    if "Other regions" in region_order7:
        region_order7.remove("Other regions")
        region_order7.append("Other regions")

    REGION_COLOR7      = "#d0d5de"
    OTHER_REGION_COLOR7= "#e8e8e8"
    GROUP_COLORS_LIGHT7 = {k: desaturate(v, 0.4) for k, v in GROUP_COLORS.items()}

    reg_sums7  = sun_agg.groupby("region", observed=True)["value"].sum()
    rg_sums7   = sun_agg.groupby(["region","group"], observed=True)["value"].sum()

    ids7, labels7, parents7, vals7, colors7 = [], [], [], [], []
    total7 = reg_sums7.sum()

    for reg in region_order7:
        v = float(reg_sums7.get(reg, 0))
        ids7.append(reg); labels7.append(reg); parents7.append(""); vals7.append(v)
        colors7.append(OTHER_REGION_COLOR7 if reg == "Other regions" else REGION_COLOR7)

    for (reg, grp), v in rg_sums7.items():
        gid = f"{reg}|{grp}"
        ids7.append(gid); labels7.append(grp); parents7.append(reg); vals7.append(float(v))
        colors7.append(GROUP_COLORS.get(grp, "#999"))

    for _, row in sun_agg.iterrows():
        mid = f"{row['region']}|{row['group']}|{row['mineral']}"
        ids7.append(mid); labels7.append(row["mineral"])
        parents7.append(f"{row['region']}|{row['group']}")
        vals7.append(float(row["value"]))
        if row["mineral"].startswith("Other ("):
            colors7.append(GROUP_COLORS_LIGHT7.get(row["group"], "#bbb"))
        else:
            colors7.append(MINERAL_COLORS.get(row["mineral"],
                           GROUP_COLORS.get(row["group"], "#999")))

    # Show text only for segments that are large enough to fit label
    display_text7 = []
    for v, lbl in zip(vals7, labels7):
        pct = v / total7 * 100
        display_text7.append(lbl if pct >= 1.5 else "")

    hover7 = []
    for v, lbl in zip(vals7, labels7):
        pct = v / total7 * 100
        hover7.append(f"<b>{lbl}</b><br>Value: {fmt_usd(v)}<br>{pct:.1f}% of total")

    fig7 = go.Figure(go.Sunburst(
        ids=ids7, labels=labels7, parents=parents7, values=vals7,
        text=display_text7, textinfo="text",
        customdata=hover7,
        hovertemplate="%{customdata}<extra></extra>",
        marker=dict(colors=colors7, line=dict(width=0.8, color="white")),
        branchvalues="total",
        insidetextorientation="radial",
        maxdepth=3,
    ))
    _lay7 = base_layout(margin=dict(l=10, r=10, t=55, b=10))
    _lay7.pop("height", None)
    _lay7["autosize"] = True
    _lay7["title"] = dict(
        text=(
            "Production Value by Region and Mineral"
            "<br><sup style='font-size:11px;font-weight:normal;color:#6b7280'>"
            "Inner ring: region · Middle: mineral group · Outer: individual mineral"
            "</sup>"
        ),
        x=0.5, xanchor="center",
        font=dict(size=STYLE["title_size"], color=STYLE["title_color"]),
    )
    _lay7["annotations"] = []
    _lay7["margin"] = dict(l=10, r=10, t=65, b=10)
    fig7.update_layout(**_lay7)
    fig7.write_html(f"{NEW_OUT_DIR}/07_sunburst_region_mineral.html",
                    config=WRITE_CONFIG, include_plotlyjs="cdn",
                    default_height="100%", default_width="100%")
    print("  → 07_sunburst_region_mineral.html")

    # ══════════════════════════════════════════════════════════════════════════
    # 10  NON-COPPER VALUE BAR – grouped by mineral
    # ══════════════════════════════════════════════════════════════════════════
    print("10 Non-copper value bar …")

    noncu = []
    for col in usd_cols_active:
        if col == "USD_VALUE_CU":
            continue
        v = inv[col].sum()
        if v > 0:
            mineral, group = MINERAL_GROUPS[col]
            noncu.append(dict(mineral=mineral, group=group, value=v))
    noncu_df = pd.DataFrame(noncu).sort_values("value", ascending=True)

    fig10 = go.Figure()
    for grp in ["Industrial minerals", "Battery/strategic", "Precious metals", "Base metals"]:
        sub10 = noncu_df[noncu_df["group"] == grp]
        if len(sub10) == 0:
            continue
        fig10.add_trace(go.Bar(
            y=sub10["mineral"], x=sub10["value"],
            orientation="h", name=grp,
            marker_color=GROUP_COLORS.get(grp, "#999"),
            legendgroup=grp,
            hovertemplate="<b>%{y}</b><br>Value: %{x:$.3s}<extra></extra>",
        ))

    # Value labels
    anns10 = []
    for _, r in noncu_df.iterrows():
        anns10.append(dict(
            x=r["value"], y=r["mineral"],
            text=f"  {fmt_usd(r['value'])}",
            showarrow=False, xanchor="left",
            font=dict(size=10, color="#555", family=FONT),
        ))
    # Explanatory note – placed below x-axis to avoid legend overlap
    anns10.append(dict(
        xref="paper", yref="paper", x=0.5, y=-0.10,
        text="<i>Excludes copper — Chile's dominant export by value</i>",
        showarrow=False, xanchor="center", yanchor="top",
        font=dict(size=10, color=STYLE["subtitle_color"], family=FONT),
    ))

    _n_noncu = len(noncu_df)
    fig10.update_layout(
        **base_layout(height=680, margin=dict(l=165, r=110, t=75, b=70)),
        title=dict(
            text=(
                "Non-Copper Mineral Production Value"
                f"<br><sup style='font-size:11px;font-weight:normal;color:#6b7280'>"
                f"Estimated 2024 value (USD, log scale) · {_n_noncu} minerals · grouped by category"
                "</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=STYLE["title_size"], color=STYLE["title_color"]),
        ),
        annotations=anns10,
        barmode="stack",
        xaxis=dict(
            title=dict(text="Estimated value (USD)", font=dict(size=STYLE["axis_size"])),
            type="log", tickformat="$.2s",
            gridcolor=STYLE["grid_color"], gridwidth=STYLE["grid_width"],
            showgrid=True, zeroline=False,
        ),
        yaxis=dict(title=""),
        legend=dict(
            orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#dde1e7", borderwidth=1,
            font=dict(size=STYLE["legend_size"]),
        ),
        showlegend=True,
    )
    fig10.write_html(f"{NEW_OUT_DIR}/10_non_copper_value_bar.html",
                     config=WRITE_CONFIG, include_plotlyjs="cdn")
    print("  → 10_non_copper_value_bar.html")

    # ══════════════════════════════════════════════════════════════════════════
    # SUPPLY CHAIN MAP  ← PRIORITY
    # Chile Mineral Supply Chain 2024
    # ══════════════════════════════════════════════════════════════════════════
    print("Supply chain map (priority) …")

    # ── TUNABLE CONSTANTS ─────────────────────────────────────────────────────
    MINE_THRESHOLD  = 500e6   # domestic edges: only mines with value ≥ this
    SC_TOP_ARCS     = 30      # top N port→country export arcs to draw
    SC_CENTER_LAT   = -10.0   # initial map centre (global Pacific view)
    SC_CENTER_LON   = -30.0
    SC_PROJ_SCALE   = 1.0     # initial zoom scale (1 = full world)
    SC_CENTER_LAT_C = -29.0   # Chile domestic view centre
    SC_CENTER_LON_C = -70.5
    SC_PROJ_SCALE_C = 7.5     # zoom into Chile
    OP_DOM_ON       = 0.45    # domestic edge opacity (highlighted)
    OP_DOM_OFF      = 0.05    # domestic edge opacity (dimmed by filter)
    OP_EXP_ON       = 0.70    # export arc opacity (highlighted)
    OP_EXP_OFF      = 0.05    # export arc opacity (dimmed by filter)
    # ──────────────────────────────────────────────────────────────────────────

    # ── Node lookup: merge edges with inventory to get values ─────────────────
    mine_val_map = inv.set_index("FACILITY_NAME")["USD_VALUE_TOTAL"].to_dict()
    mine_dom_map = inv.set_index("FACILITY_NAME")["dominant_mineral"].to_dict()
    m2p    = edges[edges["EDGE_TYPE"] == "mine_to_plant"].copy()
    m2p["_mine_val"] = m2p["FROM_NAME"].map(mine_val_map).fillna(0)
    m2p    = m2p[m2p["_mine_val"] >= MINE_THRESHOLD]

    other_dom = edges[~edges["EDGE_TYPE"].isin(["mine_to_plant", "port_to_country"])]
    dom_edges = pd.concat([m2p, other_dom], ignore_index=True)

    # ── Port-to-country arc filtering ────────────────────────────────────────
    exp_aug = export_df.copy()
    exp_aug["USD_EST"] = exp_aug.apply(estimate_usd, axis=1)
    # Aggregate by (port, country) summing all commodities; find dominant commodity
    pc_comm = (exp_aug.groupby(["FROM_NAME","TO_NAME","FROM_LAT","FROM_LON",
                                 "TO_LAT","TO_LON","COMMODITIES"])["USD_EST"]
               .sum().reset_index())
    pc_total = (exp_aug.groupby(["FROM_NAME","TO_NAME","FROM_LAT","FROM_LON",
                                  "TO_LAT","TO_LON"])["USD_EST"]
                .sum().reset_index())
    _idx = pc_comm.groupby(["FROM_NAME","TO_NAME"])["USD_EST"].idxmax().dropna()
    dominant_comm = pc_comm.loc[_idx, ["FROM_NAME","TO_NAME","COMMODITIES"]]
    pc_total = pc_total.merge(dominant_comm, on=["FROM_NAME","TO_NAME"], how="left")
    # Keep top N port→country flows by value
    arc_df = pc_total.nlargest(SC_TOP_ARCS, "USD_EST")

    # ── Continent aggregation ──────────────────────────────────────────────────
    CONTINENT_CENTROIDS = {
        "Asia":          ( 35.0,  105.0),
        "Europe":        ( 50.0,   15.0),
        "North America": ( 40.0, -100.0),
        "South America": (-15.0,  -60.0),
        "Oceania":       (-25.0,  135.0),
        "Africa":        (  5.0,   20.0),
        "Middle East":   ( 28.0,   48.0),
    }
    CONTINENT_COLORS = {
        "Asia":          "#e07b39",
        "Europe":        "#4a6fa5",
        "North America": "#c0392b",
        "South America": "#8e44ad",
        "Oceania":       "#27ae60",
        "Africa":        "#c9a227",
        "Middle East":   "#16a085",
    }
    # Only center + scale — lonaxis/lataxis.range conflicts with projection.scale
    # in natural earth and produces unpredictable results.
    CONTINENT_BOUNDS_JS = {
        "Asia":          {"geo.center.lat": 30,  "geo.center.lon": 105, "geo.projection.scale": 2.2},
        "Europe":        {"geo.center.lat": 52,  "geo.center.lon": 15,  "geo.projection.scale": 3.5},
        "North America": {"geo.center.lat": 40,  "geo.center.lon":-100, "geo.projection.scale": 2.0},
        "South America": {"geo.center.lat":-15,  "geo.center.lon": -60, "geo.projection.scale": 2.5},
        "Oceania":       {"geo.center.lat":-25,  "geo.center.lon": 140, "geo.projection.scale": 2.5},
        "Africa":        {"geo.center.lat":  5,  "geo.center.lon":  20, "geo.projection.scale": 2.2},
        "Middle East":   {"geo.center.lat": 28,  "geo.center.lon":  48, "geo.projection.scale": 4.0},
    }
    COUNTRY_CONTINENT = {
        "China":"Asia","Japan":"Asia","South Korea":"Asia","India":"Asia",
        "Taiwan":"Asia","Thailand":"Asia","Philippines":"Asia","Malaysia":"Asia",
        "Indonesia":"Asia","Vietnam":"Asia","Singapore":"Asia","Cambodia":"Asia",
        "Bangladesh":"Asia","Pakistan":"Asia","Sri Lanka":"Asia","Hong Kong":"Asia",
        "USA":"North America","Canada":"North America","Mexico":"North America",
        "Panama":"North America","Costa Rica":"North America","Guatemala":"North America",
        "Honduras":"North America","El Salvador":"North America","Nicaragua":"North America",
        "Dominican Rep.":"North America","Jamaica":"North America",
        "Germany":"Europe","Spain":"Europe","France":"Europe","Italy":"Europe",
        "Netherlands":"Europe","Belgium":"Europe","Sweden":"Europe","Finland":"Europe",
        "Bulgaria":"Europe","United Kingdom":"Europe","Switzerland":"Europe",
        "Poland":"Europe","Norway":"Europe","Greece":"Europe","Portugal":"Europe",
        "Cyprus":"Europe","Austria":"Europe","Ireland":"Europe","Lithuania":"Europe",
        "Denmark":"Europe","Hungary":"Europe",
        "Brazil":"South America","Argentina":"South America","Peru":"South America",
        "Colombia":"South America","Bolivia":"South America","Ecuador":"South America",
        "Paraguay":"South America","Uruguay":"South America","Venezuela":"South America",
        "Australia":"Oceania","New Zealand":"Oceania",
        "Turkey":"Middle East","Bahrain":"Middle East","UAE":"Middle East",
        "Saudi Arabia":"Middle East","Kuwait":"Middle East","Israel":"Middle East",
        "Lebanon":"Middle East",
        "South Africa":"Africa","Namibia":"Africa","Nigeria":"Africa",
        "Ghana":"Africa","Morocco":"Africa","Algeria":"Africa","Mozambique":"Africa",
        "Congo":"Africa",
    }
    arc_df["continent"] = arc_df["TO_NAME"].map(COUNTRY_CONTINENT).fillna("Other")
    # Continent totals: sum USD_EST, centroid lat/lon, dominant commodity
    cont_arcs = (arc_df[arc_df["continent"] != "Other"]
                 .groupby(["FROM_NAME","FROM_LAT","FROM_LON","continent"])
                 .agg(USD_EST=("USD_EST","sum"),
                      dom_comm=("COMMODITIES", lambda x: x.value_counts().index[0] if len(x) else ""))
                 .reset_index())
    cont_arcs["TO_LAT"] = cont_arcs["continent"].map(
        lambda c: CONTINENT_CENTROIDS.get(c,(0,0))[0])
    cont_arcs["TO_LON"] = cont_arcs["continent"].map(
        lambda c: CONTINENT_CENTROIDS.get(c,(0,0))[1])
    arc_comm_total = arc_df.groupby("COMMODITIES")["USD_EST"].sum()  # kept for filter ranking

    # ── Build unique node list ────────────────────────────────────────────────
    # Extract all facility nodes from filtered domestic edges
    from_nodes = dom_edges[["FROM_NAME","FROM_TYPE","FROM_LAT","FROM_LON"]].rename(
        columns={"FROM_NAME":"name","FROM_TYPE":"ftype","FROM_LAT":"lat","FROM_LON":"lon"})
    to_nodes   = dom_edges[["TO_NAME","TO_TYPE","TO_LAT","TO_LON"]].rename(
        columns={"TO_NAME":"name","TO_TYPE":"ftype","TO_LAT":"lat","TO_LON":"lon"})
    port_nodes_from_arcs = arc_df[["FROM_NAME","FROM_LAT","FROM_LON"]].rename(
        columns={"FROM_NAME":"name","FROM_LAT":"lat","FROM_LON":"lon"})
    port_nodes_from_arcs["ftype"] = "port"

    facility_nodes = (pd.concat([from_nodes, to_nodes, port_nodes_from_arcs], ignore_index=True)
                      .dropna(subset=["lat","lon"])
                      .drop_duplicates(subset=["name","ftype"])
                      .copy())
    facility_nodes = facility_nodes[facility_nodes["ftype"] != "country"]
    facility_nodes["usd_val"] = facility_nodes["name"].map(mine_val_map).fillna(0)
    facility_nodes["dominant_mineral"] = facility_nodes["name"].map(mine_dom_map).fillna("Other")

    # Country destination nodes
    country_nodes = (arc_df[["TO_NAME","TO_LAT","TO_LON"]]
                     .rename(columns={"TO_NAME":"name","TO_LAT":"lat","TO_LON":"lon"})
                     .drop_duplicates(subset=["name"]).copy())
    country_nodes["ftype"] = "country"
    country_nodes["usd_val"] = 0.0

    # ── Build figure ──────────────────────────────────────────────────────────
    figsc = go.Figure()

    # Track trace indices for visibility + filter arrays
    _dom_idxs      = {}   # comm → trace index (domestic edge traces)
    _cont_idxs     = {}   # continent → trace index (continent-level arc traces)
    _ctry_det_idxs = {}   # continent → trace index (country-detail arcs, JS-only)
    _fac_idxs      = {}   # ftype → trace index (individual facility nodes)
    _port_idx  = None
    _ctry_idx  = None
    _clust_idx = None  # geographic cluster trace (Full / Export views)
    _label_idx = None  # mine label trace (Domestic view only)

    # ─ Layer 1: background choropleth ────────────────────────────────────────
    SA = ["ARG","BOL","BRA","COL","ECU","GUY","PRY","PER","SUR","URY","VEN","GUF"]
    figsc.add_trace(go.Choropleth(
        locations=SA + ["CHL"], z=[1]*len(SA) + [2],
        colorscale=[[0,"#e0e3e8"],[0.5,"#e0e3e8"],[0.5,"#f0f2f5"],[1,"#f0f2f5"]],
        locationmode="ISO-3",
        marker_line_color="#bfc5cc", marker_line_width=0.5,
        showscale=False, showlegend=False, hoverinfo="skip",
    ))

    # ─ Layer 2: domestic edges ────────────────────────────────────────────────
    # showlegend=False on all: filter buttons serve as the sole mineral key,
    # eliminating the duplicate listing in legend + buttons.
    dom_comms_ordered = sorted(dom_edges["COMMODITIES"].dropna().unique())
    for comm in dom_comms_ordered:
        sub = dom_edges[dom_edges["COMMODITIES"] == comm]
        _fl = sub["FROM_LAT"].tolist(); _tl = sub["TO_LAT"].tolist()
        _fn = sub["FROM_LON"].tolist(); _tn = sub["TO_LON"].tolist()
        _none = [None] * len(_fl)
        lats = [v for t in zip(_fl, _tl, _none) for v in t]
        lons = [v for t in zip(_fn, _tn, _none) for v in t]
        color = COMMODITY_COLORS.get(comm, "#aaaaaa")
        _dom_idxs[comm] = len(figsc.data)
        figsc.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            name=comm, legendgroup=f"comm_{comm}",
            line=dict(width=1.6, color=color),
            opacity=float(OP_DOM_ON), hoverinfo="skip",
            showlegend=False,
        ))

    # ─ Layer 3a: continent-level export arcs (one trace per continent) ────────
    # Width ∝ log(total USD flow to that continent). Click → drill down to countries.
    _cont_order = sorted(cont_arcs["continent"].unique(),
                         key=lambda c: -cont_arcs[cont_arcs["continent"]==c]["USD_EST"].sum())
    _cont_max_val = cont_arcs.groupby("continent")["USD_EST"].sum().max() if len(cont_arcs) else 1e9

    for cont in _cont_order:
        sub_c = cont_arcs[cont_arcs["continent"] == cont]
        tot_val = float(sub_c["USD_EST"].sum())
        arc_w = float(np.clip(np.log10(max(tot_val, 1e6) / 1e6) * 1.8 + 1.2, 1.2, 9.0))
        arc_lats, arc_lons = great_circle_arcs(
            zip(sub_c["FROM_LAT"], sub_c["FROM_LON"],
                sub_c["TO_LAT"],   sub_c["TO_LON"]), n_pts=15)
        color = CONTINENT_COLORS.get(cont, "#aaaaaa")
        htxt = (f"<b>{cont}</b><br>Total flow: {fmt_usd(tot_val)}"
                f"<br>{len(sub_c)} port route{'s' if len(sub_c)>1 else ''}"
                "<br><i>Click to expand countries</i>")
        _cont_idxs[cont] = len(figsc.data)
        figsc.add_trace(go.Scattergeo(
            lat=arc_lats, lon=arc_lons, mode="lines",
            name=cont, legendgroup=f"cont_{cont}",
            line=dict(width=arc_w, color=color),
            opacity=float(OP_EXP_ON),
            text=htxt, hoverinfo="text",
            hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT),
            showlegend=False,
        ))

    # ─ Layer 3b: country-level detail arcs per continent (hidden, JS-revealed) ─
    for cont in _cont_order:
        sub_ctry = arc_df[arc_df["continent"] == cont]
        arc_lats, arc_lons = great_circle_arcs(
            zip(sub_ctry["FROM_LAT"], sub_ctry["FROM_LON"],
                sub_ctry["TO_LAT"],   sub_ctry["TO_LON"]), n_pts=15)
        color = CONTINENT_COLORS.get(cont, "#aaaaaa")
        _ctry_det_idxs[cont] = len(figsc.data)
        figsc.add_trace(go.Scattergeo(
            lat=arc_lats, lon=arc_lons, mode="lines",
            name=cont + " (countries)", legendgroup=f"ctry_{cont}",
            line=dict(width=1.4, color=color, dash="dot"),
            opacity=0.75, hoverinfo="skip",
            showlegend=False, visible=False,
        ))

    # ─ Layer 4: individual facility nodes (Domestic view — hidden initially) ──
    # Each ftype has its own legendgroup so toggling is per-type.
    _ftype_order = ["mine","concentrator","sx_ew","plant","mo_plant","smelter","re_plant","port"]
    _used_ftypes = []
    for ftype in _ftype_order:
        sub_n = facility_nodes[facility_nodes["ftype"] == ftype]
        if len(sub_n) == 0:
            continue
        _used_ftypes.append(ftype)
        sizes = ([16] * len(sub_n) if ftype == "port"
                 else np.clip(np.sqrt(sub_n["usd_val"] / 1e6) * 1.6, 5, 30).tolist())
        hover_n = sub_n.apply(lambda r: (
            f"<b>{r['name']}</b><br>"
            f"Type: {FACILITY_LABELS.get(r['ftype'], r['ftype'])}<br>"
            + (f"Dominant: {r['dominant_mineral']}<br>" if r['dominant_mineral'] != 'Other' else "")
            + (f"Value: {fmt_usd(r['usd_val'])}" if r['usd_val'] > 0 else "")
        ), axis=1).tolist()
        sym = "square" if ftype == "port" else "circle"
        kw = {"legendgrouptitle_text": "Facility type"} if len(_used_ftypes) == 1 else {}
        if ftype == "port":
            _port_idx = len(figsc.data)
        _fac_idxs[ftype] = len(figsc.data)
        figsc.add_trace(go.Scattergeo(
            lat=sub_n["lat"].tolist(), lon=sub_n["lon"].tolist(),
            mode="markers", name=FACILITY_LABELS.get(ftype, ftype),
            legendgroup=f"node_{ftype}",
            marker=dict(
                size=sizes, color=FACILITY_COLORS[ftype],
                symbol=sym, opacity=0.88,
                line=dict(width=0.9, color="white"),
                sizemode="diameter",
            ),
            text=hover_n, hoverinfo="text",
            hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT),
            showlegend=True, visible=False,  # shown only in Domestic view
            **kw,
        ))

    # Record name-order for each ftype trace (same order as points in the trace)
    _ftype_pt_names = {
        ftype: facility_nodes[facility_nodes["ftype"] == ftype]["name"].tolist()
        for ftype in _fac_idxs
    }

    # ─ Layer 5: export country destination nodes ──────────────────────────────
    # Pre-compute per-country: commodity % breakdown + source Chilean ports
    _ctry_comm  = arc_df.groupby(["TO_NAME","COMMODITIES"])["USD_EST"].sum().reset_index()
    _ctry_ports = arc_df.groupby(["TO_NAME","FROM_NAME"])["USD_EST"].sum().reset_index()

    def _ctry_hover(country):
        total = arc_df[arc_df["TO_NAME"] == country]["USD_EST"].sum()
        comms = (_ctry_comm[_ctry_comm["TO_NAME"] == country]
                 .sort_values("USD_EST", ascending=False).head(4))
        comm_parts = [f"{row['COMMODITIES']} <b>{row['USD_EST']/total*100:.0f}%</b>"
                      for _, row in comms.iterrows() if row["USD_EST"] > 0]
        ports = (_ctry_ports[_ctry_ports["TO_NAME"] == country]
                 .sort_values("USD_EST", ascending=False).head(3)["FROM_NAME"].tolist())
        return (f"<b>{country}</b>"
                f"<br>Total: <b>{fmt_usd(total)}</b>"
                f"<br>{'  ·  '.join(comm_parts)}"
                f"<br><span style='color:#6b7280'>Via: {', '.join(ports)}</span>")

    _ctry_idx = len(figsc.data)
    country_hover = [_ctry_hover(r["name"]) for _, r in country_nodes.iterrows()]
    figsc.add_trace(go.Scattergeo(
        lat=country_nodes["lat"].tolist(), lon=country_nodes["lon"].tolist(),
        mode="markers", name="Export country",
        legendgroup="node_country",
        marker=dict(size=8, color=FACILITY_COLORS["country"],
                    opacity=0.72, line=dict(width=0.6, color="white")),
        text=country_hover, hoverinfo="text",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT),
        showlegend=True,
    ))

    # ─ Layer 6: geographic cluster trace (Full / Export views) ───────────────
    # Pre-cluster facility nodes into 2.5° grid cells.
    # At global zoom individual dots are too small; this shows aggregate geography.
    # Switching Domestic ↔ Full/Export gives the "zoom = merge/split" effect.
    _GRID_DEG = 2.5
    _MERGE_VAL  = 300e6   # clusters below this get merged or dropped
    _MERGE_DEG  = 3.6     # ≈ 400 km — merge if nearest big cluster is within this

    _fcn = facility_nodes.copy()
    _fcn["_gb_lat"] = (_fcn["lat"] / _GRID_DEG).round(0) * _GRID_DEG
    _fcn["_gb_lon"] = (_fcn["lon"] / _GRID_DEG).round(0) * _GRID_DEG

    # First pass: compute raw clusters
    _raw_cl = (
        _fcn.groupby(["_gb_lat","_gb_lon"])
            .agg(lat=("lat","mean"), lon=("lon","mean"), total_val=("usd_val","sum"))
            .reset_index()
    )
    _big_cl   = _raw_cl[_raw_cl["total_val"] >= _MERGE_VAL]
    _small_cl = _raw_cl[_raw_cl["total_val"] <  _MERGE_VAL]

    # Reassign small-cluster facilities to nearest big cluster (if within range)
    for _, srow in _small_cl.iterrows():
        if len(_big_cl) == 0:
            continue
        dists = np.sqrt((_big_cl["lat"] - srow["lat"])**2 +
                        (_big_cl["lon"] - srow["lon"])**2)
        nearest_dist = dists.min()
        _nearest_idx = dists.idxmin() if dists.notna().any() else None
        if nearest_dist <= _MERGE_DEG and _nearest_idx is not None:
            nb = _big_cl.loc[_nearest_idx]
            mask = ((_fcn["_gb_lat"] == srow["_gb_lat"]) &
                    (_fcn["_gb_lon"] == srow["_gb_lon"]))
            _fcn.loc[mask, "_gb_lat"] = nb["_gb_lat"]
            _fcn.loc[mask, "_gb_lon"] = nb["_gb_lon"]
        # else: leave in original cell — will be filtered out below

    # Recompute clusters from updated assignments; drop anything still below threshold
    _geo_cl = (
        _fcn.groupby(["_gb_lat","_gb_lon"])
            .agg(lat=("lat","mean"), lon=("lon","mean"),
                 total_val=("usd_val","sum"), n_fac=("name","count"))
            .reset_index()
    )
    _geo_cl = _geo_cl[_geo_cl["total_val"] >= _MERGE_VAL].copy()

    # Build per-cluster aggregate stats for JS info panel
    _cluster_lookup = {}
    for _, crow in _geo_cl.iterrows():
        cell = _fcn[(_fcn["_gb_lat"] == crow["_gb_lat"]) & (_fcn["_gb_lon"] == crow["_gb_lon"])]
        key  = f"{crow['lat']:.2f},{crow['lon']:.2f}"

        # Facility type breakdown (e.g. "2 mines · 1 concentrator")
        tc = cell[cell["ftype"] != "port"]["ftype"].value_counts()
        type_str = "  ·  ".join(
            f"{v} {FACILITY_LABELS.get(k,k).lower()}{'s' if v>1 else ''}"
            for k, v in tc.items())

        # Dominant minerals from inv (% of cluster total value)
        cell_inv = inv[inv["FACILITY_NAME"].isin(cell["name"].tolist())]
        mineral_str = ""
        if len(cell_inv) > 0:
            mvals = {MINERAL_GROUPS[c][0]: float(cell_inv[c].sum())
                     for c in usd_cols_active
                     if c in cell_inv.columns and cell_inv[c].sum() > 0}
            tot_m = sum(mvals.values())
            if tot_m > 0:
                top_m = sorted(mvals.items(), key=lambda x: -x[1])[:3]
                mineral_str = "  ·  ".join(
                    f"{m} <b>{v/tot_m*100:.0f}%</b>" for m, v in top_m)

        # Key operators (top 2 by value)
        ops_str = ""
        if len(cell_inv) > 0 and "OPERATOR_NAME" in cell_inv.columns:
            ops = (cell_inv.dropna(subset=["OPERATOR_NAME"])
                   .groupby("OPERATOR_NAME")["USD_VALUE_TOTAL"].sum()
                   .nlargest(2).index.tolist())
            ops_str = "  ·  ".join(ops)

        # Output ports (follow edges from any facility in this cell → port)
        cell_names = cell["name"].tolist()
        out_ports = (edges[edges["FROM_NAME"].isin(cell_names) &
                           (edges["TO_TYPE"] == "port")]["TO_NAME"]
                     .value_counts().head(3).index.tolist())
        ports_str = "  ·  ".join(out_ports)

        # Domestic edges connected to any facility in this cluster
        cell_names_set = set(cell["name"].tolist())
        cl_edges = dom_edges[
            dom_edges["FROM_NAME"].isin(cell_names_set) |
            dom_edges["TO_NAME"].isin(cell_names_set)
        ]
        _efl = cl_edges["FROM_LAT"].tolist(); _etl = cl_edges["TO_LAT"].tolist()
        _efn = cl_edges["FROM_LON"].tolist(); _etn = cl_edges["TO_LON"].tolist()
        _en  = [None] * len(_efl)
        edge_lats = [v for t in zip(_efl, _etl, _en) for v in t]
        edge_lons = [v for t in zip(_efn, _etn, _en) for v in t]

        # Which point indices in each ftype trace belong to this cluster
        # (cell_names_set already defined above)
        trace_pts = {
            ftype: [i for i, n in enumerate(names) if n in cell_names_set]
            for ftype, names in _ftype_pt_names.items()
        }
        trace_pts = {k: v for k, v in trace_pts.items() if v}  # drop empty

        _cluster_lookup[key] = {
            "val":       fmt_usd(crow["total_val"]),
            "n":         int(crow["n_fac"]),
            "minerals":  mineral_str,
            "types":     type_str,
            "ops":       ops_str,
            "ports":     ports_str,
            "trace_pts": trace_pts,
            "edge_lats": edge_lats,
            "edge_lons": edge_lons,
        }
    _geo_cl["sz"]   = np.clip(np.sqrt(_geo_cl["total_val"] / 1e6) * 1.4, 8, 50).tolist()
    _geo_cl["htxt"] = _geo_cl.apply(
        lambda r: (f"<b>{int(r['n_fac'])} facilit{'y' if r['n_fac']==1 else 'ies'}</b><br>"
                   f"Total value: {fmt_usd(r['total_val'])}"), axis=1).tolist()
    _clust_idx = len(figsc.data)
    figsc.add_trace(go.Scattergeo(
        lat=_geo_cl["lat"].tolist(), lon=_geo_cl["lon"].tolist(),
        mode="markers", name="Facility cluster",
        legendgroup="node_cluster",
        legendgrouptitle_text="Nodes",
        marker=dict(
            size=_geo_cl["sz"].tolist(), color="#4a6fa5",
            opacity=0.65, sizemode="diameter",
            line=dict(width=1.5, color="rgba(255,255,255,0.85)"),
        ),
        text=_geo_cl["htxt"].tolist(), hoverinfo="text",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT),
        showlegend=True,
    ))

    # ─ Layer 7: mine name labels (Domestic view only, hidden initially) ───────
    _top_mines_df = facility_nodes[facility_nodes["ftype"] == "mine"].nlargest(12, "usd_val")
    _label_idx = len(figsc.data)
    figsc.add_trace(go.Scattergeo(
        lat=_top_mines_df["lat"].tolist(), lon=_top_mines_df["lon"].tolist(),
        mode="text",
        text=[n[:22] for n in _top_mines_df["name"].tolist()],
        textfont=dict(size=8, color="#1a2744", family=FONT),
        textposition="top center",
        hoverinfo="skip", showlegend=False, visible=False,
    ))

    # ─ Layer 8: cluster-edge placeholder (empty; JS fills it on cluster click) ──
    _cluster_edge_idx = len(figsc.data)
    figsc.add_trace(go.Scattergeo(
        lat=[], lon=[], mode="lines",
        name="Cluster connections",
        line=dict(width=2.2, color="#4a6fa5"),
        opacity=0.85, hoverinfo="skip",
        showlegend=False, visible=False,
    ))

    # ── Visibility arrays ─────────────────────────────────────────────────────
    _N = len(figsc.data)

    def _vbool(true_idxs):
        v = [False] * _N
        for i in true_idxs:
            if i is not None: v[i] = True
        return v

    _dom_all     = list(_dom_idxs.values())
    _exp_all     = list(_cont_idxs.values())      # continent-level arc traces
    _ctry_det_all= list(_ctry_det_idxs.values())  # country-detail arcs (JS-only)
    _fac_all     = list(_fac_idxs.values())

    # Full view:     domestic edges + continent arcs + ports + cluster
    # Domestic view: domestic edges + individual fac + mine labels
    # Export view:   continent arcs + ports + cluster
    vis_full     = _vbool([0] + _dom_all + _exp_all + [_port_idx, _clust_idx])
    vis_domestic = _vbool([0] + _dom_all + _fac_all + [_label_idx])
    vis_export   = _vbool([0] + _exp_all + [_port_idx, _clust_idx])

    # Showlegend arrays — ports + cluster shown in Full/Export; facility types in Domestic
    sl_full = [False] * _N
    sl_full[_clust_idx] = True
    sl_full[_port_idx]  = True
    sl_export = sl_full[:]

    # Domestic: individual facility types shown in legend
    sl_domestic = [False] * _N
    for i in _fac_all: sl_domestic[i] = True

    # Opacity arrays — all values explicit Python float (no numpy types)
    def _make_op(dom_op, exp_op):
        op = [float(1.0)] * _N
        for i in _dom_all:      op[i] = float(dom_op)
        for i in _exp_all:      op[i] = float(exp_op)   # continent arcs
        for i in _ctry_det_all: op[i] = float(0.75)     # country detail arcs
        for i in _fac_all:      op[i] = float(0.88)
        op[_ctry_idx]  = float(0.72)
        op[_clust_idx] = float(0.65)
        op[_label_idx] = float(1.0)
        return op

    # In Domestic view edges are boosted to 0.82 — much more visible when zoomed in.
    _op_full     = _make_op(OP_DOM_ON, OP_EXP_ON)
    _op_domestic = _make_op(0.82,      OP_EXP_ON)
    _op_export   = _make_op(OP_DOM_ON, OP_EXP_ON)

    # Geo zoom args
    _GEO_GLOBAL = {
        "geo.center.lat": SC_CENTER_LAT,   "geo.center.lon": SC_CENTER_LON,
        "geo.projection.scale": SC_PROJ_SCALE,
        "geo.lonaxis.range": [-155, 165],  "geo.lataxis.range": [-58, 72],
    }
    _GEO_CHILE = {
        "geo.center.lat": SC_CENTER_LAT_C, "geo.center.lon": SC_CENTER_LON_C,
        "geo.projection.scale": SC_PROJ_SCALE_C,
        "geo.lonaxis.range": [-80, -60],   "geo.lataxis.range": [-50, -16],
    }
    _GEO_EXPORT = {
        "geo.center.lat": 10.0,  "geo.center.lon": -10.0,
        "geo.projection.scale": SC_PROJ_SCALE,
        "geo.lonaxis.range": [-140, 160],  "geo.lataxis.range": [-55, 70],
    }

    # ── Commodity filter buttons (opacity-based, domestic edges only) ─────────
    _all_comms = set(_dom_idxs.keys())
    _comm_rank  = {c: float(arc_comm_total.get(c, 0)) for c in _all_comms}
    _LITHIUM_COMMS = {c for c in _all_comms if "Lithium" in c}
    _FILTER_GROUPS: dict = {}
    for c in sorted(_all_comms, key=lambda x: -_comm_rank.get(x, 0)):
        if "Lithium" in c: continue
        _FILTER_GROUPS[c] = {c}
    if _LITHIUM_COMMS:
        _FILTER_GROUPS["Lithium"] = _LITHIUM_COMMS

    _top_filter = sorted(
        _FILTER_GROUPS.keys(),
        key=lambda g: -sum(_comm_rank.get(c, 0) for c in _FILTER_GROUPS[g])
    )[:7]

    def _op_filter(keep_set):
        op = [float(1.0)] * _N
        for c, i in _dom_idxs.items():
            op[i] = float(OP_DOM_ON if c in keep_set else OP_DOM_OFF)
        for i in _exp_all:      op[i] = float(OP_EXP_ON)  # continent arcs: unfiltered
        for i in _ctry_det_all: op[i] = float(0.75)
        for i in _fac_all:      op[i] = float(0.88)
        op[_ctry_idx]  = float(0.72)
        op[_clust_idx] = float(0.65)
        op[_label_idx] = float(1.0)
        return op

    _restyle_idxs = list(range(1, _N))   # skip trace 0 (choropleth — doesn't support opacity restyle)
    filter_buttons = [dict(label="All minerals", method="restyle",
                           args=[{"opacity": _op_filter(_all_comms)[1:]}, _restyle_idxs])]
    for grp_label in _top_filter:
        keep = _FILTER_GROUPS[grp_label]
        filter_buttons.append(dict(label=grp_label, method="restyle",
                                   args=[{"opacity": _op_filter(keep)[1:]}, _restyle_idxs]))

    # ── Layout ────────────────────────────────────────────────────────────────
    n_fac_shown = len(facility_nodes)
    n_dom_edges = len(dom_edges)
    n_countries = len(country_nodes)

    figsc.update_layout(
        template="plotly_white",
        paper_bgcolor="#fafafa",
        font=dict(family=FONT),
        autosize=True,
        margin=dict(l=0, r=0, t=75, b=0),
        title=dict(
            text=(
                "Chile Mineral Supply Chain 2024"
                f"<br><sup style='font-size:11px;font-weight:normal;color:#6b7280'>"
                f"Mine-to-market · {n_fac_shown} facilities · {n_dom_edges} domestic edges"
                f" · top-{SC_TOP_ARCS} export arcs · {n_countries} countries"
                "</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=18, color=STYLE["title_color"], family=FONT),
        ),
        annotations=[],
        geo=dict(
            showframe=False, showcoastlines=True, coastlinecolor="#c0c5cc",
            projection_type="natural earth",
            bgcolor="rgba(225,235,248,0.55)",
            showland=True,  landcolor="#f0f2f5",
            showocean=True, oceancolor="#d6e8f7",
            showcountries=True, countrycolor="#c8cdd4", countrywidth=0.4,
            showrivers=False, showlakes=False,
            center=dict(lat=SC_CENTER_LAT, lon=SC_CENTER_LON),
            projection_scale=SC_PROJ_SCALE,
            lonaxis=dict(range=[-155, 165]),
            lataxis=dict(range=[-58,   72]),
            resolution=50,
        ),
        legend=dict(
            yanchor="top", y=0.97, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.93)",
            bordercolor="#dde1e7", borderwidth=1,
            font=dict(size=10, family=FONT),
            tracegroupgap=4,
        ),
        updatemenus=[
            # Top-right: mineral filter dropdown (opacity-based, independent of view mode)
            dict(
                type="dropdown",
                buttons=filter_buttons,
                active=0,
                x=0.99, xanchor="right", y=0.99, yanchor="top",
                bgcolor="#ffffff", bordercolor="#c9cfd6", borderwidth=1,
                font=dict(size=11, family=FONT),
                pad=dict(r=6, t=4, b=4, l=6),
            ),
            # Bottom-right: view mode + reset
            dict(
                type="buttons", direction="right",
                buttons=[
                    dict(label="Full view", method="update",
                         args=[{"visible": vis_full,
                                "showlegend": sl_full,
                                "opacity": _op_full}, _GEO_GLOBAL]),
                    dict(label="Domestic (Chile)", method="update",
                         args=[{"visible": vis_domestic,
                                "showlegend": sl_domestic,
                                "opacity": _op_domestic}, _GEO_CHILE]),
                    dict(label="Export flows", method="update",
                         args=[{"visible": vis_export,
                                "showlegend": sl_export,
                                "opacity": _op_export}, _GEO_EXPORT]),
                    dict(label="Reset", method="update",
                         args=[{"visible": vis_full,
                                "showlegend": sl_full,
                                "opacity": _op_full}, _GEO_GLOBAL]),
                ],
                x=0.99, xanchor="right", y=0.01, yanchor="bottom",
                bgcolor="#f0f2f5", bordercolor="#c9cfd6", borderwidth=1,
                font=dict(size=11, family=FONT),
                pad=dict(r=6, t=6, b=6, l=6),
            ),
            # Bottom-left: dezoom / back to full view
            dict(
                type="buttons", direction="left",
                buttons=[
                    dict(label="← Back", method="update",
                         args=[{"visible": vis_full,
                                "showlegend": sl_full,
                                "opacity": _op_full}, _GEO_GLOBAL]),
                ],
                x=0.01, xanchor="left", y=0.01, yanchor="bottom",
                bgcolor="#f0f2f5", bordercolor="#c9cfd6", borderwidth=1,
                font=dict(size=11, family=FONT),
                pad=dict(r=6, t=6, b=6, l=6),
            ),
        ],
    )

    # Write HTML then inject JS safety net for Reset / view-switch opacity.
    # plotly_buttonclicked fires after the button action; the setTimeout ensures
    # the restyle runs after Plotly's own update settles.
    import json as _json
    _html_path = f"{NEW_OUT_DIR}/chile_supply_chain_map.html"
    figsc.write_html(
        _html_path,
        config=dict(displayModeBar="hover", displaylogo=False, responsive=True),
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        default_height="100%",
    )
    _op_by_label = {
        "Full view":        _op_full,
        "Domestic (Chile)": _op_domestic,
        "Export flows":     _op_export,
        "Reset":            _op_full,
    }
    _js_sc = (
        "<style>\n"
        "#sc-info-panel{"
        "position:fixed;right:14px;top:50%;transform:translateY(-50%);"
        "background:#fff;border:1px solid #dde1e7;border-radius:8px;"
        "padding:12px 15px;font-family:'IBM Plex Sans',sans-serif;font-size:12px;"
        "width:240px;box-shadow:0 4px 14px rgba(0,0,0,0.13);z-index:9999;"
        "display:none;pointer-events:none;line-height:1.5;"
        "}\n"
        "#sc-info-panel .sc-panel-title{font-weight:600;font-size:13px;color:#1a2744;margin-bottom:6px;}\n"
        "#sc-info-panel .sc-row{display:flex;justify-content:space-between;gap:8px;}"
        "#sc-info-panel .sc-type{color:#6b7280;min-width:80px;}"
        "#sc-info-panel .sc-val{color:#1a2744;text-align:right;font-size:11px;}"
        "</style>\n"
        "<div id='sc-info-panel'></div>\n"
        "<script>\n"
        "(function waitForPlotly() {\n"
        "  var gd = document.querySelector('.js-plotly-plot');\n"
        "  if (!gd || !gd._fullLayout) { setTimeout(waitForPlotly, 200); return; }\n"
        "\n"
        "  var OP         = " + _json.dumps(_op_by_label) + ";\n"
        "  var CLUST_IDX  = " + str(_clust_idx) + ";\n"
        "  var CTRY_IDX   = " + str(_ctry_idx) + ";\n"
        "  var VIS_DOM    = " + _json.dumps(vis_domestic) + ";\n"
        "  var SL_DOM     = " + _json.dumps(sl_domestic) + ";\n"
        "  var OP_DOM     = " + _json.dumps(_op_domestic) + ";\n"
        "  var VIS_FULL   = " + _json.dumps(vis_full) + ";\n"
        "  var SL_FULL    = " + _json.dumps(sl_full) + ";\n"
        "  var OP_FULL    = " + _json.dumps(_op_full) + ";\n"
        "  var CONT_IDXS  = " + _json.dumps(list(_cont_idxs.values())) + ";\n"
        "  var CONT_NAMES = " + _json.dumps(list(_cont_idxs.keys())) + ";\n"
        "  var CTRY_DET   = " + _json.dumps({k: v for k, v in _ctry_det_idxs.items()}) + ";\n"
        "  var CONT_BOUNDS= " + _json.dumps(CONTINENT_BOUNDS_JS) + ";\n"
        "  var CLUST_DATA = " + _json.dumps(_cluster_lookup) + ";\n"
        "  var FAC_IDXS        = " + _json.dumps({k: v for k, v in _fac_idxs.items()}) + ";\n"
        "  var FAC_NPTS        = " + _json.dumps({k: len(v) for k, v in _ftype_pt_names.items()}) + ";\n"
        "  var DOM_IDXS        = " + _json.dumps(list(_dom_idxs.values())) + ";\n"
        "  var CLUST_EDGE_IDX  = " + str(_cluster_edge_idx) + ";\n"
        "  var panel = document.getElementById('sc-info-panel');\n"
        "\n"
        "  function hidePanel() { panel.style.display = 'none'; }\n"
        "\n"
        "  function showPanel(d) {\n"
        "    var html = '<div class=\\'sc-panel-title\\'>' + d.n + ' facilit' + (d.n===1?'y':'ies') + '  ·  ' + d.val + '</div>';\n"
        "    if (d.minerals) html += '<div class=\\'sc-row\\'><span class=\\'sc-type\\'>Minerals</span><span>' + d.minerals + '</span></div>';\n"
        "    if (d.types)    html += '<div class=\\'sc-row\\'><span class=\\'sc-type\\'>Facilities</span><span>' + d.types + '</span></div>';\n"
        "    if (d.ops)      html += '<div class=\\'sc-row\\'><span class=\\'sc-type\\'>Operators</span><span>' + d.ops + '</span></div>';\n"
        "    if (d.ports)    html += '<div class=\\'sc-row\\'><span class=\\'sc-type\\'>Exports via</span><span>' + d.ports + '</span></div>';\n"
        "    panel.innerHTML = html;\n"
        "    panel.style.display = 'block';\n"
        "  }\n"
        "\n"
        "  // View-switch buttons: fix trace opacity + reset any per-point marker.opacity\n"
        "  gd.on('plotly_buttonclicked', function(data) {\n"
        "    if (!data || !data.button) return;\n"
        "    var op = OP[data.button.label];\n"
        "    if (op) {\n"
        "      setTimeout(function() {\n"
        "        var tIdxs = gd.data.map(function(_, i) { return i; });\n"
        "        Plotly.restyle(gd, {opacity: op}, tIdxs);\n"
        "        // Reset per-point marker opacity + clear cluster edge trace\n"
        "        Object.values(FAC_IDXS).forEach(function(tIdx) {\n"
        "          Plotly.restyle(gd, {'marker.opacity': 1}, [tIdx]);\n"
        "        });\n"
        "        Plotly.restyle(gd, {lat: [[]], lon: [[]]}, [CLUST_EDGE_IDX]);\n"
        "      }, 80);\n"
        "    }\n"
        "    hidePanel();\n"
        "  });\n"
        "\n"
        "  gd.on('plotly_click', function(data) {\n"
        "    if (!data || !data.points || !data.points.length) return;\n"
        "    var pt = data.points[0];\n"
        "    var evt = data.event;\n"
        "\n"
        "  // Helper: restyle all traces then relayout\n"
        "  function restyleThenLayout(traceAttrs, layoutArgs) {\n"
        "    var tIdxs = gd.data.map(function(_, i) { return i; });\n"
        "    return Plotly.restyle(gd, traceAttrs, tIdxs)\n"
        "      .then(function() { return Plotly.relayout(gd, layoutArgs); });\n"
        "  }\n"
        "\n"
        "    // ── Cluster click: show only this cluster's facilities + edges ───────\n"
        "    if (pt.curveNumber === CLUST_IDX) {\n"
        "      var key = pt.lat.toFixed(2) + ',' + pt.lon.toFixed(2);\n"
        "      var d = CLUST_DATA[key];\n"
        "      if (d) showPanel(d);\n"
        "      var lat = pt.lat, lon = pt.lon, span = " + str(_GRID_DEG) + ";\n"
        "      var trPts = (d && d.trace_pts) || {};\n"
        "\n"
        "      // Build visibility: VIS_DOM base, but hide all domestic edges\n"
        "      // and show cluster-edge placeholder + hide cluster bubbles\n"
        "      var newVis = VIS_DOM.slice();\n"
        "      DOM_IDXS.forEach(function(i) { newVis[i] = false; });\n"
        "      newVis[CLUST_EDGE_IDX] = true;\n"
        "      newVis[CLUST_IDX]      = false;\n"
        "\n"
        "      // Load cluster edge data into placeholder trace\n"
        "      var edgeLats = (d && d.edge_lats) || [];\n"
        "      var edgeLons = (d && d.edge_lons) || [];\n"
        "      Plotly.restyle(gd, {lat: [edgeLats], lon: [edgeLons]}, [CLUST_EDGE_IDX]);\n"
        "\n"
        "      var tIdxs = gd.data.map(function(_, i) { return i; });\n"
        "      Plotly.restyle(gd, {visible: newVis, showlegend: SL_DOM, opacity: OP_DOM}, tIdxs)\n"
        "        .then(function() { return Plotly.relayout(gd, {\n"
        "          'geo.center.lat': lat, 'geo.center.lon': lon,\n"
        "          'geo.lataxis.range': [lat-span, lat+span],\n"
        "          'geo.lonaxis.range': [lon-span, lon+span]\n"
        "        }); })\n"
        "        .then(function() {\n"
        "          // Per-point opacity: only cluster facilities visible\n"
        "          var restyles = [];\n"
        "          Object.keys(FAC_IDXS).forEach(function(ftype) {\n"
        "            var tIdx  = FAC_IDXS[ftype];\n"
        "            var nPts  = FAC_NPTS[ftype];\n"
        "            var clIdx = trPts[ftype] || [];\n"
        "            var opArr = new Array(nPts).fill(0.04);\n"
        "            clIdx.forEach(function(i) { opArr[i] = 0.92; });\n"
        "            restyles.push(Plotly.restyle(gd, {'marker.opacity': [opArr]}, [tIdx]));\n"
        "          });\n"
        "          return Promise.all(restyles);\n"
        "        });\n"
        "      return;\n"
        "    }\n"
        "\n"
        "    // ── Continent arc click → drill down to country arcs ──────────────\n"
        "    var contPos = CONT_IDXS.indexOf(pt.curveNumber);\n"
        "    if (contPos === -1) { hidePanel(); return; }\n"
        "    var cont = CONT_NAMES[contPos];\n"
        "    hidePanel();\n"
        "\n"
        "    // Build new visibility: hide continent arcs, hide other country details,\n"
        "    // show this continent's country details + country nodes\n"
        "    var newVis = gd.data.map(function(t, i) { return t.visible !== false; });\n"
        "    CONT_IDXS.forEach(function(i)     { newVis[i] = false; });\n"
        "    Object.values(CTRY_DET).forEach(function(i) { newVis[i] = false; });\n"
        "    newVis[CTRY_DET[cont]] = true;\n"
        "    newVis[CTRY_IDX] = true;\n"
        "\n"
        "    var bounds = CONT_BOUNDS[cont] || {};\n"
        "    restyleThenLayout({visible: newVis}, bounds);\n"
        "  });\n"
        "\n"
        "  // Hide info panel on background click\n"
        "  gd.on('plotly_clickannotation', hidePanel);\n"
        "  document.addEventListener('click', function(e) {\n"
        "    if (!gd.contains(e.target)) hidePanel();\n"
        "  });\n"
        "})();\n"
        "</script>"
    )
    with open(_html_path, "r") as _f:
        _html = _f.read()
    _html = _html.replace("</body>", _js_sc + "\n</body>")
    with open(_html_path, "w") as _f:
        _f.write(_html)
    print("  → chile_supply_chain_map.html")


    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nAll figures written to:\n  {NEW_OUT_DIR}\n")
    for fname in sorted(os.listdir(NEW_OUT_DIR)):
        if fname.endswith(".html"):
            size_kb = os.path.getsize(os.path.join(NEW_OUT_DIR, fname)) / 1024
            print(f"  {fname:<45}  {size_kb:>7.0f} KB")
