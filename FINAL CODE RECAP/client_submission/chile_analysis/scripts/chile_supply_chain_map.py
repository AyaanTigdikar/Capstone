#!/usr/bin/env python3
"""
chile_supply_chain_map.py
Generates only the Chile Mineral Supply Chain interactive map.

Reads:  intermediary/_pipeline_state_6.pkl
Writes: outputs/chile_supply_chain_map.html

Run from anywhere:  python3 scripts/chile_supply_chain_map.py
"""

import sys, os, pickle, math, json as _json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── PATHS ─────────────────────────────────────────────────────────────────────
PKL_PATH   = "/Users/leoss/Desktop/GitHub/Capstone/Case studies/Chile/intermediary/_pipeline_state_6.pkl"
DIR_OUTPUT = "/Users/leoss/Desktop/GitHub/Capstone/Case studies/Chile/outputs"

# ── DESIGN ────────────────────────────────────────────────────────────────────
FONT = "IBM Plex Sans, system-ui, -apple-system, sans-serif"

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

FACILITY_COLORS = {
    "mine":         "#1d4e89",
    "concentrator": "#4a86c8",
    "smelter":      "#e07b39",
    "plant":        "#1a9850",
    "mo_plant":     "#8b5cf6",
    "sx_ew":        "#16a085",
    "re_plant":     "#b83030",
    "port":         "#e74c3c",
    "country":      "#9e9e9e",
}

FACILITY_LABELS = {
    "mine":         "Mine",
    "concentrator": "Concentrator",
    "smelter":      "Smelter",
    "plant":        "Processing plant",
    "mo_plant":     "Mo plant",
    "sx_ew":        "SX-EW plant",
    "re_plant":     "Re plant",
    "port":         "Export port",
    "country":      "Export country",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def fmt_usd(val):
    if val >= 1e9:  return f"${val/1e9:.1f}B"
    if val >= 1e6:  return f"${val/1e6:.0f}M"
    if val >= 1e3:  return f"${val/1e3:.0f}K"
    return f"${val:.0f}"


def great_circle_arcs(rows_iter, n_pts=25):
    all_lats, all_lons = [], []
    for (phi1d, lam1d, phi2d, lam2d) in rows_iter:
        p1, l1 = math.radians(phi1d), math.radians(lam1d)
        p2, l2 = math.radians(phi2d), math.radians(lam2d)
        d = 2 * math.asin(math.sqrt(
            math.sin((p2 - p1) / 2) ** 2
            + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2
        ))
        if d < 0.001:
            all_lats += [phi1d, phi2d, None]
            all_lons += [lam1d, lam2d, None]
            continue
        t = np.linspace(0, 1, n_pts)
        A = np.sin((1 - t) * d) / math.sin(d)
        B = np.sin(t * d) / math.sin(d)
        x = A * math.cos(p1) * math.cos(l1) + B * math.cos(p2) * math.cos(l2)
        y = A * math.cos(p1) * math.sin(l1) + B * math.cos(p2) * math.sin(l2)
        z = A * math.sin(p1) + B * math.sin(p2)
        lats = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        lons = np.degrees(np.arctan2(y, x))
        all_lats += lats.tolist() + [None]
        all_lons += lons.tolist() + [None]
    return all_lats, all_lons


PRODUCT_FORM_PRICES = {"cathode": 9_200, "concentrate": 2_800, "blister": 8_800}

def estimate_usd(row):
    val  = row.get("EXPORT_VALUE", 0)
    unit = str(row.get("EXPORT_UNIT", ""))
    pf   = str(row.get("PRODUCT_FORM", ""))
    comm = str(row.get("COMMODITIES", ""))
    if not val or pd.isna(val): return 0
    if unit in ("$FOB", "$USD"):  return val
    if unit == "$M_FOB":          return val * 1e6
    if comm == "Copper" and unit == "kMT":
        return val * 1_000 * PRODUCT_FORM_PRICES.get(pf, 5_000)
    if unit == "MT": return val * 46_954
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
        if not vals: return "Other", "Industrial minerals"
        top = max(vals, key=vals.get)
        return MINERAL_GROUPS.get(top, ("Other", "Industrial minerals"))

    inv[["dominant_mineral", "mineral_group"]] = pd.DataFrame(
        inv.apply(dominant_mineral, axis=1).tolist(), index=inv.index
    )

    os.makedirs(DIR_OUTPUT, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SUPPLY CHAIN MAP
    # ══════════════════════════════════════════════════════════════════════════
    print("Building supply chain map ...")

    # ── TUNABLE CONSTANTS ─────────────────────────────────────────────────────
    MINE_THRESHOLD  = 500e6   # domestic edges: only mines with value >= this
    SC_TOP_ARCS     = 30      # top N port-to-country export arcs to draw
    SC_CENTER_LAT   = -10.0
    SC_CENTER_LON   = -30.0
    SC_PROJ_SCALE   = 1.0
    SC_CENTER_LAT_C = -29.0
    SC_CENTER_LON_C = -70.5
    SC_PROJ_SCALE_C = 7.5
    OP_DOM_ON       = 0.45
    OP_DOM_OFF      = 0.05
    OP_EXP_ON       = 0.70

    # ── Node lookups ──────────────────────────────────────────────────────────
    mine_val_map = inv.set_index("FACILITY_NAME")["USD_VALUE_TOTAL"].to_dict()
    mine_dom_map = inv.set_index("FACILITY_NAME")["dominant_mineral"].to_dict()

    m2p = edges[edges["EDGE_TYPE"] == "mine_to_plant"].copy()
    m2p["_mine_val"] = m2p["FROM_NAME"].map(mine_val_map).fillna(0)
    m2p = m2p[m2p["_mine_val"] >= MINE_THRESHOLD]

    other_dom = edges[~edges["EDGE_TYPE"].isin(["mine_to_plant", "port_to_country"])]
    dom_edges = pd.concat([m2p, other_dom], ignore_index=True)

    # ── Port-to-country arcs ──────────────────────────────────────────────────
    exp_aug = export_df.copy()
    exp_aug["USD_EST"] = exp_aug.apply(estimate_usd, axis=1)

    pc_comm = (exp_aug.groupby(["FROM_NAME","TO_NAME","FROM_LAT","FROM_LON",
                                 "TO_LAT","TO_LON","COMMODITIES"])["USD_EST"]
               .sum().reset_index())
    pc_total = (exp_aug.groupby(["FROM_NAME","TO_NAME","FROM_LAT","FROM_LON",
                                  "TO_LAT","TO_LON"])["USD_EST"]
                .sum().reset_index())
    _idx = pc_comm.groupby(["FROM_NAME","TO_NAME"])["USD_EST"].idxmax().dropna()
    dominant_comm = pc_comm.loc[_idx, ["FROM_NAME","TO_NAME","COMMODITIES"]]
    pc_total = pc_total.merge(dominant_comm, on=["FROM_NAME","TO_NAME"], how="left")
    arc_df = pc_total.nlargest(SC_TOP_ARCS, "USD_EST")

    # ── Continent aggregation ─────────────────────────────────────────────────
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
    cont_arcs = (arc_df[arc_df["continent"] != "Other"]
                 .groupby(["FROM_NAME","FROM_LAT","FROM_LON","continent"])
                 .agg(USD_EST=("USD_EST","sum"),
                      dom_comm=("COMMODITIES", lambda x: x.value_counts().index[0] if len(x) else ""))
                 .reset_index())
    cont_arcs["TO_LAT"] = cont_arcs["continent"].map(
        lambda c: CONTINENT_CENTROIDS.get(c,(0,0))[0])
    cont_arcs["TO_LON"] = cont_arcs["continent"].map(
        lambda c: CONTINENT_CENTROIDS.get(c,(0,0))[1])
    arc_comm_total = arc_df.groupby("COMMODITIES")["USD_EST"].sum()

    # ── Build facility nodes (ALL from inv + ports) ───────────────────────────
    _FTYPE_MAP = {
        "mine": "mine", "Mine": "mine",
        "concentrator": "concentrator", "Concentrator": "concentrator",
        "sx_ew": "sx_ew", "SX-EW": "sx_ew", "sx-ew": "sx_ew",
        "plant": "plant", "Plant": "plant", "processing": "plant",
        "mo_plant": "mo_plant", "Mo plant": "mo_plant",
        "smelter": "smelter", "Smelter": "smelter",
        "re_plant": "re_plant", "Re plant": "re_plant",
        "port": "port", "Port": "port",
    }
    _ftype_col = "FACILITY_TYPE" if "FACILITY_TYPE" in inv.columns else "FACTYPE"
    inv_nodes = inv[["FACILITY_NAME", _ftype_col, "lat", "lon",
                      "USD_VALUE_TOTAL", "dominant_mineral"]].copy()
    inv_nodes = inv_nodes.rename(columns={
        "FACILITY_NAME":   "name",
        _ftype_col:        "ftype_raw",
        "USD_VALUE_TOTAL": "usd_val",
    })
    inv_nodes["ftype"] = inv_nodes["ftype_raw"].map(_FTYPE_MAP)
    inv_nodes["ftype"] = inv_nodes["ftype"].fillna(inv_nodes["ftype_raw"].str.lower())
    inv_nodes = inv_nodes.dropna(subset=["lat","lon","ftype"])
    inv_nodes = inv_nodes[["name","ftype","lat","lon","usd_val","dominant_mineral"]]

    # Ports from export arc origins
    port_nodes_from_arcs = arc_df[["FROM_NAME","FROM_LAT","FROM_LON"]].rename(
        columns={"FROM_NAME":"name","FROM_LAT":"lat","FROM_LON":"lon"})
    port_nodes_from_arcs["ftype"] = "port"
    port_nodes_from_arcs["usd_val"] = 0.0
    port_nodes_from_arcs["dominant_mineral"] = "Other"

    _port_parts = [port_nodes_from_arcs]
    _plat = next((c for c in ports_df.columns if "LAT"  in c.upper()), None)
    _plon = next((c for c in ports_df.columns if "LON"  in c.upper()), None)
    _pnm  = next((c for c in ports_df.columns if "NAME" in c.upper()), None)
    if _plat and _plon and _pnm:
        _pdf = ports_df[[_pnm, _plat, _plon]].copy()
        _pdf.columns = ["name","lat","lon"]
        _pdf["ftype"] = "port"; _pdf["usd_val"] = 0.0; _pdf["dominant_mineral"] = "Other"
        _port_parts.append(_pdf)

    facility_nodes = (pd.concat([inv_nodes] + _port_parts, ignore_index=True)
                      .dropna(subset=["lat","lon"])
                      .drop_duplicates(subset=["name","ftype"])
                      .copy())
    facility_nodes = facility_nodes[facility_nodes["ftype"] != "country"]
    facility_nodes["usd_val"] = facility_nodes["name"].map(mine_val_map).fillna(
        facility_nodes["usd_val"])
    facility_nodes["dominant_mineral"] = facility_nodes["name"].map(mine_dom_map).fillna(
        facility_nodes["dominant_mineral"])

    # Uncomment to inspect unexpected ftype values:
    # print(facility_nodes["ftype"].value_counts())

    # Country destination nodes
    country_nodes = (arc_df[["TO_NAME","TO_LAT","TO_LON"]]
                     .rename(columns={"TO_NAME":"name","TO_LAT":"lat","TO_LON":"lon"})
                     .drop_duplicates(subset=["name"]).copy())
    country_nodes["ftype"] = "country"
    country_nodes["usd_val"] = 0.0

    # ── Build figure ──────────────────────────────────────────────────────────
    figsc = go.Figure()

    _dom_idxs      = {}
    _cont_idxs     = {}
    _ctry_det_idxs = {}
    _fac_idxs      = {}
    _port_idx  = None
    _ctry_idx  = None
    _clust_idx = None
    _label_idx = None

    # Layer 1: background choropleth
    SA = ["ARG","BOL","BRA","COL","ECU","GUY","PRY","PER","SUR","URY","VEN","GUF"]
    figsc.add_trace(go.Choropleth(
        locations=SA + ["CHL"], z=[1]*len(SA) + [2],
        colorscale=[[0,"#e0e3e8"],[0.5,"#e0e3e8"],[0.5,"#f0f2f5"],[1,"#f0f2f5"]],
        locationmode="ISO-3",
        marker_line_color="#bfc5cc", marker_line_width=0.5,
        showscale=False, showlegend=False, hoverinfo="skip",
    ))

    # Layer 2: domestic edges (per commodity, filtered at mine threshold)
    dom_comms_ordered = sorted(dom_edges["COMMODITIES"].dropna().unique())
    for comm in dom_comms_ordered:
        sub = dom_edges[dom_edges["COMMODITIES"] == comm]
        _fl = sub["FROM_LAT"].tolist(); _tl = sub["TO_LAT"].tolist()
        _fn = sub["FROM_LON"].tolist(); _tn = sub["TO_LON"].tolist()
        _none = [None] * len(_fl)
        lats = [v for t in zip(_fl, _tl, _none) for v in t]
        lons = [v for t in zip(_fn, _tn, _none) for v in t]
        _dom_idxs[comm] = len(figsc.data)
        figsc.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            name=comm, legendgroup=f"comm_{comm}",
            line=dict(width=1.6, color=COMMODITY_COLORS.get(comm, "#aaaaaa")),
            opacity=float(OP_DOM_ON), hoverinfo="skip",
            showlegend=False,
        ))

    # Layer 3a: continent-level export arcs
    _cont_order = sorted(cont_arcs["continent"].unique(),
                         key=lambda c: -cont_arcs[cont_arcs["continent"]==c]["USD_EST"].sum())
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

    # Layer 3b: country-level detail arcs (hidden; JS-revealed on continent click)
    for cont in _cont_order:
        sub_ctry = arc_df[arc_df["continent"] == cont]
        arc_lats, arc_lons = great_circle_arcs(
            zip(sub_ctry["FROM_LAT"], sub_ctry["FROM_LON"],
                sub_ctry["TO_LAT"],   sub_ctry["TO_LON"]), n_pts=15)
        _ctry_det_idxs[cont] = len(figsc.data)
        figsc.add_trace(go.Scattergeo(
            lat=arc_lats, lon=arc_lons, mode="lines",
            name=cont + " (countries)", legendgroup=f"ctry_{cont}",
            line=dict(width=1.4, color=CONTINENT_COLORS.get(cont, "#aaaaaa"), dash="dot"),
            opacity=0.75, hoverinfo="skip",
            showlegend=False, visible=False,
        ))

    # Layer 4: individual facility nodes (Domestic view, one trace per ftype)
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
            showlegend=True, visible=False,
            **kw,
        ))

    _ftype_pt_names = {
        ftype: facility_nodes[facility_nodes["ftype"] == ftype]["name"].tolist()
        for ftype in _fac_idxs
    }

    # Layer 5: export country nodes
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
                f"<br>{'  .  '.join(comm_parts)}"
                f"<br><span style='color:#6b7280'>Via: {', '.join(ports)}</span>")

    _ctry_idx = len(figsc.data)
    figsc.add_trace(go.Scattergeo(
        lat=country_nodes["lat"].tolist(), lon=country_nodes["lon"].tolist(),
        mode="markers", name="Export country",
        legendgroup="node_country",
        marker=dict(size=8, color=FACILITY_COLORS["country"],
                    opacity=0.72, line=dict(width=0.6, color="white")),
        text=[_ctry_hover(r["name"]) for _, r in country_nodes.iterrows()],
        hoverinfo="text",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT),
        showlegend=True,
    ))

    # Layer 6: geographic cluster bubbles (Full / Export views)
    _GRID_DEG  = 2.5
    _MERGE_VAL = 300e6
    _MERGE_DEG = 3.6

    _fcn = facility_nodes.copy()
    _fcn["_gb_lat"] = (_fcn["lat"] / _GRID_DEG).round(0) * _GRID_DEG
    _fcn["_gb_lon"] = (_fcn["lon"] / _GRID_DEG).round(0) * _GRID_DEG

    _raw_cl = (_fcn.groupby(["_gb_lat","_gb_lon"])
               .agg(lat=("lat","mean"), lon=("lon","mean"), total_val=("usd_val","sum"))
               .reset_index())
    _big_cl   = _raw_cl[_raw_cl["total_val"] >= _MERGE_VAL]
    _small_cl = _raw_cl[_raw_cl["total_val"] <  _MERGE_VAL]

    for _, srow in _small_cl.iterrows():
        if len(_big_cl) == 0: continue
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

    _geo_cl = (_fcn.groupby(["_gb_lat","_gb_lon"])
               .agg(lat=("lat","mean"), lon=("lon","mean"),
                    total_val=("usd_val","sum"), n_fac=("name","count"))
               .reset_index())
    _geo_cl = _geo_cl[_geo_cl["total_val"] >= _MERGE_VAL].copy()

    _cluster_lookup = {}
    for _, crow in _geo_cl.iterrows():
        cell = _fcn[(_fcn["_gb_lat"] == crow["_gb_lat"]) & (_fcn["_gb_lon"] == crow["_gb_lon"])]
        key  = f"{crow['lat']:.2f},{crow['lon']:.2f}"

        tc = cell[cell["ftype"] != "port"]["ftype"].value_counts()
        type_str = "  .  ".join(
            f"{v} {FACILITY_LABELS.get(k,k).lower()}{'s' if v>1 else ''}"
            for k, v in tc.items())

        cell_inv = inv[inv["FACILITY_NAME"].isin(cell["name"].tolist())]
        mineral_str = ""
        if len(cell_inv) > 0:
            mvals = {MINERAL_GROUPS[c][0]: float(cell_inv[c].sum())
                     for c in usd_cols_active
                     if c in cell_inv.columns and cell_inv[c].sum() > 0}
            tot_m = sum(mvals.values())
            if tot_m > 0:
                top_m = sorted(mvals.items(), key=lambda x: -x[1])[:3]
                mineral_str = "  .  ".join(
                    f"{m} <b>{v/tot_m*100:.0f}%</b>" for m, v in top_m)

        ops_str = ""
        if len(cell_inv) > 0 and "OPERATOR_NAME" in cell_inv.columns:
            ops = (cell_inv.dropna(subset=["OPERATOR_NAME"])
                   .groupby("OPERATOR_NAME")["USD_VALUE_TOTAL"].sum()
                   .nlargest(2).index.tolist())
            ops_str = "  .  ".join(ops)

        cell_names = cell["name"].tolist()
        out_ports = (edges[edges["FROM_NAME"].isin(cell_names) &
                           (edges["TO_TYPE"] == "port")]["TO_NAME"]
                     .value_counts().head(3).index.tolist())
        ports_str = "  .  ".join(out_ports)

        cell_names_set = set(cell_names)
        cl_edges = dom_edges[
            dom_edges["FROM_NAME"].isin(cell_names_set) |
            dom_edges["TO_NAME"].isin(cell_names_set)
        ]
        _efl = cl_edges["FROM_LAT"].tolist(); _etl = cl_edges["TO_LAT"].tolist()
        _efn = cl_edges["FROM_LON"].tolist(); _etn = cl_edges["TO_LON"].tolist()
        _en  = [None] * len(_efl)
        edge_lats = [v for t in zip(_efl, _etl, _en) for v in t]
        edge_lons = [v for t in zip(_efn, _etn, _en) for v in t]

        trace_pts = {
            ftype: [i for i, n in enumerate(names) if n in cell_names_set]
            for ftype, names in _ftype_pt_names.items()
        }

        _cluster_lookup[key] = {
            "val": fmt_usd(crow["total_val"]), "n": int(crow["n_fac"]),
            "minerals": mineral_str, "types": type_str,
            "ops": ops_str, "ports": ports_str,
            "trace_pts": {k: v for k, v in trace_pts.items() if v},
            "edge_lats": edge_lats, "edge_lons": edge_lons,
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

    # Layer 7: mine name labels (top 12 by value, Domestic view only)
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

    # Layer 8: cluster-edge placeholder (empty; filled by JS on cluster click)
    _cluster_edge_idx = len(figsc.data)
    figsc.add_trace(go.Scattergeo(
        lat=[], lon=[], mode="lines",
        name="Cluster connections",
        line=dict(width=2.2, color="#4a6fa5"),
        opacity=0.85, hoverinfo="skip",
        showlegend=False, visible=False,
    ))

    # ── Visibility + opacity arrays ───────────────────────────────────────────
    _N = len(figsc.data)

    def _vbool(true_idxs):
        v = [False] * _N
        for i in true_idxs:
            if i is not None: v[i] = True
        return v

    _dom_all      = list(_dom_idxs.values())
    _exp_all      = list(_cont_idxs.values())
    _ctry_det_all = list(_ctry_det_idxs.values())
    _fac_all      = list(_fac_idxs.values())

    vis_full     = _vbool([0] + _dom_all + _exp_all + [_port_idx, _clust_idx])
    vis_domestic = _vbool([0] + _dom_all + _fac_all + [_label_idx])
    vis_export   = _vbool([0] + _exp_all + [_port_idx, _clust_idx])

    sl_full = [False] * _N
    sl_full[_clust_idx] = True
    sl_full[_port_idx]  = True
    sl_export = sl_full[:]

    sl_domestic = [False] * _N
    for i in _fac_all: sl_domestic[i] = True

    def _make_op(dom_op, exp_op):
        op = [float(1.0)] * _N
        for i in _dom_all:      op[i] = float(dom_op)
        for i in _exp_all:      op[i] = float(exp_op)
        for i in _ctry_det_all: op[i] = float(0.75)
        for i in _fac_all:      op[i] = float(0.88)
        op[_ctry_idx]  = float(0.72)
        op[_clust_idx] = float(0.65)
        op[_label_idx] = float(1.0)
        return op

    _op_full     = _make_op(OP_DOM_ON, OP_EXP_ON)
    _op_domestic = _make_op(0.82,      OP_EXP_ON)
    _op_export   = _make_op(OP_DOM_ON, OP_EXP_ON)

    _GEO_GLOBAL = {
        "geo.center.lat": SC_CENTER_LAT, "geo.center.lon": SC_CENTER_LON,
        "geo.projection.scale": SC_PROJ_SCALE,
        "geo.lonaxis.range": [-155, 165], "geo.lataxis.range": [-58, 72],
    }
    _GEO_CHILE = {
        "geo.center.lat": SC_CENTER_LAT_C, "geo.center.lon": SC_CENTER_LON_C,
        "geo.projection.scale": SC_PROJ_SCALE_C,
        "geo.lonaxis.range": [-80, -60], "geo.lataxis.range": [-50, -16],
    }
    _GEO_EXPORT = {
        "geo.center.lat": 10.0, "geo.center.lon": -10.0,
        "geo.projection.scale": SC_PROJ_SCALE,
        "geo.lonaxis.range": [-140, 160], "geo.lataxis.range": [-55, 70],
    }

    # ── Commodity filter dropdown ─────────────────────────────────────────────
    _all_comms     = set(_dom_idxs.keys())
    _comm_rank     = {c: float(arc_comm_total.get(c, 0)) for c in _all_comms}
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
        for i in _exp_all:      op[i] = float(OP_EXP_ON)
        for i in _ctry_det_all: op[i] = float(0.75)
        for i in _fac_all:      op[i] = float(0.88)
        op[_ctry_idx]  = float(0.72)
        op[_clust_idx] = float(0.65)
        op[_label_idx] = float(1.0)
        return op

    _restyle_idxs = list(range(1, _N))
    filter_buttons = [dict(label="All minerals", method="restyle",
                           args=[{"opacity": _op_filter(_all_comms)[1:]}, _restyle_idxs])]
    for grp_label in _top_filter:
        filter_buttons.append(dict(label=grp_label, method="restyle",
                                   args=[{"opacity": _op_filter(_FILTER_GROUPS[grp_label])[1:]},
                                         _restyle_idxs]))

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
                f"Mine-to-market  {n_fac_shown} facilities  {n_dom_edges} domestic edges"
                f"  top-{SC_TOP_ARCS} export arcs  {n_countries} countries"
                "</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=18, color="#1a2744", family=FONT),
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
            lataxis=dict(range=[-58, 72]),
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
            dict(
                type="dropdown",
                buttons=filter_buttons,
                active=0,
                x=0.99, xanchor="right", y=0.99, yanchor="top",
                bgcolor="#ffffff", bordercolor="#c9cfd6", borderwidth=1,
                font=dict(size=11, family=FONT),
                pad=dict(r=6, t=4, b=4, l=6),
            ),
            dict(
                type="buttons", direction="right",
                buttons=[
                    dict(label="Full view", method="update",
                         args=[{"visible": vis_full, "showlegend": sl_full,
                                "opacity": _op_full}, _GEO_GLOBAL]),
                    dict(label="Domestic (Chile)", method="update",
                         args=[{"visible": vis_domestic, "showlegend": sl_domestic,
                                "opacity": _op_domestic}, _GEO_CHILE]),
                    dict(label="Export flows", method="update",
                         args=[{"visible": vis_export, "showlegend": sl_export,
                                "opacity": _op_export}, _GEO_EXPORT]),
                    dict(label="Reset", method="update",
                         args=[{"visible": vis_full, "showlegend": sl_full,
                                "opacity": _op_full}, _GEO_GLOBAL]),
                ],
                x=0.99, xanchor="right", y=0.01, yanchor="bottom",
                bgcolor="#f0f2f5", bordercolor="#c9cfd6", borderwidth=1,
                font=dict(size=11, family=FONT),
                pad=dict(r=6, t=6, b=6, l=6),
            ),
            dict(
                type="buttons", direction="left",
                buttons=[
                    dict(label="Back", method="update",
                         args=[{"visible": vis_full, "showlegend": sl_full,
                                "opacity": _op_full}, _GEO_GLOBAL]),
                ],
                x=0.01, xanchor="left", y=0.01, yanchor="bottom",
                bgcolor="#f0f2f5", bordercolor="#c9cfd6", borderwidth=1,
                font=dict(size=11, family=FONT),
                pad=dict(r=6, t=6, b=6, l=6),
            ),
        ],
    )

    # ── Write HTML and inject JS ───────────────────────────────────────────────
    _html_path = os.path.join(DIR_OUTPUT, "chile_supply_chain_map.html")
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
        "    var html = '<div class=\"sc-panel-title\">' + d.n + ' facilit' + (d.n===1?'y':'ies') + '  ·  ' + d.val + '</div>';\n"
        "    if (d.minerals) html += '<div class=\"sc-row\"><span class=\"sc-type\">Minerals</span><span>' + d.minerals + '</span></div>';\n"
        "    if (d.types)    html += '<div class=\"sc-row\"><span class=\"sc-type\">Facilities</span><span>' + d.types + '</span></div>';\n"
        "    if (d.ops)      html += '<div class=\"sc-row\"><span class=\"sc-type\">Operators</span><span>' + d.ops + '</span></div>';\n"
        "    if (d.ports)    html += '<div class=\"sc-row\"><span class=\"sc-type\">Exports via</span><span>' + d.ports + '</span></div>';\n"
        "    panel.innerHTML = html;\n"
        "    panel.style.display = 'block';\n"
        "  }\n"
        "\n"
        "  gd.on('plotly_buttonclicked', function(data) {\n"
        "    if (!data || !data.button) return;\n"
        "    var op = OP[data.button.label];\n"
        "    if (op) {\n"
        "      setTimeout(function() {\n"
        "        var tIdxs = gd.data.map(function(_, i) { return i; });\n"
        "        Plotly.restyle(gd, {opacity: op}, tIdxs);\n"
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
        "\n"
        "    function restyleThenLayout(traceAttrs, layoutArgs) {\n"
        "      var tIdxs = gd.data.map(function(_, i) { return i; });\n"
        "      return Plotly.restyle(gd, traceAttrs, tIdxs)\n"
        "        .then(function() { return Plotly.relayout(gd, layoutArgs); });\n"
        "    }\n"
        "\n"
        "    if (pt.curveNumber === CLUST_IDX) {\n"
        "      var key = pt.lat.toFixed(2) + ',' + pt.lon.toFixed(2);\n"
        "      var d = CLUST_DATA[key];\n"
        "      if (d) showPanel(d);\n"
        "      var lat = pt.lat, lon = pt.lon, span = " + str(_GRID_DEG) + ";\n"
        "      var trPts = (d && d.trace_pts) || {};\n"
        "      var newVis = VIS_DOM.slice();\n"
        "      DOM_IDXS.forEach(function(i) { newVis[i] = false; });\n"
        "      newVis[CLUST_EDGE_IDX] = true;\n"
        "      newVis[CLUST_IDX]      = false;\n"
        "      var edgeLats = (d && d.edge_lats) || [];\n"
        "      var edgeLons = (d && d.edge_lons) || [];\n"
        "      Plotly.restyle(gd, {lat: [edgeLats], lon: [edgeLons]}, [CLUST_EDGE_IDX]);\n"
        "      var tIdxs = gd.data.map(function(_, i) { return i; });\n"
        "      Plotly.restyle(gd, {visible: newVis, showlegend: SL_DOM, opacity: OP_DOM}, tIdxs)\n"
        "        .then(function() { return Plotly.relayout(gd, {\n"
        "          'geo.center.lat': lat, 'geo.center.lon': lon,\n"
        "          'geo.lataxis.range': [lat-span, lat+span],\n"
        "          'geo.lonaxis.range': [lon-span, lon+span]\n"
        "        }); })\n"
        "        .then(function() {\n"
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
        "    var contPos = CONT_IDXS.indexOf(pt.curveNumber);\n"
        "    if (contPos === -1) { hidePanel(); return; }\n"
        "    var cont = CONT_NAMES[contPos];\n"
        "    hidePanel();\n"
        "    var newVis = gd.data.map(function(t, i) { return t.visible !== false; });\n"
        "    CONT_IDXS.forEach(function(i)     { newVis[i] = false; });\n"
        "    Object.values(CTRY_DET).forEach(function(i) { newVis[i] = false; });\n"
        "    newVis[CTRY_DET[cont]] = true;\n"
        "    newVis[CTRY_IDX] = true;\n"
        "    var bounds = CONT_BOUNDS[cont] || {};\n"
        "    restyleThenLayout({visible: newVis}, bounds);\n"
        "  });\n"
        "\n"
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

    size_kb = os.path.getsize(_html_path) / 1024
    print(f"\nWritten: {_html_path}  ({size_kb:.0f} KB)")
    print(f"  {n_fac_shown} facilities  {n_dom_edges} domestic edges  {n_countries} export countries")
