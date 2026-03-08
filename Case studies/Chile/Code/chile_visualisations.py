#!/usr/bin/env python3
"""
chile_visualisations.py
Standalone script: generates 10 interactive Plotly HTML charts.

Reads:  output/intermediary/_pipeline_state_6.pkl
Writes: output/charts/*.html  (10 files)

Run:  python3 chile_visualisations.py
"""

import sys, os, pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
try:
    from pipeline_utils import DIR_OUTPUT, DIR_INTERMED
except ImportError:
    DIR_OUTPUT   = "/Users/leoss/Desktop/Website-/Portfolio/Website-/projects/Chile/output"
    DIR_INTERMED = os.path.join(DIR_OUTPUT, "intermediary")

PKL_PATH = os.path.join(DIR_INTERMED, "_pipeline_state_6.pkl")
OUT_DIR  = os.path.join(DIR_OUTPUT, "charts")

"""
Chile Mineral Supply Chain: Production & Export Visualisations
==============================================================
Generates 10 standalone Plotly HTML charts from _pipeline_state_6.pkl.

Charts produced:
  01  Treemap         – Production value by mineral category > mineral
  02  Horizontal bar  – Top 20 facilities (with Cu/non-Cu toggle buttons)
  03  Scattergeo      – Facility map (sized by value, coloured by mineral group)
  04  Port map        – Chilean ports with copper export volumes by product form
  05  Choropleth      – Copper export destinations (world map)
  06  Stacked bar     – Regional value breakdown by mineral group
  07  Sunburst        – Region > Mineral Group > Mineral drilldown
  08  Grouped bar     – Export destinations for 4 exported commodities
  09  Scatter         – Copper mines: output vs non-copper by-product value
  10  Horizontal bar  – Non-copper mineral values (log scale)

Requirements: plotly, pandas, numpy
Run: python3 chile_viz.py
Reads: _pipeline_state_6.pkl (from uploads or local)
Writes: 10 HTML files to /mnt/user-data/outputs/ (or ./outputs/ locally)
"""

import pickle, os, sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── STYLE ────────────────────────────────────────────────────────────────────

STYLE = {
    'font_family': 'IBM Plex Sans, -apple-system, BlinkMacSystemFont, sans-serif',
    'tick_size': 11,
    'axis_title_size': 13,
    'legend_size': 11,
    'annotation_size': 10,
    'title_color': '#1a2744',
    'template': 'plotly_white',
    'plot_bg': '#fafafa',
    'paper_bg': '#fafafa',
    'chart_height': 550,
    'chart_height_small': 420,
    'chart_height_tall': 700,
    'margin': dict(l=60, r=40, t=10, b=50),
    'margin_map': dict(l=10, r=10, t=10, b=60),
    'margin_bar': dict(l=160, r=130, t=10, b=50),
    'grid_color': '#e5e7eb',
    'grid_width': 0.5,
    'zero_line_color': '#c9cfd6',
    'geo': dict(
        showframe=False, showcoastlines=True, coastlinecolor='#c9cfd6',
        projection_type='natural earth', bgcolor='rgba(0,0,0,0)',
        showland=True, landcolor='#f0f2f5', showcountries=True,
        countrycolor='#dde1e7', countrywidth=0.5,
    ),
    'choropleth_line_color': '#c9cfd6',
    'choropleth_line_width': 0.5,
    'colorbar': dict(len=0.7, thickness=15),
}

WRITE_CONFIG = {'displayModeBar': False, 'responsive': True}

# ── MINERAL GROUPING ─────────────────────────────────────────────────────────

MINERAL_GROUPS = {
    'USD_VALUE_CU':        ('Copper',            'Base metals'),
    'USD_VALUE_MO':        ('Molybdenum',        'Base metals'),
    'USD_VALUE_FE':        ('Iron',              'Base metals'),
    'USD_VALUE_ZN':        ('Zinc',              'Base metals'),
    'USD_VALUE_PB':        ('Lead',              'Base metals'),
    'USD_VALUE_AU':        ('Gold',              'Precious metals'),
    'USD_VALUE_AG':        ('Silver',            'Precious metals'),
    'USD_VALUE_LICO3':     ('Lithium Carbonate', 'Battery/strategic'),
    'USD_VALUE_LIOH':      ('Lithium Hydroxide', 'Battery/strategic'),
    'USD_VALUE_LISO4':     ('Lithium Sulfate',   'Battery/strategic'),
    'USD_VALUE_IO':        ('Iodine',            'Battery/strategic'),
    'USD_VALUE_NO3':       ('Nitrates',          'Industrial minerals'),
    'USD_VALUE_ULEXITE':   ('Ulexite',           'Industrial minerals'),
    'USD_VALUE_BORICACID': ('Boric Acid',        'Industrial minerals'),
    'USD_VALUE_KCL':       ('Potash',            'Industrial minerals'),
    'USD_VALUE_SALT':      ('Salt',              'Industrial minerals'),
    'USD_VALUE_CUSO4':     ('Copper Sulfate',    'Industrial minerals'),
    'USD_VALUE_LIMESTONE': ('Limestone',         'Industrial minerals'),
    'USD_VALUE_COQUINA':   ('Coquina',           'Industrial minerals'),
    'USD_VALUE_WHITECACO3':('White CaCO3',       'Industrial minerals'),
    'USD_VALUE_GYPSUM':    ('Gypsum',            'Industrial minerals'),
    'USD_VALUE_PUMICITE':  ('Pumicite',          'Industrial minerals'),
    'USD_VALUE_QUARTZ':    ('Quartz',            'Industrial minerals'),
    'USD_VALUE_SILICASAND':('Silica Sand',       'Industrial minerals'),
    'USD_VALUE_BAUXCLAY':  ('Bauxitic Clay',     'Industrial minerals'),
    'USD_VALUE_KAOLIN':    ('Kaolin',            'Industrial minerals'),
    'USD_VALUE_BENTONITE': ('Bentonite',         'Industrial minerals'),
    'USD_VALUE_DIATOMITE': ('Diatomite',         'Industrial minerals'),
    'USD_VALUE_DOLOMITE':  ('Dolomite',          'Industrial minerals'),
    'USD_VALUE_TALC':      ('Talc',              'Industrial minerals'),
    'USD_VALUE_PERLITE':   ('Perlite',           'Industrial minerals'),
    'USD_VALUE_PEAT':      ('Peat',              'Industrial minerals'),
    'USD_VALUE_PHOSPHATE': ('Phosphate Rocks',   'Industrial minerals'),
    'USD_VALUE_ZEOLITE':   ('Zeolite',           'Industrial minerals'),
}

GROUP_COLORS = {
    'Base metals':         '#4a6fa5',
    'Precious metals':     '#d4853b',
    'Battery/strategic':   '#2e7d4a',
    'Industrial minerals': '#c23a3a',
}

# Finer colours for individual minerals (used in chart 2)
MINERAL_COLORS = {
    'Copper':            '#4a6fa5',
    'Molybdenum':        '#7a9dc4',
    'Iron':              '#3d4f5f',
    'Zinc':              '#a0b0c0',
    'Lead':              '#8899aa',
    'Gold':              '#d4853b',
    'Silver':            '#e6b980',
    'Lithium Carbonate': '#2e7d4a',
    'Lithium Hydroxide': '#5aa87a',
    'Lithium Sulfate':   '#8cc9a0',
    'Iodine':            '#1a6040',
    'Nitrates':          '#c23a3a',
    'Potash':            '#d46b6b',
    'Salt':              '#e89a9a',
    'Boric Acid':        '#a03030',
    'Ulexite':           '#b84545',
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def base_layout(**kwargs):
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
        'Libertador General Bernardo O\'Higgins': 'O\'Higgins',
        'Metropolitana de Santiago': 'Metropolitana',
        'Magallanes y de la Ant\u00e1rtica Chilena': 'Magallanes',
        'Arica y Parinacota': 'Arica y Parinacota',
        'Ays\u00e9n del General Carlos Ib\u00e1\u00f1ez del Campo': 'Ays\u00e9n (Ib\u00e1\u00f1ez)',
    }
    return mapping.get(name, name)




if __name__ == '__main__':
    # ── LOAD DATA ─────────────────────────────────────────────────────────────────
    # PKL_PATH and OUT_DIR defined at module level above.
    if not os.path.exists(PKL_PATH):
        print(f"ERROR: pipeline state not found:\n  {PKL_PATH}")
        print("Run Notebooks A -> B -> C first.")
        sys.exit(1)

    print(f"Loading {PKL_PATH}")
    with open(PKL_PATH, "rb") as _f:
        state = pickle.load(_f)

    inv       = state['inv'].copy()
    edges     = state['edges'].copy()
    ports_df  = state['ports_df'].copy()
    export_df = state['export_df'].copy()

    inv['lat'] = inv['LATITUD'].astype(float)
    inv['lon'] = inv['LONGITUD'].astype(float)

    usd_cols = [c for c in inv.columns if c.startswith('USD_VALUE_') and c != 'USD_VALUE_TOTAL']
    usd_cols_active = [c for c in usd_cols if inv[c].sum() > 0]

    def dominant_mineral(row):
        vals = {c: row[c] for c in usd_cols_active if pd.notna(row[c]) and row[c] > 0}
        if not vals:
            return 'None', 'None'
        top = max(vals, key=vals.get)
        mineral, group = MINERAL_GROUPS.get(top, ('Other', 'Other'))
        return mineral, group

    inv['dominant_mineral'], inv['mineral_group'] = zip(*inv.apply(dominant_mineral, axis=1))

    valued = inv[inv['USD_VALUE_TOTAL'] > 0].copy()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Saving charts to: {OUT_DIR}\n')


    # ======================================================================
    # CHART 1: TREEMAP
    # ======================================================================

    print("Chart 1: Treemap...")

    tree_data = []
    for col in usd_cols_active:
        val = inv[col].sum()
        if val > 0:
            mineral, group = MINERAL_GROUPS[col]
            tree_data.append({'mineral': mineral, 'group': group, 'value': val})

    tree_df = pd.DataFrame(tree_data).sort_values('value', ascending=False)

    fig1 = px.treemap(
        tree_df, path=['group', 'mineral'], values='value',
        color='group', color_discrete_map=GROUP_COLORS,
    )
    fig1.update_traces(
        hovertemplate='<b>%{label}</b><br>Value: %{value:$.3s}<extra></extra>',
        marker=dict(cornerradius=3),
    )
    fig1.update_layout(**base_layout(height=560, margin=dict(l=10, r=10, t=10, b=10)),
                       coloraxis_showscale=False)
    fig1.write_html(f'{OUT_DIR}/01_treemap_mineral_value.html', config=WRITE_CONFIG)


    # ======================================================================
    # CHART 2: TOP FACILITIES – collapsed equal-weight clusters
    # ======================================================================

    print("Chart 2: Top facilities bar...")

    # --- Simplified mineral mapping ---
    SIMPLE_MINERALS = {
        'USD_VALUE_CU':    'Copper',
        'USD_VALUE_MO':    'Molybdenum',
        'USD_VALUE_FE':    'Iron',
        'USD_VALUE_ZN':    'Other',
        'USD_VALUE_PB':    'Other',
        'USD_VALUE_AU':    'Gold',
        'USD_VALUE_AG':    'Silver',
        'USD_VALUE_LICO3': 'Lithium',
        'USD_VALUE_LIOH':  'Lithium',
        'USD_VALUE_LISO4': 'Lithium',
        'USD_VALUE_IO':    'Iodine',
        'USD_VALUE_NO3':   'Nitrates',
        'USD_VALUE_KCL':   'Potash',
    }
    for col in usd_cols_active:
        if col not in SIMPLE_MINERALS:
            SIMPLE_MINERALS[col] = 'Other'

    SIMPLE_GROUPS = {
        'Copper': 'Base metals', 'Molybdenum': 'Base metals',
        'Iron': 'Base metals', 'Gold': 'Precious metals',
        'Silver': 'Precious metals', 'Lithium': 'Battery/strategic',
        'Iodine': 'Battery/strategic', 'Nitrates': 'Industrial minerals',
        'Potash': 'Industrial minerals', 'Other': 'Industrial minerals',
    }

    SIMPLE_COLORS = {
        'Copper': '#4a6fa5', 'Molybdenum': '#7a9dc4', 'Iron': '#3d4f5f',
        'Gold': '#d4853b', 'Silver': '#e6b980', 'Lithium': '#2e7d4a',
        'Iodine': '#1a6040', 'Nitrates': '#c23a3a', 'Potash': '#d46b6b',
        'Other': '#999999',
    }

    # --- Compute simplified per-facility values ---
    pool = valued.copy()
    for smin in SIMPLE_COLORS:
        src_cols = [c for c in usd_cols_active if SIMPLE_MINERALS.get(c) == smin]
        pool[f'S_{smin}'] = pool[src_cols].sum(axis=1) if src_cols else 0

    s_cols = [f'S_{m}' for m in SIMPLE_COLORS]
    pool['S_TOTAL'] = pool[s_cols].sum(axis=1)
    pool['S_NONCU'] = pool['S_TOTAL'] - pool['S_Copper']

    # --- Collapse equal-weight clusters ---
    # Facilities with identical S_TOTAL in the same region are from equal-weight
    # allocation. Group them into a single row.
    # Build proportion signature for each facility
    def mineral_signature(row):
        total = row['S_TOTAL']
        if total <= 0:
            return 'zero'
        shares = []
        for m in sorted(SIMPLE_COLORS.keys()):
            share = round(row[f'S_{m}'] / total, 2)
            shares.append(f"{share:.2f}")
        return '|'.join(shares)

    pool['_round_total'] = pool['S_TOTAL'].round(-3)
    pool['_mineral_sig'] = pool.apply(mineral_signature, axis=1)
    pool['_cluster_key'] = pool['_round_total'].astype(str) + '|' + pool['_mineral_sig']

    # For each cluster of size > 1, keep one representative row and rename
    cluster_counts = pool.groupby('_cluster_key').size()
    multi_clusters = cluster_counts[cluster_counts > 1].index

    rows_to_drop = []
    for ck in multi_clusters:
        cluster = pool[pool['_cluster_key'] == ck]
        n = len(cluster)
        # Keep the first row, rename it
        keep_idx = cluster.index[0]
        drop_idxs = cluster.index[1:]
        rows_to_drop.extend(drop_idxs)

        # Build a short label
        region_short = short_region(pool.loc[keep_idx, 'REGION'])
        # Find the dominant mineral for labeling
        mineral_vals = {m: pool.loc[keep_idx, f'S_{m}'] for m in SIMPLE_COLORS
                        if pool.loc[keep_idx, f'S_{m}'] > 0}
        top_mineral = max(mineral_vals, key=mineral_vals.get) if mineral_vals else ''

        base_name = pool.loc[keep_idx, 'FACILITY_NAME'][:18]
        pool.loc[keep_idx, 'FACILITY_NAME'] = f"{base_name} + {n-1} others"

    pool = pool.drop(rows_to_drop)

    # --- Top 80 pool ---
    pool = pool.nlargest(80, 'S_TOTAL')
    pool['short_name'] = pool['FACILITY_NAME'].str[:30]

    # De-duplicate short names
    dupes = pool['short_name'].duplicated(keep=False)
    if dupes.any():
        pool.loc[dupes, 'short_name'] = pool.loc[dupes].apply(
            lambda r: f"{r['short_name'][:26]} ({r['REGION'][:3]})", axis=1)

    all_names = pool['short_name'].tolist()

    # --- Mineral display order ---
    mineral_display = [m for m in
        sorted(SIMPLE_COLORS.keys(), key=lambda m: -pool[f'S_{m}'].sum())
        if pool[f'S_{m}'].sum() > 0]

    # --- Build traces ---
    traces = []
    trace_names = []
    for mineral in mineral_display:
        group = SIMPLE_GROUPS[mineral]
        vals = pool[f'S_{mineral}'].values
        traces.append(go.Bar(
            y=all_names, x=vals, name=mineral, orientation='h',
            marker_color=SIMPLE_COLORS[mineral],
            legendgroup=group,legendgrouptitle_text=group,
            hovertemplate='%{y}: %{x:$.3s}<extra>' + mineral + '</extra>',
        ))
        trace_names.append(mineral)

    fig2 = go.Figure(data=traces)

    # --- Button builder ---
    def make_btn(label, vis, sort_col, n=10):
        top_n = pool.nlargest(n, sort_col)
        ordered = top_n.sort_values(sort_col, ascending=True)['short_name'].tolist()
        rest = [nm for nm in all_names if nm not in ordered]
        cat = ordered + rest

        anns = []
        for _, row in top_n.iterrows():
            v = row[sort_col]
            if v > 0:
                anns.append(dict(
                    x=v, y=row['short_name'], text=f"  {fmt_usd(v)}",
                    showarrow=False, xanchor='left',
                    font=dict(size=10, color='#555'),
                ))

        return dict(label=label, method='update', args=[
            {'visible': vis},
            {'annotations': anns,
            'xaxis.autorange': True,
            'yaxis.categoryarray': cat, 'yaxis.range': [-0.5, n - 0.5]},
        ])

    vis_all  = [True] * len(traces)
    vis_nocu = [tn != 'Copper' for tn in trace_names]

    buttons = [
        make_btn('  All minerals  ', vis_all, 'S_TOTAL'),
        make_btn('  Exclude copper  ', vis_nocu, 'S_NONCU'),
    ]

    init = make_btn('init', vis_all, 'S_TOTAL')

    fig2.update_layout(
        **base_layout(height=620, margin=dict(l=210, r=80, t=50, b=80)),
        barmode='stack',
        xaxis=dict(
            title='Estimated Value (USD)',
            gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
            tickformat='$.2s',
        ),
        yaxis=dict(
            title='',
            categoryorder='array',
            categoryarray=init['args'][1]['yaxis.categoryarray'],
            range=[-0.5, 9.5],
        ),
        legend=dict(
            orientation='h', yanchor='top', y=-0.12, xanchor='center', x=0.50,
            font=dict(size=STYLE['legend_size']),
        ),
        annotations=init['args'][1]['annotations'],
        updatemenus=[dict(
            type='buttons', direction='left',
            x=1.0, xanchor='right', y=1.22, yanchor='top',
            bgcolor='#f0f2f5', bordercolor='#c9cfd6', borderwidth=1,
            font=dict(size=11),
            buttons=buttons,
        )],
    )
    fig2.write_html(f'{OUT_DIR}/02_top_facilities_bar.html', config=WRITE_CONFIG)
    # Inject JS: force xaxis rescale on legend click
    js_patch = """
    <script>
    (function waitForPlotly() {
        var gd = document.querySelector('.js-plotly-plot');
        if (!gd || !gd._fullData) {
            setTimeout(waitForPlotly, 200);
            return;
        }

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
                    x: allNames[name], y: name, text: '  ' + fmtUsd(allNames[name]),
                    showarrow: false, xanchor: 'left',
                    font: {size: 10, color: '#555'}
                };
            });

            // Always show 10 slots worth of space
            var startIdx = sorted.indexOf(top10[0]);
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

        gd.on('plotly_legenddoubleclick', function() {
            return false;
        });

        gd.on('plotly_buttonclicked', function() {
            setTimeout(rescaleAndSort, 50);
        });
    })();
    </script>
    """
    with open(f'{OUT_DIR}/02_top_facilities_bar.html', 'r') as f:
        html = f.read()
    html = html.replace('</body>', js_patch + '</body>')
    with open(f'{OUT_DIR}/02_top_facilities_bar.html', 'w') as f:
        f.write(html)

    # ======================================================================
    # CHART 3: SCATTERGEO – Facility map
    # ======================================================================

    print("Chart 3: Facility map...")

    # ── TUNABLE INITIAL VIEW ──────────────────────────────────────────────
    CENTER_LAT = -26.5
    CENTER_LON = -70.5
    ZOOM_SCALE = 4.6
    # ──────────────────────────────────────────────────────────────────────

    map_df = valued.copy()
    map_df['size_val'] = np.sqrt(map_df['USD_VALUE_TOTAL'] / 1e6) * 1.2
    map_df['size_val'] = map_df['size_val'].clip(lower=3, upper=35)
    map_df['hover'] = map_df.apply(
        lambda r: (
            f"<b>{r['FACILITY_NAME']}</b><br>"
            f"Region: {short_region(r['REGION'])}<br>"
            f"Type: {r['FACILITY_TYPE']}<br>"
            f"Primary: {r['dominant_mineral']}<br>"
            f"Total Value: {fmt_usd(r['USD_VALUE_TOTAL'])}"
        ), axis=1
    )

    SA_COUNTRIES = ['ARG', 'BOL', 'BRA', 'COL', 'ECU', 'GUY', 'PRY',
                    'PER', 'SUR', 'URY', 'VEN', 'GUF']

    fig3 = go.Figure()

    fig3.add_trace(go.Choropleth(
        locations=SA_COUNTRIES, z=[1] * len(SA_COUNTRIES),
        locationmode='ISO-3',
        colorscale=[[0, '#dcdcdc'], [1, '#dcdcdc']],
        marker_line_color='#c9cfd6', marker_line_width=0.5,
        showscale=False, showlegend=False, hoverinfo='skip',
    ))

    fig3.add_trace(go.Choropleth(
        locations=['CHL'], z=[1],
        locationmode='ISO-3',
        colorscale=[[0, '#f7f8fa'], [1, '#f7f8fa']],
        marker_line_color='#aab0b8', marker_line_width=0.8,
        showscale=False, showlegend=False, hoverinfo='skip',
    ))

    for group in ['Base metals', 'Precious metals', 'Battery/strategic', 'Industrial minerals']:
        sub = map_df[map_df['mineral_group'] == group]
        if len(sub) == 0:
            continue
        fig3.add_trace(go.Scattergeo(
            lat=sub['lat'], lon=sub['lon'], text=sub['hover'], hoverinfo='text',
            name=group,
            marker=dict(size=sub['size_val'], color=GROUP_COLORS[group], opacity=0.75,
                        line=dict(width=0.5, color='white'), sizemode='diameter'),
        ))

    _layout3 = base_layout(margin=dict(l=0, r=0, t=0, b=0))
    _layout3.pop('height', None)
    _layout3['autosize'] = True
    fig3.update_layout(
        **_layout3,
        geo=dict(
            scope='south america',
            showframe=False, showcoastlines=True, coastlinecolor='#c9cfd6',
            bgcolor='rgba(0,0,0,0)',
            showocean=True, oceancolor='#e8eef6',
            showland=True, landcolor='#dcdcdc',
            showcountries=True, countrycolor='#c9cfd6', countrywidth=0.5,
            center=dict(lat=CENTER_LAT, lon=CENTER_LON),
            projection_scale=ZOOM_SCALE,
            resolution=50,
            domain=dict(x=[0, 1], y=[0, 1]),
        ),
        # Legend: bottom-right corner of the map
        legend=dict(
            yanchor='bottom', y=0.03, xanchor='right', x=0.98,
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='#dde1e7', borderwidth=1,
            font=dict(size=STYLE['legend_size']),
        ),
        # Reset view button: top-right corner of the map
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(
                label='Reset view',
                method='relayout',
                args=[{
                    'geo.center.lat': CENTER_LAT,
                    'geo.center.lon': CENTER_LON,
                    'geo.projection.scale': ZOOM_SCALE,
                }],
            )],
            x=0.98, xanchor='right', y=0.97, yanchor='top',
            bgcolor='#f0f2f5', bordercolor='#c9cfd6', borderwidth=1,
            font=dict(size=11, family=STYLE['font_family']),
        )],
    )
    fig3.write_html(f'{OUT_DIR}/03_facility_map.html', config=WRITE_CONFIG,
                    default_height='100%', default_width='100%')

    # ======================================================================
    # CHART 4: PORT MAP – copper exports by product form
    # ======================================================================

    print("Chart 4: Port map...")

    cu_exp = export_df[export_df['COMMODITIES'] == 'Copper'].copy()
    port_vols = cu_exp.groupby(['FROM_NAME', 'PRODUCT_FORM'])['EXPORT_VALUE'].sum().reset_index()

    fig4 = go.Figure()
    product_colors = {'concentrate': '#4a6fa5', 'cathode': '#2e7d4a', 'blister': '#d4853b'}

    for form, color in product_colors.items():
        sub = port_vols[port_vols['PRODUCT_FORM'] == form].copy()
        sub = sub.merge(ports_df[['name', 'lat', 'lon']], left_on='FROM_NAME', right_on='name')
        if len(sub) == 0:
            continue
        sub['size'] = np.sqrt(sub['EXPORT_VALUE']) * 3
        sub['size'] = sub['size'].clip(lower=5, upper=45)
        fig4.add_trace(go.Scattergeo(
            lat=sub['lat'], lon=sub['lon'],
            text=sub.apply(
                lambda r: f"<b>{r['FROM_NAME']}</b><br>{form.title()}: {r['EXPORT_VALUE']:.0f} kMT",
                axis=1),
            hoverinfo='text', name=form.title(),
            marker=dict(size=sub['size'], color=color, opacity=0.7,
                        line=dict(width=0.8, color='white'), sizemode='diameter'),
        ))

    fig4.update_layout(
        **base_layout(height=700, margin=dict(l=0, r=0, t=10, b=0)),
        geo=dict(
            scope='south america',
            showframe=False, showcoastlines=True, coastlinecolor='#c9cfd6',
            bgcolor='rgba(0,0,0,0)', showland=True, landcolor='#f0f2f5',
            showcountries=True, countrycolor='#dde1e7', countrywidth=0.5,
            lonaxis=dict(range=[-76, -66]), lataxis=dict(range=[-56, -17]),
            resolution=50,
        ),
        legend=dict(yanchor='top', y=0.95, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.85)', font=dict(size=STYLE['legend_size'])),
    )
    fig4.write_html(f'{OUT_DIR}/04_port_map.html', config=WRITE_CONFIG)


    # ======================================================================
    # CHART 5: CHOROPLETH – Export destinations by commodity
    # ======================================================================
    # Layout:
    #   Row 1 (buttons):  "All Minerals" + top 3 by export value
    #   Row 2 (dropdown): remaining minerals with >$1M exports, sorted by value
    # ======================================================================

    print("Chart 5: Export choropleth (all commodities)...")

    # --- Country name -> ISO3 mapping (for Plotly choropleth locations) ------
    COUNTRY_ISO = {
        "China": "CHN", "Japan": "JPN", "South Korea": "KOR", "USA": "USA",
        "Brazil": "BRA", "India": "IND", "Germany": "DEU", "Spain": "ESP",
        "France": "FRA", "Italy": "ITA", "Netherlands": "NLD", "Belgium": "BEL",
        "Sweden": "SWE", "Bulgaria": "BGR", "Finland": "FIN", "Canada": "CAN",
        "Mexico": "MEX", "Taiwan": "TWN", "Thailand": "THA", "Philippines": "PHL",
        "Malaysia": "MYS", "Indonesia": "IDN", "Vietnam": "VNM", "Peru": "PER",
        "Colombia": "COL", "Argentina": "ARG", "Turkey": "TUR",
        "United Kingdom": "GBR", "Switzerland": "CHE", "Singapore": "SGP",
        "Greece": "GRC", "Portugal": "PRT", "Panama": "PAN", "Bahrain": "BHR",
        "UAE": "ARE", "Hong Kong": "HKG", "Poland": "POL", "Norway": "NOR",
        "Costa Rica": "CRI", "Cambodia": "KHM", "South Africa": "ZAF",
        "Namibia": "NAM", "Bangladesh": "BGD", "Bolivia": "BOL", "Ecuador": "ECU",
        "Dominican Rep.": "DOM", "Paraguay": "PRY", "Australia": "AUS",
        "Congo": "COG", "Guatemala": "GTM", "Denmark": "DNK", "Uruguay": "URY",
        "Honduras": "HND", "Cyprus": "CYP", "Saudi Arabia": "SAU",
        "Austria": "AUT", "New Zealand": "NZL", "Ireland": "IRL",
        "Nigeria": "NGA", "Ghana": "GHA", "Pakistan": "PAK", "Morocco": "MAR",
        "Jamaica": "JAM", "Algeria": "DZA", "Mozambique": "MOZ", "Hungary": "HUN",
        "El Salvador": "SLV", "Nicaragua": "NIC", "Venezuela": "VEN",
        "Lebanon": "LBN", "Kuwait": "KWT", "Sri Lanka": "LKA", "Israel": "ISR",
        "Lithuania": "LTU",
    }

    PRODUCT_FORM_PRICES = {
        'cathode': 9_200, 'concentrate': 2_800, 'blister': 8_800,
    }

    # --- Colour palettes per commodity (top minerals get distinct ones) ------
    COMM_COLORS = {
        'Copper':         [[0.0, '#e8edf3'], [0.3, '#a0b0c0'], [0.6, '#4a6fa5'], [1.0, '#1a2744']],
        'Molybdenum':     [[0.0, '#fdf0e4'], [0.3, '#e6b980'], [0.6, '#d4853b'], [1.0, '#6b421d']],
        'Lithium':        [[0.0, '#e8f5ec'], [0.3, '#8cc9a0'], [0.6, '#2e7d4a'], [1.0, '#143d22']],
        'Iodine':         [[0.0, '#fbe8e8'], [0.3, '#d98a8a'], [0.6, '#c23a3a'], [1.0, '#5e1a1a']],
        'Iron':           [[0.0, '#eae6e1'], [0.3, '#b5a48e'], [0.6, '#7d6644'], [1.0, '#3e3220']],
        'Gold':           [[0.0, '#fdf8ec'], [0.3, '#e8d48c'], [0.6, '#c9a832'], [1.0, '#6b5a18']],
        'Silver':         [[0.0, '#eef0f2'], [0.3, '#b0b8c0'], [0.6, '#6e7e8e'], [1.0, '#343e48']],
        'Nitrate':        [[0.0, '#ecf2e8'], [0.3, '#a0c890'], [0.6, '#509038'], [1.0, '#284a1c']],
        'Boron':          [[0.0, '#f5eef0'], [0.3, '#c8a0b0'], [0.6, '#8a4860'], [1.0, '#442430']],
        'Salt':           [[0.0, '#ecf0f5'], [0.3, '#90a8c4'], [0.6, '#3868a0'], [1.0, '#1c3450']],
        'Potash':         [[0.0, '#f0ece8'], [0.3, '#c4a890'], [0.6, '#886038'], [1.0, '#44301c']],
        'Rhenium':        [[0.0, '#f0ecf5'], [0.3, '#a898c4'], [0.6, '#5840a0'], [1.0, '#2c2050']],
        'Lead':           [[0.0, '#eef0ee'], [0.3, '#9ab09a'], [0.6, '#487048'], [1.0, '#243824']],
        'Zinc':           [[0.0, '#eef2f0'], [0.3, '#90c0b0'], [0.6, '#389070'], [1.0, '#1c4838']],
        'Copper Sulfate': [[0.0, '#e8f0f3'], [0.3, '#80b0c0'], [0.6, '#2870a5'], [1.0, '#143844']],
        'Sulfuric Acid':  [[0.0, '#f5f0e8'], [0.3, '#c4b080'], [0.6, '#886820'], [1.0, '#443410']],
        'Selenium':       [[0.0, '#f3eef0'], [0.3, '#c0a0a8'], [0.6, '#904858'], [1.0, '#48242c']],
        '_all':           [[0.0, '#f0ece4'], [0.3, '#c4a870'], [0.6, '#8a6020'], [1.0, '#453010']],
    }
    _DEFAULT_SCALE = [[0.0, '#eef1f5'], [0.3, '#8ea0b8'], [0.6, '#3d6090'], [1.0, '#1a3050']]

    # --- Estimate USD value per export edge -----------------------------------
    def estimate_usd(row):
        """Convert export edge value to USD regardless of source unit."""
        val = row.get('EXPORT_VALUE', 0)
        unit = str(row.get('EXPORT_UNIT', ''))
        pform = str(row.get('PRODUCT_FORM', ''))
        comm = str(row.get('COMMODITIES', ''))

        if val == 0 or pd.isna(val):
            return 0

        # Aduanas edges: already in $FOB
        if unit == '$FOB':
            return val

        # Comtrade edges: already in $USD
        if unit == '$USD':
            return val

        # COCHILCO edges with $M_FOB (Li, I from Tabla 11)
        if unit == '$M_FOB':
            return val * 1e6

        # COCHILCO copper: volume in kMT, needs price conversion
        if comm == 'Copper' and unit == 'kMT':
            price = PRODUCT_FORM_PRICES.get(pform, 5_000)
            return val * 1_000 * price

        # COCHILCO Mo: volume in MT
        if unit == 'MT':
            return val * 46_954  # Mo oxide price from pipeline

        # Fallback: treat as raw value
        return val

    # --- Build per-commodity country-level data --------------------------------
    comm_country_data = {}

    for comm in export_df['COMMODITIES'].unique():
        sub = export_df[export_df['COMMODITIES'] == comm].copy()
        if len(sub) == 0:
            continue

        sub['USD_EST'] = sub.apply(estimate_usd, axis=1)
        total_usd = sub['USD_EST'].sum()
        if total_usd < 1e6:
            continue  # skip minerals with <$1M total exports

        by_country = sub.groupby('TO_NAME')['USD_EST'].sum().reset_index()
        by_country.columns = ['country', 'usd']
        by_country['iso'] = by_country['country'].map(COUNTRY_ISO)
        by_country = by_country.dropna(subset=['iso'])
        by_country = by_country[by_country['usd'] > 0]
        by_country['log_usd'] = np.log10(by_country['usd'].clip(lower=1))
        by_country['hover'] = by_country.apply(
            lambda r: f"<b>{r['country']}</b><br>{fmt_usd(r['usd'])}", axis=1)
        comm_country_data[comm] = (by_country, total_usd)

    # --- Sort by total export value and split into tiers -----------------------
    sorted_comms = sorted(comm_country_data.keys(), key=lambda c: -comm_country_data[c][1])

    # Build "All" aggregate
    all_frames = [comm_country_data[c][0][['country', 'usd', 'iso']].copy() for c in sorted_comms]
    all_combined = pd.concat(all_frames, ignore_index=True)
    all_combined = all_combined.groupby(['country', 'iso'])['usd'].sum().reset_index()
    all_combined['log_usd'] = np.log10(all_combined['usd'].clip(lower=1))
    all_combined['hover'] = all_combined.apply(
        lambda r: f"<b>{r['country']}</b><br>{fmt_usd(r['usd'])}", axis=1)

    top3 = sorted_comms[:3]
    rest = sorted_comms[3:]

    # Trace order: [All, Top1, Top2, Top3, Rest0, Rest1, ...]
    trace_labels = ['All Minerals'] + top3 + rest
    n_traces = len(trace_labels)

    print(f"  Commodities with exports: {len(sorted_comms)}")
    print(f"  Top 3: {', '.join(top3)}")
    print(f"  Remaining: {len(rest)} minerals")
    for c in sorted_comms:
        _, total = comm_country_data[c]
        n = len(comm_country_data[c][0])
        print(f"    {c:<20} {n:>3} countries  ${total/1e6:>10,.1f}M")

    # --- Build figure ---------------------------------------------------------
    fig5 = go.Figure()

    for idx, label in enumerate(trace_labels):
        if label == 'All Minerals':
            cdf = all_combined
            cscale = COMM_COLORS['_all']
        else:
            cdf = comm_country_data[label][0]
            cscale = COMM_COLORS.get(label, _DEFAULT_SCALE)

        fig5.add_trace(go.Choropleth(
            locations=cdf['iso'],
            z=cdf['log_usd'],
            text=cdf['hover'],
            hoverinfo='text',
            colorscale=cscale,
            marker_line_color=STYLE['choropleth_line_color'],
            marker_line_width=STYLE['choropleth_line_width'],
            colorbar=dict(
                title=dict(text='USD (log scale)', font=dict(size=11)),
                tickvals=[5, 6, 7, 8, 9, 10, 11],
                ticktext=['$100K', '$1M', '$10M', '$100M', '$1B', '$10B', '$100B'],
                **STYLE['colorbar'],
            ),
            visible=(idx == 0),  # "All Minerals" visible by default
        ))

    # --- Controls: button row (All + Top 3) + dropdown (remaining) -----------

    def make_vis(active_idx):
        """Return visibility list with only trace at active_idx shown."""
        vis = [False] * n_traces
        vis[active_idx] = True
        return vis

    # Row 1: "All" + top 3 as buttons
    buttons_top = []
    for i, label in enumerate(['All Minerals'] + top3):
        buttons_top.append(dict(
            label=f'  {label}  ',
            method='update',
            args=[{'visible': make_vis(i)}],
        ))

    # Row 2: dropdown for remaining minerals
    dropdown_items = []
    for i, label in enumerate(rest):
        trace_idx = 4 + i  # offset past All + 3 buttons
        total_m = comm_country_data[label][1] / 1e6
        dropdown_items.append(dict(
            label=f'{label} (${total_m:,.0f}M)',
            method='update',
            args=[{'visible': make_vis(trace_idx)}],
        ))

    update_menus = [
        # Buttons row: All + Top 3
        dict(
            type='buttons', direction='left',
            x=0.5, xanchor='center', y=1.10, yanchor='top',
            bgcolor='#f0f2f5', bordercolor='#c9cfd6', borderwidth=1,
            font=dict(size=11, family=STYLE['font_family']),
            buttons=buttons_top,
        ),
    ]

    # Only add dropdown if there are remaining minerals
    if dropdown_items:
        update_menus.append(dict(
            type='dropdown', direction='down',
            x=1.0, xanchor='right', y=1.10, yanchor='top',
            bgcolor='#f0f2f5', bordercolor='#c9cfd6', borderwidth=1,
            font=dict(size=11, family=STYLE['font_family']),
            buttons=dropdown_items,
        ))

    fig5.update_layout(
        **base_layout(height=500, margin=dict(l=0, r=0, t=65, b=0)),
        geo=dict(**STYLE['geo']),
        updatemenus=update_menus,
    )
    fig5.write_html(f'{OUT_DIR}/05_export_choropleth.html', config=WRITE_CONFIG)
    print(f"  Saved: 05_export_choropleth.html ({n_traces} traces)")

    # ======================================================================
    # CHART 6B: THREE-COLUMN TILE CARTOGRAM – Value vs Area vs Population
    # ======================================================================

    print("Chart 6b: Three-column tile cartogram...")

    REGION_ORDER_NS = [
        'Arica y Parinacota',
        'Tarapacá',
        'Antofagasta',
        'Atacama',
        'Coquimbo',
        'Valparaíso',
        'Metropolitana',
        "O'Higgins",
        'Maule',
        'Biobío',
        'Aysén (Ibáñez)',
        'Magallanes',
    ]

    REGION_AREA_KM2 = {
        'Arica y Parinacota': 16873,
        'Tarapacá': 42226,
        'Antofagasta': 126049,
        'Atacama': 75176,
        'Coquimbo': 40580,
        'Valparaíso': 16396,
        'Metropolitana': 15403,
        "O'Higgins": 16387,
        'Maule': 30296,
        'Biobío': 23890,
        'Aysén (Ibáñez)': 108494,
        'Magallanes': 132291,
    }

    REGION_POP = {
        'Arica y Parinacota': 239126,
        'Tarapacá': 336769,
        'Antofagasta': 622640,
        'Atacama': 312486,
        'Coquimbo': 771085,
        'Valparaíso': 1825757,
        'Metropolitana': 7314176,
        "O'Higgins": 918751,
        'Maule': 1042989,
        'Biobío': 2114286,
        'Aysén (Ibáñez)': 108328,
        'Magallanes': 164661,
    }

    # --- Build per-region total mineral value (merge Santiago into Metropolitana) ---
    region_vals = inv.groupby('REGION')['USD_VALUE_TOTAL'].sum()
    region_vals.index = region_vals.index.map(short_region)
    if 'Santiago' in region_vals.index and 'Metropolitana' in region_vals.index:
        region_vals['Metropolitana'] += region_vals['Santiago']
        region_vals = region_vals.drop('Santiago')
    elif 'Santiago' in region_vals.index:
        region_vals = region_vals.rename(index={'Santiago': 'Metropolitana'})

    # --- Tile parameters (calibrated for ~70 total tiles each) ---
    GRID_WIDTH = 6
    TILE_PAD = 0.06
    GAP_ROWS = 0.5

    VALUE_TILE = 1e9
    AREA_TILE = 10000
    POP_TILE = 250000

    MINERAL_COLOR = '#4a6fa5'
    AREA_COLOR = '#a8b8c8'
    POP_COLOR = '#c4a882'

    # --- Column x positions (tighter spacing for container fit) ---
    COL1_START = 0
    COL2_START = GRID_WIDTH + 1.8
    COL3_START = COL2_START + GRID_WIDTH + 1.8
    LABEL_RIGHT = -0.4

    # --- Tile builders ---
    def value_tiles(region):
        val = region_vals.get(region, 0)
        if val <= 0:
            return []
        n = max(1, round(val / VALUE_TILE))
        hover = f"<b>{region}</b><br>{fmt_usd(val)}"
        return [hover] * n

    def area_tiles(region):
        area = REGION_AREA_KM2.get(region, 0)
        if area == 0:
            return []
        n = max(1, round(area / AREA_TILE))
        hover = f"<b>{region}</b><br>{area:,.0f} km²"
        return [hover] * n

    def pop_tiles(region):
        pop = REGION_POP.get(region, 0)
        if pop == 0:
            return []
        n = max(1, round(pop / POP_TILE))
        hover = f"<b>{region}</b><br>{pop:,.0f} people"
        return [hover] * n

    def rows_needed(tile_list):
        if not tile_list:
            return 0
        return (len(tile_list) - 1) // GRID_WIDTH + 1

    # --- Layout ---
    all_tiles = []
    annotations = []
    current_y = 0

    for region in reversed(REGION_ORDER_NS):
        vt = value_tiles(region)
        at = area_tiles(region)
        pt = pop_tiles(region)

        if not vt and not at and not pt:
            continue

        max_rows = max(rows_needed(vt), rows_needed(at), rows_needed(pt))
        v_off = (max_rows - rows_needed(vt)) / 2
        a_off = (max_rows - rows_needed(at)) / 2
        p_off = (max_rows - rows_needed(pt)) / 2

        for i, hover in enumerate(vt):
            all_tiles.append({
                'x': COL1_START + i % GRID_WIDTH,
                'y': current_y + v_off + i // GRID_WIDTH,
                'color': MINERAL_COLOR, 'hover': hover,
            })

        for i, hover in enumerate(at):
            all_tiles.append({
                'x': COL2_START + i % GRID_WIDTH,
                'y': current_y + a_off + i // GRID_WIDTH,
                'color': AREA_COLOR, 'hover': hover,
            })

        for i, hover in enumerate(pt):
            all_tiles.append({
                'x': COL3_START + i % GRID_WIDTH,
                'y': current_y + p_off + i // GRID_WIDTH,
                'color': POP_COLOR, 'hover': hover,
            })

        label_y = current_y + max_rows / 2
        annotations.append(dict(
            x=LABEL_RIGHT, y=label_y,
            text=f"<b>{region}</b>",
            showarrow=False, xanchor='right',
            font=dict(size=11, color='#333', family=STYLE['font_family']),
        ))

        current_y += max_rows + GAP_ROWS

    tiles_df = pd.DataFrame(all_tiles)

    # --- Build figure ---
    fig6b = go.Figure()

    for _, t in tiles_df.iterrows():
        fig6b.add_shape(
            type='rect',
            x0=t['x'] + TILE_PAD, x1=t['x'] + 1 - TILE_PAD,
            y0=t['y'] + TILE_PAD, y1=t['y'] + 1 - TILE_PAD,
            fillcolor=t['color'],
            opacity=0.85, line=dict(width=0), layer='below',
        )

    fig6b.add_trace(go.Scatter(
        x=tiles_df['x'] + 0.5, y=tiles_df['y'] + 0.5,
        mode='markers',
        marker=dict(size=20, color='rgba(0,0,0,0)'),
        text=tiles_df['hover'], hoverinfo='text',
        showlegend=False,
    ))

    # --- Column headers ---
    annotations.append(dict(
        x=COL1_START + GRID_WIDTH / 2, y=current_y + 0.8,
        text="<b>Mineral Value</b><br><span style='font-size:10px;color:#777'>1 tile ≈ $1B</span>",
        showarrow=False, xanchor='center',
        font=dict(size=14, color=MINERAL_COLOR, family=STYLE['font_family']),
    ))
    annotations.append(dict(
        x=COL2_START + GRID_WIDTH / 2, y=current_y + 0.8,
        text="<b>Surface Area</b><br><span style='font-size:10px;color:#777'>1 tile ≈ 10,000 km²</span>",
        showarrow=False, xanchor='center',
        font=dict(size=14, color='#6a7a8a', family=STYLE['font_family']),
    ))
    annotations.append(dict(
        x=COL3_START + GRID_WIDTH / 2, y=current_y + 0.8,
        text="<b>Population</b><br><span style='font-size:10px;color:#777'>1 tile ≈ 250k people</span>",
        showarrow=False, xanchor='center',
        font=dict(size=14, color='#9a7a5a', family=STYLE['font_family']),
    ))

    total_width = COL3_START + GRID_WIDTH + 1

    fig6b.update_layout(
        **base_layout(height=max(720, int(current_y * 13) + 100),
                      margin=dict(l=120, r=20, t=10, b=10)),
        annotations=annotations,
        xaxis=dict(
            visible=False,
            range=[LABEL_RIGHT - 1.2, total_width + 0.5],
            constrain='domain',
        ),
        yaxis=dict(
            visible=False,
            range=[-1, current_y + 2],
            scaleanchor='x',
            scaleratio=1,
        ),
        showlegend=False,
        autosize=True,
    )
    fig6b.write_html(f'{OUT_DIR}/06b_regional_tile_cartogram.html', config=WRITE_CONFIG,
                     default_width='100%', default_height='100%')


    # ======================================================================
    # CHART 7: SUNBURST – Region > Group > Mineral (patched)
    # ======================================================================

    print("Chart 7: Sunburst...")

    sun_records = []
    for _, row in valued.iterrows():
        region = short_region(row['REGION'])
        for col in usd_cols_active:
            val = row[col]
            if pd.notna(val) and val > 0:
                mineral, group = MINERAL_GROUPS[col]
                sun_records.append({'region': region, 'group': group,
                                    'mineral': mineral, 'value': val})

    sun_df = pd.DataFrame(sun_records)
    sun_agg = sun_df.groupby(['region', 'group', 'mineral'])['value'].sum().reset_index()

    # --- Collapse small regions into "Other" ---
    region_totals = sun_agg.groupby('region')['value'].sum().sort_values(ascending=False)
    total_value = region_totals.sum()
    TOP_REGION_SHARE = 0.02

    top_regions = region_totals[region_totals / total_value >= TOP_REGION_SHARE].index.tolist()
    sun_agg['region'] = sun_agg['region'].apply(lambda r: r if r in top_regions else 'Other regions')
    sun_agg = sun_agg.groupby(['region', 'group', 'mineral'], observed=True)['value'].sum().reset_index()

    # --- Collapse small minerals into "Other" per region+group ---
    MINERAL_THRESHOLD = 300e6  # $300M

    def collapse_minerals(df):
        rows = []
        for (region, group), grp in df.groupby(['region', 'group'], observed=True):
            big = grp[grp['value'] >= MINERAL_THRESHOLD]
            small = grp[grp['value'] < MINERAL_THRESHOLD]
            for _, r in big.iterrows():
                rows.append(r.to_dict())
            if len(small) > 0:
                other_val = small['value'].sum()
                if other_val > 0:
                    rows.append({
                        'region': region, 'group': group,
                        'mineral': f'Other ({group.split()[0].lower()})',
                        'value': other_val
                    })
        return pd.DataFrame(rows)

    sun_agg = collapse_minerals(sun_agg)

    # --- Sort regions by total value (largest first) ---
    region_order = sun_agg.groupby('region', observed=True)['value'].sum().sort_values(ascending=False).index.tolist()
    if 'Other regions' in region_order:
        region_order.remove('Other regions')
        region_order.append('Other regions')

    sun_agg['region'] = pd.Categorical(sun_agg['region'], categories=region_order, ordered=True)
    sun_agg = sun_agg.sort_values(['region', 'group', 'mineral'])

    # --- Build go.Sunburst manually for full color control ---
    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    REGION_COLOR = '#d5d8dc'
    OTHER_REGION_COLOR = '#e8e8e8'

    # Lighter shades for "Other (group)" entries
    GROUP_COLORS_LIGHT = {
        'Base metals':         '#a3b5cc',
        'Precious metals':     '#e6c49a',
        'Battery/strategic':   '#7ab893',
        'Industrial minerals': '#d98a8a',
    }

    # Precompute sums
    region_sums = sun_agg.groupby('region', observed=True)['value'].sum()
    rg_sums = sun_agg.groupby(['region', 'group'], observed=True)['value'].sum()

    # Level 1: Regions
    for region in region_order:
        ids.append(region)
        labels.append(region)
        parents.append('')
        values.append(float(region_sums.get(region, 0)))
        colors.append(OTHER_REGION_COLOR if region == 'Other regions' else REGION_COLOR)

    # Level 2: Region > Group
    for (region, group), val in rg_sums.items():
        gid = f"{region}|{group}"
        ids.append(gid)
        labels.append(group)
        parents.append(region)
        values.append(float(val))
        colors.append(GROUP_COLORS.get(group, '#999'))

    # Level 3: Region > Group > Mineral
    for _, row in sun_agg.iterrows():
        mid = f"{row['region']}|{row['group']}|{row['mineral']}"
        ids.append(mid)
        labels.append(row['mineral'])
        parents.append(f"{row['region']}|{row['group']}")
        values.append(float(row['value']))
        # Use lighter shade for "Other" entries
        if row['mineral'].startswith('Other ('):
            colors.append(GROUP_COLORS_LIGHT.get(row['group'], '#bbb'))
        else:
            colors.append(GROUP_COLORS.get(row['group'], '#999'))

    # Build display text: hide labels for small segments
    display_text = []
    for v, lbl in zip(values, labels):
        if v < 500e6:
            display_text.append('')
        else:
            display_text.append(lbl)

    fig7 = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        text=display_text,
        textinfo='text',
        marker=dict(colors=colors, line=dict(width=0.5, color='white')),
        branchvalues='total',
        hovertemplate='<b>%{label}</b><br>Value: %{value:$.3s}<extra></extra>',
        insidetextorientation='radial',
        maxdepth=3,
    ))

    _layout7 = base_layout(margin=dict(l=10, r=10, t=10, b=10))
    _layout7.pop('height', None)
    _layout7['autosize'] = True
    fig7.update_layout(**_layout7)
    fig7.write_html(f'{OUT_DIR}/07_sunburst_region_mineral.html', config=WRITE_CONFIG,
                    default_height='100%', default_width='100%')

    # ======================================================================
    # CHART 8: 2x2 PANEL – Export destinations per commodity
    # ======================================================================

    print("Chart 8: Export destinations...")

    COMM_COLORS = {
        'Copper': '#4a6fa5', 'Lithium': '#2e7d4a',
        'Iodine': '#c23a3a', 'Molybdenum': '#d4853b',
    }

    comm_order = ['Copper', 'Lithium', 'Iodine', 'Molybdenum']
    unit_labels = {'kMT': 'kMT', '$M_FOB': '$M FOB', 'MT': 'MT'}

    fig8 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{c} ({unit_labels.get(export_df[export_df["COMMODITIES"]==c]["EXPORT_UNIT"].iloc[0], "")})'
            for c in comm_order
        ],
        vertical_spacing=0.15, horizontal_spacing=0.15,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for comm, (row, col) in zip(comm_order, positions):
        sub = export_df[export_df['COMMODITIES'] == comm]
        by_dest = sub.groupby('TO_NAME')['EXPORT_VALUE'].sum().nlargest(10).sort_values()
        unit = sub['EXPORT_UNIT'].iloc[0]

        fig8.add_trace(go.Bar(
            y=by_dest.index, x=by_dest.values, orientation='h',
            marker_color=COMM_COLORS.get(comm, '#999'), name=comm, showlegend=False,
            hovertemplate='%{y}: %{x:.1f}<extra>' + comm + '</extra>',
        ), row=row, col=col)

        ax_idx = (row - 1) * 2 + col
        xref = 'x' if ax_idx == 1 else f'x{ax_idx}'
        yref = 'y' if ax_idx == 1 else f'y{ax_idx}'

        for country, val in by_dest.items():
            if unit == 'kMT':
                label = f"  {val:.0f}"
            elif unit == '$M_FOB':
                label = f"  ${val:.0f}M"
            else:
                label = f"  {val:,.0f}"
            fig8.add_annotation(
                x=val, y=country, text=label,
                showarrow=False, xanchor='left', font=dict(size=9, color='#555'),
                xref=xref, yref=yref,
            )

    fig8.update_layout(
        **base_layout(height=700, margin=dict(l=120, r=60, t=30, b=40)),
    )
    for i in range(1, 5):
        fig8.update_xaxes(gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
                          row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
    fig8.write_html(f'{OUT_DIR}/08_export_destinations.html', config=WRITE_CONFIG)


    # ======================================================================
    # CHART 9: SCATTER – Copper output vs non-Cu by-product value
    # ======================================================================

    print("Chart 9: Company scatter...")

    cu_mines = inv[inv['COCHILCO_CU_2024_KMT'] > 0].copy()
    cu_mines['cu_kmt'] = cu_mines['COCHILCO_CU_2024_KMT']
    non_cu_cols_list = [c for c in usd_cols_active if c != 'USD_VALUE_CU']
    cu_mines['non_cu_value'] = cu_mines[non_cu_cols_list].sum(axis=1).clip(lower=0)
    cu_mines['region_short'] = cu_mines['REGION'].map(short_region)

    COMPANY_COLORS = {
        'Escondida':               '#c23a3a',
        'Collahuasi':              '#4a6fa5',
        u'Divisi\u00f3n El Teniente':    '#2e7d4a',
        'Los Pelambres':           '#d4853b',
        u'Divisi\u00f3n Chuquicamata':   '#8b5c3c',
        u'Divisi\u00f3n Radomiro Tomic': '#3d4f5f',
    }
    cu_mines['company_label'] = cu_mines['COCHILCO_COMPANY'].apply(
        lambda x: x if x in COMPANY_COLORS else 'Other')

    fig9 = go.Figure()
    for company in list(COMPANY_COLORS.keys()) + ['Other']:
        sub = cu_mines[cu_mines['company_label'] == company]
        if len(sub) == 0:
            continue
        color = COMPANY_COLORS.get(company, '#aaa')
        fig9.add_trace(go.Scatter(
            x=sub['cu_kmt'], y=sub['non_cu_value'] / 1e6,
            mode='markers+text', name=company,
            text=sub['FACILITY_NAME'].str[:15], textposition='top center',
            textfont=dict(size=8, color='#555'),
            marker=dict(
                size=np.sqrt(sub['USD_VALUE_TOTAL'] / 1e6) * 1.8,
                color=color, opacity=0.7,
                line=dict(width=0.5, color='white'),
                sizemode='diameter', sizemin=5,
            ),
            hovertemplate=(
                '<b>%{text}</b><br>Cu: %{x:.0f} kMT<br>'
                'Non-Cu Value: $%{y:.0f}M<extra>' + company + '</extra>'
            ),
        ))

    fig9.update_layout(
        **base_layout(height=520, margin=dict(l=70, r=40, t=10, b=60)),
        xaxis=dict(title='Copper Production (kMT)',
                   gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width']),
        yaxis=dict(title='Non-Copper Production Value ($M)',
                   gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width']),
        legend=dict(yanchor='top', y=0.98, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.85)', font=dict(size=STYLE['legend_size'])),
    )
    fig9.write_html(f'{OUT_DIR}/09_copper_byproduct_scatter.html', config=WRITE_CONFIG)


    # ======================================================================
    # CHART 10: HORIZONTAL BAR – Non-copper mineral values (log scale)
    # ======================================================================

    print("Chart 10: Non-copper mineral bar...")

    non_cu = []
    for col in usd_cols_active:
        if col == 'USD_VALUE_CU':
            continue
        val = inv[col].sum()
        if val > 0:
            mineral, group = MINERAL_GROUPS[col]
            non_cu.append({'mineral': mineral, 'group': group, 'value': val})

    non_cu_df = pd.DataFrame(non_cu).sort_values('value')

    fig10 = go.Figure()
    fig10.add_trace(go.Bar(
        y=non_cu_df['mineral'], x=non_cu_df['value'], orientation='h',
        marker_color=non_cu_df['group'].map(GROUP_COLORS),
        hovertemplate='%{y}: %{x:$.3s}<extra></extra>',
    ))

    for _, row in non_cu_df.iterrows():
        fig10.add_annotation(
            x=row['value'], y=row['mineral'],
            text=f"  {fmt_usd(row['value'])}",
            showarrow=False, xanchor='left', font=dict(size=9, color='#555'),
        )

    fig10.update_layout(
        **base_layout(height=650, margin=dict(l=150, r=80, t=10, b=50)),
        xaxis=dict(title='Estimated Value (USD)',
                   gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'],
                   tickformat='$.2s', type='log'),
        yaxis=dict(title=''),
        showlegend=False,
    )
    fig10.write_html(f'{OUT_DIR}/10_non_copper_value_bar.html', config=WRITE_CONFIG)


    # ======================================================================

    print(f"\nDone. 10 charts written to {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.html'):
            print(f"  {f}")
