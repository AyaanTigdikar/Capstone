"""
maps.py
=======
One choropleth map per regression specification.

Run from project root:
    python3 robustness-forest/maps.py

Output: robustness-forest/outputs/maps/
    map_spec1_original_54.png
    map_spec2_adj_rent_54.png
    map_spec3_3pct_38.png
    map_spec4_1pct_58.png
    map_spec5_all_nonhic_93.png
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
import wbgapi as _wb
import plotly.graph_objects as go

_eco = _wb.economy.DataFrame()
HIC  = set(_eco[_eco['incomeLevel'] == 'HIC'].index)
from viz_utils import FONT, BG, NAVY, GRID, base_layout, save, INCLUDE_LIST

OUT_MAPS = os.path.join(ROOT, 'robustness-forest', 'outputs', 'maps')
os.makedirs(OUT_MAPS, exist_ok=True)

GULF = {'ARE', 'BHR', 'KWT', 'OMN', 'QAT', 'SAU', 'IRQ', 'IRN', 'YEM'}
RENT_ADJ = 'NR rents excl. forest (% of GDP)'

# ── load adjusted master for 1995 threshold checks ────────────────────────────
adj = pd.read_csv(os.path.join(ROOT, 'intermediary', 'Master.csv'),
                  dtype={'Country Code': str})
name_lu = adj[['Country Code', 'Country Name']].drop_duplicates().set_index('Country Code')['Country Name']
d95 = adj[adj['Year'] == 1995].set_index('Country Code')

# ── sample sets ───────────────────────────────────────────────────────────────
S1 = set(INCLUDE_LIST)
S2 = set(INCLUDE_LIST)   # same countries, adjusted rent variable
S3 = set(INCLUDE_LIST)   # same countries, 3% threshold spec
S4 = set(INCLUDE_LIST)   # same countries, 2% threshold spec
S5 = set(INCLUDE_LIST)   # same countries, 1% threshold spec
S6 = set(INCLUDE_LIST)   # same countries, all non-HIC spec

# ── colours ───────────────────────────────────────────────────────────────────
C_IN   = '#4a6fa5'   # blue  — in sample
C_GULF = '#d4853b'   # orange — Gulf states
C_DROP = '#c23a3a'   # red   — was in original 54 but dropped
C_OUT  = '#e0e0e0'   # light grey — not in sample

GEO = dict(
    showframe=False,
    showcoastlines=True, coastlinecolor='#bbb',
    landcolor='#f2f2f2',
    showocean=True, oceancolor='#ddeeff',
    showlakes=False,
    projection_type='natural earth',
)

WRITE_CONFIG = {'displayModeBar': False, 'responsive': True}


def make_map(sample_set, title, subtitle, fname, highlight_dropped=None):
    """
    sample_set       : set of ISO-3 codes IN this spec
    title            : main title string
    subtitle         : smaller subtitle (shown as <sup>)
    fname            : output filename (no extension)
    highlight_dropped: set of codes to show in red (were in 54, now dropped)
    """
    # classify every country we know about
    all_codes = set(d95.index) | S1
    rows = []
    for cc in all_codes:
        gulf = cc in GULF
        dropped = highlight_dropped and cc in highlight_dropped
        in_samp = cc in sample_set

        if gulf and in_samp:
            cat, col = 'Gulf state (in sample)', C_GULF
        elif in_samp:
            cat, col = 'In sample', C_IN
        elif dropped:
            cat, col = 'Dropped (forest-driven)', C_DROP
        else:
            continue   # don't plot — stays as land colour

        rows.append({'cc': cc, 'name': name_lu.get(cc, cc),
                     'cat': cat, 'col': col})

    df = pd.DataFrame(rows)
    fig = go.Figure()

    for cat, col in [('In sample', C_IN),
                     ('Gulf state (in sample)', C_GULF),
                     ('Dropped (forest-driven)', C_DROP)]:
        sub = df[df['cat'] == cat]
        if sub.empty:
            continue
        fig.add_trace(go.Choropleth(
            locations=sub['cc'],
            z=[1] * len(sub),
            locationmode='ISO-3',
            colorscale=[[0, col], [1, col]],
            showscale=False,
            name=cat,
            text=sub['name'],
            hovertemplate='<b>%{text}</b><br>' + cat + '<extra></extra>',
            marker_line_color='white',
            marker_line_width=0.6,
        ))

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor=BG,
        font=dict(family=FONT, size=11, color=NAVY),
        geo=GEO,
        height=440,
        margin=dict(l=0, r=0, t=80, b=10),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01,
            xanchor='center', x=0.5, font=dict(size=11),
            bgcolor='rgba(255,255,255,0.85)',
        ),
        title=dict(
            text=f'{title}<br><sup>{subtitle}</sup>',
            font=dict(size=14), x=0.5, y=0.97, yanchor='top',
        ),
    )

    path = os.path.join(OUT_MAPS, fname)
    fig.write_html(f'{path}.html', config=WRITE_CONFIG)
    print(f'  Saved: {path}.html')
    try:
        fig.write_image(f'{path}.png', width=1100, height=440, scale=3)
        print(f'  Saved: {path}.png')
    except Exception as e:
        print(f'  PNG skipped ({e})')


# ══════════════════════════════════════════════════════════════════════════════
# MAP 1 — Original 54 countries (total rents, baseline)
# ══════════════════════════════════════════════════════════════════════════════
make_map(
    sample_set=S1,
    title='Spec 1 — Original Sample (54 Countries)',
    subtitle='Total NR rents ≥ 5% of GDP in 1995 (incl. forest rents) · Baseline specification',
    fname='map_spec1_original_54',
)
print('✓ Map 1')

# ══════════════════════════════════════════════════════════════════════════════
# MAP 2 — Same 54 countries, adjusted rent variable
# ══════════════════════════════════════════════════════════════════════════════
make_map(
    sample_set=S2,
    title='Spec 2 — Same 54 Countries, Adjusted Rent Variable',
    subtitle='Same sample as Spec 1 · NR rents variable excludes forest rents',
    fname='map_spec2_adj_rent_54',
)
print('✓ Map 2')

# ══════════════════════════════════════════════════════════════════════════════
# MAP 3 — Re-selected countries (≥3% adj rents or Gulf)
# ══════════════════════════════════════════════════════════════════════════════
make_map(
    sample_set=S3,
    title=f'Spec 3 — Same 54 Countries, ≥3% Adj. Rent Threshold',
    subtitle='Same sample as baseline · Sensitivity: adj. NR rents variable, ≥3% threshold',
    fname='map_spec3_3pct',
)
print('✓ Map 3')

# ══════════════════════════════════════════════════════════════════════════════
# MAP 4 — Same 54 countries, ≥2% adj rents spec
# ══════════════════════════════════════════════════════════════════════════════
make_map(
    sample_set=S4,
    title=f'Spec 4 — Same 54 Countries, ≥2% Adj. Rent Threshold',
    subtitle='Same sample as baseline · Sensitivity: adj. NR rents variable, ≥2% threshold',
    fname='map_spec4_2pct',
)
print('✓ Map 4')

# ══════════════════════════════════════════════════════════════════════════════
# MAP 5 — Same 54 countries, ≥1% adj rents spec
# ══════════════════════════════════════════════════════════════════════════════
make_map(
    sample_set=S5,
    title=f'Spec 5 — Same 54 Countries, ≥1% Adj. Rent Threshold',
    subtitle='Same sample as baseline · Sensitivity: adj. NR rents variable, ≥1% threshold',
    fname='map_spec5_1pct',
)
print('✓ Map 5')

# ══════════════════════════════════════════════════════════════════════════════
# MAP 6 — Same 54 countries, all non-HIC spec
# ══════════════════════════════════════════════════════════════════════════════
make_map(
    sample_set=S6,
    title=f'Spec 6 — Same 54 Countries, All Non-HIC Specification',
    subtitle='Same sample as baseline · Sensitivity: broader non-HIC panel specification',
    fname='map_spec6_all_nonhic',
)
print('✓ Map 6')

print(f'\n✓ maps.py complete — 6 maps in robustness-forest/outputs/maps/')
