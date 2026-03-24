"""
viz_utils.py — Single source of truth for all Capstone visualization scripts.

Import at the top of every viz_*.py:
    from viz_utils import (
        PALETTE, CLUSTER_COLORS, CLUSTER_LABELS, WRITE_CONFIG,
        base_layout, save,
        load_master, load_clusters, load_nr, load_nb5,
        INCLUDE_LIST, resource_rich_codes,
        analyze_variable_missingness, analyze_country_missingness,
        build_sample,
    )

WHY THIS FILE EXISTS
--------------------
The five original viz scripts had three inconsistencies that caused different
results for the same chart:

1. Cluster labels hardcoded differently per file (old names, now unified):
     viz_improved_1_7.py  → {0: 'No Oil, No Minerals',  1: 'Some Oil, No Minerals', ...}
     viz_charts_14_24.py  → {0: 'Some Oil, No Minerals', 1: 'No Oil, No Minerals', ...}
   Clusters 0 and 1 were swapped, and cluster 3 was named differently.
   Labels renamed in 2026-03 to: Petrostates / Oil Exporters / Major Producers / Gold & Coal.

2. viz_improved_1_7.py re-ran KMeans from scratch with its own centroid
   re-ordering, which can produce different cluster assignments than the
   clusters1995.csv produced by NB4.

3. viz_diagnostic.py built the sample using NR rents > 5% from
   master_data_wide.csv (NB2 logic), while all other files used the
   hardcoded 54-country include_list from NB4. Different countries appeared
   in diagnostic charts vs analysis charts.

SOLUTION
--------
- Cluster labels are derived at runtime from clusters1995.csv (which NB4 wrote),
  not hardcoded. CLUSTER_LABELS and CLUSTER_COLORS are built once here.
- All files use the same INCLUDE_LIST (the 54-country NB4 sample).
- All data loaders are centralised here so every script reads from the same file
  with the same filters.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths  (all relative to the project root: FINAL CODE RECAP/)
# ---------------------------------------------------------------------------
_INTR  = 'intermediary'
_NB5   = 'Final/NB5'
_BOOT  = 'intermediary/bootstrap'


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
FONT         = 'IBM Plex Sans, -apple-system, BlinkMacSystemFont, sans-serif'
BG           = '#ffffff'
NAVY         = '#1a2744'
GRID         = '#e5e7eb'
WRITE_CONFIG = {'displayModeBar': False, 'responsive': True}

PALETTE = dict(
    blue        = '#4a6fa5',
    red         = '#c23a3a',
    green       = '#2e7d4a',
    orange      = '#d4853b',
    grey        = '#999999',
    light_blue  = '#7a9dc4',
    light_red   = '#d46b6b',
    light_green = '#5aa87a',
    purple      = '#7a5c9e',
    teal        = '#3a8fa5',
    gold        = '#d4a017',
    # model aliases
    lasso       = '#c23a3a',
    ridge       = '#4a6fa5',
    en          = '#2e7d4a',
    rf          = '#d4853b',
)

# Fixed colour sequence for the 4 clusters (index → hex).
# Colors are keyed by label name (label-stable, not cluster-ID-stable).
_LABEL_COLORS = {
    'Petrostates':           '#d4853b',   # orange
    'Oil Exporters':         '#4a6fa5',   # blue
    'Major Producers':       '#2e7d4a',   # green  (k=4 default)
    'Forestry Intensive':    '#c23a3a',   # red
    # k=5
    'Diversified Producers': '#2e7d4a',   # green  (replaces Major Producers in k=5/6)
    'Mining Exporters':      '#7a5c9e',   # purple
    # k=6
    'Oil & Minerals':        '#3a8fa5',   # teal
}


# ---------------------------------------------------------------------------
# 54-country analysis sample (NB4 include_list — single source of truth)
# ---------------------------------------------------------------------------
INCLUDE_LIST = [
    'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
    'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
    'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
    'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
    'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
    'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
]

# Territories dropped during NB1 cleaning (not countries)
_NOT_COUNTRIES = [
    'HKG', 'MAC', 'PRI', 'VIR', 'GUM', 'ASM', 'CYM', 'BMU',
    'GRL', 'MAF', 'SXM', 'CUW', 'ABW', 'FRO', 'MNP', 'PYF',
]


# ---------------------------------------------------------------------------
# Cluster label + colour maps — derived from clusters1995.csv at import time
# ---------------------------------------------------------------------------
def _build_cluster_maps(csv_path: str):
    """
    Read clusters1995.csv (produced by NB4) and derive:
      CLUSTER_LABELS  : dict  cluster_id (int) → label string
      CLUSTER_COLORS  : dict  cluster_id (int) → hex colour
      LABEL_TO_COLOR  : dict  label string     → hex colour

    This is the only place cluster labels are defined, so all scripts that
    import from here agree on which cluster is which.

    Falls back to NB4's rank-based defaults if the CSV is absent.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Build mapping from the actual data
        mapping = (
            df[['Cluster', 'ClusterLabels']]
            .drop_duplicates()
            .sort_values('Cluster')
        )
        labels  = dict(zip(mapping['Cluster'].astype(int),
                           mapping['ClusterLabels']))
        l2c     = {lbl: _LABEL_COLORS.get(lbl, '#aaa') for lbl in labels.values()}
        colors  = {k: l2c[v] for k, v in labels.items()}
        return labels, colors, l2c

    # Fallback: NB4 rank-based defaults
    fallback_labels = {
        0: 'Oil Exporters',
        1: 'Forestry Intensive',
        2: 'Major Producers',
        3: 'Petrostates',
    }
    fallback_l2c    = {lbl: _LABEL_COLORS.get(lbl, '#aaa') for lbl in fallback_labels.values()}
    fallback_colors = {k: fallback_l2c[v] for k, v in fallback_labels.items()}
    return fallback_labels, fallback_colors, fallback_l2c


_clusters_csv = os.path.join(_INTR, 'clusters1995.csv')
CLUSTER_LABELS, CLUSTER_COLORS, LABEL_TO_COLOR = _build_cluster_maps(_clusters_csv)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def base_layout(**kw) -> dict:
    """Return a Plotly layout dict with shared style defaults applied."""
    d = dict(
        template    = 'plotly_white',
        plot_bgcolor  = BG,
        paper_bgcolor = BG,
        font          = dict(family=FONT, size=11, color=NAVY),
        margin        = dict(l=60, r=40, t=40, b=50),
    )
    d.update(kw)
    return d


def save(fig: go.Figure, name: str, out_dir: str, w: int = 1100, h: int = 600) -> None:
    """Write a Plotly figure to <out_dir>/<n>.png."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    try:
        fig.write_image(f"{path}.png", width=w, height=h, scale=2)
        print(f"  Saved: {path}.png")
    except Exception as e:
        print(f"  PNG failed ({e})")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_master() -> pd.DataFrame:
    """
    Load intermediary/Master.csv — the fully imputed, population-merged panel.
    Adds prod_pc = Total_Production_Value / Population for convenience.
    """
    path = os.path.join(_INTR, 'Master.csv')
    df   = pd.read_csv(path)
    df['prod_pc'] = df['Total_Production_Value'] / df['Population'].replace(0, np.nan)
    return df


def load_master_wide() -> pd.DataFrame:
    """
    Load intermediary/master_data_wide.csv — pre-imputation wide panel (NB1 output).
    Used only by diagnostic / missingness charts.
    """
    path = os.path.join(_INTR, 'master_data_wide.csv')
    df   = pd.read_csv(path)
    df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], inplace=True)
    return df


def load_clusters(year: str = '1995') -> pd.DataFrame:
    """
    Load intermediary/clusters<year>.csv (e.g. clusters1995.csv, clusters2019.csv).
    Cluster column is cast to int to match CLUSTER_LABELS keys.
    """
    path = os.path.join(_INTR, f'clusters{year}.csv')
    df   = pd.read_csv(path)
    df['Cluster'] = df['Cluster'].astype(int)
    # Re-attach authoritative labels from CLUSTER_LABELS (overrides any stale CSV values)
    df['ClusterLabels'] = df['Cluster'].map(CLUSTER_LABELS)
    return df


def load_nr() -> pd.DataFrame:
    """Load intermediary/NaturalResource.csv."""
    return pd.read_csv(os.path.join(_INTR, 'NaturalResource.csv'))


def load_nb5(filename: str) -> pd.DataFrame:
    """Load a CSV from Final/NB5/ (model outputs)."""
    return pd.read_csv(os.path.join(_NB5, filename))


def load_bootstrap(filename: str) -> pd.DataFrame:
    """Load a CSV from intermediary/bootstrap/."""
    return pd.read_csv(os.path.join(_BOOT, filename))


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------
def resource_rich_codes() -> list:
    """Return ISO3 codes in the analysis sample (from clusters1995.csv if available,
    else falls back to INCLUDE_LIST)."""
    path = os.path.join(_INTR, 'clusters1995.csv')
    if os.path.exists(path):
        return pd.read_csv(path)['Country Code'].unique().tolist()
    return list(INCLUDE_LIST)


def build_sample(df: pd.DataFrame, use_include_list: bool = True) -> pd.DataFrame:
    """
    Filter a DataFrame to the analysis sample.

    Parameters
    ----------
    df               : any DataFrame with a 'Country Code' column
    use_include_list : if True (default), filter to INCLUDE_LIST — the 54-country
                       NB4 sample.  If False, use the NR rents > 5% + not_countries
                       exclusion logic from NB2 (for diagnostic charts only).

    Notes
    -----
    Always use use_include_list=True for analysis charts (clustering, ML, regressions).
    use_include_list=False is only appropriate for viz_diagnostic.py, where the
    pre-imputation master_data_wide.csv is the input.
    """
    if use_include_list:
        return df[df['Country Code'].isin(INCLUDE_LIST)].copy()

    # NB2-style: NR rents > 5% in 1995, exclude territories
    rents_col = 'Total natural resources rents (% of GDP)'
    if rents_col not in df.columns:
        raise ValueError(f"Column '{rents_col}' not found — required for NB2-style sample.")

    qualifying = (
        df[df['Year'] == 1995]
        .dropna(subset=[rents_col])
        .loc[lambda x: x[rents_col] > 5, 'Country Code']
        .unique()
    )
    return df[
        df['Country Code'].isin(qualifying) &
        ~df['Country Code'].isin(_NOT_COUNTRIES)
    ].copy()


# ---------------------------------------------------------------------------
# Missingness utilities (shared by viz_diagnostic.py — eliminates copy-paste)
# ---------------------------------------------------------------------------
def analyze_variable_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-variable missingness summary for a panel DataFrame.
    Skips Country Code, Country Name, Year columns.

    Returns a DataFrame sorted by % Missing descending with columns:
    Variable, Valid Obs, Missing, % Missing, Countries, Years, Year Range
    """
    id_cols  = {'Country Code', 'Country Name', 'Year'}
    data_cols = [c for c in df.columns if c not in id_cols]
    rows = []
    for col in data_cols:
        valid = df[col].notna()
        n_obs = valid.sum()
        n_miss = (~valid).sum()
        pct   = 100 * n_miss / len(df)
        yr    = df.loc[valid, 'Year'].dropna() if 'Year' in df.columns else pd.Series(dtype=float)
        rows.append({
            'Variable':   col,
            'Valid Obs':  int(n_obs),
            'Missing':    int(n_miss),
            '% Missing':  round(pct, 1),
            'Countries':  int(df.loc[valid, 'Country Code'].nunique()) if 'Country Code' in df.columns else None,
            'Years':      int(yr.nunique()) if len(yr) else 0,
            'Year Range': f"{int(yr.min())}–{int(yr.max())}" if len(yr) else 'N/A',
        })
    return pd.DataFrame(rows).sort_values('% Missing', ascending=False).reset_index(drop=True)


def analyze_country_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-country missingness summary for a panel DataFrame.

    Returns a DataFrame sorted by % Missing descending with columns:
    Code, Country, Rows, Years Covered, % Missing, Complete Vars,
    Vars with Data, Total Vars
    """
    id_cols   = {'Country Code', 'Country Name', 'Year'}
    data_cols = [c for c in df.columns if c not in id_cols]
    n_vars    = len(data_cols)
    rows = []
    for code, grp in df.groupby('Country Code'):
        total     = grp[data_cols].size
        missing   = grp[data_cols].isna().sum().sum()
        pct       = 100 * missing / total if total else 0
        name_vals = grp['Country Name'].dropna()
        rows.append({
            'Code':          code,
            'Country':       name_vals.iloc[0] if len(name_vals) else code,
            'Rows':          len(grp),
            'Years Covered': grp['Year'].nunique() if 'Year' in grp.columns else 0,
            '% Missing':     round(pct, 1),
            'Complete Vars': int(sum(grp[c].notna().all() for c in data_cols)),
            'Vars with Data':int(sum(grp[c].notna().any() for c in data_cols)),
            'Total Vars':    n_vars,
        })
    return pd.DataFrame(rows).sort_values('% Missing', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature label shortener (used in ML charts 8–10, 19–20)
# ---------------------------------------------------------------------------
_FEAT_SHORT = {
    'Domestic credit to private sector (% of GDP)':                        'Domestic Credit',
    'Access to electricity (% of population)':                             'Electricity Access',
    'Human capital index':                                                  'Human Capital',
    'HCI_x_ProductionValue':                                               'HC x Production',
    'GFCF_x_ProductionValue':                                              'GFCF x Production',
    'Rule of law index':                                                    'Rule of Law',
    'Political stability — estimate':                                  'Political Stability',
    'Trade (% of GDP)':                                                     'Trade',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'Capital Formation',
    'Urban population (% of total population)':                           'Urban Population',
    'Use of IMF credit (DOD, current US$)':                               'IMF Credit',
    'Total_Production_Value_Per_Capita':                                   'Prod Value p.c.',
    'Capital depreciation rate':                                           'Depreciation',
    'Landlocked':                                                          'Landlocked',
    'Real interest rate (%)':                                              'Interest Rate',
    'Inflation, consumer prices (annual %)':                               'Inflation',
    'GDP per capita (constant prices, PPP)':                               'GDP per Capita',
    'Total natural resources rents (% of GDP)':                           'NR Rents (% GDP)',
    'Oil rents (% of GDP)':                                                'Oil Rents',
    'Mineral rents (% of GDP)':                                            'Mineral Rents',
    'Natural gas rents (% of GDP)':                                        'Gas Rents',
    'Forest rents (% of GDP)':                                             'Forest Rents',
    'Economic Complexity Index':                                           'ECI',
    'Adjusted savings: gross savings (% of GNI)':                        'Savings',
    'Government revenue':                                                   'Gov Revenue',
    'Share of investment in GDP':                                          'Investment Share',
    'Life expectancy at birth, total (years)':                            'Life Expectancy',
    'Death rate, crude (per 1,000 people)':                               'Death Rate',
    'Manufacturing, value added (% of GDP)':                              'Manufacturing',
    'Agriculture, forestry, and fishing, value added (% of GDP)':        'Agriculture (% GDP)',
    'Mobile cellular subscriptions (per 100 people)':                     'Mobile Subs',
    'Political corruption index':                                          'Pol. Corruption',
    'Property rights':                                                     'Property Rights',
    'Services, value added (% of GDP)':                                   'Services (% GDP)',
    'Industry (including construction), value added (% of GDP)':         'Industry (% GDP)',
    'prod_pc':                                                             'Prod Value p.c.',
    'Lending interest rate (%)':                                           'Lending Rate',
    'L1_ECI':                                                              'Lagged ECI',
    'Inflation_roll5':                                                     'Inflation (5yr avg)',
    'RealRate_roll5':                                                      'Real Rate (5yr avg)',
    'Resource_HHI':                                                        'Resource HHI',
    'RuleOfLaw_x_ProductionValue':                                         'RuleLaw x Production',
}


def shorten_feat(name: str, max_len: int = 34) -> str:
    """Return a short display label for a feature name."""
    return _FEAT_SHORT.get(name, name[:max_len])
