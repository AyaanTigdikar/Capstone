#!/usr/bin/env python3
"""
run_loco.py
===========
Leave-One-Country-Out (LOCO) analysis for the Moody's capstone.
Location:  scripts/run_loco.py
Run from:  /Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP

WHAT IT DOES
------------
For each of the 54 countries in the sample, the script removes all
observations belonging to that country and refits the Elastic Net model
using the same specification as in NB5 (same features, same temporal CV,
same scaler-on-train-only approach). It records:

  1. Elastic Net coefficients for every feature in every leave-one-out run
  2. Test-set R² for every leave-one-out run

From these it produces:
  - Coefficient stability chart  (Figure 13 in Appendix F)
  - R² stability chart           (Figure 14 in Appendix F)
  - CSV outputs in intermediary/loco/

PREREQUISITES
-------------
intermediary/Master.csv must exist (output of NB3).

USAGE
-----
    cd "/Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP"
    python scripts/run_loco.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_END = 2014
TEST_START = 2015
TOP_K = 12            # number of features shown in the coefficient chart

LOCO_DIR = "intermediary/loco"
FIG_DIR = os.path.join("Final", "NB5")
os.makedirs(LOCO_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

INCLUDE = [
    'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
    'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
    'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
    'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
    'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
    'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE LISTS (must match NB5 cell 4 exactly)
# ══════════════════════════════════════════════════════════════════════════════

base_features = [
    'Total_Production_Value_Per_Capita',
    'Human capital index',
    'Rule of law index',
    'Political stability — estimate',
    'Trade (% of GDP)',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP',
    'Share of investment in GDP',
    'Domestic credit to private sector (% of GDP)',
    'Landlocked',
    'Urban population (% of total population)',
    'Government revenue',
    'Capital depreciation rate',
    'Use of IMF credit (DOD, current US$)',
    'Real interest rate (%)',
    'Inflation, consumer prices (annual %)',
    'Access to electricity (% of population)',
    'Adjusted savings: gross savings (% of GNI)',
    'L1_ECI',
    'Forestry rents (% of GDP)',
]
interaction_features = ['HCI_x_ProductionValue', 'RuleOfLaw_x_ProductionValue']
all_features = base_features + interaction_features

NAME_MAP = {
    'Human capital index':                                                 'Human Capital',
    'Rule of law index':                                                   'Rule of Law',
    'Political stability — estimate':                                      'Political Stability',
    'Trade (% of GDP)':                                                    'Trade',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'Capital Formation',
    'Share of investment in GDP':                                           'Investment Share',
    'Domestic credit to private sector (% of GDP)':                        'Domestic Credit',
    'Landlocked':                                                          'Landlocked',
    'Urban population (% of total population)':                            'Urban Population',
    'Government revenue':                                                  'Gov Revenue',
    'Capital depreciation rate':                                           'Depreciation',
    'Use of IMF credit (DOD, current US$)':                                'IMF Credit',
    'Real interest rate (%)':                                              'Interest Rate',
    'Inflation, consumer prices (annual %)':                               'Inflation',
    'Access to electricity (% of population)':                             'Electricity',
    'Adjusted savings: gross savings (% of GNI)':                          'Savings',
    'Total_Production_Value_Per_Capita':                                    'Prod Value p.c.',
    'HCI_x_ProductionValue':                                               'HC x Production',
    'RuleOfLaw_x_ProductionValue':                                         'RuleLaw x Production',
    'L1_ECI':                                                              'Lagged ECI',
    'Forestry rents (% of GDP)':                                           'Forestry Rents',
}

def shorten(f):
    return NAME_MAP.get(f, f[:22])


# ══════════════════════════════════════════════════════════════════════════════
# PANEL TEMPORAL CV (copied from NB5 cell 6)
# ══════════════════════════════════════════════════════════════════════════════

class PanelTemporalCV:
    def __init__(self, years, n_splits=5, gap=1, min_train_years=8):
        self.years = np.asarray(years)
        self.n_splits = n_splits
        self.gap = gap
        unique_years = np.sort(np.unique(self.years))
        earliest_cutoff = unique_years[0] + min_train_years - 1
        latest_cutoff = unique_years[-1] - gap - 1
        if earliest_cutoff > latest_cutoff:
            raise ValueError("Year range too narrow for given parameters.")
        self.cutoffs = np.unique(
            np.linspace(earliest_cutoff, latest_cutoff, n_splits).astype(int)
        )
        self.n_splits = len(self.cutoffs)

    def split(self, X=None, y=None, groups=None):
        for cutoff in self.cutoffs:
            train_idx = np.where(self.years <= cutoff)[0]
            val_idx = np.where(self.years > cutoff + self.gap)[0]
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION (replicates NB5 cells 2 + 4)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_full_df():
    """Load Master.csv and apply all NB5 feature engineering."""
    master = pd.read_csv("intermediary/Master.csv")
    df = master[
        (master['Year'] >= 1995)
        & (master['Year'] <= 2019)
        & (master['Country Code'].isin(INCLUDE))
    ].copy()
    df = df.sort_values(['Country Code', 'Year']).reset_index(drop=True)

    # Per-capita production value
    df['Total_Production_Value_Per_Capita'] = (
        df['Total_Production_Value'] / df['Population']
    )

    # ECI targets
    df['L1_ECI'] = df.groupby('Country Code')['Economic Complexity Index'].shift(1)
    df['ECI_delta'] = df['Economic Complexity Index'] - df['L1_ECI']
    df = df.dropna(subset=['L1_ECI', 'Economic Complexity Index', 'ECI_delta'])

    # Log transforms (before interactions)
    log_cols = [
        'Human capital index',
        'Total_Production_Value_Per_Capita',
        'Gross fixed capital formation, all, Constant prices, Percent of GDP',
        'Government revenue',
        'Use of IMF credit (DOD, current US$)',
        'Forestry rents (% of GDP)',
    ]
    df[log_cols] = np.log1p(df[log_cols]).replace([np.inf, -np.inf], np.nan)

    # Interaction terms (grand-mean centred)
    hci_mean = df['Human capital index'].mean()
    prod_mean = df['Total_Production_Value_Per_Capita'].mean()
    rol_mean = df['Rule of law index'].mean()

    df['HCI_x_ProductionValue'] = (
        (df['Human capital index'] - hci_mean)
        * (df['Total_Production_Value_Per_Capita'] - prod_mean)
    )
    df['RuleOfLaw_x_ProductionValue'] = (
        (df['Rule of law index'] - rol_mean)
        * (df['Total_Production_Value_Per_Capita'] - prod_mean)
    )

    df = df.dropna(subset=all_features)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LOCO LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_loco(df):
    """
    For each country, exclude it and refit Elastic Net on the remaining
    sample. Record scaled coefficients and test-set R-squared.
    """
    countries = sorted(df['Country Code'].unique())
    N = len(countries)

    # Also fit the full-sample model for the baseline R²
    full_train = df[df['Year'] <= TRAIN_END]
    full_test = df[df['Year'] >= TEST_START]
    scaler_full = StandardScaler()
    X_train_full = scaler_full.fit_transform(full_train[all_features].values)
    X_test_full = scaler_full.transform(full_test[all_features].values)
    y_train_full = full_train['Economic Complexity Index'].values
    y_test_full = full_test['Economic Complexity Index'].values
    tscv_full = PanelTemporalCV(full_train['Year'].values, n_splits=5, gap=1, min_train_years=8)

    en_full = ElasticNetCV(
        l1_ratio=[0.5], cv=tscv_full, random_state=42, max_iter=10000
    ).fit(X_train_full, y_train_full)
    full_r2 = r2_score(y_test_full, en_full.predict(X_test_full))
    full_coefs = dict(zip(all_features, en_full.coef_))

    print(f"  Full-sample Elastic Net: test R² = {full_r2:.4f}, "
          f"alpha = {en_full.alpha_:.4f}")

    # LOCO loop
    coef_rows = []
    r2_rows = []
    t0 = time.time()

    for i, cc in enumerate(countries, 1):
        df_loo = df[df['Country Code'] != cc].copy()

        train_loo = df_loo[df_loo['Year'] <= TRAIN_END]
        test_loo = df_loo[df_loo['Year'] >= TEST_START]

        if len(train_loo) < 30 or len(test_loo) < 10:
            print(f"  WARNING: skipping {cc} (train={len(train_loo)}, test={len(test_loo)})")
            continue

        scaler_loo = StandardScaler()
        X_tr = scaler_loo.fit_transform(train_loo[all_features].values)
        X_te = scaler_loo.transform(test_loo[all_features].values)
        y_tr = train_loo['Economic Complexity Index'].values
        y_te = test_loo['Economic Complexity Index'].values

        tscv_loo = PanelTemporalCV(
            train_loo['Year'].values, n_splits=5, gap=1, min_train_years=8
        )

        en_loo = ElasticNetCV(
            l1_ratio=[0.5], cv=tscv_loo, random_state=42, max_iter=10000
        ).fit(X_tr, y_tr)

        pred = en_loo.predict(X_te)
        test_r2 = r2_score(y_te, pred)

        row = {'excluded_country': cc}
        for feat, c in zip(all_features, en_loo.coef_):
            row[feat] = c
        coef_rows.append(row)

        r2_rows.append({
            'excluded_country': cc,
            'test_r2': test_r2,
            'n_train': len(train_loo),
            'n_test': len(test_loo),
            'n_countries': df_loo['Country Code'].nunique(),
            'alpha': en_loo.alpha_,
        })

        if i % 10 == 0 or i == 1:
            elapsed = time.time() - t0
            eta = (N - i) / max(i, 1) * elapsed
            print(f"  {i:>2d}/{N}  excluded={cc}  R²={test_r2:.4f}  "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    coef_df = pd.DataFrame(coef_rows)
    r2_df = pd.DataFrame(r2_rows)

    coef_df.to_csv(os.path.join(LOCO_DIR, "loco_coefs.csv"), index=False)
    r2_df.to_csv(os.path.join(LOCO_DIR, "loco_r2.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n  {len(coef_rows)} LOCO runs completed in {elapsed:.1f}s")
    print(f"  Saved: {LOCO_DIR}/loco_coefs.csv, {LOCO_DIR}/loco_r2.csv")

    return coef_df, r2_df, full_coefs, full_r2


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_coefficient_stability(coef_df, full_coefs, full_r2):
    """
    Figure 13: horizontal bar chart of mean LOCO coefficient for the top-K
    features (by full-sample absolute coefficient), with min-max whiskers.
    """
    feat_cols = [c for c in coef_df.columns if c != 'excluded_country']

    # Rank features by full-sample absolute coefficient, exclude Lagged ECI
    ranked = sorted(
        [(f, abs(full_coefs.get(f, 0))) for f in feat_cols if f != 'L1_ECI'],
        key=lambda x: x[1], reverse=True,
    )
    top_feats = [f for f, _ in ranked[:TOP_K]]

    means = coef_df[top_feats].mean()
    mins = coef_df[top_feats].min()
    maxs = coef_df[top_feats].max()
    short = [shorten(f) for f in top_feats]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y_pos = np.arange(len(top_feats))

    # Whiskers (min-max range)
    for j, feat in enumerate(top_feats):
        ax.plot(
            [mins[feat], maxs[feat]], [j, j],
            color='#4a6fa5', linewidth=1.5, solid_capstyle='round',
        )

    # Mean dots
    colors = ['#c23a3a' if means[feat] < 0 else '#4a6fa5' for feat in top_feats]
    ax.scatter(means.values, y_pos, c=colors, s=50, zorder=5, edgecolors='white', linewidths=0.5)

    # Full-sample coefficient as reference markers
    for j, feat in enumerate(top_feats):
        ax.scatter(
            full_coefs.get(feat, 0), j,
            marker='|', c='#333333', s=80, zorder=6, linewidths=1.2,
        )

    ax.axvline(0, color='#c9cfd6', linewidth=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Elastic Net coefficient (standardised)', fontsize=11)
    ax.tick_params(axis='x', labelsize=10)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4a6fa5',
               markersize=7, label='LOCO mean'),
        Line2D([0], [0], marker='|', color='#333333', markersize=8,
               markeredgewidth=1.2, linestyle='None', label='Full-sample'),
        Line2D([0], [0], color='#4a6fa5', linewidth=1.5, label='Min-max range'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              frameon=True, framealpha=0.9, edgecolor='#e5e7eb')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig13_loco_coef_stability")
    fig.savefig(f"{path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{path}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}.png / .pdf")


def plot_r2_stability(r2_df, full_r2):
    """
    Figure 14: bar chart of test R² for each LOCO run, with full-sample
    baseline shown as a horizontal line.
    """
    r2_sorted = r2_df.sort_values('test_r2', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(r2_sorted))

    # Colour bars by distance from full-sample R²
    bar_colors = []
    for _, row in r2_sorted.iterrows():
        diff = full_r2 - row['test_r2']
        if diff > 0.02:
            bar_colors.append('#c23a3a')      # notable drop
        elif diff < -0.005:
            bar_colors.append('#2e7d4a')      # improvement when excluded
        else:
            bar_colors.append('#4a6fa5')      # stable

    ax.bar(x_pos, r2_sorted['test_r2'], color=bar_colors, width=0.7, edgecolor='white', linewidth=0.3)
    ax.axhline(full_r2, color='#333333', linewidth=1.2, linestyle='--', label=f'Full sample (R² = {full_r2:.3f})')

    # Set y-axis to start from a sensible floor
    y_min = max(0, r2_sorted['test_r2'].min() - 0.03)
    ax.set_ylim(y_min, min(1.0, r2_sorted['test_r2'].max() + 0.02))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(r2_sorted['excluded_country'], rotation=90, fontsize=7)
    ax.set_ylabel('Test R² (Elastic Net)', fontsize=11)
    ax.set_xlabel('Excluded country', fontsize=11)
    ax.legend(fontsize=9, loc='lower left', frameon=True, framealpha=0.9, edgecolor='#e5e7eb')
    ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig14_loco_r2_stability")
    fig.savefig(f"{path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{path}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(coef_df, r2_df, full_coefs, full_r2):
    print(f"\n{'=' * 70}")
    print("  LOCO SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  Full-sample test R²: {full_r2:.4f}")
    print(f"  LOCO R² range:       [{r2_df['test_r2'].min():.4f}, {r2_df['test_r2'].max():.4f}]")
    print(f"  LOCO R² mean:        {r2_df['test_r2'].mean():.4f}")
    print(f"  LOCO R² std:         {r2_df['test_r2'].std():.4f}")

    # Countries whose exclusion changes R² most
    r2_df_s = r2_df.copy()
    r2_df_s['delta_r2'] = r2_df_s['test_r2'] - full_r2
    biggest_drop = r2_df_s.sort_values('delta_r2').head(5)
    biggest_gain = r2_df_s.sort_values('delta_r2', ascending=False).head(5)

    print("\n  Largest R² drops when excluded (influential for fit):")
    for _, row in biggest_drop.iterrows():
        print(f"    {row['excluded_country']}  R²={row['test_r2']:.4f}  "
              f"(delta={row['delta_r2']:+.4f})")

    print("\n  Largest R² gains when excluded:")
    for _, row in biggest_gain.iterrows():
        print(f"    {row['excluded_country']}  R²={row['test_r2']:.4f}  "
              f"(delta={row['delta_r2']:+.4f})")

    # Coefficient stability for top features
    feat_cols = [c for c in coef_df.columns if c != 'excluded_country']
    ranked = sorted(
        [(f, abs(full_coefs.get(f, 0))) for f in feat_cols if f != 'L1_ECI'],
        key=lambda x: x[1], reverse=True,
    )
    top_feats = [f for f, _ in ranked[:TOP_K]]

    print(f"\n  Coefficient stability (top {TOP_K}, excl. Lagged ECI):")
    print(f"  {'Feature':<30} {'Full':>8} {'Mean':>8} {'Min':>8} {'Max':>8} {'SD':>8}")
    print("  " + "-" * 74)
    for feat in top_feats:
        vals = coef_df[feat]
        fc = full_coefs.get(feat, np.nan)
        print(f"  {shorten(feat):<30} {fc:>+8.4f} {vals.mean():>+8.4f} "
              f"{vals.min():>+8.4f} {vals.max():>+8.4f} {vals.std():>8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Ensure working directory is FINAL CODE RECAP, not scripts/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir).lower() == "scripts":
        os.chdir(os.path.dirname(script_dir))

    print("=" * 70)
    print("  LOCO ANALYSIS — Elastic Net Stability")
    print(f"  Working dir: {os.getcwd()}")
    print("=" * 70)

    t_total = time.time()

    print("\nLoading and preparing data...")
    df = prepare_full_df()
    print(f"  {df['Country Code'].nunique()} countries, {len(df):,} obs")
    print(f"  Train: {len(df[df['Year'] <= TRAIN_END]):,} | "
          f"Test: {len(df[df['Year'] >= TEST_START]):,}")

    print("\nRunning LOCO analysis...")
    coef_df, r2_df, full_coefs, full_r2 = run_loco(df)

    # CSVs are already saved inside run_loco().
    # Plots are wrapped so a matplotlib issue doesn't lose the data.
    print("\nGenerating figures...")
    try:
        plot_coefficient_stability(coef_df, full_coefs, full_r2)
    except Exception as e:
        print(f"  WARNING: coefficient stability plot failed: {e}")
        print(f"  CSVs are saved in {LOCO_DIR}/ — replot manually if needed.")

    try:
        plot_r2_stability(r2_df, full_r2)
    except Exception as e:
        print(f"  WARNING: R2 stability plot failed: {e}")
        print(f"  CSVs are saved in {LOCO_DIR}/ — replot manually if needed.")

    print_summary(coef_df, r2_df, full_coefs, full_r2)

    total = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE  |  Total time: {total:.1f}s ({total / 60:.1f} min)")
    print("=" * 70)
