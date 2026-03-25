#!/usr/bin/env python3
"""
run_loco.py
===========
Leave-One-Country-Out (LOCO) for LASSO, Ridge, ElasticNet, and Random Forest.
Feature set aligned with run_bootstrap.py (24 features: base 19 + rolling 3 + interactions 2).

Run from:  /Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP
    python robustness/run_loco.py

Outputs
-------
intermediary/loco/loco_r2.csv          — test R2 per country per model
intermediary/loco/loco_coefs_en.csv    — ElasticNet coefficients per LOO run
intermediary/loco/loco_coefs_lasso.csv — LASSO coefficients per LOO run
intermediary/loco/loco_coefs_ridge.csv — Ridge coefficients per LOO run
Final/NB5/fig13_loco_coef_stability.*  — coefficient stability chart (EN)
Final/NB5/fig14_loco_r2_stability.*    — R2 stability chart (all 4 models)
"""

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_END  = 2014
TEST_START = 2015
TOP_K      = 12

LOCO_DIR = "intermediary/loco"
FIG_DIR  = os.path.join("Final", "NB5")
os.makedirs(LOCO_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

INCLUDE = [
    'AGO','ARE','AZE','BFA','BHR','BOL','CHL','CIV','CMR',
    'COD','COG','DZA','ECU','EGY','ETH','GAB','GHA','GIN',
    'GNQ','IDN','IRN','IRQ','KAZ','KEN','KWT','LAO','LBR',
    'LBY','MDG','MLI','MMR','MNG','MOZ','MWI','MYS','NER',
    'NGA','OMN','PNG','QAT','RUS','RWA','SAU','TCD','TGO',
    'TTO','TZA','UGA','UZB','VEN','VNM','YEM','ZMB','ZWE',
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE LISTS  —  aligned with run_bootstrap.py
# ─────────────────────────────────────────────────────────────────────────────

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
new_features         = ['Inflation_roll5', 'RealRate_roll5', 'Resource_HHI']
interaction_features = ['HCI_x_ProductionValue', 'RuleOfLaw_x_ProductionValue']
all_features         = base_features + new_features + interaction_features  # 24

NAME_MAP = {
    'Total_Production_Value_Per_Capita':                                    'Prod Value p.c.',
    'Human capital index':                                                  'Human Capital',
    'Rule of law index':                                                    'Rule of Law',
    'Political stability — estimate':                                       'Political Stability',
    'Trade (% of GDP)':                                                     'Trade',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP':  'Capital Formation',
    'Share of investment in GDP':                                           'Investment Share',
    'Domestic credit to private sector (% of GDP)':                        'Domestic Credit',
    'Landlocked':                                                           'Landlocked',
    'Urban population (% of total population)':                            'Urban Population',
    'Government revenue':                                                   'Gov Revenue',
    'Capital depreciation rate':                                            'Depreciation',
    'Use of IMF credit (DOD, current US$)':                                'IMF Credit',
    'Real interest rate (%)':                                               'Real Interest Rate',
    'Inflation, consumer prices (annual %)':                               'Inflation (annual)',
    'Access to electricity (% of population)':                             'Electricity Access',
    'Adjusted savings: gross savings (% of GNI)':                          'Gross Savings',
    'L1_ECI':                                                              'Lagged ECI',
    'Forestry rents (% of GDP)':                                           'Forestry Rents',
    'Inflation_roll5':                                                      'Inflation (5yr avg)',
    'RealRate_roll5':                                                       'Real Rate (5yr avg)',
    'Resource_HHI':                                                         'Resource HHI',
    'HCI_x_ProductionValue':                                               'HC × Production',
    'RuleOfLaw_x_ProductionValue':                                         'RuleLaw × Production',
}

def sname(f):
    return NAME_MAP.get(f, f[:22])


# ─────────────────────────────────────────────────────────────────────────────
# PANEL TEMPORAL CV  (identical to NB5 / run_bootstrap.py)
# ─────────────────────────────────────────────────────────────────────────────

class PanelTemporalCV:
    def __init__(self, years, n_splits=5, gap=1, min_train_years=8):
        self.years = np.asarray(years)
        unique_years = np.sort(np.unique(self.years))
        earliest = unique_years[0] + min_train_years - 1
        latest   = unique_years[-1] - gap - 1
        if earliest > latest:
            raise ValueError("Year range too narrow.")
        self.cutoffs  = np.unique(np.linspace(earliest, latest, n_splits).astype(int))
        self.n_splits = len(self.cutoffs)
        self.gap      = gap

    def split(self, X=None, y=None, groups=None):
        for c in self.cutoffs:
            tr = np.where(self.years <= c)[0]
            va = np.where(self.years > c + self.gap)[0]
            if len(tr) > 0 and len(va) > 0:
                yield tr, va

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION  —  aligned with run_bootstrap.prepare_ml_df
# ─────────────────────────────────────────────────────────────────────────────

def prepare_full_df():
    master = pd.read_csv("intermediary/Master.csv")
    df = master[
        master['Year'].between(1995, 2019) &
        master['Country Code'].isin(INCLUDE)
    ].copy().sort_values(['Country Code', 'Year']).reset_index(drop=True)

    df['Total_Production_Value_Per_Capita'] = (
        df['Total_Production_Value'] / df['Population']
    )

    # Rolling macro controls
    df['Inflation_roll5'] = (
        df.groupby('Country Code')['Inflation, consumer prices (annual %)']
          .transform(lambda x: x.rolling(5, min_periods=3).mean())
    )
    df['RealRate_roll5'] = (
        df.groupby('Country Code')['Real interest rate (%)']
          .transform(lambda x: x.rolling(5, min_periods=3).mean())
    )

    # Resource concentration HHI
    rents_cols = [
        'Oil rents (% of GDP)', 'Natural gas rents (% of GDP)',
        'Mineral rents (% of GDP)', 'Forestry rents (% of GDP)',
    ]
    total_rents = df['Total natural resources rents (% of GDP)'].replace(0, np.nan)
    df['Resource_HHI'] = sum(
        (df[col] / total_rents) ** 2 for col in rents_cols
    )

    # ECI targets and lag
    df['L1_ECI']    = df.groupby('Country Code')['Economic Complexity Index'].shift(1)
    df['ECI_delta'] = df['Economic Complexity Index'] - df['L1_ECI']
    df = df.dropna(subset=['L1_ECI', 'Economic Complexity Index', 'ECI_delta'])

    # Log transforms
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
    hci_mean  = df['Human capital index'].mean()
    prod_mean = df['Total_Production_Value_Per_Capita'].mean()
    rol_mean  = df['Rule of law index'].mean()

    df['HCI_x_ProductionValue']      = (df['Human capital index'] - hci_mean) * (df['Total_Production_Value_Per_Capita'] - prod_mean)
    df['RuleOfLaw_x_ProductionValue'] = (df['Rule of law index']  - rol_mean)  * (df['Total_Production_Value_Per_Capita'] - prod_mean)

    df = df.dropna(subset=all_features)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FULL-SAMPLE BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def fit_full_sample(df):
    train = df[df['Year'] <= TRAIN_END]
    test  = df[df['Year'] >= TEST_START]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(train[all_features].values)
    X_te   = scaler.transform(test[all_features].values)
    y_tr   = train['Economic Complexity Index'].values
    y_te   = test['Economic Complexity Index'].values

    tscv = PanelTemporalCV(train['Year'].values, n_splits=5, gap=1, min_train_years=8)

    lasso = LassoCV(cv=tscv, random_state=42, max_iter=10000).fit(X_tr, y_tr)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=tscv).fit(X_tr, y_tr)
    en    = ElasticNetCV(l1_ratio=[0.5], cv=tscv, random_state=42, max_iter=10000).fit(X_tr, y_tr)
    rf    = RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42, n_jobs=-1
    ).fit(X_tr, y_tr)

    models = {'LASSO': lasso, 'Ridge': ridge, 'EN': en, 'RF': rf}
    r2s    = {name: r2_score(y_te, m.predict(X_te)) for name, m in models.items()}
    coefs  = {
        'LASSO': dict(zip(all_features, lasso.coef_)),
        'Ridge': dict(zip(all_features, ridge.coef_)),
        'EN':    dict(zip(all_features, en.coef_)),
        'RF':    dict(zip(all_features, rf.feature_importances_)),
    }
    return r2s, coefs


# ─────────────────────────────────────────────────────────────────────────────
# LOCO LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_loco(df, full_r2s, full_coefs):
    countries = sorted(df['Country Code'].unique())
    N = len(countries)

    r2_rows, coef_en_rows, coef_lasso_rows, coef_ridge_rows = [], [], [], []
    t0 = time.time()

    for i, cc in enumerate(countries, 1):
        df_loo  = df[df['Country Code'] != cc].copy()
        tr_loo  = df_loo[df_loo['Year'] <= TRAIN_END]
        te_loo  = df_loo[df_loo['Year'] >= TEST_START]

        if len(tr_loo) < 30 or len(te_loo) < 10:
            print(f"  WARNING: skipping {cc} (train={len(tr_loo)}, test={len(te_loo)})")
            continue

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(tr_loo[all_features].values)
        X_te   = scaler.transform(te_loo[all_features].values)
        y_tr   = tr_loo['Economic Complexity Index'].values
        y_te   = te_loo['Economic Complexity Index'].values

        tscv = PanelTemporalCV(tr_loo['Year'].values, n_splits=5, gap=1, min_train_years=8)

        lasso = LassoCV(cv=tscv, random_state=42, max_iter=10000).fit(X_tr, y_tr)
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=tscv).fit(X_tr, y_tr)
        en    = ElasticNetCV(l1_ratio=[0.5], cv=tscv, random_state=42, max_iter=10000).fit(X_tr, y_tr)
        rf    = RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42, n_jobs=-1
        ).fit(X_tr, y_tr)

        r2_rows.append({
            'excluded_country': cc,
            'LASSO_r2':  r2_score(y_te, lasso.predict(X_te)),
            'Ridge_r2':  r2_score(y_te, ridge.predict(X_te)),
            'EN_r2':     r2_score(y_te, en.predict(X_te)),
            'RF_r2':     r2_score(y_te, rf.predict(X_te)),
            'n_train': len(tr_loo), 'n_test': len(te_loo),
            'n_countries': df_loo['Country Code'].nunique(),
        })

        en_row    = {'excluded_country': cc, **dict(zip(all_features, en.coef_))}
        lasso_row = {'excluded_country': cc, **dict(zip(all_features, lasso.coef_))}
        ridge_row = {'excluded_country': cc, **dict(zip(all_features, ridge.coef_))}
        coef_en_rows.append(en_row)
        coef_lasso_rows.append(lasso_row)
        coef_ridge_rows.append(ridge_row)

        if i % 10 == 0 or i == 1:
            elapsed = time.time() - t0
            eta     = (N - i) / max(i, 1) * elapsed
            print(f"  {i:>2d}/{N}  excluded={cc}  "
                  f"EN={r2_rows[-1]['EN_r2']:.4f}  RF={r2_rows[-1]['RF_r2']:.4f}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    r2_df    = pd.DataFrame(r2_rows)
    en_df    = pd.DataFrame(coef_en_rows)
    lasso_df = pd.DataFrame(coef_lasso_rows)
    ridge_df = pd.DataFrame(coef_ridge_rows)

    r2_df.to_csv(   os.path.join(LOCO_DIR, "loco_r2.csv"),           index=False)
    en_df.to_csv(   os.path.join(LOCO_DIR, "loco_coefs_en.csv"),     index=False)
    lasso_df.to_csv(os.path.join(LOCO_DIR, "loco_coefs_lasso.csv"),  index=False)
    ridge_df.to_csv(os.path.join(LOCO_DIR, "loco_coefs_ridge.csv"),  index=False)

    print(f"\n  {len(r2_rows)} LOCO runs in {time.time()-t0:.1f}s — CSVs saved to {LOCO_DIR}/")
    return r2_df, en_df, lasso_df, ridge_df


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'LASSO': '#4a6fa5', 'Ridge': '#7aa0c4',
    'EN':    '#2e7d4a', 'RF':    '#c23a3a',
}


def plot_coef_stability(en_df, lasso_df, full_coefs, full_r2s):
    """Figure 13 — coefficient stability for EN and LASSO (top K features by EN magnitude)."""
    feat_cols = [c for c in en_df.columns if c != 'excluded_country']

    ranked = sorted(
        [(f, abs(full_coefs['EN'].get(f, 0))) for f in feat_cols if f != 'L1_ECI'],
        key=lambda x: x[1], reverse=True,
    )
    top_feats = [f for f, _ in ranked[:TOP_K]]
    short     = [sname(f) for f in top_feats]
    y_pos     = np.arange(len(top_feats))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.subplots_adjust(wspace=0.04)

    for ax, (label, coef_df, model_key, clr) in zip(axes, [
        ('ElasticNet', en_df,    'EN',    '#2e7d4a'),
        ('LASSO',      lasso_df, 'LASSO', '#4a6fa5'),
    ]):
        means = coef_df[top_feats].mean()
        mins  = coef_df[top_feats].min()
        maxs  = coef_df[top_feats].max()

        for j, feat in enumerate(top_feats):
            ax.plot([mins[feat], maxs[feat]], [j, j],
                    color=clr, linewidth=1.5, alpha=0.6, solid_capstyle='round')

        dot_colors = ['#c23a3a' if means[feat] < 0 else clr for feat in top_feats]
        ax.scatter(means.values, y_pos, c=dot_colors, s=55, zorder=5,
                   edgecolors='white', linewidths=0.5)

        for j, feat in enumerate(top_feats):
            ax.scatter(full_coefs[model_key].get(feat, 0), j,
                       marker='|', c='#333333', s=90, zorder=6, linewidths=1.2)

        ax.axvline(0, color='#c9cfd6', linewidth=0.8, zorder=0)
        ax.set_title(label, fontsize=12, fontweight='semibold', pad=8)
        ax.set_xlabel('Standardised coefficient', fontsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(short, fontsize=9)
    axes[0].invert_yaxis()

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555', markersize=7, label='LOCO mean'),
        Line2D([0], [0], marker='|', color='#333333', markersize=9,
               markeredgewidth=1.2, linestyle='None', label='Full-sample'),
        Line2D([0], [0], color='#999', linewidth=1.5, label='Min-max range'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9,
                   frameon=True, framealpha=0.9, edgecolor='#e5e7eb')

    fig.suptitle('LOCO Coefficient Stability — Penalised ML Models', fontsize=13,
                 fontweight='semibold', y=1.01)

    path = os.path.join(FIG_DIR, "fig13_loco_coef_stability")
    fig.savefig(f"{path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{path}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}.png / .pdf")


def plot_r2_stability(r2_df, full_r2s):
    """Figure 14 — test R2 per excluded country for all 4 models."""
    # Sort by mean R2 across models
    r2_df = r2_df.copy()
    r2_df['mean_r2'] = r2_df[['LASSO_r2', 'Ridge_r2', 'EN_r2', 'RF_r2']].mean(axis=1)
    r2_df = r2_df.sort_values('mean_r2').reset_index(drop=True)

    model_cols = [('LASSO', 'LASSO_r2'), ('Ridge', 'Ridge_r2'),
                  ('EN', 'EN_r2'), ('RF', 'RF_r2')]

    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(r2_df))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for (label, col), off in zip(model_cols, offsets):
        ax.bar(x_pos + off * width, r2_df[col], width=width,
               color=COLORS[label], label=label, edgecolor='white', linewidth=0.2)
        ax.axhline(full_r2s[label], color=COLORS[label],
                   linewidth=1.0, linestyle='--', alpha=0.6)

    y_min = max(0, r2_df[['LASSO_r2','Ridge_r2','EN_r2','RF_r2']].values.min() - 0.03)
    ax.set_ylim(y_min, 1.01)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(r2_df['excluded_country'], rotation=90, fontsize=7)
    ax.set_ylabel('Test R²', fontsize=11)
    ax.set_xlabel('Excluded country', fontsize=11)
    ax.legend(fontsize=9, loc='lower left', frameon=True, framealpha=0.9, edgecolor='#e5e7eb')
    ax.tick_params(axis='y', labelsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('LOCO Test R² Stability — All ML Models', fontsize=13,
                 fontweight='semibold', pad=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig14_loco_r2_stability")
    fig.savefig(f"{path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{path}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}.png / .pdf")


def print_summary(r2_df, en_df, full_r2s, full_coefs):
    print(f"\n{'='*70}")
    print("  LOCO SUMMARY")
    print(f"{'='*70}")
    for label, col in [('LASSO','LASSO_r2'),('Ridge','Ridge_r2'),('EN','EN_r2'),('RF','RF_r2')]:
        vals = r2_df[col]
        print(f"  {label:<8}  full={full_r2s[label]:.4f}  "
              f"LOCO mean={vals.mean():.4f}  range=[{vals.min():.4f},{vals.max():.4f}]  "
              f"SD={vals.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir).lower() in ("robustness", "scripts"):
        os.chdir(os.path.dirname(script_dir))

    print("="*70)
    print("  LOCO ANALYSIS — LASSO / Ridge / ElasticNet / Random Forest")
    print(f"  Features: {len(all_features)} | Working dir: {os.getcwd()}")
    print("="*70)

    t_total = time.time()

    print("\nLoading data...")
    df = prepare_full_df()
    print(f"  {df['Country Code'].nunique()} countries, {len(df):,} obs  "
          f"(train: {(df.Year<=TRAIN_END).sum()}, test: {(df.Year>=TEST_START).sum()})")

    print("\nFitting full-sample baselines...")
    full_r2s, full_coefs = fit_full_sample(df)
    for k, v in full_r2s.items():
        print(f"  {k:<8} test R² = {v:.4f}")

    print("\nRunning LOCO...")
    r2_df, en_df, lasso_df, ridge_df = run_loco(df, full_r2s, full_coefs)

    print("\nGenerating figures...")
    try:
        plot_coef_stability(en_df, lasso_df, full_coefs, full_r2s)
    except Exception as e:
        print(f"  WARNING coef plot failed: {e}")

    try:
        plot_r2_stability(r2_df, full_r2s)
    except Exception as e:
        print(f"  WARNING r2 plot failed: {e}")

    print_summary(r2_df, en_df, full_r2s, full_coefs)

    total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  DONE  |  {total:.1f}s ({total/60:.1f} min)")
    print("="*70)
