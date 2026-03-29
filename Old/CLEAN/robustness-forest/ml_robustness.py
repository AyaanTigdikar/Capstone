"""
ml_robustness.py
================
Robustness check for ML models: re-run LASSO, Ridge, Elastic Net, RF
across four sample definitions, keeping algorithms and hyperparameters
IDENTICAL to run_nb5.py.

Samples compared:
  A — Original 54-country list (NB4)
  B — Adjusted rent sample, 3% threshold + Gulf (38 countries)
  C — Adjusted rent sample, 1% threshold + Gulf (58 countries)
  D — All countries in Master.csv with sufficient data (≈128 countries)

Outputs (robustness-forest/outputs/):
  ml_r2_comparison.csv          — train/test R² per model per sample
  ml_feature_importance.csv     — normalised feature importances (LASSO coef, RF importance)
  ml_coef_signs.csv             — sign stability across samples

Run from project root:
    python3 robustness-forest/ml_robustness.py
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from viz_utils import INCLUDE_LIST

OUT = os.path.join(ROOT, 'robustness-forest', 'outputs')
os.makedirs(OUT, exist_ok=True)

# ── Exact constants from run_nb5.py ──────────────────────────────────────────
TRAIN_END  = 2014
TEST_START = 2015
GULF = {'ARE', 'BHR', 'KWT', 'OMN', 'QAT', 'SAU', 'IRQ', 'IRN', 'YEM'}

BASE_FEATURES = [
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
    'L1_ECI',   # restored — mirrors NB5; excluded from importance display
]
INTERACTION_FEATURES = ['HCI_x_ProductionValue', 'GFCF_x_ProductionValue']
ALL_FEATURES = BASE_FEATURES + INTERACTION_FEATURES
IMPORTANCE_EXCLUDE = {'L1_ECI'}   # trained on but not reported

LOG_COLS = [
    'Human capital index',
    'Total_Production_Value_Per_Capita',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP',
    'Government revenue',
    'Use of IMF credit (DOD, current US$)',
]

SHORT = {
    'Total_Production_Value_Per_Capita': 'Prod. Value p.c.',
    'Human capital index': 'Human Capital',
    'Rule of law index': 'Rule of Law',
    'Political stability — estimate': 'Pol. Stability',
    'Trade (% of GDP)': 'Trade',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP': 'GFCF',
    'Share of investment in GDP': 'Investment Share',
    'Domestic credit to private sector (% of GDP)': 'Domestic Credit',
    'Landlocked': 'Landlocked',
    'Urban population (% of total population)': 'Urban Pop.',
    'Government revenue': 'Gov. Revenue',
    'Capital depreciation rate': 'Depreciation',
    'Use of IMF credit (DOD, current US$)': 'IMF Credit',
    'Real interest rate (%)': 'Real Rate',
    'Inflation, consumer prices (annual %)': 'Inflation',
    'Access to electricity (% of population)': 'Electricity',
    'Adjusted savings: gross savings (% of GNI)': 'Savings',
    'HCI_x_ProductionValue':  'HCI × Prod.',
    'GFCF_x_ProductionValue': 'GFCF × Prod.',
    'L1_ECI':                 'Lagged ECI',
}


# ── PanelTemporalCV — identical copy from run_nb5.py ─────────────────────────
class PanelTemporalCV:
    def __init__(self, years, n_splits=5, gap=1, min_train_years=8):
        self.years = np.asarray(years)
        self.n_splits = n_splits
        self.gap = gap
        self.min_train_years = min_train_years
        unique_years = np.sort(np.unique(self.years))
        earliest_cutoff = unique_years[0] + self.min_train_years - 1
        latest_cutoff   = unique_years[-1] - self.gap - 1
        self.cutoffs = np.unique(
            np.linspace(earliest_cutoff, latest_cutoff, n_splits).astype(int)
        )
        self.n_splits = len(self.cutoffs)

    def split(self, X=None, y=None, groups=None):
        for cutoff in self.cutoffs:
            train_idx = np.where(self.years <= cutoff)[0]
            val_idx   = np.where(self.years >  cutoff + self.gap)[0]
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def __iter__(self):
        return self.split()


# ── Feature engineering — identical to run_nb5.py ────────────────────────────
def engineer_features(df):
    df = df.copy()
    df['Total_Production_Value_Per_Capita'] = (
        df['Total_Production_Value'] / df['Population'].replace(0, np.nan)
    )
    df['L1_ECI']    = df.groupby('Country Code')['Economic Complexity Index'].shift(1)
    df['ECI_delta'] = df['Economic Complexity Index'] - df['L1_ECI']
    df = df.dropna(subset=['Economic Complexity Index'])

    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0)).replace([np.inf, -np.inf], np.nan)

    # Grand-mean centred interactions (means computed on THIS sample)
    _hci_mean  = df['Human capital index'].mean()
    _prod_mean = df['Total_Production_Value_Per_Capita'].mean()
    _gfcf_mean = df['Gross fixed capital formation, all, Constant prices, Percent of GDP'].mean()
    df['HCI_x_ProductionValue']  = (df['Human capital index'] - _hci_mean)  * \
                                    (df['Total_Production_Value_Per_Capita'] - _prod_mean)
    df['GFCF_x_ProductionValue'] = (df['Gross fixed capital formation, all, Constant prices, Percent of GDP'] - _gfcf_mean) * \
                                    (df['Total_Production_Value_Per_Capita'] - _prod_mean)
    df = df.dropna(subset=ALL_FEATURES)
    return df


# ── Run models for one sample ─────────────────────────────────────────────────
def run_sample(df, label):
    print(f"\n{'='*60}")
    print(f"Sample: {label}  ({df['Country Code'].nunique()} countries, {len(df):,} obs)")

    train_df = df[df['Year'] <= TRAIN_END].copy()
    test_df  = df[df['Year'] >= TEST_START].copy()
    print(f"  Train: {len(train_df):,} obs  |  Test: {len(test_df):,} obs")

    if len(test_df) < 10:
        print("  Skipping — not enough test observations.")
        return None

    train_years = train_df['Year'].values
    tscv = PanelTemporalCV(train_years, n_splits=5, gap=1, min_train_years=8)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[ALL_FEATURES].values)
    X_test  = scaler.transform(test_df[ALL_FEATURES].values)
    y_train = train_df['Economic Complexity Index'].values
    y_test  = test_df['Economic Complexity Index'].values

    lasso   = LassoCV(cv=tscv, random_state=42, max_iter=10000).fit(X_train, y_train)
    ridge   = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=tscv).fit(X_train, y_train)
    elastic = ElasticNetCV(l1_ratio=[0.5], cv=tscv, random_state=42,
                           max_iter=10000).fit(X_train, y_train)
    rf      = RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=10,
        random_state=42, n_jobs=-1, oob_score=True,
    ).fit(X_train, y_train)

    models = {'LASSO': lasso, 'Ridge': ridge, 'Elastic Net': elastic, 'Random Forest': rf}

    # ── R² ───────────────────────────────────────────────────────────────────
    r2_rows = []
    for mname, m in models.items():
        r2_tr = r2_score(y_train, m.predict(X_train))
        r2_te = r2_score(y_test,  m.predict(X_test))
        r2_rows.append({'Sample': label, 'Model': mname,
                        'R2_train': round(r2_tr, 4), 'R2_test': round(r2_te, 4)})
        print(f"  {mname:<15} train R²={r2_tr:.4f}  test R²={r2_te:.4f}")

    # ── Feature importances (raw coefs / RF importances) — L1_ECI excluded ──
    imp_rows = []
    for feat, lasso_c, ridge_c, en_c, rf_i in zip(
        ALL_FEATURES,
        lasso.coef_, ridge.coef_, elastic.coef_, rf.feature_importances_,
    ):
        if feat in IMPORTANCE_EXCLUDE:
            continue
        imp_rows.append({
            'Sample': label, 'Feature': feat,
            'Feature_short': SHORT.get(feat, feat),
            'LASSO_coef': round(lasso_c, 5),
            'Ridge_coef': round(ridge_c, 5),
            'EN_coef':    round(en_c, 5),
            'RF_imp':     round(rf_i, 5),
        })

    return pd.DataFrame(r2_rows), pd.DataFrame(imp_rows)


# ── Define samples ────────────────────────────────────────────────────────────
master_raw = pd.read_csv(os.path.join(ROOT, 'intermediary', 'Master.csv'),
                         dtype={'Country Code': str})
master_raw['Year'] = master_raw['Year'].astype(int)
master_raw = master_raw[(master_raw['Year'] >= 1995) & (master_raw['Year'] <= 2019)]

# Merge adjusted rent column for sample selection
adj_rents = pd.read_csv(os.path.join(OUT, 'master_adj.csv'),
                        dtype={'Country Code': str},
                        usecols=['Country Code', 'Year',
                                 'NR rents excl. forest (% of GDP)'])
master_raw = master_raw.merge(adj_rents, on=['Country Code', 'Year'], how='left')

RENT_ADJ = 'NR rents excl. forest (% of GDP)'
d95 = master_raw[master_raw['Year'] == 1995]
s3  = set(d95[d95[RENT_ADJ] >= 3.0]['Country Code']) | GULF
s2  = set(d95[d95[RENT_ADJ] >= 2.0]['Country Code']) | GULF
s1  = set(d95[d95[RENT_ADJ] >= 1.0]['Country Code']) | GULF

# Exclude HIC countries (excl. Gulf + original 54) from Sample D
import wbgapi as _wb
_eco = _wb.economy.DataFrame()
_hic = set(_eco[_eco['incomeLevel'] == 'HIC'].index.tolist())
_hic_excl = _hic - GULF - set(INCLUDE_LIST)
master_raw_dev = master_raw[~master_raw['Country Code'].isin(_hic_excl)].copy()

SAMPLES = {
    'A — Original 54 (NB4)':              master_raw[master_raw['Country Code'].isin(INCLUDE_LIST)].copy(),
    'B — Adj. rent, 3% sample (38 ctry)': master_raw[master_raw['Country Code'].isin(s3)].copy(),
    'C — Adj. rent, 2% sample':           master_raw[master_raw['Country Code'].isin(s2)].copy(),
    'D — Adj. rent, 1% sample (58 ctry)': master_raw[master_raw['Country Code'].isin(s1)].copy(),
    'E — All non-HIC countries':           master_raw_dev.copy(),
}

# ── Run all samples ───────────────────────────────────────────────────────────
all_r2  = []
all_imp = []

for label, df_raw in SAMPLES.items():
    df = engineer_features(df_raw)
    result = run_sample(df, label)
    if result is not None:
        all_r2.append(result[0])
        all_imp.append(result[1])

r2_df  = pd.concat(all_r2,  ignore_index=True)
imp_df = pd.concat(all_imp, ignore_index=True)

r2_df.to_csv(os.path.join(OUT, 'ml_r2_comparison.csv'),        index=False)
imp_df.to_csv(os.path.join(OUT, 'ml_feature_importance.csv'),  index=False)
print(f'\n  Saved ml_r2_comparison.csv')
print(f'  Saved ml_feature_importance.csv')
print('\n✓ ml_robustness.py complete')
