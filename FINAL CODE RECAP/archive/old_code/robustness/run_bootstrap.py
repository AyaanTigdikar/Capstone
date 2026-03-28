#!/usr/bin/env python3
"""
run_bootstrap.py
================
Self-contained bootstrap pipeline for the Moody's capstone.
Run from:  /Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP

PREREQUISITE
------------
Before running this script, add ONE line to NB3 right after the
sample-selection cell (Cell 23, the one that ends with
`cmaster = df.copy()`). The line to add:

    cmaster.to_csv("intermediary/master_pre_imputation.csv", index=False)

Then re-run NB3 so that file is created. Everything else this script
needs (master_data_imputed.csv, Master.csv, PopulationWDI.csv) should
already exist from prior runs.

WHAT IT DOES
------------
Phase 1 (NB3):  Generates B=200 country-resampled, re-imputed datasets.
Phase 2 (NB6):  Fits Models 3a/3b on each, reports bootstrap CIs.
Phase 3 (NB5):  Fits ML models on each, reports bootstrap CIs.

NB6 runs before NB5 because it is much faster (~2 min vs ~40 min).

USAGE
-----
    cd "/Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP"
    python run_bootstrap.py

Optional flags (edit the CONFIGURATION block below):
    B           = 200     # bootstrap iterations (100 is fine if short on time)
    SKIP_NB3    = False   # set True to skip Phase 1 if bootstrap files exist
    SKIP_NB5    = False   # set True to skip Phase 3 (the slow one)
"""

import os
import sys
import io
import time
import glob
import warnings
import contextlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from types import SimpleNamespace

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

B = 200
KNN_NEIGHBORS = 5
RANDOM_SEED = 12345
ALPHA = 0.05        # for 95% CIs
TRAIN_END = 2014
TEST_START = 2015

SKIP_NB3 = False    # set True if intermediary/bootstrap/Master_b*.csv exist
SKIP_NB5 = False    # set True to skip the slow ML phase

BOOT_DIR = "intermediary/bootstrap"
os.makedirs(BOOT_DIR, exist_ok=True)

# 54-country sample
INCLUDE = [
    'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
    'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
    'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
    'LBY', 'MDG', 'MLI', 'MMR', 'MNG', 'MOZ', 'MWI', 'MYS', 'NER',
    'NGA', 'OMN', 'PNG', 'QAT', 'RUS', 'RWA', 'SAU', 'TCD', 'TGO',
    'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE',
]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FUNCTIONS (extracted from NB3, NB5, NB6)
# ══════════════════════════════════════════════════════════════════════════════

# ── Imputation constants ──────────────────────────────────────────────────────

EXCLUDE_VARS = [
    'Landlocked',
    'Subsoil_Metals_Dominant',
    'Hydrocarbons_Dominant',
    'Precious_Metals_Dominant',
    'Total_Reserves_Value',
    'Total_Reserves',
]

EXCLUDE_VARS_KNN = [
    'Landlocked',
    'Subsoil_Metals_Dominant',
    'Hydrocarbons_Dominant',
    'Precious_Metals_Dominant',
    'Total_Production_Value',
    'Total_Reserves_Value',
    'Total_Production',
    'Total_Reserves',
]

MAX_GAP = 3


# ── Imputation functions (copied from NB3 cells 25, 28) ──────────────────────

def impute_linear_interpolation(df, max_gap=MAX_GAP, quiet=False):
    id_cols = ['Country Code', 'Country Name', 'Year']
    data_cols = [c for c in df.columns if c not in id_cols and c not in EXCLUDE_VARS]

    df_sorted = df.sort_values(['Country Code', 'Year']).reset_index(drop=True)
    imputed_dfs = []

    for country in df_sorted['Country Code'].unique():
        country_data = df_sorted[df_sorted['Country Code'] == country].copy()
        for col in data_cols:
            series = country_data[col].values.copy().astype(float)
            observed_mask = ~np.isnan(series)
            if not observed_mask.any():
                continue

            obs_indices = np.where(observed_mask)[0]
            first_obs = obs_indices[0]
            last_obs = obs_indices[-1]

            # Interior interpolation
            interior = pd.Series(series)
            interior = interior.interpolate(
                method='linear', limit=max_gap, limit_direction='forward',
                limit_area='inside',
            )
            series = interior.values

            # Leading extrapolation
            if first_obs > 0 and first_obs <= max_gap:
                fill_series = pd.Series(series)
                fill_series = fill_series.interpolate(
                    method='linear', limit=max_gap, limit_direction='backward',
                    limit_area=None,
                )
                series[:first_obs] = fill_series.values[:first_obs]

            # Trailing extrapolation
            n = len(series)
            trailing_gap = n - 1 - last_obs
            if trailing_gap > 0 and trailing_gap <= max_gap:
                fill_series = pd.Series(series)
                fill_series = fill_series.interpolate(
                    method='linear', limit=max_gap, limit_direction='forward',
                    limit_area=None,
                )
                series[last_obs + 1:] = fill_series.values[last_obs + 1:]

            country_data[col] = series
        imputed_dfs.append(country_data)

    return pd.concat(imputed_dfs, ignore_index=True)


def apply_knn_imputer(df, n_neighbors=5):
    id_cols = ['Country Code', 'Country Name', 'Year']
    numeric_cols = [
        c for c in df.columns
        if c not in id_cols and c not in EXCLUDE_VARS_KNN
        and df[c].dtype in ['float64', 'int64']
    ]

    was_missing = df[numeric_cols].isna().copy()

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols].fillna(df[numeric_cols].mean())),
        columns=numeric_cols,
    )
    df_scaled[df[numeric_cols].isna()] = np.nan

    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    imputed_array = imputer.fit_transform(df_scaled)
    imputed_values = scaler.inverse_transform(imputed_array)

    df_final = df.copy()
    df_final[numeric_cols] = imputed_values

    knn_mask = was_missing & df_final[numeric_cols].notna()
    return df_final, knn_mask


# ── PanelTemporalCV (copied from NB5 cell 6) ─────────────────────────────────

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


# ── NB6 regression variable lists ────────────────────────────────────────────

reg3_input = [
    'log_HCI', 'log_GFCF', 'Political stability — estimate',
    'Rule of law index', 'log_Production_Value',
    'Forestry rents (% of GDP)', 'Trade (% of GDP)',
]
INTERACT_VARS = [
    'log_HCI_x_log_Production', 'log_GFCF_x_log_Production',
    'log_HCI_x_forestry_rents', 'log_GFCF_x_forestry_rents',
]


# ── NB5 feature lists ────────────────────────────────────────────────────────

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
all_features = base_features + new_features + interaction_features


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: BOOTSTRAP IMPUTATION (NB3)
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1():
    print("\n" + "#" * 70)
    print("# PHASE 1: Bootstrap Imputation (NB3)")
    print("#" * 70)

    pre_imp_path = "intermediary/master_pre_imputation.csv"
    if not os.path.exists(pre_imp_path):
        print(
            f"\nERROR: {pre_imp_path} not found.\n"
            "Add this line to NB3 right after 'cmaster = df.copy()':\n\n"
            '    cmaster.to_csv("intermediary/master_pre_imputation.csv", index=False)\n\n'
            "Then re-run NB3 and try again."
        )
        sys.exit(1)

    cmaster = pd.read_csv(pre_imp_path)
    print(f"Loaded pre-imputation panel: {cmaster.shape[0]:,} rows, "
          f"{cmaster['Country Code'].nunique()} countries")

    # Load population data
    pop_path = "rawdata/PopulationWDI.csv"
    if not os.path.exists(pop_path):
        print(f"ERROR: {pop_path} not found. Cannot merge population.")
        sys.exit(1)

    df_pop = pd.read_csv(pop_path)
    df_pop = df_pop[df_pop["Country Code"].notna()].copy()
    id_cols_pop = ["Country Name", "Country Code"]
    year_cols = [col for col in df_pop.columns if "[YR" in col]
    df_long = pd.melt(df_pop, id_vars=id_cols_pop, value_vars=year_cols,
                       var_name="Year_raw", value_name="Population")
    df_long["Year"] = df_long["Year_raw"].str.extract(r"(\d{4})").astype(int)
    df_long["Population"] = pd.to_numeric(df_long["Population"], errors="coerce")
    df_long = df_long[["Country Code", "Year", "Population"]].dropna()
    print(f"Population data: {df_long.shape[0]:,} rows")

    # Bootstrap loop
    rng = np.random.RandomState(RANDOM_SEED)
    countries = cmaster["Country Code"].unique()
    N = len(countries)
    log_rows = []
    t0 = time.time()

    print(f"\nStarting {B} bootstrap iterations...\n")

    for b in range(1, B + 1):
        drawn = rng.choice(countries, size=N, replace=True)

        # Build bootstrap panel with suffixed duplicates
        frames = []
        counts = {}
        for cc in drawn:
            counts[cc] = counts.get(cc, 0) + 1
            chunk = cmaster[cmaster["Country Code"] == cc].copy()
            chunk["_original_cc"] = cc
            if counts[cc] > 1:
                suffix = f"_{counts[cc]}"
                chunk["Country Code"] = cc + suffix
                chunk["Country Name"] = chunk["Country Name"].astype(str) + suffix
            frames.append(chunk)

        boot_panel = pd.concat(frames, ignore_index=True)

        # Pull _original_cc out before imputation (it's a string column
        # and the imputer would choke trying to cast it to float)
        original_cc = boot_panel[["_original_cc"]].copy()
        boot_panel = boot_panel.drop(columns=["_original_cc"])

        # Impute
        boot_interp = impute_linear_interpolation(boot_panel, max_gap=MAX_GAP)
        boot_imp, _ = apply_knn_imputer(boot_interp, n_neighbors=KNN_NEIGHBORS)

        # Reattach original codes and merge population
        boot_imp["_original_cc"] = original_cc["_original_cc"].values
        boot_imp = boot_imp.merge(
            df_long.rename(columns={"Country Code": "_original_cc"}),
            on=["_original_cc", "Year"],
            how="left",
        )
        boot_imp["Country Code"] = boot_imp["_original_cc"]
        boot_imp.drop(columns=["_original_cc"], inplace=True)

        # Save
        out_path = os.path.join(BOOT_DIR, f"Master_b{b:03d}.csv")
        boot_imp.to_csv(out_path, index=False)

        log_rows.append({
            "b": b, "unique_drawn": len(set(drawn)), "n_rows": len(boot_imp),
        })

        if b % 25 == 0 or b == 1:
            elapsed = time.time() - t0
            eta = (B - b) / (b / elapsed)
            print(f"  b={b:>3d}/{B}  |  {len(set(drawn))}/{N} unique  |  "
                  f"{elapsed:.0f}s elapsed  |  ETA {eta:.0f}s")

    pd.DataFrame(log_rows).to_csv(
        os.path.join(BOOT_DIR, "bootstrap_log.csv"), index=False
    )
    print(f"\nPhase 1 complete: {B} files in {BOOT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: BOOTSTRAP REGRESSIONS (NB6)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_regression_df(filepath):
    master = pd.read_csv(filepath)
    df = master[master["Country Code"].isin(INCLUDE)].copy()
    if len(df) < 50:
        return None

    df["Total_Production_Value_Per_Capita"] = (
        df["Total_Production_Value"] / df["Population"]
    )
    df["log_HCI"] = np.log1p(df["Human capital index"])
    df["log_GFCF"] = np.log1p(
        df["Gross fixed capital formation, all, Constant prices, Percent of GDP"]
    )
    df["log_Production_Value"] = np.log1p(df["Total_Production_Value_Per_Capita"])

    # Grand-mean centre log variables and forestry rents (matches NB6)
    for col in ["log_HCI", "log_GFCF", "log_Production_Value"]:
        df[f"{col}_c"] = df[col] - df[col].mean()
    df["forestry_rents_c"] = (
        df["Forestry rents (% of GDP)"] - df["Forestry rents (% of GDP)"].mean()
    )

    df["log_HCI_x_log_Production"]  = df["log_HCI_c"]  * df["log_Production_Value_c"]
    df["log_GFCF_x_log_Production"] = df["log_GFCF_c"] * df["log_Production_Value_c"]
    df["log_HCI_x_forestry_rents"]  = df["log_HCI_c"]  * df["forestry_rents_c"]
    df["log_GFCF_x_forestry_rents"] = df["log_GFCF_c"] * df["forestry_rents_c"]

    df = df.sort_values(["Country Code", "Year"]).reset_index(drop=True)
    df["ECI_lag1"] = df.groupby("Country Code")["Economic Complexity Index"].shift(1)
    return df


def fit_ols(df, include_lag=False):
    vars_list = reg3_input + INTERACT_VARS
    if include_lag:
        vars_list = vars_list + ["ECI_lag1"]
    cols = vars_list + ["Economic Complexity Index"]
    reg_df = df[cols].dropna()
    if len(reg_df) < 30:
        return None
    y = reg_df["Economic Complexity Index"]
    X = sm.add_constant(reg_df[vars_list])
    fit = sm.OLS(y, X).fit()
    result = {"n_obs": int(fit.nobs), "r2": fit.rsquared}
    for var in fit.params.index:
        result[f"coef__{var}"] = fit.params[var]
        result[f"se__{var}"] = fit.bse[var]
    return result


def fit_driscoll_kraay(y, X, time, groups):
    raw = sm.OLS(y, X).fit()
    robust = raw.get_robustcov_results(
        cov_type='HAC-Groupsum', time=time, groups=groups,
        maxlags=2, kernel='bartlett', use_correction=True,
    )
    return SimpleNamespace(
        params=pd.Series(robust.params, index=X.columns),
        bse=pd.Series(robust.bse, index=X.columns),
        tvalues=pd.Series(robust.tvalues, index=X.columns),
        pvalues=pd.Series(robust.pvalues, index=X.columns),
        nobs=robust.nobs, rsquared=robust.rsquared,
    )


def run_phase2():
    print("\n" + "#" * 70)
    print("# PHASE 2: Bootstrap Regressions (NB6)")
    print("#" * 70)

    boot_files = sorted(glob.glob(os.path.join(BOOT_DIR, "Master_b*.csv")))
    if not boot_files:
        print("ERROR: No bootstrap files found. Run Phase 1 first.")
        return
    print(f"Found {len(boot_files)} bootstrap files.")

    # ── Fit original models from Master.csv for comparison (clustered SE, matches NB6)
    print("\nFitting original Models 3a/3b on Master.csv...")
    orig_df = prepare_regression_df("intermediary/Master.csv")

    def fit_clustered(y, X, groups):
        raw = sm.OLS(y, X).fit()
        robust = raw.get_robustcov_results(cov_type='cluster', groups=groups)
        return SimpleNamespace(
            params=pd.Series(robust.params, index=X.columns),
            bse=pd.Series(robust.bse, index=X.columns),
            tvalues=pd.Series(robust.tvalues, index=X.columns),
            pvalues=pd.Series(robust.pvalues, index=X.columns),
            nobs=robust.nobs, rsquared=raw.rsquared,
        )

    cols_3a = reg3_input + INTERACT_VARS + ['Economic Complexity Index', 'Country Code', 'Year']
    r3a_df = orig_df[cols_3a].dropna()
    y3a = r3a_df['Economic Complexity Index']
    X3a = sm.add_constant(r3a_df[reg3_input + INTERACT_VARS])
    m3a = fit_clustered(y3a, X3a, r3a_df['Country Code'].values)
    print(f"  Model 3a: N={int(m3a.nobs)}, R2={m3a.rsquared:.4f}")

    cols_3b = cols_3a + ['ECI_lag1']
    r3b_df = orig_df[cols_3b].dropna()
    y3b = r3b_df['Economic Complexity Index']
    X3b = sm.add_constant(r3b_df[reg3_input + INTERACT_VARS + ['ECI_lag1']])
    m3b = fit_clustered(y3b, X3b, r3b_df['Country Code'].values)
    print(f"  Model 3b: N={int(m3b.nobs)}, R2={m3b.rsquared:.4f}")

    # ── Bootstrap loop ────────────────────────────────────────────────────
    results_3a, results_3b = [], []
    skipped = 0
    t0 = time.time()

    for i, fpath in enumerate(boot_files, 1):
        df_b = prepare_regression_df(fpath)
        if df_b is None:
            skipped += 1
            continue
        r3a = fit_ols(df_b, include_lag=False)
        r3b = fit_ols(df_b, include_lag=True)
        if r3a:
            r3a["b"] = i
            results_3a.append(r3a)
        if r3b:
            r3b["b"] = i
            results_3b.append(r3b)

        if i % 50 == 0 or i == 1:
            elapsed = time.time() - t0
            print(f"  b={i:>3d}/{len(boot_files)}  |  {elapsed:.1f}s elapsed")

    df_3a = pd.DataFrame(results_3a)
    df_3b = pd.DataFrame(results_3b)
    df_3a.to_csv(os.path.join(BOOT_DIR, "nb6_boot_coefs_3a.csv"), index=False)
    df_3b.to_csv(os.path.join(BOOT_DIR, "nb6_boot_coefs_3b.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n  {len(boot_files) - skipped} successful, {skipped} skipped in {elapsed:.1f}s")

    # ── Report ────────────────────────────────────────────────────────────
    lo, hi = ALPHA / 2, 1 - ALPHA / 2

    for label, boot_df, orig_model, var_list in [
        ("MODEL 3a (no lag)", df_3a, m3a, ["const"] + reg3_input + INTERACT_VARS),
        ("MODEL 3b (with lag)", df_3b, m3b, ["const"] + reg3_input + INTERACT_VARS + ["ECI_lag1"]),
    ]:
        print(f"\n{'=' * 95}")
        print(f"  {label}")
        print(f"{'=' * 95}")
        print(f"  {'Variable':<40} {'Orig':>8} {'Clust-SE':>8} "
              f"{'Bt med':>8} {'Bt SE':>8} {'CI lo':>8} {'CI hi':>8} {'Sign%':>6}")
        print("  " + "-" * 93)

        summary_rows = []
        for var in var_list:
            coef_col = f"coef__{var}"
            if coef_col not in boot_df.columns:
                continue
            vals = boot_df[coef_col].dropna()
            if len(vals) == 0:
                continue

            orig_c = orig_model.params.get(var, np.nan)
            orig_se = orig_model.bse.get(var, np.nan)
            bt_med = vals.median()
            bt_se = vals.std()
            ci_lo = vals.quantile(lo)
            ci_hi = vals.quantile(hi)
            sign_pct = (vals > 0).mean() if orig_c > 0 else (vals < 0).mean() if orig_c < 0 else np.nan
            sig = "*" if (ci_lo > 0 or ci_hi < 0) else " "

            print(f"  {var:<40} {orig_c:>+8.4f} {orig_se:>8.4f} "
                  f"{bt_med:>+8.4f} {bt_se:>8.4f} {ci_lo:>+8.4f} {ci_hi:>+8.4f} "
                  f"{sign_pct:>5.0%} {sig}")

            summary_rows.append({
                "Model": label, "Variable": var,
                "Original_Coef": orig_c, "Clustered_SE": orig_se,
                "Boot_Median": bt_med, "Boot_SE": bt_se,
                "Boot_CI_lo": ci_lo, "Boot_CI_hi": ci_hi,
                "Sign_Stability": sign_pct,
            })

        r2_vals = boot_df["r2"].dropna()
        print(f"\n  R2: original={orig_model.rsquared:.4f}  "
              f"boot median={r2_vals.median():.4f}  "
              f"CI=[{r2_vals.quantile(lo):.4f}, {r2_vals.quantile(hi):.4f}]")

    # Save combined summary
    all_summary = []
    for label, boot_df, orig_model, var_list in [
        ("3a", df_3a, m3a, ["const"] + reg3_input + INTERACT_VARS),
        ("3b", df_3b, m3b, ["const"] + reg3_input + INTERACT_VARS + ["ECI_lag1"]),
    ]:
        for var in var_list:
            coef_col = f"coef__{var}"
            if coef_col not in boot_df.columns:
                continue
            vals = boot_df[coef_col].dropna()
            if len(vals) == 0:
                continue
            orig_c = orig_model.params.get(var, np.nan)
            all_summary.append({
                "Model": label, "Variable": var,
                "Original_Coef": orig_c,
                "Clustered_SE": orig_model.bse.get(var, np.nan),
                "Boot_Median": vals.median(), "Boot_SE": vals.std(),
                "Boot_CI_lo": vals.quantile(lo), "Boot_CI_hi": vals.quantile(hi),
                "Sign_Stability": (vals > 0).mean() if orig_c > 0 else (vals < 0).mean(),
            })
    pd.DataFrame(all_summary).to_csv(
        os.path.join(BOOT_DIR, "nb6_boot_summary.csv"), index=False
    )
    print(f"\n  Saved: {BOOT_DIR}/nb6_boot_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: BOOTSTRAP ML (NB5)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_ml_df(filepath):
    master = pd.read_csv(filepath)
    df = master[master["Country Code"].isin(INCLUDE)].copy()
    if len(df) == 0:
        return None

    df["Total_Production_Value_Per_Capita"] = (
        df["Total_Production_Value"] / df["Population"]
    )
    df = df.sort_values(["Country Code", "Year"]).reset_index(drop=True)

    # ── Rolling macro controls (NB5 cell 4) ──────────────────────────────
    df["Inflation_roll5"] = (
        df.groupby("Country Code")["Inflation, consumer prices (annual %)"]
          .transform(lambda x: x.rolling(5, min_periods=3).mean())
    )
    df["RealRate_roll5"] = (
        df.groupby("Country Code")["Real interest rate (%)"]
          .transform(lambda x: x.rolling(5, min_periods=3).mean())
    )

    # ── Resource concentration HHI (NB5 cell 4) ─────────────────────────
    rents_cols = [
        "Oil rents (% of GDP)", "Natural gas rents (% of GDP)",
        "Mineral rents (% of GDP)", "Forestry rents (% of GDP)",
    ]
    total_rents = df["Total natural resources rents (% of GDP)"].replace(0, np.nan)
    df["Resource_HHI"] = sum(
        (df[col] / total_rents) ** 2 for col in rents_cols
    )

    # ── ECI targets ──────────────────────────────────────────────────────
    df["L1_ECI"] = df.groupby("Country Code")["Economic Complexity Index"].shift(1)
    df["ECI_delta"] = df["Economic Complexity Index"] - df["L1_ECI"]
    df = df.dropna(subset=["L1_ECI", "Economic Complexity Index", "ECI_delta"])

    # ── Log transforms (before interactions) ─────────────────────────────
    log_cols = [
        "Human capital index", "Total_Production_Value_Per_Capita",
        "Gross fixed capital formation, all, Constant prices, Percent of GDP",
        "Government revenue", "Use of IMF credit (DOD, current US$)",
        "Forestry rents (% of GDP)",
    ]
    df[log_cols] = np.log1p(df[log_cols]).replace([np.inf, -np.inf], np.nan)

    # ── Interaction terms (grand-mean centred) ───────────────────────────
    hci_mean = df["Human capital index"].mean()
    prod_mean = df["Total_Production_Value_Per_Capita"].mean()
    rol_mean = df["Rule of law index"].mean()

    df["HCI_x_ProductionValue"] = (
        (df["Human capital index"] - hci_mean)
        * (df["Total_Production_Value_Per_Capita"] - prod_mean)
    )
    df["RuleOfLaw_x_ProductionValue"] = (
        (df["Rule of law index"] - rol_mean)
        * (df["Total_Production_Value_Per_Capita"] - prod_mean)
    )

    df = df.dropna(subset=all_features)
    return df if len(df) >= 50 else None


def fit_ml_models(df):
    train_df = df[df["Year"] <= TRAIN_END]
    test_df = df[df["Year"] >= TEST_START]
    if len(train_df) < 30 or len(test_df) < 10:
        return None

    y_train = train_df["Economic Complexity Index"].values
    y_test = test_df["Economic Complexity Index"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[all_features].values)
    X_test = scaler.transform(test_df[all_features].values)

    tscv = PanelTemporalCV(train_df["Year"].values, n_splits=5, gap=1, min_train_years=8)

    lasso = LassoCV(cv=tscv, random_state=42, max_iter=10000).fit(X_train, y_train)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=tscv).fit(X_train, y_train)
    elastic = ElasticNetCV(
        l1_ratio=[0.5], cv=tscv, random_state=42, max_iter=10000
    ).fit(X_train, y_train)
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)

    models = {"LASSO": lasso, "Ridge": ridge, "Elastic Net": elastic, "Random Forest": rf}

    metrics = {"n_train": len(train_df), "n_test": len(test_df),
               "n_countries": df["Country Code"].nunique()}
    coefs = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        metrics[f"{name}_test_r2"] = r2_score(y_test, pred)
        metrics[f"{name}_test_rmse"] = np.sqrt(mean_squared_error(y_test, pred))
        if hasattr(model, "coef_"):
            for feat, c in zip(all_features, model.coef_):
                coefs[f"{name}__{feat}"] = c

    importances = dict(zip(all_features, rf.feature_importances_))
    lasso_sel = {feat: int(c != 0) for feat, c in zip(all_features, lasso.coef_)}

    return {"metrics": metrics, "coefs": coefs,
            "importances": importances, "lasso_selected": lasso_sel}


def run_phase3():
    print("\n" + "#" * 70)
    print("# PHASE 3: Bootstrap ML (NB5)")
    print("#" * 70)

    boot_files = sorted(glob.glob(os.path.join(BOOT_DIR, "Master_b*.csv")))
    if not boot_files:
        print("ERROR: No bootstrap files found.")
        return
    print(f"Found {len(boot_files)} bootstrap files. This phase takes ~30-45 min.\n")

    all_metrics, all_coefs, all_imp, all_sel = [], [], [], []
    skipped = 0
    t0 = time.time()

    for i, fpath in enumerate(boot_files, 1):
        df_b = prepare_ml_df(fpath)
        if df_b is None:
            skipped += 1
            continue

        result = fit_ml_models(df_b)
        if result is None:
            skipped += 1
            continue

        result["metrics"]["b"] = i
        all_metrics.append(result["metrics"])
        result["coefs"]["b"] = i
        all_coefs.append(result["coefs"])
        result["importances"]["b"] = i
        all_imp.append(result["importances"])
        result["lasso_selected"]["b"] = i
        all_sel.append(result["lasso_selected"])

        if i % 25 == 0 or i == 1:
            elapsed = time.time() - t0
            eta = (len(boot_files) - i) / (i / elapsed) if i > 0 else 0
            print(f"  b={i:>3d}/{len(boot_files)}  |  {elapsed:.0f}s elapsed  |  ETA {eta:.0f}s")

    # Save
    metrics_df = pd.DataFrame(all_metrics)
    coefs_df = pd.DataFrame(all_coefs)
    imp_df = pd.DataFrame(all_imp)
    sel_df = pd.DataFrame(all_sel)

    metrics_df.to_csv(os.path.join(BOOT_DIR, "nb5_boot_metrics.csv"), index=False)
    coefs_df.to_csv(os.path.join(BOOT_DIR, "nb5_boot_coefs.csv"), index=False)
    imp_df.to_csv(os.path.join(BOOT_DIR, "nb5_boot_importances.csv"), index=False)
    sel_df.to_csv(os.path.join(BOOT_DIR, "nb5_boot_lasso_selection.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n  {len(boot_files) - skipped} successful, {skipped} skipped in {elapsed:.1f}s")

    # ── Report ────────────────────────────────────────────────────────────
    lo, hi = ALPHA / 2, 1 - ALPHA / 2

    print(f"\n{'=' * 70}")
    print("  BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print(f"{'=' * 70}")

    print("\n  Test R2:")
    for name in ["LASSO", "Ridge", "Elastic Net", "Random Forest"]:
        col = f"{name}_test_r2"
        vals = metrics_df[col].dropna()
        print(f"    {name:<16}  median={vals.median():.4f}  "
              f"CI=[{vals.quantile(lo):.4f}, {vals.quantile(hi):.4f}]  "
              f"SD={vals.std():.4f}")

    print("\n  RF Feature Importance (top 10 by median):")
    feat_cols = [c for c in imp_df.columns if c != "b"]
    medians = imp_df[feat_cols].median().sort_values(ascending=False)
    for feat in medians.head(10).index:
        vals = imp_df[feat].dropna()
        print(f"    {feat[:42]:<44}  median={vals.median():.4f}  "
              f"CI=[{vals.quantile(lo):.4f}, {vals.quantile(hi):.4f}]")

    print("\n  LASSO Selection Frequency:")
    feat_cols_sel = [c for c in sel_df.columns if c != "b"]
    sel_freq = sel_df[feat_cols_sel].mean().sort_values(ascending=False)
    for feat, freq in sel_freq.items():
        tag = " STABLE" if freq > 0.90 else " unstable" if freq < 0.50 else ""
        print(f"    {feat[:42]:<44}  {freq:.0%}{tag}")

    print(f"\n  Saved: {BOOT_DIR}/nb5_boot_*.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  CAPSTONE BOOTSTRAP PIPELINE")
    print(f"  B={B}  |  seed={RANDOM_SEED}  |  KNN k={KNN_NEIGHBORS}")
    print(f"  Working dir: {os.getcwd()}")
    print("=" * 70)

    t_total = time.time()

    if not SKIP_NB3:
        run_phase1()
    else:
        n_existing = len(glob.glob(os.path.join(BOOT_DIR, "Master_b*.csv")))
        print(f"\nPhase 1 skipped. {n_existing} bootstrap files found in {BOOT_DIR}/")

    run_phase2()

    if not SKIP_NB5:
        run_phase3()
    else:
        print("\nPhase 3 (NB5) skipped.")

    total = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"  ALL DONE  |  Total time: {total / 60:.1f} minutes")
    print("=" * 70)
