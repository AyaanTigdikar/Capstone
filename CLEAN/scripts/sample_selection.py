"""
sample_selection.py
───────────────────
Data-driven country selection for the Moody's capstone panel.

Two stages, called at different points in the NB3 pipeline:

  STAGE 1 -- ELIGIBILITY (before imputation):
    1. Extractive rents >= threshold in 1995 (Gulf states guaranteed)
    2. Not classified as high-income by the World Bank in 1995 (Gulf exempt)
    3. ECI (the dependent variable) must have >= 1 non-missing observation

  STAGE 2 -- DATA QUALITY (after imputation):
    4. Raw missingness <= ceiling (computed on pre-imputation data)
    5. Number of fully missing variables <= ceiling (pre-imputation)
    6. KNN imputation reliance <= ceiling (post-imputation)

Gulf states can be dropped by data-quality filters (stage 2).

Usage inside NB3:
─────────────────
    from sample_selection import apply_eligibility_filters, apply_quality_filters

    # Cell 23 -- after Part A produces nrpa and feasible_countries:
    cmaster, elig_diag = apply_eligibility_filters(
        master_path="intermediary/master_data_wide.csv",
        nr_agg=nrpa,
        feasible=feasible_countries,
    )
    cmaster_pre = cmaster.copy()   # snapshot for quality diagnostics

    # ... run interpolation + KNN ...

    # Cell 30 -- after imputation:
    cmaster_imp, quality_diag = apply_quality_filters(
        cmaster_pre, cmaster_imp, knn_mask,
        max_missingness=15.0,
        max_fully_missing=5,
        max_knn_pct=10.0,
    )

All thresholds can be overridden via keyword arguments.
"""

import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_MIN_RENTS         = 1.0    # extractive rents as % of GDP, 1995
DEFAULT_MAX_MISSINGNESS   = 15.0   # % of total cells (pre-imputation)
DEFAULT_MAX_KNN           = 10.0   # % of cells filled by KNN
DEFAULT_MAX_FULLY_MISSING = 5      # variables with zero observations for a country

# ── Gulf states: guaranteed past eligibility filters, subject to data quality ─
GULF = {"ARE", "BHR", "KWT", "OMN", "QAT", "SAU", "IRQ", "IRN", "YEM"}

# ── Non-sovereign territories (always excluded) ──────────────────────────────
NOT_COUNTRIES = {
    "HKG", "MAC", "PRI", "VIR", "GUM", "ASM", "CYM", "BMU",
    "GRL", "MAF", "SXM", "CUW", "ABW", "FRO", "MNP", "PYF",
}

# ── Manual exclusions ─────────────────────────────────────────────────────────
# Large diversified economies where population size dilutes per-capita
# production to near-zero, distorting PCA-based clustering. These countries
# pass the ≥1% extractive rents threshold but are not resource-dependent
# in the sense this study is designed to explain.
MANUAL_EXCLUDE = {"CHN", "IND"}

# ── 1995 World Bank high-income economies (ISO3) ─────────────────────────────
# Gulf HICs are included here for completeness but are exempted by the Gulf
# guarantee in the eligibility filter logic.
HIC_1995 = {
    # OECD high-income
    "AUS", "AUT", "BEL", "CAN", "CHE", "DEU", "DNK", "ESP", "FIN",
    "FRA", "GBR", "GRC", "IRL", "ISL", "ITA", "JPN", "LUX", "NLD",
    "NOR", "NZL", "PRT", "SWE", "USA",
    # Non-OECD high-income
    "ARE", "BHR", "BHS", "BRB", "BRN", "CYP", "GNQ", "ISR", "KWT",
    "MLT", "QAT", "SAU", "SGP", "SVN", "TTO",
    # Territories classified as high-income (overlap with NOT_COUNTRIES)
    "HKG", "MAC", "PRI", "BMU", "CYM", "GRL", "VIR", "GUM",
}

# ── Variables retained for the econometric and ML pipeline ────────────────────
KEEP_VARS = [
    "Country Code", "Country Name", "Year",
    # Governance
    "Access to electricity (% of population)",
    "Adjusted savings: gross savings (% of GNI)",
    "Agriculture",
    "Capital depreciation rate",
    "Clientelism index",
    "Death rates, crude per 1000 people",
    "Domestic credit to private sector (% of GDP)",
    "Economic Complexity Index",
    "GDP per capita (constant prices, PPP)",
    "Government revenue",
    "Gross fixed capital formation, all, Constant prices, Percent of GDP",
    "Human capital index",
    "Industry",
    "Inflation, consumer prices (annual %)",
    "Landlocked",
    "Lending interest rate (%)",
    "Life expectancy at birth, total (years)",
    "Manufacturing",
    "Mineral rents (% of GDP)",
    "Mobile cellular subscriptions (per 100 people)",
    "Natural gas rents (% of GDP)",
    "Oil rents (% of GDP)",
    "Forest rents (% of GDP)",
    "Coal rents (% of GDP)",
    "Extractive_NR_Rents",
    "Political corruption index",
    "Political stability \u2014 estimate",
    "Primary net lending, General government, Percent of GDP",
    "Property rights",
    "Real interest rate (%)",
    "Rule of law index",
    "Services",
    "Share of consumption in GDP",
    "Share of government spending in GDP",
    "Share of investment in GDP",
    "Total natural resources rents (% of GDP)",
    "Trade (% of GDP)",
    "Urban population (% of total population)",
    "Use of IMF credit (DOD, current US$)",
    # NR variables (from Part A)
    "Total_Production",
    "Total_Reserves",
    "Total_Production_Value",
    "Total_Reserves_Value",
    "Hydrocarbons_Dominant",
    "Subsoil_Metals_Dominant",
    "Precious_Metals_Dominant",
]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: ELIGIBILITY (before imputation)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_eligibility_filters(
    master_path,
    nr_agg,
    feasible,
    min_rents=DEFAULT_MIN_RENTS,
    keep_vars=None,
):
    """
    Apply eligibility filters 1-3 (rents, HIC exclusion, ECI requirement).
    No data-quality filtering here; that happens after imputation.

    Parameters
    ----------
    master_path : str
        Path to master_data_wide.csv (NB1 output).
    nr_agg : DataFrame
        Country-year NR aggregates from NB3 Part A.
    feasible : set
        Country codes that have NR production data (from Part A).
    min_rents : float
        Minimum extractive rents as % of GDP in 1995.
    keep_vars : list or None
        Variable list; defaults to KEEP_VARS.

    Returns
    -------
    cmaster : DataFrame
        Eligible panel (all countries passing filters 1-3), ready for imputation.
    diagnostics : DataFrame
        One row per candidate country with filter outcomes.
    """
    if keep_vars is None:
        keep_vars = KEEP_VARS

    feasible = set(feasible)

    # ── Load and merge ────────────────────────────────────────────────────────
    df = pd.read_csv(master_path)
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)

    df = pd.merge(
        df,
        nr_agg.drop(columns=["Country"], errors="ignore"),
        on=["Country Code", "Year"],
        how="left",
    )

    # ── Extractive rents ──────────────────────────────────────────────────────
    RENT_COL   = "Total natural resources rents (% of GDP)"
    FOREST_COL = "Forest rents (% of GDP)"

    df["Extractive_NR_Rents"] = (
        df[RENT_COL].fillna(0) - df[FOREST_COL].fillna(0)
    ).clip(lower=0)

    rents_1995 = df[df["Year"] == 1995].dropna(subset=[RENT_COL])
    rent_pass = set(
        rents_1995[rents_1995["Extractive_NR_Rents"] >= min_rents]["Country Code"]
    )

    # ── Build candidate pool ──────────────────────────────────────────────────
    candidates = feasible & (rent_pass | GULF)
    candidates -= NOT_COUNTRIES
    candidates -= MANUAL_EXCLUDE

    print("=" * 70)
    print("SAMPLE SELECTION -- Stage 1: Eligibility filters")
    print("=" * 70)
    print(f"\nFeasible countries (have NR data):       {len(feasible)}")
    print(f"Pass rent threshold (>= {min_rents}%):         {len(rent_pass)}")
    print(f"Gulf states (guaranteed for eligibility): {len(GULF)}")
    print(f"Candidate pool after rent + Gulf:         {len(candidates)}")

    # ── Filter 2: exclude 1995 HICs (Gulf exempt) ─────────────────────────────
    hic_drop = candidates & HIC_1995 - GULF
    candidates -= hic_drop
    if hic_drop:
        print(f"\nDropped as 1995 high-income (Gulf exempt): {sorted(hic_drop)}")
    print(f"After HIC exclusion:                      {len(candidates)}")

    # ── Select variables and restrict to candidates ───────────────────────────
    available = [v for v in keep_vars if v in df.columns]
    missing_vars = [v for v in keep_vars if v not in df.columns]
    if missing_vars:
        print(f"\nWarning: {len(missing_vars)} requested variables not in data:")
        for v in missing_vars:
            print(f"  - {v}")

    df = df[available]
    df = df[df["Country Code"].isin(candidates)]

    # Structural zeros
    for col in ["Use of IMF credit (DOD, current US$)",
                "Death rates, crude per 1000 people"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── Compute per-country diagnostics ───────────────────────────────────────
    id_cols = ["Country Code", "Country Name", "Year"]
    data_cols = [c for c in df.columns if c not in id_cols]

    diag_rows = []
    for code in sorted(df["Country Code"].unique()):
        sub = df[df["Country Code"] == code]
        name = sub["Country Name"].iloc[0]

        total_cells = len(sub) * len(data_cols)
        missing_cells = sub[data_cols].isna().sum().sum()
        pct_missing = 100 * missing_cells / total_cells if total_cells > 0 else 100
        fully_missing = sum(sub[col].isna().all() for col in data_cols)

        eci_col = "Economic Complexity Index"
        eci_obs = sub[eci_col].notna().sum() if eci_col in sub.columns else 0

        rent_row = rents_1995[rents_1995["Country Code"] == code]
        rent_val = rent_row["Extractive_NR_Rents"].values[0] if len(rent_row) > 0 else np.nan

        diag_rows.append({
            "Country Code": code,
            "Country Name": name,
            "Gulf": code in GULF,
            "Extr. Rents 1995": round(rent_val, 2) if not np.isnan(rent_val) else np.nan,
            "Raw Miss. (%)": round(pct_missing, 1),
            "Fully Missing Vars": fully_missing,
            "ECI Obs": eci_obs,
            "Years": sub["Year"].nunique(),
            "Pass Rents": code in rent_pass or code in GULF,
            "Pass HIC": code not in (HIC_1995 - GULF),
        })

    diagnostics = pd.DataFrame(diag_rows)

    # ── Filter 3: require ECI ─────────────────────────────────────────────────
    no_eci = set(diagnostics[diagnostics["ECI Obs"] == 0]["Country Code"])
    if no_eci:
        print(f"\nDropped for zero ECI observations:         {sorted(no_eci)}")
    diagnostics["Pass ECI"] = diagnostics["ECI Obs"] > 0

    # ── Remove only ECI-less countries; keep everyone else for imputation ─────
    survivors = candidates - no_eci
    df = df[df["Country Code"].isin(survivors)]

    print(f"\nEligible sample (entering imputation):     {len(survivors)} countries")
    print(f"Panel dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing cells (pre-imputation): {df[data_cols].isna().sum().sum():,}")

    # ── Save intermediate outputs ─────────────────────────────────────────────
    diagnostics.to_csv("intermediary/sample_diagnostics.csv", index=False)
    print(f"\nSaved: intermediary/sample_diagnostics.csv")

    return df.copy().reset_index(drop=True), diagnostics


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: DATA QUALITY (after imputation)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_quality_filters(
    cmaster_pre,
    cmaster_imp,
    knn_mask,
    max_missingness=DEFAULT_MAX_MISSINGNESS,
    max_fully_missing=DEFAULT_MAX_FULLY_MISSING,
    max_knn_pct=DEFAULT_MAX_KNN,
):
    """
    Apply data-quality filters 4-6 after imputation.

    Filters 4-5 use pre-imputation data (cmaster_pre) to compute raw
    missingness and fully missing variable counts. Filter 6 uses the
    KNN mask from imputation to compute KNN reliance.

    Parameters
    ----------
    cmaster_pre : DataFrame
        Pre-imputation panel (snapshot from before interpolation/KNN).
    cmaster_imp : DataFrame
        Post-imputation panel.
    knn_mask : DataFrame
        Boolean mask where True = cell was filled by KNN.
    max_missingness : float
        Maximum share of missing cells (%) in pre-imputation data.
    max_fully_missing : int
        Maximum number of variables with zero observations for a country.
    max_knn_pct : float
        Maximum share of cells (%) filled by KNN.

    Returns
    -------
    cmaster_final : DataFrame
        Panel with countries failing quality filters removed.
    quality_diag : DataFrame
        Per-country quality diagnostics.
    """
    id_cols = ["Country Code", "Country Name", "Year"]
    data_cols = [c for c in cmaster_pre.columns if c not in id_cols]
    numeric_cols = list(knn_mask.columns)

    print("=" * 70)
    print("SAMPLE SELECTION -- Stage 2: Data quality filters (post-imputation)")
    print("=" * 70)

    # ── Compute per-country diagnostics ───────────────────────────────────────
    diag_rows = []
    for code in sorted(cmaster_pre["Country Code"].unique()):
        # Pre-imputation stats (filters 4-5)
        sub_pre = cmaster_pre[cmaster_pre["Country Code"] == code]
        name = sub_pre["Country Name"].iloc[0]
        total_cells = len(sub_pre) * len(data_cols)
        missing_cells = sub_pre[data_cols].isna().sum().sum()
        pct_missing = 100 * missing_cells / total_cells if total_cells > 0 else 100
        fully_missing = sum(sub_pre[col].isna().all() for col in data_cols)

        # KNN reliance (filter 6)
        mask_rows = cmaster_imp["Country Code"] == code
        knn_cells = knn_mask.loc[mask_rows].sum().sum()
        total_knn_cells = mask_rows.sum() * len(numeric_cols)
        knn_pct = 100 * knn_cells / total_knn_cells if total_knn_cells > 0 else 0

        diag_rows.append({
            "Country Code": code,
            "Country Name": name,
            "Gulf": code in GULF,
            "Raw Miss. (%)": round(pct_missing, 1),
            "Fully Missing Vars": fully_missing,
            "KNN Reliance (%)": round(knn_pct, 2),
            "Pass Missingness": pct_missing <= max_missingness,
            "Pass FullyMissing": fully_missing <= max_fully_missing,
            "Pass KNN": knn_pct <= max_knn_pct,
        })

    quality_diag = pd.DataFrame(diag_rows)
    quality_diag["Pass All Quality"] = (
        quality_diag["Pass Missingness"]
        & quality_diag["Pass FullyMissing"]
        & quality_diag["Pass KNN"]
    )

    # ── Report ────────────────────────────────────────────────────────────────
    n_before = len(quality_diag)

    dropped_miss = quality_diag[~quality_diag["Pass Missingness"]]["Country Code"].tolist()
    dropped_fmv = quality_diag[
        quality_diag["Pass Missingness"] & ~quality_diag["Pass FullyMissing"]
    ]["Country Code"].tolist()
    dropped_knn = quality_diag[
        quality_diag["Pass Missingness"]
        & quality_diag["Pass FullyMissing"]
        & ~quality_diag["Pass KNN"]
    ]["Country Code"].tolist()

    all_dropped = set(quality_diag[~quality_diag["Pass All Quality"]]["Country Code"])

    print(f"\nCountries entering quality filters:  {n_before}")
    print(f"Thresholds: missingness <= {max_missingness}%, "
          f"fully missing vars <= {max_fully_missing}, "
          f"KNN reliance <= {max_knn_pct}%")
    print()

    # Print per-country table
    print(f"{'Country':<8} {'Name':<30s} {'Miss%':>6s} {'FMV':>4s} "
          f"{'KNN%':>6s}  Status")
    print("-" * 70)
    for _, row in quality_diag.sort_values("Raw Miss. (%)", ascending=False).iterrows():
        status = "ok" if row["Pass All Quality"] else "DROPPED"
        reasons = []
        if not row["Pass Missingness"]:
            reasons.append("miss")
        if not row["Pass FullyMissing"]:
            reasons.append("fmv")
        if not row["Pass KNN"]:
            reasons.append("knn")
        reason_str = f" ({','.join(reasons)})" if reasons else ""
        print(
            f"{row['Country Code']:<8} "
            f"{str(row['Country Name'])[:30]:<30s} "
            f"{row['Raw Miss. (%)']:>5.1f}% "
            f"{row['Fully Missing Vars']:>4d} "
            f"{row['KNN Reliance (%)']:>5.1f}%  "
            f"{status}{reason_str}"
        )

    if all_dropped:
        print(f"\nDropped countries ({len(all_dropped)}):")
        if dropped_miss:
            print(f"  Raw missingness > {max_missingness}%:    {sorted(dropped_miss)}")
        if dropped_fmv:
            print(f"  Fully missing vars > {max_fully_missing}:      {sorted(dropped_fmv)}")
        if dropped_knn:
            print(f"  KNN reliance > {max_knn_pct}%:        {sorted(dropped_knn)}")
    else:
        print(f"\nNo countries dropped by quality filters.")

    # ── Apply filter ──────────────────────────────────────────────────────────
    survivors = set(quality_diag[quality_diag["Pass All Quality"]]["Country Code"])
    cmaster_final = cmaster_imp[
        cmaster_imp["Country Code"].isin(survivors)
    ].reset_index(drop=True)

    n_after = cmaster_final["Country Code"].nunique()
    print(f"\nFinal sample: {n_after} countries, "
          f"{cmaster_final.shape[0]:,} rows x {cmaster_final.shape[1]} columns")

    # Gulf states status
    gulf_in = GULF & survivors
    gulf_out = GULF - survivors
    if gulf_in:
        print(f"Gulf states in sample:   {sorted(gulf_in)}")
    if gulf_out:
        print(f"Gulf states dropped:     {sorted(gulf_out)} (data quality)")

    # ── Save ──────────────────────────────────────────────────────────────────
    final_codes = sorted(cmaster_final["Country Code"].unique().tolist())
    pd.DataFrame({"Country Code": final_codes}).to_csv(
        "intermediary/sample_countries_final.csv", index=False
    )
    quality_diag.sort_values("Raw Miss. (%)", ascending=False).to_csv(
        "intermediary/quality_diagnostics.csv", index=False
    )
    print(f"\nSaved: intermediary/sample_countries_final.csv")
    print(f"Saved: intermediary/quality_diagnostics.csv")

    return cmaster_final, quality_diag


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_selection_summary(elig_diag, quality_diag=None):
    """
    Print a consolidated summary of all filters.
    Call after both stages are complete.
    """
    print("\n" + "=" * 70)
    print("CONSOLIDATED SAMPLE SELECTION SUMMARY")
    print("=" * 70)

    n_total = len(elig_diag)
    n_pass_eci = elig_diag["Pass ECI"].sum()

    print(f"\n  Stage 1 -- Eligibility:")
    print(f"    Candidates (feasible + rent/Gulf):  {n_total}")
    print(f"    Pass ECI requirement:               {n_pass_eci}")
    print(f"    Entering imputation:                {n_pass_eci}")

    if quality_diag is not None:
        n_entering = len(quality_diag)
        n_pass_miss = quality_diag["Pass Missingness"].sum()
        n_pass_fmv = quality_diag["Pass FullyMissing"].sum()
        n_pass_knn = quality_diag["Pass KNN"].sum()
        n_final = quality_diag["Pass All Quality"].sum()

        print(f"\n  Stage 2 -- Data quality (post-imputation):")
        print(f"    Countries entering filters:         {n_entering}")
        print(f"    Pass raw missingness:               {n_pass_miss}")
        print(f"    Pass fully missing vars:            {n_pass_fmv}")
        print(f"    Pass KNN reliance:                  {n_pass_knn}")
        print(f"\n  FINAL SAMPLE: {n_final} countries")

        # Dropped by reason
        dropped_eci = elig_diag[~elig_diag["Pass ECI"]]["Country Code"].tolist()
        dropped_quality = quality_diag[~quality_diag["Pass All Quality"]]["Country Code"].tolist()

        if dropped_eci or dropped_quality:
            print(f"\n  All dropped countries:")
            if dropped_eci:
                print(f"    No ECI data:           {', '.join(sorted(dropped_eci))}")
            if dropped_quality:
                print(f"    Data quality:          {', '.join(sorted(dropped_quality))}")
    else:
        print(f"\n  (Stage 2 not yet applied; run apply_quality_filters after imputation)")
