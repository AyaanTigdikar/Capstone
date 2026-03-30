# Capstone Code Submission: Resource Curse & Economic Complexity

**Project:** Does Natural Resource Dependence Hinder Economic Complexity?
**Sample:** 54-country panel, 1995–2019
**Submitted:** March 2026

---

## Folder Structure

```
client_submission/
├── README.md
├── main_analysis/
│   ├── 0_NR_extraction_FINAL.ipynb
│   ├── 1_cleaning_master_data_FINAL.ipynb
│   ├── 2_Imputing_FINAL.ipynb
│   ├── 3_Clustering_FINAL.ipynb
│   ├── 4_ML_FINAL.ipynb
│   ├── 5_Regressions_Unified.ipynb
│   ├── 6_Viz_Descriptive_Clustering.ipynb
│   ├── 7_Viz_ML.ipynb
│   ├── scripts/
│   │   ├── standardize_country.py   ← country name harmonisation (used by NB0/NB1)
│   │   └── viz_utils.py             ← shared Plotly styling helpers
│   ├── rawdata/
│   │   ├── base_dataset.xlsx
│   │   ├── Statistical Review of World Energy Narrow File-1.csv
│   │   ├── Oil Gas Coal Uranium Price.xlsx
│   │   ├── PopulationWDI.csv
│   │   ├── production_values_w_prices-EM.csv
│   │   └── Minerals/
│   └── intermediary/                ← empty; populated as notebooks run
├── chile_analysis/
│   ├── Chile_A_Setup_Production.ipynb
│   ├── Chile_B_Supply_Chain.ipynb
│   ├── Chile_C_Validation_Analysis.ipynb
│   ├── scripts/
│   │   ├── chile_supply_chain_map.py
│   │   ├── chile_visualisations.py
│   │   ├── commodity_prices_2024.py
│   │   ├── pipeline_constants.py
│   │   └── pipeline_utils.py
│   ├── inputs/
│   ├── intermediary/
│   └── outputs/
```

> **Note on NB0 and NB1:** These come from `v2/` (the data extraction/cleaning layer that feeds the v1 analysis). The `v1/` folder starts at NB3 because it assumes a pre-built master dataset. Run NB0 → NB1 first if building from raw sources.

---

## Environment

- **Python:** 3.10
- **Core packages:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `statsmodels`, `plotly`, `geopandas`, `scipy`, `linearmodels`, `wbgapi`

```bash
pip install pandas numpy scikit-learn xgboost statsmodels plotly geopandas scipy linearmodels wbgapi
```

---

## Main Analysis - Execution Order

Run notebooks in numerical order. Each reads from intermediary outputs of the previous step.

---

### NB0 - `0_NR_extraction_FINAL.ipynb` - Natural Resource Data Extraction

**What it does:**
Pulls raw natural resource production, consumption, reserves, and prices from five sources and merges them into a single long-format panel.

- **Oil/gas/coal:** reads `Statistical Review of World Energy Narrow File-1.csv` (variables: `oilprod_kbd`, `gasprod_bcm`, `coalprod_mt`); converts units (oil kbd → bbl/yr ×365,000; gas bcm → MMBtu ×35.315M×1.037; coal Mt → tonnes ×1M)
- **Minerals:** reads production/reserves tables from `base_dataset.xlsx` (EI P-R sheets: Cobalt, Lithium, Graphite, etc.) and OWID files; USGS price data from `ds140-*.xlsx`
- **Prices:** loads `Oil Gas Coal Uranium Price.xlsx` and `natural resource prices.xlsx`; applies priority ranking ConsolidatedPrices > EI > OWID > GasPrice (fills gaps only, does not overwrite)
- Country names standardised to ISO3 via custom `add_iso3()` mapping; decade-range rows skipped via regex; non-resource rows dropped

**Inputs:** `rawdata/Statistical Review of World Energy Narrow File-1.csv`, `rawdata/base_dataset.xlsx`, `rawdata/Minerals/`, `rawdata/Oil Gas Coal Uranium Price.xlsx`
**Output:** `intermediary/natural_resources_production_values.csv` - long format (Country, Year, Resource, Metric, Value) covering 20+ resources, 1990–2024

---

### NB1 - `1_cleaning_master_data_FINAL.ipynb` - Master Dataset Construction

**What it does:**
Downloads and caches economic indicators from six data sources via API, then merges them into a country × year panel.

- **World Bank:** 24 variables (NR rents, GFCF, trade, employment, infrastructure) via `wbgapi`
- **IMF WEO:** GDP per capita, government revenue, debt, structural balance
- **IMF ICSD:** 3 GFCF components (P51G: government, private, PPP); primary net lending from FAD_FM dataset
- **ECI:** Economic Complexity Index from GitHub dataset
- **V-Dem:** 11 governance indices (polyarchy, corruption, rule of law, etc.)
- **PWT 11.0:** human capital, TFP, savings shares, depreciation
- **CEPII:** landlocked indicator
- All downloads cached to `intermediary/cache/`; `FORCE_REFRESH=True` to re-download. API calls use retry logic (max 3 attempts, 2-second pause).
- Year filter: 1995–2019; aggregate regions excluded; GFCF components summed across sectors

**Inputs:** API calls (World Bank, IMF, V-Dem, PWT, CEPII); local cache on repeat runs
**Outputs:** `intermediary/master_data_long.csv`, `intermediary/master_data_wide.csv`

---

### NB2 - `2_Imputing_FINAL.ipynb` - Missing Value Imputation

**What it does:**
Fills gaps in sparse mineral production data using a two-pass strategy, then integrates NR data with the master economic panel.

**Pass 1 - share-based backfilling:**
- For each mineral, finds the earliest year where country coverage exceeds 70% of global production
- Extracts each country's share in that threshold year; backfills prior years using constant shares × global production
- Sensitivity tested at 60/70/80% thresholds; 70% chosen as baseline
- Example: Tin reaches 70% coverage in 1997 → 1995–1996 filled using 1997 country shares

**Pass 2 - KNN imputation:**
- `sklearn.KNeighborsRegressor(n_neighbors=5)` on remaining gaps
- Features scaled before fitting; imputation done within-country then cross-country

Additional steps:
- Removes Petroleum (overlaps Oil) and Platinum Group (coverage issues)
- Per-capita production calculated using `PopulationWDI.csv`
- Diagnostic output: `knn_reliance_by_country.csv` records what share of each country's data was imputed

**Inputs:** `intermediary/natural_resources_production_values.csv`, `intermediary/master_data_wide.csv`, `rawdata/PopulationWDI.csv`
**Outputs:** `intermediary/Master.csv`, `intermediary/NaturalResource.csv`, `intermediary/master_data_imputed.csv`, `intermediary/NRCleanData.csv`, `intermediary/knn_reliance_by_country.csv`

---

### NB3 - `3_Clustering_FINAL.ipynb` - Country Clustering

**What it does:**
Classifies 54 resource-rich developing countries into resource profile clusters using dimensionality reduction and K-Means.

**Pipeline (run three times: 1995 snapshot, 2019 snapshot, aggregated 1995/1999/2005):**
1. Pivot `NaturalResource.csv` to wide (rows = Country×Year, columns = resources)
2. Divide all resource columns by population → per-capita production
3. `log1p()` transform on per-capita values to compress outliers
4. `PCA(n_components=2)` - PC1 loads on oil/gas, PC2 loads on copper/gold/coal
5. `KMeans(k=5)` - k selected via silhouette score tested over k=2..8 on 1995 data

**Cluster labelling** (reference-country anchoring, not manual relabelling):
- SAU → Petrostates, NGA → Oil Exporters, RUS → Major Producers, CHL → Mining Exporters, COD → Forestry Intensive
- Each reference country's cluster ID is looked up and assigned the label; handles missing references gracefully

**Outputs:** `intermediary/clusters_k5_1995.csv`, `intermediary/clusters_k5_agg.csv`, `intermediary/clusters1995.csv`, `intermediary/clusters2019.csv`, `intermediary/clustersagg.csv`
Charts: PCA biplot, loadings heatmap, cluster choropleth, animated Rosling-style ECI vs log GDP (1995–2019)

---

### NB4 - `4_ML_FINAL.ipynb` - Machine Learning Models

**What it does:**
Trains four supervised models to predict Economic Complexity Index and its annual change from institutional, macroeconomic, and NR production variables.

**Feature engineering (19 base features → ~30 after transforms):**
- `log1p` applied to: human capital, per-capita production value, GFCF, government revenue, IMF credit, forestry rents
- 5-year rolling averages (min_periods=3) on inflation and real interest rate
- Resource HHI: Σ(rents_i / total_rents)² across oil/gas/mineral/forestry rents
- Interaction terms (mean-centered): HCI × Production Value, GFCF × Production Value
- L1_ECI: ECI lagged 1 year within country
- After all transforms + dropna: 1,100 obs across 51 countries

**Cross-validation:** `PanelTemporalCV` expanding window - train: Year ≤ 2014 (993 obs), test: Year ≥ 2015 (107 obs); 5 folds, min_train_years=8, gap=1 year

**Models (both ECI level and ΔECI targets):**
- `LassoCV(max_iter=10000)` - learned α ≈ 0.01–0.03
- `RidgeCV(alphas=logspace(-3, 3, 100))`
- `ElasticNetCV(l1_ratio=0.5, max_iter=10000)`
- `RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=10, oob_score=True)`

Test R²: Lasso ~0.75, RF ~0.65. Feature importance: min-max normalised |coefficients| for linear models; MDI for RF.

**Inputs:** `intermediary/Master.csv`, `intermediary/clustersagg.csv`
**Outputs:** model objects and results in-memory (used directly by NB7)

---

### NB5 - `5_Regressions_Unified.ipynb` - Econometric Regressions

**What it does:**
Estimates six pooled OLS specifications of ECI on institutional and NR production variables, all with country-clustered standard errors.

**Variable transforms:** `log1p` on HCI, GFCF, per-capita production value; all interaction terms mean-centred on the 54-country sample means; ECI shifted by (min + 1) then log for log-scale AR specs (ECI contains negatives).

**Model specifications:**
- **Model 1:** Full variable set (~20 vars: GDP pc, savings, agriculture/industry shares, employment, credit, inflation, life expectancy, etc.) - kitchen-sink baseline, clustered SE by country
- **Model 2:** AR baseline - only ECI_lag1 as predictor; R² ~0.97, establishes how much of ECI is just persistence
- **Model 3a:** Parsimonious - 7 vars (log_HCI, log_GFCF, political stability, rule of law, log_production_value_pc, trade, forestry rents) + 4 interaction terms (HCI×Production, GFCF×Production, HCI×Forestry, GFCF×Forestry); no ECI lag
- **Model 3b:** Model 3a + ECI_lag1 (controls for persistence, isolates within-country dynamics)
- **Model 3c:** Model 3b with all 7 regressors and interactions lagged one period instead of contemporaneous (addresses endogeneity)
- **Model 4a/4b:** Model 3b/3c + resource-type dummies (Hydrocarbons_Dominant, Subsoil_Metals_Dominant, Precious_Metals_Dominant) + electricity access
- **Model 5:** log(ECI) ~ log(ECI)_lag1 - log-scale AR persistence benchmark
- **Model 6:** ΔECI as dependent variable (first differences) with extended controls - tests whether changes in inputs drive changes in complexity

Also includes: residual diagnostics (QQ plots, Durbin-Watson), HTML regression tables exported for publication, and a robustness re-estimation of Models 3a/3b on the full (non-restricted) sample for comparison.

**Inputs:** `intermediary/Master.csv`, `intermediary/clusters1995.csv`
**Outputs:** `intermediary/high_resource_countries.csv` (54-country filtered panel with cluster labels, used by NB6–7); coefficient tables, VIF table, HTML regression tables, ECI distribution and trajectory charts printed inline

---

### NB6 - `6_Viz_Descriptive_Clustering.ipynb` - Descriptive & Clustering Charts

**What it does:**
Four publication charts for the data and clustering sections.

- **Sample map** (`chart000_sample_map.png`): choropleth highlighting the 54 countries in the sample
- **Correlation chart** (`02_correlations_with_eci.png`): horizontal bar plot of Pearson correlations between 31 variables and ECI, grouped into 5 categories (Resource Rents, Macro & Structure, Finance, Human Capital, Governance), sorted by magnitude
- **PCA loadings heatmap** (`26_pca_loadings_heatmap.png`): top 20 resources by loading magnitude; PC1 = oil/gas axis, PC2 = copper/gold/coal axis
- **PCA biplot** (`03_pca_biplot_1995.png`): 1995 country scatter on PC1–PC2 coloured by cluster, with top 10 resource loading vectors overlaid

**Inputs:** `intermediary/Master.csv`, `intermediary/NaturalResource.csv`, `intermediary/sample_countries_final.csv`, `intermediary/clustersagg.csv`
**Outputs:** 4 PNG files to `outputs/charts/descriptive/`

---

### NB7 - `7_Viz_ML.ipynb` - ML Charts

**What it does:**
Four charts summarising ML model performance and feature importance.

- **Feature importance consensus** (`07_ml_feature_importance_consensus.png`): top 15 features, min-max normalised importance averaged across Lasso/Ridge/Elastic Net; bars coloured by feature category
- **Standardised coefficients** (`08_ml_standardised_coefficients.png`): grouped bar chart comparing standardised coefficients across the three linear models (inputs StandardScaler-transformed before fitting)
- **Train/test R² comparison** (`09_ml_performance_comparison.png`): side-by-side bars for train and test R² for all four models, both ECI level and ΔECI targets
- **Random Forest importance** (`11_ml_random_forest_importance.png`): MDI importance for top features from RF fit

Same train/test split and model specs as NB4 (Year ≤ 2014 / ≥ 2015).

**Inputs:** `intermediary/Master.csv`, `intermediary/high_resource_countries.csv`, `intermediary/clusters_k5_agg.csv`
**Outputs:** 4 PNG files to `outputs/charts/ml/`

---

## Chile Analysis - Execution Order

Independent pipeline; does not share data with the main 54-country analysis. Run A → B → C.

---

### Chile A - `Chile_A_Setup_Production.ipynb` - Inventory & Production Setup

**What it does:**
Loads the Chilean facilities inventory and COCHILCO production statistics, matches production figures to individual mines, and saves a pickled pipeline state for downstream notebooks.

- Loads `Chile_Minerals_Inventory.csv` (200+ mines, concentrators, smelters, ports) and `Chile_Mine_Plant_Links.csv`; removes links from idle mines; backs up both files to `_backup.csv`
- Tags all Codelco facilities by keyword-matching division names in the inventory
- **Copper matching:** reads COCHILCO production by company from `COCHILCO_Production_2005_2024.xlsx`; searches inventory for each company name/deposit; assigns production figures to matched mines (coverage ~95% of national total)
- **Molybdenum matching:** parses Tabla 4.2 from COCHILCO Excel (concentrado vs óxido); applies split allocations from `MO_SPLIT` constants (e.g., "Chuquicamata y Radomiro Tomic" → 40% Chuquicamata / 60% Radomiro Tomic)
- **Lithium rebuild:** reclassifies Salar de Atacama facilities from Potash to Lithium (USGS misclassification); rebuilds lithium links from scratch using distance thresholds (210 km for active mines, 80 km for prospects)
- Prices sourced from COCHILCO Anuario, implied FOB prices, and USGS benchmarks

**Inputs:** `chile_analysis/inputs/Chile_Minerals_Inventory.csv` (or in-repo equivalent), `COCHILCO_Production_2005_2024.xlsx`, `Anuario-de-Estadisticas-del-Cobre-y-otros-Minerales-2005-2024.xlsx`
**Outputs:** `intermediary/Chile_Minerals_Inventory.csv` (adds `COCHILCO_CU_2024_KMT`, `COCHILCO_MO_2024_MT`, `OPERATOR_NAME`), `intermediary/Chile_Mine_Plant_Links.csv`, pickled pipeline state

---

### Chile B - `Chile_B_Supply_Chain.ipynb` - Supply Chain Construction

**What it does:**
Builds the full directed supply chain graph (mine → concentrator/SX-EW → smelter → port) from the inventory and link tables.

**Processing stage classification (vectorized):**
- Facility types mapped to stages: Mine→extraction, Concentrator→concentration, SX-EW Plant→sx_ew, Smelter→smelting, Refinery→refining
- Refined by facility name keywords where type is ambiguous

**Product form assignment:**
- mine_to_plant edges inherit from plant type: SX-EW → cathode_sxew, Smelter → blister, Refinery → cathode_er, Concentrator → concentrate

**Downstream edge construction (4 categories):**
1. **Concentrator → Smelter:** named feeds (e.g., Escondida concentrate → Altonorte smelter) + regional feed (within 300 km for custom smelters)
2. **Smelter → Port:** based on `smelter.export_ports` list in constants
3. **Concentrator → Port:** checks `DEDICATED_PORT` override first, then nearest port by product type
4. **SX-EW → Port:** checks `CODELCO_CATHODE_ROUTING` override (e.g., El Teniente cathode → Ventanas port), then `DEDICATED_PORT`, then nearest cathode port

Unified edge table columns: `FROM_NAME`, `FROM_TYPE`, `FROM_LAT/LON`, `TO_NAME`, `TO_TYPE`, `TO_LAT/LON`, `EDGE_TYPE`, `PRODUCT_FORM`, `COMMODITIES`, `DISTANCE_KM`

**Inputs:** pickled state from Chile A
**Outputs:** `intermediary/Chile_Ports.csv`, `intermediary/Chile_Supply_Chain_Edges.csv`, `intermediary/Chile_Export_Destinations.csv`, `intermediary/Chile_Downstream_Links.csv`, `intermediary/Chile_Supply_Chain_Summary.csv`; updated pipeline pickle

---

### Chile C - `Chile_C_Validation_Analysis.ipynb` - Validation & Repair

**What it does:**
Validates the supply chain graph and patches any construction errors before the data is used for visualisation or analysis.

- Checks copper production coverage: sums assigned production across matched mines vs COCHILCO national total; flags uncovered production
- Traces end-to-end paths (mine → concentrator → smelter → port) for each commodity; identifies facilities with no downstream edge
- Consolidates duplicate entities (same facility under different name variants)
- Patches missing edges where path tracing finds dead ends
- Compares modelled port assignments against geodesically optimal assignments (haversine distance matrix); generates port distance comparison charts
- Cross-references against Comtrade trade flow data (`Comtrade_vs_Salidas_Validation.csv`)

**Inputs:** pickled state from Chile B, `intermediary/Chile_Supply_Chain_Edges.csv`
**Outputs:** `intermediary/Chile_Network_Metrics.csv`, `intermediary/Mine_Port_Distance_Matrix.csv`, `intermediary/Mine_Optimal_Port_Assignments.csv`, `intermediary/Port_Distance_Comparison.csv`, `intermediary/Comtrade_vs_Salidas_Validation.csv`; port comparison charts in `chile_analysis/outputs/`

---

### Chile standalone scripts

| Script | What it does |
|--------|-------------|
| `chile_supply_chain_map.py` | Interactive Plotly map of the full supply chain graph (nodes = facilities, edges = flows, sized by production volume) → `outputs/chile_supply_chain_map.html` |
| `chile_visualisations.py` | All other Chile charts: treemap of mineral export value, top facilities bar chart, facility map, export choropleth by region, tile cartogram, sunburst (region × mineral), non-copper value bar |
| `commodity_prices_2024.py` | Price lookup tables for 2024 USD valuations (Cu, Li, Mo, Fe, Au, Ag, etc.) |
| `pipeline_constants.py` | Shared constants: file paths, column names, mineral lists, `MO_SPLIT` rules, `CODELCO_CATHODE_ROUTING`, smelter/port metadata |
| `pipeline_utils.py` | Shared utility functions: distance calculations, inventory search helpers, state load/save |

Run directly:
```bash
python3.10 chile_analysis/scripts/chile_supply_chain_map.py
python3.10 chile_analysis/scripts/chile_visualisations.py
```

---

## Shared Scripts (`main_analysis/scripts/`)

| Script | What it does |
|--------|-------------|
| `standardize_country.py` | Harmonises country name strings to ISO3 across datasets (World Bank, UN, and common variant spellings). Auto-imported by NB0 and NB1 via `sys.path`. |
| `viz_utils.py` | Shared Plotly styling: IBM Plex Sans font, colour palette, margin defaults used across all visualisation notebooks |

---

## Notes

- **Sample integrity:** The 54-country sample in `main_analysis/` is fixed. Do not filter it further in the main pipeline.
- **Intermediary files:** Notebooks read from CSVs written by earlier steps. If a notebook fails to find an input, run the preceding step first.
- **Chile pipeline is self-contained:** No shared data with the 54-country analysis; can be run independently.
- **Outputs:** Interactive charts are `.html`; static charts are `.png` in `outputs/charts/` subdirectories.
