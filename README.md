# Capstone Project: Moody's

### Members:
Ayaan Tigdikar <br>
Carlos Gallegos <br>
Emilio Gonzalez-Islas <br>
Ignacio Orueta <br>
Leonardo Luksic <br>

In this Capstone project, we address the following questions:

•	What factors explain why some countries have been more successful than others? <br>
•	What policy levers and institutional conditions support or hinder the transition? <br>
•	How sustainable are these shifts in terms of fiscal revenues, employment and export diversification? How have they supported or hindered credit quality? 

## Data sources:
   The data used for this project comes from 3 sources: <br><br>
        1) <a href="https://comtradeplus.un.org/" target="_blank">COMTRADE</a>: This source aggregates detailed global annual and monthly trade statistics 
        by product and trading partner. It covers approximately 200 countries and represents more than 99% of the world's merchandise trade. 
        It also has an API developer tool, which I integrated to my workflow. 
        <br><br>
        2) <a href="https://atlas.hks.harvard.edu/data-downloads" target="_blank">Harvard Atlas of Economic Complexity</a>: Using COMTRADE data,
        this website provides information about the Economic Complexity Index (ECI), Economic Complexity Rankings, and the Product Complexity Index (PCI).
        <br><br>
        3) <a href="https://data.worldbank.org/" target="_blank">World Bank</a>: This source offers data on various health, inequality, and environmental outcomes.
        These datasets were analyzed with the previously mentioned data.
        4) To add 


## Methodology:
### 1) Quantitative analysis:

#### a) Econometric analysis

#### b) Machine learning analysis:

##### i) Cluster analysis: this allow us to filter countries for our analysis, keeping ony emerging markets with a high level of natural resources.

##### ii) LASSO/Random Forest: here, we analyze the most important variables for ECI upgrading. 

### 2) Qualitative analysis:
#### - Comparative analysis

 ## Main findings

 ## Conclusion

---

## Pipeline Reference (`FINAL CODE RECAP/`)

> All production notebooks live in the `FINAL CODE RECAP/` directory.
> Run them **in order** (NB0 → NB6). Each notebook must be run with its working directory set to `FINAL CODE RECAP/` so that relative paths resolve correctly.

### Notebook sequence

| # | File | What it does |
|---|------|-------------|
| 0 | `0_NR_extraction_FINAL.ipynb` | Extracts natural resource production volumes and prices from EI, OWID, USGS, and gas-benchmark sources; computes annualised production values (USD); outputs `intermediary/natural_resources_production_values.csv` and `natural_resources_final_clean.csv`. |
| 1a | `1_cleaning_master_data_FINAL.ipynb` | Downloads 7 data sources (World Bank, IMF WEO, IMF ICSD, ECI, V-Dem, PWT, CEPII) and merges them into a master panel (1995–2019, ~208 countries); outputs `intermediary/master_data_wide.csv` and `master_data_long.csv`. Uses caching (`intermediary/cache/`); set `FORCE_REFRESH = True` to re-download. |
| 1b | `1_NR_source_conflict_resolution.ipynb` | Documentation-only notebook recording manual resolution of 18 high-discrepancy (>100%) source conflicts between OWID and EI. No code outputs. |
| 2 | `2_MissingCheck_FINAL.ipynb` | Analyses variable- and country-level missingness patterns; produces 6 static and 2 interactive figures in `Graphics/NB2/`. Informs the `omit_countries` list in NB3. |
| 3 | `3_Imputing_FINAL.ipynb` | Backfills NR production gaps, applies IEA 2021 price adjustments, filters to 54 resource-dependent countries, imputes missing values (linear interpolation capped at 3-year gaps + KNN k=5); outputs `intermediary/Master.csv` and `intermediary/NaturalResource.csv`. |
| 4 | `4_Clustering_FINAL.ipynb` | Classifies 54 countries into 4 resource-profile clusters via PCA (2 components) + K-Means (k=4, validated by silhouette); outputs `intermediary/clusters1995.csv`, `clusters2019.csv`, `clustersagg.csv`, and charts in `Graphics/NB4/`. |
| 5 | `5_ML_FINAL.ipynb` | Fits LASSO, Ridge, Elastic Net, and Random Forest to predict log(ECI) on the 54-country panel; reports in-sample R², VIF diagnostics, coefficient comparison, and feature importance; outputs to `Graphics/NB5/` and `intermediary/coefficient_summary_table.csv`. |
| 6 | `6_Regressions_FINAL.ipynb` | Descriptive statistics, FE-OLS regressions (including HCI × Production and GFCF × Production interactions), and a supplementary two-way FE model with country + year fixed effects and country-clustered SEs; outputs to `Graphics/NB6/` and `intermediary/high_resource_countries.csv`. |

### Required Python packages

```
pandas numpy scipy scikit-learn statsmodels plotly matplotlib seaborn
wbgapi weo rdata requests pathlib
```

Install via:
```bash
pip install pandas numpy scipy scikit-learn statsmodels plotly matplotlib seaborn wbgapi weo rdata requests
```

### How to run

```bash
cd "FINAL CODE RECAP"
jupyter nbconvert --to notebook --execute 0_NR_extraction_FINAL.ipynb --inplace
jupyter nbconvert --to notebook --execute 1_cleaning_master_data_FINAL.ipynb --inplace
jupyter nbconvert --to notebook --execute 2_MissingCheck_FINAL.ipynb --inplace
jupyter nbconvert --to notebook --execute 3_Imputing_FINAL.ipynb --inplace
jupyter nbconvert --to notebook --execute 4_Clustering_FINAL.ipynb --inplace
jupyter nbconvert --to notebook --execute 5_ML_FINAL.ipynb --inplace
jupyter nbconvert --to notebook --execute 6_Regressions_FINAL.ipynb --inplace
```

NB0 and NB1 require an active internet connection (data downloads). NB1 caches results in `intermediary/cache/`.

### Data flow

```
                     ┌──────────────────────────────────────────────┐
rawdata/             │  EI CSV/Excel, OWID, USGS, Gas benchmarks     │
                     └────────────────────┬─────────────────────────┘
                                          ▼
                              NB0: 0_NR_extraction
                                          │
                        natural_resources_final_clean.csv
                                          │
 APIs / GitHub ───► NB1: 1_cleaning ─────┼──► master_data_wide.csv
                                          │
                         NB2: 2_MissingCheck (diagnostic only)
                                          │
                              NB3: 3_Imputing
                                    │        │
                              Master.csv   NaturalResource.csv
                                    │        │
                    ┌───────────────┘        ▼
                    │                NB4: 4_Clustering
                    │                    clusters*.csv
                    │                        │
                    └──────────┬─────────────┘
                               ▼
                    NB5: 5_ML_FINAL  ──► Graphics/NB5/, coefficient_summary_table.csv
                               │
                    NB6: 6_Regressions_FINAL ──► Graphics/NB6/, high_resource_countries.csv
```

### Final outputs

| Output | Location | Notebook |
|--------|----------|---------|
| Figures (ML) | `Graphics/NB5/` | NB5 |
| Coefficient summary table | `intermediary/coefficient_summary_table.csv` | NB5 |
| Figures (regressions) | `Graphics/NB6/` | NB6 |
| Regression dataset | `intermediary/high_resource_countries.csv` | NB6 |
| Cluster maps | `Graphics/NB4/` | NB4 |
| Missingness charts | `Graphics/NB2/` | NB2 |