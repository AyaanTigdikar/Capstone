# Data Sources — Capstone Project

## Notebook 0: `0_NR_extraction_FINAL.ipynb` — Natural Resources Pipeline

Builds `natural_resources_production_values.csv` covering oil, gas, coal, and 13+ minerals.

| # | Source | Label | What it provides | File |
|---|--------|-------|-----------------|------|
| 1 | **Energy Institute — Statistical Review of World Energy (Narrow CSV)** | `EI_CSV` | Oil, Gas, Coal production & consumption; Cobalt, Lithium, Graphite, Rare Earth production & reserves | `rawdata/Statistical Review of World Energy Narrow File-1.csv` |
| 2 | **Energy Institute — Excel workbook** | `EI_Excel` | Production & Reserves sheets for 13 minerals (Cobalt, Lithium, Graphite, Rare Earth, Copper, Manganese, Nickel, Zinc, Platinum Group, Bauxite, Aluminium, Tin, Vanadium) | `rawdata/base_dataset.xlsx` |
| 3 | **Energy Institute — Excel price sheets** | `EI_Prices` | Oil prices (1861–present, $/bbl); Coal NW Europe benchmark ($/t); Cobalt, Lithium, Nickel, Graphite prices | `rawdata/base_dataset.xlsx` (sheets: *Oil crude prices since 1861*, *Coal & Uranium - Prices*, *Mineral Commodity Prices*) |
| 4 | **Our World in Data (OWID) — Mineral CSVs** | `OWID` | Production, reserves, and unit-value prices for 20+ minerals | `rawdata/Minerals/<mineral-name>/<mineral-name>.csv` |
| 5 | **USGS ds140 — Mineral Unit Values** | `USGS` | Mineral unit values ($/t) | `rawdata/USGS/ds140-*.xlsx` *(optional — not present on disk; pipeline skips gracefully)* |
| 6 | **Oil / Gas / Coal benchmark prices** | `GasPrice` | Brent crude ($/bbl); Coal PCOALAUUSDA ($/t); Henry Hub, TTF, JKM gas ($/MMBtu) — averaged to a single per-resource world price | `rawdata/Oil Gas Coal Uranium Price.xlsx` |
| 7 | **Consolidated Prices (working file)** | `ConsolidatedPrices` | Pre-harmonised wide-format prices for 19 minerals + Oil + Coal + Gas, 1990–2024 *(highest priority for all price series)* | `workingdata/natural resource prices.xlsx` |

**Source priority for prices:** ConsolidatedPrices → USGS → EI_Prices → GasPrice → OWID → EI_Excel → EI_CSV
**Source priority for production/reserves:** best longitudinal coverage per country wins (EI_CSV / OWID / EI_Excel)

---

## Notebook 1: `1_cleaning_master_data_FINAL.ipynb` — Master Panel Construction

Builds `master_data_long.csv` / `master_data_wide.csv` covering 208 countries, 1995–2019.

| # | Source | Variables used | Access method |
|---|--------|---------------|---------------|
| 1 | **World Bank Indicators (WBI)** | 26 indicators: resource rents (oil/gas/mineral/total), GDP structure (manuf., industry, agri., services), savings, finance (credit, interest rates, inflation), employment by sector, infrastructure (electricity, mobile), trade, demographics | `wbgapi` Python package (live API) |
| 2 | **IMF World Economic Outlook (WEO)** — April 2024 vintage | GDP per capita (PPP, constant 2017 USD); general government revenue (% GDP); general government net debt (% GDP); structural fiscal balance (% potential GDP) | `weo` Python package (downloads April 2024 vintage) |
| 3 | **IMF Investment and Capital Stock Dataset (ICSD)** | Gross fixed capital formation — government, private, PPP sectors (summed, % GDP, constant prices) | GitHub raw CSV (`AyaanTigdikar/Capstone`) |
| 3b | **IMF Fiscal Affairs Dept — FAD_FM dataset** | Primary net lending, general government (% GDP) | GitHub raw CSV (`AyaanTigdikar/Capstone`) |
| 4 | **Economic Complexity Index (ECI)** — Atlas of Economic Complexity, HS92 | ECI score per country-year *(dependent variable)* | GitHub raw CSV (`AyaanTigdikar/Capstone/rawdata/growth_proj_eci_rankings.csv`) |
| 5 | **Varieties of Democracy (V-Dem)** | 12 indicators: 5 democracy indices (electoral, liberal, participatory, deliberative, egalitarian); rule of law; property rights; political corruption; accountability; clientelism; political stability (WGI); civil war | `vdemdata` R package `.RData` file (GitHub) |
| 6 | **Penn World Table 11.0 (PWT)** | Human capital index; capital stock; TFP (constant & welfare-relevant); consumption/investment/government expenditure shares; capital depreciation rate | GitHub raw Excel (`AyaanTigdikar/Capstone/rawdata/pwt110.xlsx`) |
| 7 | **CEPII GeoDist** | Landlocked dummy (time-invariant, expanded across all years) | GitHub raw XLS (`AyaanTigdikar/Capstone/rawdata/geo_cepii.xls`) |
| 8 | **Natural Resource Production Values** *(from NB0)* | Production & reserves by resource category (Hydrocarbons, Metals) per country-year | Pre-processed CSV (`rawdata/production_values_w_prices-EM.csv`) |

---

## Coverage Summary

| Panel dimension | Value |
|----------------|-------|
| Years | 1995–2019 |
| Countries | ~208 (non-aggregate World Bank economies, excl. 16 territories) |
| Total variables | ~60 (26 WB + 4 IMF WEO + 2 IMF ICSD + 1 ECI + 12 V-Dem + 8 PWT + 1 CEPII + NR production combos) |
| Variables used in analysis | 29 flagged as `Important_vars` |
