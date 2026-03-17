# Chilean Mineral Supply Chain Pipeline: Complete Reference

## Overview

This pipeline builds a directed network of Chile's mineral supply chain, connecting mines to processing facilities, export ports, and destination countries. It runs as 6 sequential Jupyter notebooks (Parts 0-5), each saving state via pickle files that the next part loads. All code runs locally on macOS.

---

## File Locations

**Base directory:** `/Users/leoss/Desktop/GitHub/Capstone/Case studies/Chile`  
**Working directory:** `{BASE_DIR}/Preliminary/` (all intermediate and final outputs)  
**Data directory:** `{BASE_DIR}/data/`

### Input Data Files

| File | Description |
|------|-------------|
| `COCHILCO_Production_2005_2024.xlsx` | Cleaned COCHILCO Anuario with named sheets (A_National_Production, B1_Copper_by_Company, C_* regional sheets) |
| `Anuario-de-Estadisticas-del-Cobre-y-otros-Minerales-2005-2024.xlsx` | Original COCHILCO Anuario (used for Mo parsing from Tabla 4.2, export destinations from Tablas 18.2/19.2/20.2/23.2/11) |
| `salidas_2024_clean.csv` | Chilean customs (Aduanas) export records, semicolon-delimited, with HS codes, port codes, country codes, FOB values |
| `tablas_de_codigos.xlsx` | Aduanas lookup tables (sheets: Países, Puertos, Vías de Transporte, Regiones) mapping numeric codes to names |
| `chile_mineral_trade_combined.csv` | UN Comtrade export data for cross-validation |
| `Chile_Minerals_Inventory.csv` | Georeferenced facility inventory (mines, plants, smelters) with columns: FACILITY_NAME, FACILITY_TYPE, LATITUD, LONGITUD, REGION, OPERATOR_NAME, COMMODITY_LIST_STR/ALL_COMMODITIES_RAW, resource/reserve columns |
| `Chile_Mine_Plant_Links.csv` | Pre-existing mine-to-plant linkages with MINE_NAME, PLANT_NAME, coordinates, SHARED_COMMODITIES, DISTANCE_KM |

### Output Files (in `Preliminary/`)

| File | Description |
|------|-------------|
| `Chile_Minerals_Inventory.csv` | Enriched inventory with production columns and CHAIN_STAGE |
| `Chile_Mine_Plant_Links.csv` | Cleaned links with PRODUCT_FORM |
| `Chile_Supply_Chain_Edges.csv` | **Main output.** Unified directed edge table (all edge types) |
| `Chile_Downstream_Links.csv` | Subset of edges excluding mine_to_plant |
| `Chile_Export_Destinations.csv` | Port-to-country export edges only |
| `Chile_Ports.csv` | Port reference table (maritime + air) |
| `Chile_Port_Shares_Aduanas.csv` | Aduanas-derived FOB share by port and product type |
| `Comtrade_vs_Salidas_Validation.csv` | Cross-validation of Aduanas vs Comtrade totals |
| `Mine_Optimal_Port_Assignments.csv` | Distance-based port assignment per mine |
| `Mine_Port_Distance_Matrix.csv` | Full mine x port haversine distance matrix |
| `Port_Distance_Comparison.csv` | Actual vs modelled port share comparison |
| `Port_Comparison_Chart.png` | Bar chart visualization of port share comparison |

### Pipeline State Files

Each notebook saves a pickle: `_pipeline_state_0.pkl` through `_pipeline_state_5.pkl`. Each contains all accumulated variables (inv, links, edges, constants, etc.).

---

## Shared Constants (defined in Part 0, passed through all parts)

### `COMPANY_TO_DEPOSIT` (dict)
Maps COCHILCO company names to inventory search terms. Covers all ~30 major copper producers (7 Codelco divisions + ~23 private mines). Used to match COCHILCO production figures to inventory facility records.

### `CODELCO_EXTRA_SEARCH` (dict)
Extra search terms for Codelco divisions that have non-mine associated facilities: Andina -> "Rio Blanco", Ministro Hales -> "Mansa Mina", Salvador -> "El Salvador"/"Potrerillos".

### `SMELTERS` (list of dicts)
6 Chilean copper smelters, each with: name, search terms, operator, smelter_type (integrated/custom), region, lat/lon, has_refinery flag, feeds_from_mines list (or feeds_from_region for custom smelters), output_product (cathode/blister), export_ports list.

Smelters: Chuquicamata (Codelco, integrated), Potrerillos (Codelco, integrated), Caletones (Codelco, integrated), Altonorte (Glencore, custom), Paipote/H.V. Lira (ENAMI, custom), Chagres (Anglo American, integrated).

### `PORTS` (list of dicts)
13 maritime export ports with name, region, lat/lon, products handled, key users. Key ports: Coloso (Escondida dedicated), Angamos, Mejillones, Antofagasta (ATI), Iquique, Patache, Barquito, Caldera, Coquimbo, Los Vilos (Los Pelambres dedicated), Ventanas, San Antonio, San Vicente.

### `SMELTER_NAME_MAP` (dict)
Maps short canonical smelter names to long inventory facility names for edge standardization.

### `DEDICATED_PORT` (dict)
Contractual mine-to-port overrides that bypass nearest-port logic: Escondida -> Coloso, Los Pelambres -> Los Vilos.

### `CODELCO_CATHODE_ROUTING` (dict)
Codelco cathode consolidation routing: Chuquicamata/Radomiro Tomic/Ministro Hales/Gabriela Mistral -> Angamos; Salvador -> Barquito; El Teniente/Andina -> San Antonio.

### `MATCH_DISAMBIGUATION` (dict)
Preferred exact facility names when substring matching returns multiple: Cerro Negro (not Cerro Negro Norte), Andacollo -> Carmen de Andacollo (not Andacollo Oro).

### `IRON_MINE_NAMES`, `ZINC_MINE_NAMES` (lists)
Known iron mines (Los Colorados, El Algarrobo, Cerro Negro Norte, Romeral, Tofo) and zinc mine (El Toqui) for direct assignment.

---

## Shared Utility Functions (redefined in every notebook)

- **`haversine_km(lat1, lon1, lat2, lon2)`**: Great-circle distance in km.
- **`parse_comm_list(val)`**: Splits comma-separated commodity string into list.
- **`add_commodity(row_idx, commodity, df, col)`**: Appends commodity to a comma-separated cell if not already present.
- **`nearest_port(lat, lon, product_type)`**: Finds closest port from PORTS list, filtering by product type ("concentrate" or "cathode").
- **`search_inventory(inv_df, terms, require_mine=False)`**: Searches inventory FACILITY_NAME by substring terms, optionally restricted to mines.
- **`section_header(title)`**: Prints formatted section divider.

---

## Part 0: Setup & Data Loading

**Notebook:** `Chile_Part0_Setup.ipynb`  
**Saves:** `_pipeline_state_0.pkl`

### Operations
1. Defines all constants listed above.
2. Loads `Chile_Minerals_Inventory.csv` and `Chile_Mine_Plant_Links.csv` from `Preliminary/`. Creates `_backup.csv` copies on first run and always reloads from backups (guarantees clean state).
3. Identifies `comm_col` as either `COMMODITY_LIST_STR` or `ALL_COMMODITIES_RAW` depending on which exists in inventory.
4. Removes links from idle mines (FACILITY_TYPE == "Mine (idle)") from the links table. Stores `idle_mines` set.
5. Saves state pickle with: inv, links, comm_col, idle_mines, all constants.

### Key Variables After Part 0
- `inv`: DataFrame, full facility inventory (~hundreds of rows)
- `links`: DataFrame, mine-to-plant linkages (idle mines removed)
- `comm_col`: string, name of the commodity list column in inv

---

## Part 1: Production Matching (Cu, Mo, Li)

**Notebook:** `Chile_Part1_Production.ipynb`  
**Loads:** `_pipeline_state_0.pkl`  
**Saves:** `_pipeline_state_1.pkl`

### Section 2A: Codelco Division Mapping
Iterates `COMPANY_TO_DEPOSIT` entries starting with "División". For each, searches inventory using division terms + CODELCO_EXTRA_SEARCH. Sets OPERATOR_NAME to "Codelco" where blank.

### Section 2B: Copper Production Matching
- Reads sheet `B1_Copper_by_Company` from COCHILCO_PATH. Extracts latest year (2024).
- Creates columns `COCHILCO_CU_2024_KMT` and `COCHILCO_COMPANY` on inv.
- For each company in COMPANY_TO_DEPOSIT, looks up 2024 production in kMT, then matches to inventory mine by searching facility names. Uses MATCH_DISAMBIGUATION for ambiguous cases.
- Reports matched count and coverage percentage vs TOTAL row.

### Section 2C: Molybdenum Production Matching
- Reads `Tabla 4.2` from the original COCHILCO Anuario workbook (openpyxl, read_only).
- Parses year header row, then extracts company-level Mo production. Handles parent/child company rows (e.g., "CONCENTRADO" sub-rows summed into parent).
- Uses `MO_MAP` (dict mapping COCHILCO company names to inventory search terms) to match ~13 Mo producers.
- Creates column `COCHILCO_MO_2024_MT` on inv.

### Section 2D: Lithium Fixes and Link Cleanup
- Fixes commodity lists: adds "Lithium" to USGS Potash-listed Salar facilities, adds "Potassium" to Salar de Atacama.
- Rebuilds all lithium mine-to-plant links from scratch:
  - Identifies Li mines (active extraction with Lithium in commodity list and coordinates).
  - Identifies Li plants (non-extraction facilities with Lithium and coordinates), deduplicates by rounded lat/lon.
  - Links mines to plants within distance caps: 210 km for active mines (Salar de Atacama to Salar del Carmen is ~200-207 km), 80 km for prospects.
  - Removes old Li links, appends new, deduplicates by (MINE_NAME, rounded plant location).

---

## Part 1 continued (in same notebook, cells 3-5): Production Matching (Au, Ag, Fe, Zn)

**Loads:** `_pipeline_state_1.pkl`  
**Saves:** `_pipeline_state_2.pkl`

### Section 2E: Non-Copper Mineral Production

**Strategy:** COCHILCO has no company-level tables for Au/Ag/Fe/Zn. Uses regional production from `C_*` sheets, allocated to individual mines weighted by Sernageomin resource/reserve tonnage.

**Column setup:** Creates `COCHILCO_AU_2024_KG`, `COCHILCO_AG_2024_KG`, `COCHILCO_FE_2024_KMT`, `COCHILCO_ZN_2024_MT` on inv.

**`MINERAL_COLS` dict:** For each mineral, stores: column name, unit, COCHILCO pattern (e.g. "ORO"), commodity keyword, resource_col, reserve_col.

**`REGION_SHEET_MAP` dict:** Maps C_* sheet names to region string patterns for filtering inventory by region (14 regions).

**Process per mineral:**
1. Parse regional production values from C_* sheets (openpyxl read_only, find 2024 column, scan rows for matching mineral pattern).
2. For each producing region, find active mines with matching commodity and region. Compute weights from resource tonnage (priority) or reserve tonnage (fallback). Mines with neither get weight 0.
3. Allocate regional production proportionally by weight. Add commodity tag via `add_commodity()`.

---

## Part 2: Processing Stages & Supply Chain Edges

**Notebook:** `Chile_Part2_Processing_Edges.ipynb`  
**Loads:** `_pipeline_state_2.pkl`  
**Saves:** `_pipeline_state_3.pkl`

### Section 3A: Processing Stage Classification
Adds `CHAIN_STAGE` column to inv using `STAGE_MAP`: Mine (active) -> extraction, Concentrator -> concentration, SX-EW Plant -> sx_ew, Smelter -> smelting, Refinery -> refining, Processing/Pellet/Grinding/Steel Plant -> processing, others -> other.

Refines "processing" stage using facility name keywords: smelter/fundici -> smelting, refin/electro -> refining, sx-ew/leach/cathode -> sx_ew, concentrat/flotation/mill -> concentration.

### Section 3B: Smelter Matching to Inventory
Matches each of the 6 SMELTERS to an inventory facility by searching facility names. Stores mapping in `smelter_inv_map` (dict: canonical name -> inventory FACILITY_NAME). Sets matched facilities' CHAIN_STAGE to "smelting". Saves ports_df as CSV.

### Section 3C: Product Form Assignment
Adds `PRODUCT_FORM` column to links table. Classifies based on PLANT_TYPE and PLANT_NAME keywords:
- sx-ew/leach/cathode -> "cathode_sxew"
- smelter/fundici -> "blister"
- refin/electro -> "cathode_er"
- concentrat/flotation/mill -> "concentrate"
- Default: "concentrate"

### Section 3D: Build Downstream Edges
Constructs `downstream_edges` list with 4 edge types:

**A. Concentrator -> Smelter:** For integrated smelters, matches concentrators by feeds_from_mines search terms. For custom smelters (Altonorte, Paipote), matches regional concentrators within 300 km not already fed to another smelter. Tracks `smelter_fed` set to avoid double-assignment.

**B. Smelter -> Port:** Each smelter connected to its export_ports from SMELTERS config.

**C. Concentrator -> Port:** For concentrators not feeding a smelter. Checks DEDICATED_PORT overrides first, then nearest concentrate-handling port.

**D. SX-EW -> Port:** For all sx_ew facilities. Checks CODELCO_CATHODE_ROUTING first, then DEDICATED_PORT, then nearest cathode-handling port.

### Section 3E: Unified Edge Table
Combines upstream (mine_to_plant from links) and downstream edges into single `edges` DataFrame with common columns: FROM_NAME, FROM_TYPE, FROM_LAT, FROM_LON, TO_NAME, TO_TYPE, TO_LAT, TO_LON, EDGE_TYPE, PRODUCT_FORM, COMMODITIES, DISTANCE_KM.

Enriches downstream edge commodities from inventory using `get_facility_commodities()`.

---

## Part 3: Export Destinations

**Notebook:** `Chile_Part3_Export_Destinations.ipynb`  
**Loads:** `_pipeline_state_3.pkl`  
**Saves:** `_pipeline_state_4.pkl`

### Export Destination Parsing (from COCHILCO Anuario)
Parses country-level export volumes using `parse_destination_table()`:
- `cu_refined` from Tabla 18.2 (kMT)
- `cu_blister` from Tabla 19.2 (kMT)
- `cu_concentrate` from Tabla 20.2 (kMT)
- `mo_concentrate` from Tabla 23.2 (MT)
- `li_exports` and `io_exports` from Tabla 11 ($M FOB), parsed via `parse_nonmetallic_column()` with column index.

**`COUNTRY_COORDS` dict:** Lat/lon for ~35 destination countries (major port cities).  
**`COUNTRY_ALIAS` dict:** Spanish -> English country name mapping.

### Section 4A: Aduanas-Derived Port Shares
Loads `salidas_2024_clean.csv` (auto-detects semicolon delimiter). Filters to export operation codes (EXPORT_OP_CODES set). Maps columns (hs, port, country, fob) by keyword search in column names.

**`HS_MINERAL_MAP` dict:** Maps HS code prefixes (4-8 digit) to (commodity, product_form) tuples. Covers: 2603 -> Cu concentrate, 7402 -> Cu blister, 7403 -> Cu cathode, 2613 -> Mo, 2601 -> Fe, 7108 -> Au, 7106 -> Ag, 2836*/2825*/2842* -> Li compounds, 2801 -> Iodine, 2528 -> Boron, 2602 -> Mn, 2834 -> Nitrate, 2841 -> Rhenium.

**`classify_hs(code_str)`**: Tries 8/6/4-digit prefix matches against HS_MINERAL_MAP (also tries zero-trimmed variants).

Loads lookup tables from `tablas_de_codigos.xlsx` via `parse_codigos_sheet()`. Applies `ADUANAS_PORT_ALIAS` and `ADUANAS_MANUAL_PORTS` to map port codes to standard names.

Computes `PORT_PRODUCT_MAP` (dict: product_type -> {port: FOB share}) from actual Aduanas data for copper products and all other commodities.

Builds `aduanas_port_country` DataFrame: aggregated FOB by (COMMODITY, PRODUCT_FORM, PORT_NAME, COUNTRY_EN).

### Section 4B: Comtrade Cross-Validation
Loads `chile_mineral_trade_combined.csv`. Compares Comtrade FOB totals against Aduanas Salidas FOB by commodity. Flags mismatches (ratio outside 0.8-1.2). Also does country-level comparison for copper using ISO3_TO_NAME mapping.

### Export Edge Construction
**`build_export_edges()`**: Creates port-to-country edges using COCHILCO destination volumes distributed across ports by PORT_PRODUCT_MAP shares.

**`build_aduanas_edges()`**: Creates port-to-country edges directly from Aduanas port-country aggregation (preferred when available).

For 6 configured commodities (EXPORT_CONFIGS): uses Aduanas direct edges if available, falls back to COCHILCO proportional distribution. Also processes ADUANAS_ONLY_COMMODITIES (Gold, Silver, Iron, Manganese, Boron, Nitrate, Rhenium) from Aduanas data only.

Adds `AIR_PORTS` (Santiago, Antofagasta, Iquique, Arica, Puerto Montt air freight) to ports_df for gold/silver shipments.

Export edges have extra columns: EXPORT_VALUE, EXPORT_UNIT, DESTINATION_TOTAL.

---

## Part 4: Cleanup, Validation & Save

**Notebook:** `Chile_Part4_Cleanup_Validation.ipynb`  
**Loads:** `_pipeline_state_4.pkl`  
**Saves:** `_pipeline_state_5.pkl` + all final CSV files

### Section 5A: Smelter Name Standardization
Replaces canonical smelter names with inventory facility names in edges FROM_NAME/TO_NAME columns using SMELTER_NAME_MAP.

### Section 5B: Missing Link Repairs
- **Andacollo Oro mine:** If no link exists, finds nearest SX-EW plant within 50 km and creates mine-to-plant link + edge.
- **5B.2: Missing smelter-to-port edges:** Adds s2p edges for Caletones, H.V. Lira, and Las Ventanas (long inventory names that weren't matched in Part 2's smelter_inv_map).
- **5B.3: Caserones mine-to-plant link:** Caserones (124.6 kMT) had no edge. Matches by name to its onsite concentrator, or by proximity within 50 km, or creates direct mine-to-port edge as fallback.

### Section 5C: Deduplication
Drops duplicate edges on (FROM_NAME, TO_NAME, EDGE_TYPE, COMMODITIES, PRODUCT_FORM), keeping first.

### Section 5D: Validation
- Reports production coverage for Cu, Mo, Au, Ag, Fe, Zn.
- **Path traceability check:** For each Cu producer, verifies whether a path exists from mine through processing to a port. Uses set-based lookups (no nested loops). Handles facilities where COCHILCO production is attached to the SX-EW/concentrator record rather than a separate mine record. Reports disconnected mines and their tonnage.

### Section 5E: Diagnostics Summary
Edge counts by type, downstream coverage by commodity, smelter connectivity check (mine-to-plant and smelter-to-port edge counts per smelter).

### Section 5F: Save
Sorts edges by (EDGE_TYPE, FROM_NAME, TO_NAME). Exports 6 CSV files to Preliminary/. Serializes full state to `_pipeline_state_5.pkl`.

---

## Part 5: Distance-Based Port Assignment Analysis

**Notebook:** `Chile_Part5_Distance_Analysis.ipynb`  
**Loads:** `_pipeline_state_5.pkl`  
**Saves:** comparison CSVs + distance matrix + chart

### Product Type Classification
Classifies each mine's output product using `PRODUCT_TYPE_OVERRIDE` dict (highest priority), then commodity keywords in inventory, then nearby facility stages, then links table mode. Categories: concentrate, cathode, blister.

**`PRODUCT_TYPE_OVERRIDE` dict:** ~30 entries. Codelco divisions and pure SX-EW operations -> cathode. Third-party concentrate shippers -> concentrate.

### Vectorized Distance Matrix
Computes full mine x port haversine distance matrix using numpy broadcasting (no per-row loops). Stored as `distance_df` (mines as rows, ports as columns).

### Port Assignment
`assign_port()` function: DEDICATED_PORT override > CODELCO_CATHODE_ROUTING (for cathode) > nearest port by distance.

### Simulated Port Shares
Groups mines by product_type, sums COCHILCO production per assigned port, computes shares. Compared against actual Aduanas shares from `Chile_Port_Shares_Aduanas.csv`.

### Comparison and Visualization
Computes differences (actual - modelled), mean absolute difference, total volume mismatch per product type. Generates grouped bar chart (actual vs modelled) for concentrate, cathode, blister.

---

## Edge Types in Final `edges` DataFrame

| EDGE_TYPE | FROM_TYPE | TO_TYPE | Description |
|-----------|-----------|---------|-------------|
| mine_to_plant | mine | plant | Upstream: mine to concentrator/SX-EW/processing |
| concentrate_to_smelter | concentrator | smelter | Named feeds + regional (custom smelters, <300 km) |
| smelter_to_port | smelter | port | Smelter output to export port |
| concentrate_to_port | concentrator | port | Direct concentrate export (no smelter step) |
| sxew_to_port | sx_ew | port | Cathode from SX-EW to port |
| port_to_country | port | country | Export to destination country |

---

## Key Methodological Notes

1. **Production allocation for Au/Ag/Fe/Zn** is weighted by resource/reserve tonnage (not equally split). Mines lacking both are excluded.
2. **Lithium links** use distance caps (210 km active, 80 km prospect) rather than generic nearest-facility matching.
3. **Export edges** prefer Aduanas direct port-country flows over COCHILCO proportional distribution when available.
4. **Port assignment** uses a three-tier hierarchy: contractual overrides > Codelco consolidation routing > nearest haversine distance.
5. **Custom smelters** (Altonorte, Paipote) receive feed from regional concentrators within 300 km, not from named mines.
6. **Idle mines** are kept in inventory but removed from the links/edges tables.
7. **Comtrade cross-validation** flags commodity-level FOB mismatches between Aduanas and UN Comtrade data.

---

## Dependencies

Python packages: pandas, numpy, openpyxl, matplotlib, seaborn, pickle (stdlib).
