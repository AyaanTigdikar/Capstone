# Chile Mineral Supply Chain Pipeline: Unified Code Documentation

Technical documentation for the three-notebook pipeline that constructs a supply chain network for Chile's mineral sector. The pipeline combines COCHILCO production statistics, USGS/SERNAGEOMIN facility inventories, Chilean customs (Aduanas) export data, and UN Comtrade trade flows to produce a directed graph from mine to processing plant to port to export destination, with facility-level production and USD valuations.

---

## Supporting Modules

Two standalone Python modules provide shared infrastructure across all three notebooks.

### pipeline_utils.py

Shared constants, file paths, and utility functions imported by every notebook via `from pipeline_utils import *`.

**Paths:** Defines `BASE_DIR`, `DIR_CODE`, `DIR_DATA`, `DIR_OUTPUT`, and `DIR_INTERMED` for the project directory tree. Includes fallback path resolution for COCHILCO Anuario and Aduanas Salidas files across multiple candidate locations.

**Constants:** Contains dictionaries that configure the pipeline's domain knowledge:

- `COMPANY_TO_DEPOSIT` (29 entries): Maps COCHILCO company names (e.g. "Division El Teniente") to inventory facility search terms. Used for copper production matching.
- `CODELCO_EXTRA_SEARCH` (3 entries): Additional search terms for Codelco divisions whose inventory names differ from COCHILCO labels (Andina/Rio Blanco, Ministro Hales/Mansa Mina, Salvador/Potrerillos).
- `SMELTERS` (6 entries): Defines each Chilean copper smelter with its operator, type (integrated vs. custom), coordinates, feed mines, output product form (cathode or blister), and export port assignments. The six are Chuquicamata (Codelco, integrated), Potrerillos (Codelco, integrated), Caletones (Codelco, integrated), Altonorte (Glencore, custom), Paipote/H.V. Lira (ENAMI, custom), and Chagres (Anglo American, integrated).
- `PORTS` (13 entries): Chilean mineral export ports with coordinates and product specializations. Ranges from Iquique in the north to San Vicente in the south.
- `SMELTER_NAME_MAP`: Maps canonical smelter names to their long-form inventory equivalents, since the USGS inventory uses descriptive names (e.g. "Caletones smelter (anodes). refinery (fire-refined ingots), and SX-EW plant").
- `DEDICATED_PORT`: Overrides nearest-port assignment for specific mines (Escondida -> Coloso, Los Pelambres -> Los Vilos).
- `CODELCO_CATHODE_ROUTING` (7 entries): Routes Codelco SX-EW cathode to specific consolidation ports rather than the nearest port.
- `IRON_MINE_NAMES` and `ZINC_MINE_NAMES`: Lists of known iron and zinc mine names for commodity-specific matching.

**Functions:**

- `haversine_km(lat1, lon1, lat2, lon2)`: Great-circle distance between two points.
- `parse_comm_list(val)`: Splits comma-separated commodity strings into lists.
- `nearest_port(lat, lon, product_type)`: Finds the nearest port from the PORTS list that handles the specified product type ("concentrate" or "cathode").
- `search_inventory(inv_df, terms, require_mine)`: Searches the inventory by facility name keywords.
- `load_state(part_number)` / `save_state(state, part_number)`: Serializes and deserializes pipeline state dictionaries as pickle files in the intermediary output directory. State files are named `_pipeline_state_{N}.pkl` where N ranges from 0 to 6.
- `unpack_state(state)`: Convenience function returning `(inv, links, comm_col, idle_mines, edges, ports_df)`.

### commodity_prices_2024.py

Parses 2024 commodity prices from the COCHILCO Anuario de Estadisticas del Cobre y otros Minerales (an Excel workbook) and converts them to standardized USD/MT or USD/kg units.

**Three Anuario tables are parsed:**

- **Tabla 96** (Copper): LME Grade A Settlement price in cents US$/lb, converted to USD/MT using 2,204.62 lb/MT.
- **Tabla 97** (Other metals): Exchange-specific prices for Gold, Silver, Aluminum, Nickel, Lead, Tin, Zinc, Molybdenum, Platinum. Each mineral uses a preferred exchange source (e.g. Handy & Harman for gold/silver, LME for base metals). Precious metals (Au, Ag, Pt) are converted to USD/kg using 32.15 troy oz/kg; others to USD/MT.
- **Tabla 98** (Non-metallic minerals): Annual min/max price bands for Lithium Carbonate, Iodine, Boric Acid, and Alumina. The midpoint of each band is used.

**Module-level exports:** `COMMODITY_PRICES_2024` (USD/MT dict), `COMMODITY_PRICES_2024_PER_KG` (USD/kg for precious metals), `COMMODITY_PRICE_UNITS`, `COMMODITY_PRICE_SOURCE`, `COMMODITY_PRICE_BANDS`.

---

## Notebook A: Setup and Production Matching

**Input:** Inventory CSV (461 facilities), Mine-Plant Links CSV (1,370 links), COCHILCO Anuario Excel workbook.
**Output:** `_pipeline_state_2.pkl` containing the enriched inventory with production columns and USD valuations.

### Cell 1: Load and Filter

Reads the backup inventory and links CSVs. Identifies 80 idle mines and removes their 263 associated links to prevent ghost edges in the supply chain graph. Saves `_pipeline_state_0.pkl`.

### Cell 3 (Sections 2A through 2D): Copper, Molybdenum, Lithium

**2A. Codelco Division Mapping:** Iterates over the 7 Codelco divisions in `COMPANY_TO_DEPOSIT`, searching the inventory by name to find the matching mine and associated processing facilities. Records the mapping for use in copper matching.

**2B. Copper Production Matching:** Opens the COCHILCO production workbook and reads company-level copper production from the latest year (2024). For each of 31 COCHILCO company entries, uses `COMPANY_TO_DEPOSIT` to find the corresponding inventory mine and writes the production value into a new `COCHILCO_CU_2024_KMT` column. Includes disambiguation logic for entries like "Las Cenizas" (which maps to multiple mines/plants) and the Chuquicamata/Radomiro Tomic 40/60 Molybdenum split. Total matched: 5,281.6 of 5,506.0 kMT (95.9%).

**2C. Molybdenum Production Matching:** Parses Tabla 4.2 from the Anuario for company-level Mo production. Applies a 40/60 split to the combined Chuquicamata-Radomiro Tomic entry. Writes `COCHILCO_MO_2024_MT` column. 12 facilities matched, 37,901.5 of 38,486.9 MT (98.5%).

**2D. Lithium Fixes and Link Cleanup:** Addresses lithium-specific inventory issues (duplicate Chemetall Foote entries, Salar de Atacama mine linkages). Removes additional orphan links discovered during the matching process. Saves `_pipeline_state_1.pkl`.

### Cell 5 (Section 2E): Extended Mineral Production Matching

Expands production matching from 6 minerals to 32, covering both metallic and non-metallic categories. Defines the `MINERAL_COLS` dictionary (32 entries), each specifying:

- A COCHILCO Spanish-language pattern to search for in the regional `C_*` sheets of the production workbook
- A column name for the production data (e.g. `COCHILCO_IO_2024_MT`)
- A weighting strategy: resource/reserve weighting for metallic minerals, equal weighting for non-metallics
- An inventory commodity keyword for facility matching

**Two allocation strategies:**

- **Resource/reserve weighting** (5 metallic minerals: Au, Ag, Fe, Zn, Pb): Regional production from COCHILCO `C_*` sheets is distributed across active mines in each region proportional to their resource or reserve tonnage from the SERNAGEOMIN inventory.
- **Equal weighting** (27 non-metallic minerals): Regional production is split equally among facilities in the matching region tagged with the relevant commodity keyword.

**Pattern matching:** Uses longest-pattern-first sorting and word-boundary checks to prevent substring collisions (e.g. "ORO" matching inside "BORO").

**Double-counting prevention:** Aggregate categories that sum their subcategories are excluded from facility matching: "Lithium Compounds" (sum of LiCO3 + LiOH + Li2SO4), "Calcium Carbonate" (Limestone + Coquina + White CaCO3), "Clay" (Bauxitic Clay + Bentonite + Kaolin), "Silica Ores" (Quartz + Silica Sand). These appear only in the national summary.

### Cell 7 (Section 2F): Pricing and Valuation

Assigns a USD price to each mineral from a four-tier hierarchy:

1. **COCHILCO Anuario (Tabla 96-98):** Cu, Mo, Au, Ag, Zn, Pb, LiCO3, Iodine, Boric Acid. Parsed by `commodity_prices_2024.py`.
2. **Derived from Tabla 98:** LiOH (1.1x LiCO3 price), LiSO4 (0.7x LiCO3).
3. **Implied FOB unit values** (Tabla 7 export values / Tabla 9 export volumes): Salt ($22/MT), Sodium Nitrate ($1,258/MT), Potassium Nitrate ($653/MT), Ulexite ($301/MT).
4. **Manual benchmarks** (USGS Mineral Commodity Summaries 2024): Iron ($110/MT, Platts), Potash ($260/MT), Limestone ($12/MT), Gypsum ($10/MT), and 16 other non-metallic minerals.

For each mineral with a price, creates a `USD_VALUE_*` column by multiplying facility production by unit price. Also creates a summed `USD_VALUE_TOTAL` column per facility. Total estimated value across all matched facilities: approximately $62 billion.

### Cell 9 (Section 2G): National Production Summary

Produces `Chile_National_Production_2024.csv` by parsing every row in the national production sheet with nonzero 2024 data. Each row includes the COCHILCO label, national total, unit, price, estimated value, number of facilities with allocations, allocation coverage percentage, and a boolean for whether the mineral has supply chain edges.

### Cell 11: Save State

Stores `_pipeline_state_2.pkl` with the enriched inventory (now ~163 columns), updated links, and three new keys: `commodity_prices` (all price dictionaries), `MINERAL_COLS` (mineral definitions), and `national_production` (per-mineral summary).

---

## Notebook B: Supply Chain Edge Construction

**Input:** `_pipeline_state_2.pkl` from Notebook A.
**Output:** `_pipeline_state_5.pkl` containing the unified edge table.

### Cell 1: Load State

Loads state from Notebook A: 461 inventory rows, 1,109 link rows (post idle-mine filter).

### Cell 3 (Sections 3A through 3E): Copper Supply Chain

**3A. Processing Stage Classification:** Assigns a `CHAIN_STAGE` label to each facility using a two-pass approach. First pass maps `FACILITY_TYPE` to a stage (extraction, concentration, sx_ew, smelting, refining, processing). Second pass refines the 123 generic "Processing Plant" entries by keyword-matching facility names for smelter/refinery/SX-EW/concentrator terminology. Result: extraction (187), processing (128), extraction_idle (80), sx_ew (25), concentration (22), smelting (15), refining (4).

**3B. Smelter and Port Setup:** Matches the 6 SMELTERS definitions to inventory rows by searching facility names. Builds the `smelter_inv_map` dictionary mapping canonical smelter names to inventory names. Saves 13 ports to CSV.

**3C. Product Form Assignment:** Assigns a product form to each mine-plant link by keyword-matching plant type and name: `cathode_sxew` (571 links), `concentrate` (440), `blister` (82), `cathode_er` (16). 342 links with no keyword match default to "concentrate".

**3D. Downstream Edge Construction:** Builds four categories of copper downstream edges:

- **(A) Concentrator -> Smelter** (7 edges): Named feed mines for integrated smelters, plus regional feed within 300km for custom smelters (Altonorte, Paipote).
- **(B) Smelter -> Port** (9 edges): Each smelter's configured export ports from the SMELTERS constant.
- **(C) Concentrator -> Port** (6 edges): Concentrators not feeding a smelter are routed to the nearest concentrate port, with dedicated port overrides.
- **(D) SX-EW -> Port** (25 edges): SX-EW plants routed to cathode-capable ports, with Codelco routing overrides.

**3E. Unified Edge Table:** Converts the 1,109 mine-plant links into standardized upstream edges and concatenates with the downstream edges. Enriches downstream edges with commodity labels from the inventory. Total: 1,165 edges.

### Cell 5 (Section 4): Export Destinations

Builds port-to-country export edges from three data sources with a priority hierarchy:

**COCHILCO Anuario destination tables:** Parses product-form-specific destination tables for copper (refined, blister, concentrate), molybdenum, lithium, and iodine. Each table provides country-level volumes or FOB values.

**Aduanas Salidas customs data:** Loads Chile's 2024 customs export records (328,443 rows), filters to mineral HS codes, and maps port codes to the 13 pipeline ports. Computes port-level market share by commodity for allocating COCHILCO destination volumes to specific ports.

**Comtrade HS6 trade data:** Cross-validates Aduanas totals and provides country-level export data for additional minerals (Boron, Nitrate, Salt, Potash, Lead, Copper Sulfate, Selenium, Rhenium, Sulfuric Acid) not covered by the COCHILCO destination tables.

**Edge allocation:** For each commodity-country pair, the national export volume is distributed across ports proportional to each port's share in the Aduanas data. Creates 561 port-to-country edges (later expanded with Comtrade minerals).

### Cells 7-16 (Section 5): Non-Copper Mineral Edges

Each cell handles one mineral's domestic supply chain. All use a shared set of helper functions:

- `make_edge()`: Builds standardized edge dictionaries.
- `find_facilities(patterns)`: Searches inventory by name patterns.
- `nearest_from_list(lat, lon, port_list)`: Finds nearest port from an explicit list.

**Cell 8 (Iron):** 6 mines with COCHILCO production. Searches for nearby processing plants (pellet plants, concentration plants) within 100km. Connects mine -> plant -> port, with direct mine-to-port fallback. Includes custom port definitions for Huasco and Guayacan (not in the standard PORTS list). A post-loop sweep catches iron processing plants that received upstream links from the original inventory data but were missed by the 100km radius.

**Cell 9 (Lithium):** 1 mine (Salar de Atacama), 9 processing plants. Deduplicates plants by location, routes to nearest lithium port (Angamos, Mejillones, ATI). 3 plant-to-port edges.

**Cell 10 (Molybdenum):** 12 mines with COCHILCO Mo production, 4 Mo processing plants. Each mine is assigned to its nearest Mo plant. Then each Mo plant gets a port edge. 12 mine-to-mo_plant + 4 mo_to_port = 16 edges.

**Cell 11 (Gold and Silver):** Mines with Au/Ag production are routed to the nearest smelter or refinery from a curated pool (the 6 named smelters plus facilities containing "refin", "Gregorio", or "Biocobre" in their names). Deduplicates the pool by facility name to avoid routing to commodity-variant duplicates. Each smelter that received Au/Ag mine edges then gets an airport edge to Santiago (air) or Antofagasta (air). Gold: 20 mines -> 8 smelters -> 2 airports. Silver: 12 mines -> 6 smelters -> 2 airports.

**Cell 12 (Iodine):** Searches inventory by name patterns (caliche operations, SQM facilities, Algorta, Cosayach) and by commodity tag. 16 mines, 6 plants. Mine-to-plant edges within 150km, then plant-to-port using name-based deduplication. Ports: Angamos, Mejillones, Iquique. 6 iodine_to_port edges.

**Cell 13 (Nitrate):** Searches for nitrate/salitre/caliche facilities plus SQM and potash-tagged operations. 7 processing plants routed to nearest port from Tocopilla (custom coordinates), Iquique, and Angamos. 7 edges.

**Cell 14 (Zinc):** 2 mines with COCHILCO Zn production (El Toqui, Alhue). Custom port definitions for Puerto Aysen. 2 mine-to-port + 2 via-plant = 4 edges.

**Cell 15 (Rhenium):** Filters rhenium-tagged facilities to processing plants only (Nos plant, which is Molymet's refinery). The 4 Mo processing plants each get an edge to Nos plant, and Nos plant gets one edge to San Antonio. 4 mo_to_re_plant + 1 rhenium_to_port = 5 edges.

**Cell 16 (Boron):** Filters to active mines and processing plants within 800km of boron ports. 3 facilities (Salar de Atacama, plus 2 boric acid plants) routed to Antofagasta (ATI), Iquique, or Arica. 3 edges.

**Cell 17 (Sulfuric Acid):** Verifies existing acid edges (3 smelter-to-port edges from the copper section). No new edges added, as H2SO4 is mostly consumed domestically by SX-EW plants.

### Cell 18: Processing Dead-End Sweep

Finds all processing-stage facilities that have upstream mine_to_plant edges but zero downstream edges, excluding cement, methanol, petroleum, iron, and iodine (handled separately). Routes each to the nearest port as `processing_to_port`. Covers ENAMI custom plants (Centinela oxide, Vallenar, Osvaldo Martinez, Jose Antonio Moreno, Manuel Antonio Matta, etc.). 19 edges.

### Cell 19: Merge, Deduplicate, Save

Concatenates all new non-copper edges with the existing edge table. Deduplicates by `(FROM_NAME, TO_NAME, EDGE_TYPE, COMMODITIES, PRODUCT_FORM)`. Saves `_pipeline_state_5.pkl`.

---

## Notebook C: Validation, Distance Analysis and Summary

**Input:** `_pipeline_state_5.pkl` from Notebook B.
**Output:** `_pipeline_state_6.pkl` (final state) plus all output CSVs.

### Cell 3 (Section 5): Cleanup and Validation

**5A. Smelter Name Standardization:** Iterates over `SMELTER_NAME_MAP` and renames any edge references from canonical short names to inventory long names. Handles 3 renames where Notebook B used short names that differ from the inventory.

**5A.2. Entity Consolidation:** Remaps duplicate smelter entries to their canonical form. "Ventanas refinery and smelter" (4 commodity-variant rows) is consolidated into "Las Ventanas refinery and smelter". "Chagres smelter" (Sulfuric Acid variant) is consolidated into "Chagres smelter (anodes and blister)". Affects both edge FROM_NAME/TO_NAME and link MINE_NAME/PLANT_NAME columns.

**5B. Andacollo Oro Mine Link:** Checks whether the Andacollo Oro mine has a link to a processing plant. If missing, creates a link and edge to the nearest SX-EW plant within 50km.

**5B.2. Missing Smelter-to-Port Edges:** Adds smelter-to-port edges for three smelters whose long inventory names were not matched during Notebook B's edge construction. Caletones gets edges to San Antonio and Ventanas; Hernan Videla Lira to Barquito and Caldera; Las Ventanas to Ventanas and San Antonio.

**5B.3. Caserones Mine-to-Plant Link:** Ensures the Caserones mine (124.6 kMT copper) has a link to a processing plant. Searches for a named match first, then by proximity within 50km, with a direct mine-to-port fallback.

**5C. Deduplication:** Removes duplicate edges by the same five-column key used in Notebook B's merge.

**5D. Validation:** Verifies copper production coverage (31 records, 5,281.6 kMT matched) and performs path traceability: checks that every copper producer can reach at least one export port through the edge network. Uses set-based lookup for efficiency. Also handles the edge case where COCHILCO production is attached directly to SX-EW or concentrator records rather than separate mine records.

**5E. Diagnostics Summary:** Prints edge type counts, downstream coverage by commodity, and smelter connectivity (upstream mine-to-plant count and downstream smelter-to-port count for each smelter).

**5F. Save:** Sorts edges, writes all output CSVs, and prints remaining issues.

### Cell 5 (Section 6): Distance-Based Port Assignment Analysis

Compares the pipeline's modelled port assignments (based on haversine distance and overrides) against actual 2024 Aduanas customs data. Computes port-level shares for both modelled and actual flows, then calculates mean absolute difference and total volume mismatch for concentrate and cathode separately.

For concentrate, the mean absolute port share difference is approximately 4.3%, with a total volume mismatch of 27.6%. For cathode the figures are approximately 6.5% and 25.9%. The main discrepancies arise because the model assigns all of a mine's output to a single nearest port, while in practice mines split shipments across multiple ports for commercial or logistical reasons.

Also produces a mine-port distance matrix (31 mines x 13 ports) and optimal port assignment table.

### Cells 7-14 (Section 7): Pipeline Summary and Integrity Checks

Eight cells producing a structured summary of the final pipeline state:

1. **Inventory Summary:** Facility counts by type, geographic distribution by region, column count.
2. **Supply Chain Edges:** Edge type counts, commodity distribution.
3. **Copper Production Coverage:** Matched versus national totals, top producers.
4. **Connectivity Checks:** Ensures all smelters have upstream and downstream edges. Checks for orphan nodes.
5. **Data Quality:** Coordinate completeness, missing values in critical columns.
6. **Export Destinations:** Country coverage by commodity, top export partners.
7. **Port Utilisation:** Volume and edge count per port.
8. **Output File Check:** Verifies all expected CSV files exist with correct row counts.

### Cells 16-23: Save Final State

Saves `_pipeline_state_6.pkl` and performs any post-hoc analysis or visualization setup.

---

## Pipeline State Propagation

| State File | Producer | Contents Added |
|---|---|---|
| `_pipeline_state_0.pkl` | Cell A.1 | Inventory, links, idle mines, constants |
| `_pipeline_state_1.pkl` | Cell A.3 | Cu/Mo/Li production columns, link fixes |
| `_pipeline_state_2.pkl` | Cell A.11 | 32 mineral production columns, USD valuations, commodity_prices, MINERAL_COLS, national_production |
| `_pipeline_state_5.pkl` | Cell B.19 | Edge table (all minerals), smelter_inv_map, export_df, PORT_PRODUCT_MAP |
| `_pipeline_state_6.pkl` | Cell C.16 | Final cleaned edges, validated production, output CSVs |

---

## Output Files

| File | Records | Description |
|---|---|---|
| Chile_Minerals_Inventory.csv | 461 | Full facility inventory with ~163 columns |
| Chile_Mine_Plant_Links.csv | 1,109 | Mine-to-plant linkage (idle links removed) |
| Chile_Supply_Chain_Edges.csv | ~2,400 | Complete directed edge table |
| Chile_Downstream_Links.csv | ~1,200 | Non-upstream edges only |
| Chile_Ports.csv | 13 | Port coordinates and product types |
| Chile_Export_Destinations.csv | ~1,065 | Port-to-country export edges with values |
| Mine_Port_Distance_Matrix.csv | 31 x 13 | Haversine distances (km) |
| Mine_Optimal_Port_Assignments.csv | 31 | Modelled port assignment per mine |
| Chile_National_Production_2024.csv | ~35 | All minerals with national totals and allocation status |

---

## Commodity Coverage Summary

### Complete Chain (mine -> plant -> port -> country)

| Mineral | Upstream | Domestic Downstream | Export | Total Edges |
|---|---|---|---|---|
| Copper | mine_to_plant (844) | sxew/smelter/conc/proc_to_port (44) | port_to_country (480) | 1,368 |
| Nitrate | mine_to_plant (9) | nitrate_to_port (7) | port_to_country (147) | 163 |
| Gold | mine_to_plant (91), mine_to_smelter (20) | gold_to_airport (8), proc_to_port (5) | port_to_country (27) | 156 |
| Iodine | mine_to_plant (59) | iodine_to_port (6) | port_to_country (84) | 149 |
| Silver | mine_to_plant (55), mine_to_smelter (12) | silver_to_airport (6), proc_to_port (6) | port_to_country (24) | 104 |
| Molybdenum | mine_to_plant (37), mine_to_mo_plant (12) | mo_to_port (4) | port_to_country (27) | 83 |
| Iron | mine_to_plant (45) | iron_to_port (7) | port_to_country (21) | 78 |
| Boron | mine_to_plant (2) | boron_to_port (3) | port_to_country (62) | 67 |
| Lithium | mine_to_plant (3) | lithium_to_port (3) | port_to_country (30) | 36 |
| Rhenium | mine_to_plant (5), mo_to_re_plant (4) | rhenium_to_port (1) | port_to_country (8) | 18 |
| Zinc | mine_to_plant (4) | zinc_to_port (2) | port_to_country (4) | 10 |

### Export Only (port -> country, no domestic chain)

| Mineral | Export Edges | Countries |
|---|---|---|
| Salt | 80 | 20 |
| Copper Sulfate | 48 | 16 |
| Potash | 45 | 15 |
| Sulfuric Acid | 15 | 4 |
| Lead | 12 | 6 |
| Selenium | 2 | 1 |

### Production and Valuation Only (no supply chain edges)

Bentonite, Diatomite, Dolomite, Gypsum, Kaolin, Pumicite, Quartz, Salt, Silica Sand, White CaCO3, Zeolite, Copper Sulfate, Limestone, Coquina, Perlite, Peat, Phosphate Rocks, Bauxitic Clay. These have production columns and USD valuations in the inventory but no domestic routing edges. National totals are captured in `Chile_National_Production_2024.csv`.
