# Chile Mineral Supply Chain Pipeline: Code Summary

## Purpose

This notebook (`Chile_MainAn.ipynb`) constructs a directed supply chain graph for Chile's mineral sector. It links upstream extraction facilities (mines) through processing stages (concentrators, SX-EW plants, smelters, refineries) to export ports and final destination countries. The analysis integrates three distinct data sources: USGS/Sernageomin facility inventories, COCHILCO annual production and trade statistics (2005-2024), and Chilean customs code tables.

The pipeline is part of a broader capstone project analyzing natural resource production and supply chain structures across countries.

---

## Input Files

| File | Description |
|------|-------------|
| `Preliminary/Chile_Minerals_Inventory.csv` | Facility-level inventory from `Chile_Pipeline.py`, containing mine and plant records with coordinates, operators, and commodity info |
| `Preliminary/Chile_Mine_Plant_Links.csv` | Distance-based linkages between mines and nearby processing plants, generated upstream |
| `data/COCHILCO_Production_2005_2024.xlsx` | Extracted and cleaned production tables from COCHILCO's annual yearbook |
| `Anuario-de-Estadisticas-del-Cobre-y-otros-Minerales-2005-2024.xlsx` | Original COCHILCO yearbook (used for raw Molybdenum and export destination parsing via `openpyxl`) |
| `tablas_de_codigos.xlsx` | Chilean customs reference tables: port codes, country codes, customs offices, transport modes, trade clauses, regions, currencies, and units of measure |

---

## Pipeline Structure (7 Sections)

### Section 1: Setup

Loads all libraries (`pandas`, `numpy`, `openpyxl`, `matplotlib`, `seaborn`) and defines file paths rooted at `/Users/leoss/Desktop/GitHub/Capstone/Case studies/Chile`. Establishes two key reference dictionaries:

- **`COMPANY_TO_DEPOSIT`**: Maps COCHILCO company names (e.g., "Division El Teniente") to search terms used to match them against facility names in the inventory. Covers all major Codelco divisions plus private operators (Escondida, Collahuasi, Los Pelambres, Spence, Quebrada Blanca, etc.).
- **`CODELCO_MAP`**: Similar structure specifically for Codelco divisions, with additional metadata for operator assignment.

Also loads the inventory and links CSVs, filters out idle mine links, and defines utility functions for section headers and printing.

### Section 2: COCHILCO Integration (Copper, Molybdenum, Lithium)

This section matches COCHILCO's company-level production data to individual facilities in the inventory. It proceeds in sub-steps:

- **2A. Codelco Division Mapping**: Uses fuzzy name matching to assign Codelco as operator for facilities matching division search terms.
- **2B. Copper Production Matching**: Reads the national production table from COCHILCO (`A_National_Production` sheet), extracts the latest year's copper output per company (in thousands of metric tons), and assigns values to matching inventory records. Adds columns `COCHILCO_CU_2024_KMT` and `COCHILCO_COMPANY` to the inventory.
- **2C. Molybdenum Production**: Parses raw Molybdenum tables directly from the original COCHILCO yearbook using `openpyxl` (the cleaned Excel did not capture Mo properly). Assigns `COCHILCO_MO_2024_MT` to matching facilities.
- **2D. Lithium Fixes**: Deduplicates lithium-related links (since multiple plant types serve the same lithium operations) and cleans up lithium facility assignments.

### Section 3: Downstream Supply Chain

Classifies each facility into a processing stage and builds the downstream edge list:

- **3A. Processing Stage Classification**: Maps `FACILITY_TYPE` values to standardized `CHAIN_STAGE` labels: `extraction`, `concentration`, `sx_ew`, `smelting`, `refining`, `processing`, `other`. Applies keyword-based refinement for ambiguous "Processing Plant" entries (e.g., if the name contains "fundici" or "smelting", reclassify as smelting).
- **3B. Smelter and Port Setup**: Identifies Chile's copper smelters (Chuquicamata, Potrerillos, Paipote, Caletones, Chagres, Hernan Videla Lira, Ventanas) and major export ports. Builds a `ports_df` DataFrame with port names and coordinates. Uses a `SMELTER_NAME_MAP` to reconcile naming differences between COCHILCO and the inventory.
- **3C. Edge Construction**: Creates directed edges between facilities based on the mine-plant links and stage progression logic. Produces upstream edges (mine to concentrator/SX-EW) and downstream edges (concentrator to smelter, smelter to refinery, processing to port). Assigns `PRODUCT_FORM` labels (concentrate, cathode, blister, refined, etc.) and `EDGE_TYPE` categories.

### Section 4: Export Destinations (COCHILCO Section D)

The largest section (~29,000 chars). Parses COCHILCO's Section D destination tables directly from the original yearbook using `openpyxl`:

- Reads multiple sheets covering different product categories (copper concentrates, blister/anode, refined cathodes, copper wire, molybdenum products, lithium, iron, gold, silver).
- For each product, extracts country-level export volumes for 2024.
- Handles Spanish/English bilingual country names, region headers (Europa, America, Asia, Oceania, Africa), and various formatting inconsistencies in the original COCHILCO spreadsheets.
- Maps country names to standardized forms and assigns port-to-country edges based on which Chilean ports handle which product types.
- Also integrates Chilean customs data (`tablas_de_codigos.xlsx`) for port code lookups and share allocation (the `Chile_Port_Shares_Aduanas.csv` output).
- Produces `Chile_Export_Destinations.csv` with columns for product type, destination country, export volume, and originating port.

### Section 5: Cleanup and Validation

- **5A. Smelter Name Standardization**: Ensures edge table names match inventory names exactly, using the `SMELTER_NAME_MAP`. Handles the "Las Ventanas" vs. "Ventanas refinery" ambiguity.
- **5B. Edge Deduplication**: Removes duplicate edges by `(FROM_NAME, TO_NAME, EDGE_TYPE, COMMODITIES, PRODUCT_FORM)`.
- **5C. Path Tracing**: Validates that complete supply chain paths exist from mine to export destination. Traces paths through the edge graph for sample facilities.

Saves all final outputs: updated inventory, links, downstream links, unified supply chain edges, and export destinations.

### Section 6: Diagnostics

Runs a merged audit checking for:

- Multi-match cases where a single COCHILCO company name matched multiple inventory records (and which one was selected).
- Plant attribution cases where no mine was found and production was assigned to a processing facility instead.
- Missing matches where COCHILCO companies could not be linked to any inventory record.
- Edge consistency checks ensuring all edge endpoints exist in the inventory.

### Section 7: Port Distance Comparison

Compares actual port assignments (from the supply chain edges) against optimal assignments based on geographic distance:

- Extracts all mines with production data and coordinates.
- Determines each mine's primary product type (concentrate vs. cathode) by looking for associated processing facilities in the inventory by name matching, with manual overrides for cases where the inventory is outdated (e.g., Spence, Quebrada Blanca post-expansion).
- Computes haversine distances from each mine to all eligible ports.
- Compares the actual port used (from edges) to the nearest port, flagging cases where mines ship through distant ports.
- Saves `Port_Distance_Comparison.csv`, `Mine_Optimal_Port_Assignments.csv`, and `Mine_Port_Distance_Matrix.csv`.

---

## Output Files

All saved to the `Preliminary/` subdirectory:

| File | Content |
|------|---------|
| `Chile_Minerals_Inventory.csv` | Updated inventory with `CHAIN_STAGE`, `COCHILCO_CU_2024_KMT`, `COCHILCO_MO_2024_MT`, `COCHILCO_COMPANY` |
| `Chile_Mine_Plant_Links.csv` | Links with added `PRODUCT_FORM` column |
| `Chile_Downstream_Links.csv` | Processing-to-port and smelter-to-refinery edges |
| `Chile_Supply_Chain_Edges.csv` | Unified edge list across all 4 layers (upstream, downstream, port, export) |
| `Chile_Export_Destinations.csv` | Country-level export volumes by product type and port |
| `Chile_Ports.csv` | Port names and coordinates |
| `Chile_Port_Shares_Aduanas.csv` | Customs-derived port throughput shares |
| `Port_Distance_Comparison.csv` | Actual vs. optimal port assignment per mine |
| `Mine_Optimal_Port_Assignments.csv` | Closest port for each mine by product type |
| `Mine_Port_Distance_Matrix.csv` | Full distance matrix from mines to ports |

---

## Key Technical Decisions

- **Name matching over IDs**: COCHILCO data uses company/division names rather than facility IDs, so the pipeline relies on fuzzy string matching via `COMPANY_TO_DEPOSIT` and `CODELCO_MAP` dictionaries. This is the most fragile part of the pipeline and where most manual overrides concentrate.
- **Raw XLSX parsing for Molybdenum and exports**: The pre-cleaned COCHILCO Excel did not preserve Molybdenum tables or Section D destination data correctly, so the pipeline reads the original yearbook directly with `openpyxl`, parsing row-by-row with heuristics for header detection.
- **Product type inference**: Mines are classified as producing concentrate or cathode based on the types of processing facilities that share their name in the inventory, rather than from the links table (which reflects geographic proximity, not operational relationships).
- **Manual overrides**: Several mines have hardcoded product type overrides because recent expansions (Spence concentrator 2021, QB2 2023) are not yet reflected in the USGS/Sernageomin inventory.

---

## Dependencies

`pandas`, `numpy`, `openpyxl`, `matplotlib`, `seaborn`, `os`, `shutil`, `re`, `collections.Counter`

---

## Notes for Future Users

- The notebook is structured as a single large cell with `# %%` section markers (VS Code/Jupyter cell mode). It runs sequentially and takes moderate time due to the `openpyxl` parsing of large COCHILCO workbooks.
- The `tablas_de_codigos.xlsx` file provides the reference codes for Chilean customs data (ports, countries, regions, trade terms). It is used in Section 4 for port code resolution.
- If COCHILCO updates their yearbook format or adds new companies/divisions, the `COMPANY_TO_DEPOSIT` and `SMELTER_NAME_MAP` dictionaries will need manual updates.
- All paths are hardcoded to `/Users/leoss/Desktop/GitHub/Capstone/Case studies/Chile`. Adjust `BASE_DIR` if running from a different location.
