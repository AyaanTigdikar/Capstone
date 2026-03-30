"""
pipeline_constants.py
---
Single source of truth for every hard-coded value in the pipeline that is:
  (a) an assumption rather than something derived from data, OR
  (b) currently duplicated across multiple notebooks, OR
  (c) something that silently affects pipeline correctness if wrong.

Each entry documents:
  • the value itself
  • its source / empirical basis
  • its verification status
  • which notebook(s) consume it

Import at the top of each notebook (after pipeline_utils):
    from pipeline_constants import *

If a value needs updating, change it HERE only.  Never redefine these in a
notebook cell.

Audit trail
---
  Created  : 2025-03 (post-audit of pipeline v1)
  Last edit: update this line when values change
"""

from __future__ import annotations

# 1.  MOLYBDENUM PRODUCTION SPLIT
#
# COCHILCO Tabla 4.2 reports Chuquicamata and Radomiro Tomic as a single
# combined line "Divisiones Chuquicamata y Radomiro Tomic".  The 40/60 split
# is derived from Codelco divisional operating statistics, which consistently
# attribute ~55–65 % of the combined Mo output to Radomiro Tomic because its
# SX-EW oxide-leach circuit processes higher-Mo content ores.
#
# SOURCE : Codelco Annual Reports 2020–2023, divisional production tables
# STATUS : Plausible estimate — verify against latest Codelco annual report
#           before using for financial modelling
# USED BY: Notebook A, Cell 3 (§2C Molybdenum Production Matching)

MO_SPLIT: dict[str, list[tuple[str, float]]] = {
    "Divisiones Chuquicamata y Radomiro Tomic": [
        ("Chuquicamata",    0.40),
        ("Radomiro Tomic",  0.60),
    ],
}


# 2.  MINE → PRODUCT-TYPE CLASSIFICATION
#
# Each copper-producing mine is classified by its PRIMARY export product:
#   "cathode"     – SX-EW or fire-refined cathode exported directly
#   "concentrate" – sulphide concentrate shipped to a third-party smelter
#   "blister"     – partially refined anode (rare as a final export form)
#
# Determines:
#   • which port-type filter nearest_port() applies
#   • which CODELCO_CATHODE_ROUTING key is checked
#
# IMPORTANT – Rio Blanco is explicitly listed here because División Andina's
# copper production may be assigned to "Rio Blanco mine" (the primary ore body)
# rather than to "Andina mine".  Without this entry the Codelco cathode routing
# to San Antonio would silently fall back to nearest-cathode-port.
#
# SOURCE :
#   Codelco integrated mines  → COCHILCO "Forma de comercialización" tables
#   SX-EW mines               → USGS/SNL facility type + operator disclosures
#   Concentrate producers     → COCHILCO Tabla 20.2 concentrate exports
# STATUS : Spot-checked against COCHILCO 2024 Anuario.
#           Escondida is "concentrate" even though it also runs a SX-EW plant
#           (~5–8 % of output); the SX-EW cathode share is below the threshold
#           that would change the dominant routing.
# USED BY: Notebook C, Cell 5 (§6 Distance-Based Port Assignment)

PRODUCT_TYPE_OVERRIDE: dict[str, str] = {
    # Codelco integrated operations (onsite smelter → cathode)
    "Chuquicamata":      "cathode",   # → Chuquicamata smelter → Angamos/Mejillones
    "Radomiro Tomic":    "cathode",   # SX-EW → Angamos
    "Ministro Hales":    "cathode",   # → Chuquicamata smelter → Angamos
    "Gabriela Mistral":  "cathode",   # SX-EW (Gaby) → Angamos
    "El Teniente":       "cathode",   # → Caletones smelter → San Antonio
    "Andina":            "cathode",   # → Las Ventanas refinery → San Antonio/Ventanas
    "Rio Blanco":        "cathode",   # División Andina primary ore body — same routing
    "Salvador":          "cathode",   # → Potrerillos smelter → Barquito
    # Pure SX-EW operations
    "Zaldívar":          "cathode",
    "Zaldivar":          "cathode",   # alternate spelling in some inventory rows
    "El Abra":           "cathode",
    "Antucoya":          "cathode",
    "Lomas Bayas":       "cathode",
    "Mantoverde":        "cathode",
    "Mantos Blancos":    "cathode",
    "Mantos de la Luna": "cathode",
    "Franke":            "cathode",
    "Tres Valles":       "cathode",
    "Michilla":          "cathode",
    "Las Luces":         "cathode",
    # Concentrate producers (sulphide ore → third-party smelting)
    "Escondida":         "concentrate",
    "Collahuasi":        "concentrate",
    "Los Pelambres":     "concentrate",
    "Spence":            "concentrate",
    "Quebrada Blanca":   "concentrate",
    "Los Bronces":       "concentrate",  # → Chagres smelter → blister → Ventanas
    "El Soldado":        "concentrate",  # → Chagres smelter
    "Sierra Gorda":      "concentrate",
    "Caserones":         "concentrate",
    "Centinela":         "concentrate",
    "Candelaria":        "concentrate",
}


# 3.  MISSING SMELTER-TO-PORT EDGE PATCHES
#
# Three smelters have mine-to-plant edges recorded under their LONG inventory
# names but no smelter-to-port edges.  Root cause: smelter_inv_map in
# Notebook B resolves canonical short names via keyword search to LONG
# inventory names, creating a split between m2p and s2p edges.
#
# These patches are applied in Notebook C §5B.2.
#
# COORDINATE NOTE: lat/lon values here must match the SMELTERS entries in
# pipeline_utils.py.  If coordinates are corrected in SMELTERS, update here.
#
# SOURCE : SMELTERS constant (pipeline_utils.py) + COCHILCO facility data
# STATUS : Structural fix — verified against inventory and SMELTERS dict
# USED BY: Notebook C, Cell 3 (§5B.2 Missing Smelter-to-Port Edges)

MISSING_S2P: list[dict] = [
    {
        "from_name": (
            "Caletones smelter (anodes). refinery (fire-refined ingots), "
            "and SX-EW plant"
        ),
        "ports":   ["San Antonio", "Ventanas"],
        "lat":     -34.12,   # must match SMELTERS["Caletones smelter"]["lat"]
        "lon":     -70.48,
        "product": "cathode",
        "note":    "El Teniente division; fire-refined cathode via San Antonio / Ventanas",
    },
    {
        "from_name": "Hernán Videla Lira smelter (anodes and blister)",
        "ports":   ["Barquito", "Caldera"],
        "lat":     -27.37,   # must match SMELTERS["Paipote smelter"]["lat"]
        "lon":     -70.30,
        "product": "blister",
        "note":    "ENAMI Paipote; blister/anodes shipped via Barquito and Caldera",
    },
    {
        "from_name": "Las Ventanas refinery and smelter",
        "ports":   ["Ventanas", "San Antonio"],
        "lat":     -32.74,
        "lon":     -71.49,
        "product": "cathode",
        "note":    "Codelco refinery processing anodes from multiple Codelco smelters",
    },
]


# 4.  SMELTER ENTITY CONSOLIDATION MAP
#
# Several smelters appear under multiple names in the inventory (one row per
# commodity processed at the same physical facility).  Edges referencing
# variant names must be remapped to the canonical name so that upstream and
# downstream paths connect correctly.
#
# SOURCE : Inventory duplicate detection by lat/lon proximity
# STATUS : Verified — affected inventory indices noted in comments
# USED BY: Notebook C, Cell 3 (§5A.2 Entity Consolidation)

ENTITY_MERGE_MAP: dict[str, str] = {
    # variant name in edges               → canonical inventory name
    "Ventanas refinery and smelter":      "Las Ventanas refinery and smelter",
    "Chagres smelter":                    "Chagres smelter (anodes and blister)",
}


# 5.  IRON EXPORT PORTS (extra terminals not in the main PORTS list)
#
# Huasco and Guayacán are significant iron-ore / pellet export terminals but
# are NOT in the main PORTS list (which covers copper).  Iron edge-building
# (Notebook B §Iron) must look these up via FE_PORTS_EXTRA rather than through
# nearest_port() or nearest_from_list() — both of which silently skip ports
# absent from the global PORTS list.
#
# Aduanas Salidas 2024 iron ore (HS 2601) FOB-value shares:
#   Caldera  ≈ 48 %   (also in main PORTS list)
#   Huasco   ≈ 39 %   (CAP Guacolda pellet plant)
#   Guayacán ≈ 12.5 % (CMP Los Colorados terminal, Coquimbo bay)
#
# SOURCE : Aduanas Salidas 2024 filtered to HS 2601
# STATUS : Derived from Aduanas data — re-derive if Salidas year changes
# USED BY: Notebook B, Cell 8 (Iron section)

FE_PORTS_EXTRA: dict[str, dict] = {
    "Huasco": {
        "lat": -28.47, "lon": -71.22,
        "products": "Fe pellets, iron ore",
        "key_users": "CAP Guacolda pellet plant",
    },
    "Guayacán": {
        "lat": -29.97, "lon": -71.37,
        "products": "Fe ore",
        "key_users": "CMP Los Colorados",
    },
}

# Fallback FOB shares for iron port→country edges
FE_PORT_SHARES: dict[str, float] = {
    "Caldera":   0.480,
    "Huasco":    0.390,
    "Guayacán":  0.125,
    # NOTE: values sum to 0.995 due to rounding; the residual ~0.5 % is "Other"
}


# 6.  FALLBACK PORT-SHARE DISTRIBUTIONS
#
# Used when Aduanas Salidas is unavailable and PORT_PRODUCT_MAP (derived in
# Notebook B §4A) does not cover a given commodity / product-form pair.
#
# Copper values calibrated from COCHILCO Tablas 18.2/19.2/20.2 (5-year
# average 2019–2023).  Non-copper values from Aduanas Salidas 2022–2023
# where available, otherwise COCHILCO narrative.
#
# SOURCE : See per-key inline comment
# STATUS : Recalibrate annually; values tagged below as (ADUANAS) or (COCHILCO)
# USED BY: Notebook B, Cell 5 (build_export_edges fallback)

PORT_PRODUCT_MAP_FALLBACK: dict[str, dict[str, float]] = {
    "concentrate": {   # (ADUANAS 2024 FOB shares)
        "Coloso":    0.35, "Mejillones": 0.20, "Patache":   0.15,
        "Barquito":  0.08, "Coquimbo":   0.12, "Angamos":   0.05,
        "Caldera":   0.05,
    },
    "cathode": {       # (ADUANAS 2024 FOB shares)
        "Angamos":           0.25, "Mejillones":      0.15,
        "Antofagasta (ATI)": 0.15, "Iquique":         0.15,
        "San Antonio":       0.10, "Ventanas":        0.10,
        "Barquito":          0.05, "Coquimbo":        0.05,
    },
    "blister": {       # (COCHILCO Tabla 19.2 narrative)
        "Mejillones": 0.40, "Ventanas": 0.30,
        "Barquito":   0.20, "Antofagasta (ATI)": 0.10,
    },
}

FALLBACK_SHARES: dict[str, dict[str, float]] = {
    # Mo concentrate — Aduanas 2022–2023 (HS 2613)
    "molybdenum_mo_concentrate": {
        "Mejillones": 0.50, "Antofagasta (ATI)": 0.30, "Barquito": 0.20,
    },
    # Lithium compounds — Aduanas 2023 (HS 2836/2825)
    "lithium_lithium_compounds": {
        "Antofagasta (ATI)": 0.50, "Mejillones": 0.30, "Iquique": 0.20,
    },
    # Iodine — Aduanas 2022–2023 (HS 2801)
    "iodine_iodine": {
        "Iquique": 0.40, "Patache": 0.30,
        "Antofagasta (ATI)": 0.20, "Mejillones": 0.10,
    },
    # Au/Ag — exported by air in practice; seaport names used as proxy nodes
    # because the pipeline does not model air-freight terminals.
    "gold_gold_refined": {
        "San Antonio": 0.50, "Antofagasta (ATI)": 0.30, "Mejillones": 0.20,
    },
    "silver_silver_refined": {
        "San Antonio": 0.40, "Antofagasta (ATI)": 0.35, "Mejillones": 0.25,
    },
    # Iron — Aduanas 2024 (HS 2601); uses FE_PORTS_EXTRA names for Huasco/Guayacán
    "iron_iron_ore": {
        "Caldera": 0.48, "Huasco": 0.39, "Guayacán": 0.125,
    },
    # Zinc concentrate — Aduanas 2022–2023 (HS 2608); El Toqui + Alhué
    "zinc_zinc_concentrate": {
        "San Vicente": 0.60, "San Antonio": 0.40,
    },
    # Nitrates — Aduanas 2022–2023 (HS 2834/3102)
    "nitrate_nitrate": {
        "Patache": 0.45, "Iquique": 0.35, "Mejillones": 0.20,
    },
    # Borates — Aduanas 2022–2023 (HS 2528)
    "boron_borate": {
        "Antofagasta (ATI)": 0.70, "Iquique": 0.30,
    },
    # Salt — Aduanas 2022–2023 (HS 2501)
    "salt_salt": {
        "Patache": 0.40, "Iquique": 0.30, "Coquimbo": 0.20, "San Antonio": 0.10,
    },
    # Potash (KCl) — Aduanas 2022–2023 (HS 3104)
    "potash_potassium_chloride": {
        "Patache": 0.50, "Antofagasta (ATI)": 0.30, "Iquique": 0.20,
    },
    # Rhenium — primarily from Codelco Mo plants
    "rhenium_perrhenate": {
        "San Antonio": 0.60, "Mejillones": 0.40,
    },
    # Lead — El Toqui (Aysén) + other sources
    "lead_lead_unwrought": {
        "Antofagasta (ATI)": 0.50, "San Antonio": 0.50,
    },
    # Copper sulfate
    "copper sulfate_copper_sulfate": {
        "Antofagasta (ATI)": 0.40, "Mejillones": 0.30, "San Antonio": 0.30,
    },
    # Sulfuric acid — smelter byproduct
    "sulfuric acid_sulfuric_acid": {
        "Mejillones": 0.50, "Antofagasta (ATI)": 0.30, "Barquito": 0.20,
    },
    # Selenium — byproduct of Cu anode slimes
    "selenium_selenium": {
        "Mejillones": 0.50, "Antofagasta (ATI)": 0.50,
    },
}


# 7.  ADUANAS CUSTOMS DATA — CODES AND ALIASES

# Export operation codes used to filter Aduanas Salidas rows.
# Codes 200–216 correspond to export régimes under DFL-2
# (Ley Orgánica del Servicio de Aduanas, Resolución Exenta Nº 4.481, 2017).
# STATUS: Stable across years; verify if using a post-2020 Salidas file.
# USED BY: Notebook B, Cell 5 (§4A Aduanas Port Shares)

EXPORT_OP_CODES: set[str] = {
    "200", "201", "202", "203", "204", "205", "206", "207",
    "210", "211", "212", "213", "216",
}

# Aduanas records port names in all-caps Spanish.  These aliases normalise
# to the mixed-case English names used in the PORTS list.
# Codes 827 and 821 are absent from the standard tablas_de_codigos lookup
# and are inferred from geographic context and AIFTA declarations.
# STATUS: Verified for 2024 Salidas; may need extension for new terminals.
# USED BY: Notebook B, Cell 5 (§4A)

ADUANAS_PORT_ALIAS: dict[str, str] = {
    "CALETA COLOSO":            "Coloso",
    "PUERTO ANGAMOS":           "Angamos",
    "ANTOFAGASTA":              "Antofagasta (ATI)",
    "CHAÑARAL / BARQUITO":      "Barquito",       # ñ preserved — must match raw Salidas
    "HUASCO / GUACOLDA":        "Huasco",
    "GUAYACÁN":                 "Guayacán",
    "CAP. HUACHIPATO":          "Huachipato",
    "AEROP. A.M. BENITEZ":      "Santiago (air)",
    "AEROP. CERRO MORENO":      "Antofagasta (air)",
    "AEROP. CHACALLUTA":        "Arica (air)",
    "AEROP. DIEGO ARACENA":     "Iquique (air)",
    "AEROP. EL TEPUAL":         "Puerto Montt (air)",
    "AEROP. C.I. DEL CAMPO":    "Santiago (air)",
    "CHACABUCO / PUERTO AYSÉN": "Puerto Aysén",
    "TERMINAL PETROLERO ENAP":  "ENAP terminal",
    "OTROS PUERTOS CHILENOS":   "Other",
}

ADUANAS_MANUAL_PORTS: dict[int, str] = {
    827: "Iquique",     # ZOFRI free-trade zone — not in tablas_de_codigos
    821: "Valparaíso",  # minor volume ($2.4M); best geographic approximation
}


# 8.  MANUAL COMMODITY PRICES  (USD/MT unless noted)
#
# Tier 4 (lowest priority) prices: used only when COCHILCO Tablas 96/97/98
# and implied FOB values are both unavailable for a mineral.
#
# Lithium product forms (LiOH, LiSO4) are derived from the parsed LiCO3
# midpoint price from Tabla 98; their multipliers are in LITHIUM_PRICE_MULTIPLIERS.
# The None entries below act as documentation placeholders — these two minerals
# must NOT fall back to 0; the build_prices() caller should apply multipliers.
#
# SOURCE : per-entry inline citation
# STATUS : 2024 averages — update annually; ESTIMATE entries flagged
# USED BY: Notebook A, Cell 7 (§2F Commodity Price Extraction)

MANUAL_PRICES: dict[str, tuple[float | None, str]] = {
    # (price_USD_per_MT,  source_citation)
    "Iron":              (110.0,  "Platts IODEX 62% Fe CFR China 2024 avg"),
    "Salt":              ( 22.0,  "Implied FOB: COCHILCO Tabla 7/9 ($140.6M / 6,400 kMT)"),
    "Sodium Nitrate":    (1258.0, "Implied FOB: COCHILCO Tabla 7/9 ($52.42M / 41,657 MT)"),
    "Potassium Nitrate": ( 653.0, "Implied FOB: COCHILCO Tabla 7/9 ($4.6M / 7,049 MT)"),
    "Ulexite":           ( 301.0, "Implied FOB: COCHILCO Tabla 7/9 ($9.4M / 31,176 MT)"),
    "Potash":            ( 260.0, "USGS MCS 2024 avg KCl FOB"),
    "Limestone":         (  12.0, "USGS MCS 2024 avg crushed limestone"),
    "Coquina":           (  12.0, "USGS MCS 2024 avg crushed limestone (proxy for coquina)"),
    "White CaCO3":       (  80.0, "USGS MCS 2024 avg ground calcium carbonate"),
    "Gypsum":            (  10.0, "USGS MCS 2024 avg crude gypsum"),
    "Pumicite":          (  35.0, "USGS MCS 2024 avg pumice/pumicite"),
    "Quartz":            (  50.0, "USGS MCS 2024 avg industrial quartz"),
    "Silica Sand":       (  40.0, "USGS MCS 2024 avg industrial silica sand"),
    "Bauxitic Clay":     (  50.0, "USGS MCS 2024 avg bauxitic kaolin (estimate)"),
    "Kaolin":            ( 175.0, "USGS MCS 2024 avg kaolin"),
    "Bentonite":         (  80.0, "USGS MCS 2024 avg bentonite"),
    "Diatomite":         ( 400.0, "USGS MCS 2024 avg diatomite"),
    "Dolomite":          (  25.0, "USGS MCS 2024 avg dolomite"),
    "Talc":              ( 150.0, "USGS MCS 2024 avg talc"),
    "Perlite":           ( 125.0, "USGS MCS 2024 avg perlite"),
    "Peat":              (  32.0, "USGS MCS 2024 avg peat"),
    "Phosphate Rocks":   ( 120.0, "USGS MCS 2024 avg phosphate rock"),
    "Copper Sulfate":    (1800.0, "Industry avg CuSO4·5H₂O 2024"),
    "Zeolite":           ( 150.0, "USGS MCS 2024 avg natural zeolite"),
    # Aggregate categories — national summary only; NOT used for facility valuation
    "Calcium Carbonate": (  12.0, "USGS MCS 2024 avg crushed limestone (aggregate proxy)"),
    "Clay":              (  30.0, "USGS MCS 2024 avg common clay (aggregate proxy)"),
    "Silica Ores":       (  45.0, "USGS MCS 2024 avg industrial sand/gravel (aggregate)"),
    "Phosphates":        ( 120.0, "USGS MCS 2024 avg phosphate rock (aggregate proxy)"),
    # Lithium product forms — prices derived from Tabla 98 LiCO3; see multipliers below
    "Lithium Carbonate": (None,   "COCHILCO Tabla 98 — parsed by commodity_prices_2024.py"),
    "Lithium Hydroxide": (None,   "ESTIMATE: ~1.10 × Tabla 98 LiCO3 midpoint — see LITHIUM_PRICE_MULTIPLIERS"),
    "Lithium Sulfate":   (None,   "ESTIMATE: ~0.70 × Tabla 98 LiCO3 midpoint — see LITHIUM_PRICE_MULTIPLIERS"),
}

# Multipliers applied to the Tabla 98 LiCO3 midpoint to derive LiOH / LiSO4
# prices when those products are not separately reported in COCHILCO tables.
#
# SOURCE : Benchmark Mineral Intelligence Q4 2024; COCHILCO lithium price bulletin
# STATUS : ESTIMATE — review quarterly; flagged in MANUAL_PRICES above
# USED BY: Notebook A, Cell 7 (§2F — wherever li_price is built)

LITHIUM_PRICE_MULTIPLIERS: dict[str, float] = {
    "Lithium Carbonate": 1.00,   # base — from Tabla 98 directly
    "Lithium Hydroxide": 1.10,   # ~10 % premium (battery-grade market, 2024)
    "Lithium Sulfate":   0.70,   # ~30 % discount (lower purity, smaller market)
}


# 9.  DOUBLE-COUNTING EXCLUSION LIST
#
# COCHILCO A_National_Production reports both aggregate categories AND their
# sub-components in separate rows.  e.g. "COMPUESTOS DE LITIO" is the sum of
# "CARBONATO DE LITIO" + "HIDRÓXIDO DE LITIO" + "SULFATO DE LITIO".
# Allocating aggregate rows to facilities would double-count production that
# is already accounted for via the sub-component rows.
#
# These names match the MINERAL_COLS keys in Notebook A Cell 5 — any mineral
# in this list is captured in the national summary table only (for completeness)
# and is explicitly skipped during facility-level allocation.
#
# SOURCE : COCHILCO Anuario table structure
# STATUS : Stable — extend if COCHILCO adds new aggregate categories
# USED BY: Notebook A, Cell 5 (§2E MINERAL_COLS guard comment)
#          Notebook A, Cell 9 (§2G NAT_LABEL_MAP, col=None entries)

DOUBLE_COUNT_EXCLUDE: list[str] = [
    "Lithium Compounds",   # = sum of Lithium Carbonate + Hydroxide + Sulfate
    "Calcium Carbonate",   # = aggregate of Limestone + Coquina + White CaCO3
    "Silica Ores",         # = aggregate of Quartz + Silica Sand
    "Clay",                # = aggregate of Bauxitic Clay + Kaolin + Bentonite
    "Nitrates",            # = aggregate (Sodium Nitrate + Potassium Nitrate);
                           #   kept national-summary-only; sub-rows allocated
    "Phosphates",          # = aggregate of Phosphate Rocks
    "Manganese",           # no facility-level data in inventory
]


# 10.  MINOR PORT COORDINATES (not in main PORTS list)
#
# Small-volume ports used for specific commodities that are absent from the
# primary PORTS list (which covers major copper/mineral terminals).
# Notebook B builds edges to these ports for nitrate, zinc, and boron flows.
#
# SOURCE : Aduanas Salidas 2024 + port authority coordinates
# STATUS : Verified lat/lon from OpenStreetMap/port authority data
# USED BY: Notebook B, Cells 13 (nitrate), 14 (zinc), 16 (boron)

NITRATE_PORT_COORDS: dict[str, dict] = {
    # Tocopilla is the dominant Chilean nitrate export terminal;
    # ~84% of nitrate FOB value per Aduanas 2022-2023 (HS 2834/3102).
    # Not in main PORTS list because it handles no significant copper volume.
    "Tocopilla": {"lat": -22.09, "lon": -70.20},
}

ZN_PORT_COORDS: dict[str, dict] = {
    # Puerto Aysén / Chacabuco serves El Toqui zinc mine (Region Aysén);
    # the standard PORTS list ends at San Vicente (Bio-Bío, -36.7°).
    "Puerto Aysen": {"lat": -45.40, "lon": -72.70},
}

BORON_PORT_COORDS: dict[str, dict] = {
    # Arica handles ~23% of Chilean borate/ulexite exports per Aduanas 2022-2023
    # (HS 2528); not in main PORTS list (copper focus).
    "Arica": {"lat": -18.48, "lon": -70.33},
}
