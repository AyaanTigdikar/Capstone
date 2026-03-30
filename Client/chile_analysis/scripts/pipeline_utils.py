"""
pipeline_utils.py
Shared constants, paths, and utility functions for the Chile mineral supply
chain pipeline (Parts 0-5). Import at the top of each notebook to avoid
repeating ~70 lines of boilerplate per file.

Usage:
    from pipeline_utils import *          # imports everything below
    # or selectively:
    from pipeline_utils import haversine_km, load_state, PORTS

Hard-coded pipeline assumptions live in pipeline_constants.py, which should
also be imported in each notebook:
    from pipeline_constants import *
"""

import os
import re
import shutil
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
# BASE_DIR is derived from this file's location:  scripts/ → Chile/ (parent)
# This makes the whole project portable — copy the Chile/ folder anywhere and
# paths resolve automatically.

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_CODE        = os.path.join(BASE_DIR, "scripts")
DIR_DATA        = os.path.join(BASE_DIR, "inputs")
DIR_OUTPUT      = os.path.join(BASE_DIR, "outputs")
DIR_INTERMED    = os.path.join(BASE_DIR, "intermediary")  # pkl states, processed CSVs
DIR_TEMP        = os.path.join(DIR_INTERMED, "temporary") # scratch outputs

# Legacy alias so any code that still references DIR_PRELIM keeps working
DIR_PRELIM = DIR_INTERMED

# Data file paths

COCHILCO_PATH = os.path.join(DIR_DATA, "COCHILCO_Production_2005_2024.xlsx")

_cochilco_orig_candidates = [
    os.path.join(DIR_DATA,
                 "1771263160312_Anuario-de-Estadisticas-del-Cobre-y-otros-Minerales-2005-2024.xlsx"),
    os.path.join(DIR_DATA,
                 "Anuario-de-Estadisticas-del-Cobre-y-otros-Minerales-2005-2024.xlsx"),
]
COCHILCO_ORIG = next(
    (p for p in _cochilco_orig_candidates if os.path.exists(p)),
    _cochilco_orig_candidates[0],
)

_salidas_candidates = [
    os.path.join(DIR_DATA, "Salidas2025.csv"),
    os.path.join(DIR_DATA, "salidas_2024.csv"),
    os.path.join(DIR_DATA, "Salidas2024.csv"),
    os.path.join(DIR_DATA, "Salidas_minerals_only.csv"),
]
SALIDAS_PATH = next(
    (p for p in _salidas_candidates if os.path.exists(p)),
    _salidas_candidates[0],
)

# Shared constants

COMPANY_TO_DEPOSIT = {
    "División El Teniente":      ["El Teniente", "Teniente"],
    "División Chuquicamata":     ["Chuquicamata"],
    "División Radomiro Tomic":   ["Radomiro Tomic", "Radomiro"],
    "División Andina":           ["Andina"],
    "División Ministro Hales":   ["Ministro Hales"],
    "División Gabriela Mistral": ["Gabriela Mistral", "Gaby"],
    "División Salvador":         ["Salvador"],
    "Escondida":                 ["Escondida"],
    "Collahuasi":                ["Collahuasi"],
    "Los Pelambres":             ["Los Pelambres", "Pelambres"],
    "Spence":                    ["Spence"],
    "Quebrada Blanca":           ["Quebrada Blanca"],
    "Los Bronces":               ["Los Bronces", "Bronces"],
    "Sierra Gorda":              ["Sierra Gorda"],
    "Candelaria":                ["Candelaria"],
    "Caserones":                 ["Caserones"],
    "Centinela (Súlfuros)":      ["Centinela"],
    "El Abra":                   ["El Abra", "Abra"],
    "Zaldívar":                  ["Zaldívar", "Zaldivar"],
    "Antucoya":                  ["Antucoya"],
    "Lomas Bayas":               ["Lomas Bayas"],
    "Mantoverde":                ["Mantoverde"],
    "El Soldado":                ["El Soldado", "Soldado"],
    "Mantos Blancos":            ["Mantos Blancos"],
    "Andacollo":                 ["Carmen de Andacollo"],
    "Michilla":                  ["Michilla"],
    "Franke":                    ["Franke"],
    "Mantos de la Luna":         ["Mantos de la Luna"],
    "Cerro Negro":               ["Cerro Negro"],
    "Las Cenizas":               ["Las Cenizas", "Cenizas", "Las Luces", "Aguilucho"],
    "Tres Valles":               ["Tres Valles"],
}

CODELCO_EXTRA_SEARCH = {
    "División Andina":         ["Rio Blanco"],
    "División Ministro Hales": ["Mansa Mina"],
    "División Salvador":       ["El Salvador", "Potrerillos"],
}

SMELTERS = [
    {"name": "Chuquicamata smelter",
     "search": ["Chuquicamata SX-EW plant (oxide) and smelter",
                "Chuquicamata Division", "Chuquicamata plant"],
     "operator": "Codelco", "smelter_type": "integrated", "region": "Antofagasta",
     "lat": -22.32, "lon": -68.93, "has_refinery": True,
     "feeds_from_mines": ["Chuquicamata", "Radomiro Tomic", "Ministro Hales"],
     "output_product": "cathode", "export_ports": ["Angamos", "Mejillones"]},
    {"name": "Potrerillos smelter",
     "search": ["Potrerillos SX-EW refinery and smelter", "Potrerillos plant"],
     "operator": "Codelco", "smelter_type": "integrated", "region": "Atacama",
     "lat": -26.39, "lon": -69.46, "has_refinery": True,
     "feeds_from_mines": ["Salvador"],
     "output_product": "cathode", "export_ports": ["Barquito"]},
    {"name": "Caletones smelter",
     "search": ["El Teniente plant"],
     "operator": "Codelco", "smelter_type": "integrated", "region": "O'Higgins",
     "lat": -34.12, "lon": -70.48, "has_refinery": True,
     "feeds_from_mines": ["El Teniente"],
     "output_product": "cathode", "export_ports": ["San Antonio", "Ventanas"]},
    {"name": "Altonorte smelter",
     "search": ["Altonorte"],
     "operator": "Glencore", "smelter_type": "custom", "region": "Antofagasta",
     "lat": -23.78, "lon": -70.31, "has_refinery": False,
     "feeds_from_mines": [], "feeds_from_region": "Antofagasta",
     "output_product": "blister", "export_ports": ["Antofagasta", "Mejillones"]},
    {"name": "Paipote smelter (H.V. Lira)",
     "search": ["Paipote", "Hernan Videla", "Hernán Videla"],
     "operator": "ENAMI", "smelter_type": "custom", "region": "Atacama",
     "lat": -27.37, "lon": -70.30, "has_refinery": False,
     "feeds_from_mines": [], "feeds_from_region": "Atacama",
     "output_product": "blister", "export_ports": ["Barquito", "Caldera"]},
    {"name": "Chagres smelter",
     "search": ["Chagres"],
     "operator": "Anglo American", "smelter_type": "integrated", "region": "Valparaiso",
     "lat": -32.78, "lon": -70.97, "has_refinery": False,
     "feeds_from_mines": ["Los Bronces", "El Soldado"],
     "output_product": "blister", "export_ports": ["Ventanas"]},
]

PORTS = [
    {"name": "Coloso",            "region": "Antofagasta", "lat": -23.76, "lon": -70.45,
     "products": "Cu concentrate",             "key_users": "Escondida (dedicated)"},
    {"name": "Angamos",           "region": "Antofagasta", "lat": -23.10, "lon": -70.42,
     "products": "Cu cathode, concentrate",    "key_users": "Codelco"},
    {"name": "Mejillones",        "region": "Antofagasta", "lat": -23.10, "lon": -70.45,
     "products": "Cu concentrate, cathode, acid", "key_users": "Multiple"},
    {"name": "Antofagasta (ATI)", "region": "Antofagasta", "lat": -23.65, "lon": -70.40,
     "products": "Cu cathode, general",        "key_users": "Antofagasta Minerals"},
    {"name": "Iquique",           "region": "Tarapaca",    "lat": -20.21, "lon": -70.15,
     "products": "Cu cathode",                 "key_users": "Collahuasi, Quebrada Blanca"},
    {"name": "Patache",           "region": "Tarapaca",    "lat": -20.80, "lon": -70.22,
     "products": "Cu concentrate",             "key_users": "Collahuasi, Quebrada Blanca"},
    {"name": "Barquito",          "region": "Atacama",     "lat": -27.07, "lon": -70.84,
     "products": "Cu concentrate, cathode",    "key_users": "Codelco Salvador, ENAMI"},
    {"name": "Caldera",           "region": "Atacama",     "lat": -27.07, "lon": -70.82,
     "products": "Cu concentrate",             "key_users": "Medium miners"},
    {"name": "Coquimbo",          "region": "Coquimbo",    "lat": -29.96, "lon": -71.35,
     "products": "Cu concentrate, Fe pellets", "key_users": "CMP"},
    {"name": "Los Vilos",         "region": "Coquimbo",    "lat": -31.91, "lon": -71.51,
     "products": "Cu concentrate",             "key_users": "Los Pelambres (dedicated)"},
    {"name": "Ventanas",          "region": "Valparaiso",  "lat": -32.74, "lon": -71.49,
     "products": "Cu cathode, blister",        "key_users": "Codelco, Anglo American"},
    {"name": "San Antonio",       "region": "Valparaiso",  "lat": -33.59, "lon": -71.62,
     "products": "Cu cathode, general",        "key_users": "El Teniente, general"},
    {"name": "San Vicente",       "region": "Biobio",      "lat": -36.73, "lon": -73.13,
     "products": "Cu cathode, general",        "key_users": "Southern region"},
]

SMELTER_NAME_MAP = {
    "Chuquicamata smelter":
        "Chuquicamata SX-EW plant (oxide) and smelter",
    "Potrerillos smelter":
        "Potrerillos SX-EW refinery and smelter",
    "Caletones smelter":
        "Caletones smelter (anodes). refinery (fire-refined ingots), and SX-EW plant",
    "Altonorte smelter":
        "Altonorte smelter",
    "Paipote smelter (H.V. Lira)":
        "Hernán Videla Lira smelter (anodes and blister)",
    "Chagres smelter":
        "Chagres smelter (anodes and blister)",
}

DEDICATED_PORT = {
    "Escondida":  "Coloso",
    "Los Pelambres": "Los Vilos",
    "Pelambres":  "Los Vilos",
}

CODELCO_CATHODE_ROUTING = {
    "Chuquicamata":   "Angamos",
    "Radomiro Tomic": "Angamos",
    "Ministro Hales": "Angamos",
    "Gabriela Mistral": "Angamos",
    "Salvador":       "Barquito",
    "El Teniente":    "San Antonio",
    "Andina":         "San Antonio",
}

MATCH_DISAMBIGUATION = {
    "Cerro Negro": "Cerro Negro",
    "Andacollo":   "Carmen de Andacollo",
}

IRON_MINE_NAMES = [
    "Los Colorados", "El Algarrobo", "Cerro Negro Norte",
    "Romeral", "Tofo", "Distrito El Algarrobo",
]

ZINC_MINE_NAMES = ["El Toqui"]

# Utility functions

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def parse_comm_list(val):
    if pd.isna(val):
        return []
    return [x.strip() for x in str(val).split(",") if x.strip()]


def add_commodity(row_idx, commodity, df, col):
    current = parse_comm_list(df.at[row_idx, col])
    if commodity not in current:
        current.append(commodity)
        df.at[row_idx, col] = ", ".join(current)
        return True
    return False


def nearest_port(lat, lon, product_type="concentrate", ports=None):
    """Return (port_dict, distance_km) for the nearest matching port.

    product_type filters the candidates:
      "concentrate" — ports that handle copper concentrate
      "cathode"     — ports that handle copper cathode
      "blister"     — ports that handle blister/anodes
      anything else — no filtering; returns the geometrically nearest port
    """
    if ports is None:
        ports = PORTS
    best_dist, best_port = float("inf"), None
    for port in ports:
        prod = port["products"].lower()
        if product_type == "cathode"     and "cathode"     not in prod:
            continue
        if product_type == "concentrate" and "concentrate" not in prod:
            continue
        if product_type == "blister"     and "blister"     not in prod:
            continue
        dist = haversine_km(lat, lon, port["lat"], port["lon"])
        if dist < best_dist:
            best_dist, best_port = dist, port
    return best_port, best_dist


def section_header(title, width=65):
    print(f"\n{'=' * width}\n{title}\n{'=' * width}")


def search_inventory(inv_df, terms, require_mine=False):
    """Return sorted list of inventory indices matching any term in `terms`."""
    matched = set()
    name_lower = inv_df["FACILITY_NAME"].str.lower().str.strip()
    for term in terms:
        mask = name_lower.str.contains(term.lower(), na=False, regex=False)
        if require_mine:
            mask = mask & inv_df["FACILITY_TYPE"].str.contains("Mine", case=False, na=False)
        matched.update(inv_df[mask].index)
    return sorted(matched)


# State I/O helpers

_STATE_KEYS = [
    "inv", "links", "comm_col", "idle_mines",
    "COMPANY_TO_DEPOSIT", "CODELCO_EXTRA_SEARCH",
    "SMELTERS", "PORTS", "SMELTER_NAME_MAP",
    "DEDICATED_PORT", "CODELCO_CATHODE_ROUTING",
    "MATCH_DISAMBIGUATION", "IRON_MINE_NAMES", "ZINC_MINE_NAMES",
    "n_idle_links", "cu_total", "edges", "common_cols", "smelter_inv_map",
    "export_df", "PORT_PRODUCT_MAP", "ports_df",
    "inv_path", "links_path",
]

_STATE_DEFAULTS = {
    "inv":                     pd.DataFrame(),
    "links":                   pd.DataFrame(),
    "comm_col":                "COMMODITY_LIST_STR",
    "idle_mines":              set(),
    "COMPANY_TO_DEPOSIT":      COMPANY_TO_DEPOSIT,
    "CODELCO_EXTRA_SEARCH":    CODELCO_EXTRA_SEARCH,
    "SMELTERS":                SMELTERS,
    "PORTS":                   PORTS,
    "SMELTER_NAME_MAP":        SMELTER_NAME_MAP,
    "DEDICATED_PORT":          DEDICATED_PORT,
    "CODELCO_CATHODE_ROUTING": CODELCO_CATHODE_ROUTING,
    "MATCH_DISAMBIGUATION":    MATCH_DISAMBIGUATION,
    "IRON_MINE_NAMES":         IRON_MINE_NAMES,
    "ZINC_MINE_NAMES":         ZINC_MINE_NAMES,
    "n_idle_links":            0,
    "cu_total":                0.0,   # national Cu production total (kMT); set in Notebook A
    "edges":                   pd.DataFrame(),
    "common_cols":             ["FROM_NAME", "FROM_TYPE", "FROM_LAT", "FROM_LON",
                                "TO_NAME", "TO_TYPE", "TO_LAT", "TO_LON",
                                "EDGE_TYPE", "PRODUCT_FORM", "COMMODITIES", "DISTANCE_KM"],
    "smelter_inv_map":         {},
    "export_df":               pd.DataFrame(),
    "PORT_PRODUCT_MAP":        {},
    "ports_df":                pd.DataFrame(PORTS),
    "inv_path":                "",
    "links_path":              "",
}


def load_state(part_number: int) -> dict:
    """Load pipeline state from output/intermediary/_pipeline_state_{part_number}.pkl.

    Raises a clear RuntimeError if the file is missing (run the preceding
    notebook first) or corrupt (re-run from the last clean state).
    """
    path = os.path.join(DIR_INTERMED, f"_pipeline_state_{part_number}.pkl")
    if not os.path.exists(path):
        existing = sorted(
            f for f in os.listdir(DIR_INTERMED)
            if f.startswith("_pipeline_state_") and f.endswith(".pkl")
        ) if os.path.isdir(DIR_INTERMED) else []
        hint = f"  Available states: {existing}" if existing else "  No state files found in intermediary directory."
        raise FileNotFoundError(
            f"Pipeline state {part_number} not found at:\n  {path}\n"
            f"Run the preceding notebook to generate it.\n{hint}"
        )
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load pipeline state {part_number} from:\n  {path}\n"
            f"The file may be truncated or written by a different Python/pandas version.\n"
            f"Re-run the notebook that produces state {part_number} to regenerate it.\n"
            f"Original error: {exc}"
        ) from exc
    for key, default in _STATE_DEFAULTS.items():
        state.setdefault(key, default)
    return state


def save_state(state: dict, part_number: int) -> None:
    """Persist `state` to output/intermediary/_pipeline_state_{part_number}.pkl."""
    path = os.path.join(DIR_INTERMED, f"_pipeline_state_{part_number}.pkl")
    os.makedirs(DIR_INTERMED, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"State saved to {path}")


def save_temp(obj, filename: str) -> str:
    """Save any object or DataFrame to output/temporary/. Returns the path."""
    os.makedirs(DIR_TEMP, exist_ok=True)
    path = os.path.join(DIR_TEMP, filename)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    print(f"Temp file saved: {path}")
    return path


def unpack_state(state: dict) -> tuple:
    """Return the most commonly needed variables as a tuple for quick unpacking.

    Usage:
        inv, links, comm_col, idle_mines, edges, ports_df = unpack_state(state)
    """
    return (
        state["inv"],
        state["links"],
        state.get("comm_col", "COMMODITY_LIST_STR"),
        state.get("idle_mines", set()),
        state.get("edges", pd.DataFrame()),
        state.get("ports_df", pd.DataFrame(PORTS)),
    )
