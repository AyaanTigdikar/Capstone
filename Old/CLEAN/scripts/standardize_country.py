"""
standardize_country.py
Canonical country-name/ISO3 mapping for the Moody's capstone pipeline.

Usage (in any notebook):
    from standardize_country import standardize, add_iso3, ALIAS_TO_ISO3, ISO3_TO_WB

    # Option A: map a single name -> ISO3
    standardize("Russia")          # -> "RUS"
    standardize("DR Congo")       # -> "COD"

    # Option B: add an iso3 column to a DataFrame with a name column
    df = add_iso3(df, name_col="Country")
    #   -> adds "iso3" column, warns about unmatched names

    # Option C: get WB canonical name from ISO3
    ISO3_TO_WB["RUS"]             # -> "Russian Federation"
"""

import warnings
from typing import Optional
import pandas as pd
from pathlib import Path

# ── Load mapping from CSV (same directory as this module) ──
_MAP_PATH = Path(__file__).parent.parent / "country_mapping.csv"
_map_df = pd.read_csv(_MAP_PATH)

ALIAS_TO_ISO3 = dict(zip(_map_df["alias"], _map_df["iso3"]))
ISO3_TO_WB = (
    _map_df.dropna(subset=["wb_name"])
    .drop_duplicates(subset=["iso3"])
    .set_index("iso3")["wb_name"]
    .to_dict()
)


def standardize(name: str) -> Optional[str]:
    """Return ISO3 code for a country name/alias, or None if not found."""
    s = str(name)
    # Try exact match first (preserves trailing-space aliases like "Australia ")
    result = ALIAS_TO_ISO3.get(s)
    if result is None:
        # Fallback: stripped match
        result = ALIAS_TO_ISO3.get(s.strip())
    return result


def add_iso3(
    df: pd.DataFrame,
    name_col: str = "Country",
    iso3_col: str = "iso3",
    drop_unmatched: bool = False,
    warn: bool = True,
) -> pd.DataFrame:
    """
    Map a name column to ISO3 codes.

    Parameters
    ----------
    df : DataFrame with a country-name column.
    name_col : Column containing country names/aliases.
    iso3_col : Name of the new ISO3 column to create.
    drop_unmatched : If True, drop rows that could not be mapped.
    warn : If True, print unmatched names to stderr.

    Returns
    -------
    DataFrame with the new iso3 column added.
    """
    out = df.copy()
    # Exact match first (catches trailing-space aliases), then stripped fallback
    out[iso3_col] = out[name_col].map(
        lambda x: ALIAS_TO_ISO3.get(str(x)) or ALIAS_TO_ISO3.get(str(x).strip())
    )

    unmatched = out.loc[out[iso3_col].isna(), name_col].unique()
    # Filter out known non-country entries (World, Global, OPEC, EI aggregates, etc.)
    skip = {"World", "Global", "Total", "OPEC", "Non-OPEC", "OECD", "Non-OECD",
            "European Union", ""}
    skip_prefixes = ("Other", "Total ", "Rest of")
    real_unmatched = [
        n for n in unmatched
        if str(n).strip() not in skip and not str(n).strip().startswith(skip_prefixes)
    ]

    if real_unmatched and warn:
        warnings.warn(
            f"standardize_country: {len(real_unmatched)} unmatched names "
            f"(first 10): {real_unmatched[:10]}"
        )

    if drop_unmatched:
        out = out[out[iso3_col].notna()]

    return out


def add_wb_name(
    df: pd.DataFrame,
    iso3_col: str = "iso3",
    name_col: str = "Country Name",
) -> pd.DataFrame:
    """Add a WB canonical name column from an ISO3 column."""
    out = df.copy()
    out[name_col] = out[iso3_col].map(ISO3_TO_WB)
    return out
