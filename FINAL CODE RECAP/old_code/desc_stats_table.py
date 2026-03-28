"""
Descriptive Statistics Table — Capstone Capstone
Produces a LaTeX table matching the layout in the paper:
  Left panel  = Full Sample (all countries in Master.csv)
  Right panel = High-Resource Countries (54-country include_list)
"""

import pandas as pd
import numpy as np
import os

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
master_path = os.path.join(BASE, "intermediary", "Master.csv")
out_path    = os.path.join(BASE, "outputs", "desc_stats_table.tex")

# ── 54-country sample ────────────────────────────────────────────────────────
include_list = [
    'AGO', 'ARE', 'AZE', 'BFA', 'BHR', 'BOL', 'CHL', 'CIV', 'CMR',
    'COD', 'COG', 'DZA', 'ECU', 'EGY', 'ETH', 'GAB', 'GHA', 'GIN',
    'GNQ', 'IDN', 'IRN', 'IRQ', 'KAZ', 'KEN', 'KWT', 'LAO', 'LBR',
    'LBY', 'MDG', 'MEX', 'MNG', 'MOZ', 'MRT', 'MYS', 'NGA', 'NOR',
    'OMN', 'PER', 'PNG', 'QAT', 'RUS', 'SAU', 'SDN', 'SLE', 'SYR',
    'TCD', 'TGO', 'TKM', 'TTO', 'TZA', 'UGA', 'UZB', 'VEN', 'VNM',
    'YEM', 'ZMB', 'ZWE',
]

# ── variable definitions ─────────────────────────────────────────────────────
# (col_in_csv, label_in_table, is_binary)
CONT_VARS = [
    ("Economic Complexity Index",                            "Economic Complexity Index (ECI)",              False),
    ("Human capital index",                                  "Human Capital Index",                          False),
    ("Gross fixed capital formation, all, Constant prices, Percent of GDP",
                                                             "Gross Fixed Capital Formation (\\% of GDP)",   False),
    ("Political stability — estimate",                       "Political Stability",                          False),
    ("Rule of law index",                                    "Rule of Law Index",                            False),
    ("NR_prod_value_pc",                                     "Natural Resource Prod. Value Per Capita",      False),
    ("Trade (% of GDP)",                                     "Trade (\\% of GDP)",                           False),
    ("Access to electricity (% of population)",              "Access to Electricity (\\% of pop.)",          False),
    ("Adjusted savings: gross savings (% of GNI)",           "Gross Savings (\\% of GNI)",                   False),
    ("Domestic credit to private sector (% of GDP)",         "Domestic Credit to Private Sector (\\% of GDP)", False),
]

BIN_VARS = [
    ("Hydrocarbons_Dominant",    "Hydrocarbons Dominant (=1)"),
    ("Subsoil_Metals_Dominant",  "Subsoil Metals Dominant (=1)"),
    ("Precious_Metals_Dominant", "Precious Metals Dominant (=1)"),
]

# ── load & construct derived variables ───────────────────────────────────────
df = pd.read_csv(master_path)

# Natural resource production value per capita
df["NR_prod_value_pc"] = df["Total_Production_Value"] / df["Population"]

full = df.copy()
high = df[df["Country Code"].isin(include_list)].copy()

# ── stats helpers ────────────────────────────────────────────────────────────
def cont_row(series, label):
    s = series.dropna()
    return {
        "label": label,
        "N":      f"{len(s):,}",
        "Mean":   f"{s.mean():.3f}",
        "Std":    f"{s.std():.3f}",
        "Min":    f"{s.min():.3f}",
        "p25":    f"{s.quantile(0.25):.3f}",
        "Median": f"{s.median():.3f}",
        "p75":    f"{s.quantile(0.75):.3f}",
        "Max":    f"{s.max():.3f}",
    }

def bin_row(series, label):
    s = series.dropna()
    return {
        "label": label,
        "N":      f"{len(s):,}",
        "Mean":   f"{s.mean()*100:.1f}\\%",
        "Std":    "---", "Min": "---", "p25": "---",
        "Median": "---", "p75": "---", "Max": "---",
    }

COLS = ["N", "Mean", "Std", "Min", "p25", "Median", "p75", "Max"]

def make_rows(data):
    rows = []
    for col, label, _ in CONT_VARS:
        rows.append(cont_row(data[col], label))
    for col, label in BIN_VARS:
        rows.append(bin_row(data[col], label))
    return rows

full_rows = make_rows(full)
high_rows = make_rows(high)

# ── build LaTeX ──────────────────────────────────────────────────────────────
def fmt(v):
    return v.replace("%", "\\%") if isinstance(v, str) else v

col_spec = "l" + "r" * 8 + "  " + "r" * 8   # 1 label + 8 + 8

header_top = (
    r"\multicolumn{9}{c}{\textbf{Full Sample}} & "
    r"\multicolumn{8}{c}{\textbf{High-Resource Countries}}"
)
header_cols = " & ".join([""] + COLS + COLS[1:])  # skip second N label... actually keep both
# Actually match the image: both panels show N Mean Std Min p25 Median p75 Max
header_cols = " & ".join([""] + COLS + COLS)

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Descriptive Statistics}")
lines.append(r"\label{tab:desc_stats}")
lines.append(r"\small")
lines.append(r"\setlength{\tabcolsep}{4pt}")
lines.append(r"\begin{tabular}{" + "l" + "r"*8 + "r"*8 + "}")
lines.append(r"\toprule")
lines.append(header_top + r" \\")
lines.append(r"\cmidrule(lr){2-9}\cmidrule(lr){10-17}")
lines.append(header_cols + r" \\")
lines.append(r"\midrule")

# continuous block
lines.append(r"\multicolumn{17}{l}{\textit{Continuous variables}} \\[2pt]")
for fr, hr in zip(full_rows[:len(CONT_VARS)], high_rows[:len(CONT_VARS)]):
    row_parts = [fr["label"]]
    for c in COLS:
        row_parts.append(fr[c])
    for c in COLS:
        row_parts.append(hr[c])
    lines.append(" & ".join(row_parts) + r" \\")

lines.append(r"\midrule")

# binary block
lines.append(r"\multicolumn{17}{l}{\textit{Binary variables (share = mean)}} \\[2pt]")
for fr, hr in zip(full_rows[len(CONT_VARS):], high_rows[len(CONT_VARS):]):
    row_parts = [fr["label"]]
    for c in COLS:
        row_parts.append(fr[c])
    for c in COLS:
        row_parts.append(hr[c])
    lines.append(" & ".join(row_parts) + r" \\")

lines.append(r"\midrule")

# totals
n_full = f"{len(full):,}"
n_high = f"{len(high):,}"
c_full = f"{full['Country Code'].nunique()}"
c_high = f"{high['Country Code'].nunique()}"
lines.append(
    r"Observations & \multicolumn{8}{c}{" + n_full + r"} & \multicolumn{8}{c}{" + n_high + r"} \\"
)
lines.append(
    r"Countries    & \multicolumn{8}{c}{" + c_full + r"} & \multicolumn{8}{c}{" + c_high + r"} \\"
)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(
    r"\begin{minipage}{\textwidth}"
    r"\vspace{4pt}\footnotesize"
    r"\textit{Note:} Statistics computed on non-missing observations. "
    r"Binary variables report the share of country-year observations equal to 1. "
    r"Std.\ Dev., Min, quartiles, and Max are omitted for binary variables."
    r"\end{minipage}"
)
lines.append(r"\end{table}")

tex = "\n".join(lines)

os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
with open(out_path, "w") as f:
    f.write(tex)

print(f"LaTeX table written to: {out_path}")
print()

# ── also print a quick plain-text preview ────────────────────────────────────
print(f"{'Variable':<48} {'N':>6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Median':>10} {'Max':>10}")
print("-" * 108)
print("FULL SAMPLE")
for r in full_rows:
    print(f"  {r['label']:<46} {r['N']:>6} {r['Mean']:>10} {r['Std']:>10} {r['Min']:>10} {r['Median']:>10} {r['Max']:>10}")
print()
print("HIGH-RESOURCE COUNTRIES (54)")
for r in high_rows:
    print(f"  {r['label']:<46} {r['N']:>6} {r['Mean']:>10} {r['Std']:>10} {r['Min']:>10} {r['Median']:>10} {r['Max']:>10}")
