"""
nb2_charts.py
=============
Missingness diagnostic charts extracted from 2_MissingCheck_FINAL.ipynb.
Not run during the main pipeline. Run standalone after NB1 produces
intermediary/master_data_wide.csv.

Usage:
    cd /Users/leoss/Desktop/GitHub/Capstone/CLEAN
    python3 scripts/nb2_charts.py

Outputs: Graphics/NB2/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go

os.makedirs("Graphics/NB2", exist_ok=True)

# ── Load and filter (mirrors NB2 cell 3) ──
master = pd.read_csv("intermediary/master_data_wide.csv")
if "Unnamed: 0" in master.columns:
    master.drop(columns="Unnamed: 0", inplace=True)

RENT_COL   = "Total natural resources rents (% of GDP)"
FOREST_COL = "Forest rents (% of GDP)"
COAL_COL   = "Coal rents (% of GDP)"
THRESHOLD  = 1.0

master["Extractive_NR_Rents"] = (
    master[RENT_COL].fillna(0)
    - master[FOREST_COL].fillna(0)
    - master[COAL_COL].fillna(0)
).clip(lower=0)

rents_1995 = master[master["Year"] == 1995].dropna(subset=[RENT_COL])
resource_countries = set(
    rents_1995[rents_1995["Extractive_NR_Rents"] >= THRESHOLD]["Country Code"].tolist()
)
GULF = {"ARE", "BHR", "KWT", "OMN", "QAT", "SAU", "IRQ", "IRN", "YEM"}
resource_countries = resource_countries | GULF

not_countries = [
    "HKG", "MAC", "PRI", "VIR", "GUM", "ASM", "CYM", "BMU",
    "GRL", "MAF", "SXM", "CUW", "ABW", "FRO", "MNP", "PYF",
]

cmaster = master[
    (master["Country Code"].isin(resource_countries))
    & (~master["Country Code"].isin(not_countries))
]
print(f"Sample: {cmaster['Country Code'].nunique()} countries, {cmaster.shape[0]:,} rows")


# ======================================================================
# NB2 cell 5: analysis function
# ======================================================================
def analyze_variable_missingness(df):
    """Calculate missingness statistics for each variable in the panel."""
    data_cols = [c for c in df.columns if c not in ['Country Code', 'Country Name', 'Year']]

    results = []
    for col in data_cols:
        valid_data = df[df[col].notna()]
        n_obs = valid_data.shape[0]
        n_missing = df[col].isna().sum()
        pct_missing = (n_missing / len(df)) * 100
        n_countries = valid_data['Country Code'].nunique()
        n_years = valid_data['Year'].dropna().nunique()
        year_range = (f"{int(valid_data['Year'].min())}-{int(valid_data['Year'].max())}"
                      if n_years > 0 else "N/A")

        results.append({
            'Variable': col,
            'Valid Obs': n_obs,
            'Missing': n_missing,
            '% Missing': round(pct_missing, 1),
            'Countries': n_countries,
            'Years': n_years,
            'Year Range': year_range,
        })

    return pd.DataFrame(results).sort_values('% Missing', ascending=False)


var_missing = analyze_variable_missingness(cmaster)

print("=" * 80)
print("VARIABLE-LEVEL MISSINGNESS SUMMARY")
print("=" * 80)
print(f"\nTotal observations: {len(cmaster)}")
print(f"Total countries: {cmaster['Country Code'].nunique()}")
print(f"Total years: {cmaster['Year'].dropna().nunique()}")
print()
print(var_missing.to_string(index=False))

# Coverage buckets
print("\n--- Variable Coverage Buckets ---")
print(f"< 10% missing: {(var_missing['% Missing'] < 10).sum()}")
print(f"10-30% missing: {((var_missing['% Missing'] >= 10) & (var_missing['% Missing'] < 30)).sum()}")
print(f"30-50% missing: {((var_missing['% Missing'] >= 30) & (var_missing['% Missing'] < 50)).sum()}")
print(f"> 50% missing: {(var_missing['% Missing'] >= 50).sum()}")

# ======================================================================
# NB2 cell 7: analysis function
# ======================================================================
def analyze_country_missingness(df):
    """Calculate missingness statistics for each country in the panel."""
    data_cols = [c for c in df.columns if c not in ['Country Code', 'Country Name', 'Year']]

    results = []
    for code in df['Country Code'].unique():
        country_data = df[df['Country Code'] == code]
        country_name = country_data['Country Name'].iloc[0]

        n_rows = len(country_data)
        total_cells = n_rows * len(data_cols)
        missing_cells = country_data[data_cols].isna().sum().sum()
        pct_missing = (missing_cells / total_cells) * 100

        complete_vars = sum(country_data[col].notna().all() for col in data_cols)
        vars_with_data = sum(country_data[col].notna().any() for col in data_cols)
        years_covered = country_data['Year'].dropna().nunique()

        results.append({
            'Code': code,
            'Country': country_name,
            'Rows': n_rows,
            'Years Covered': years_covered,
            '% Missing': round(pct_missing, 1),
            'Complete Vars': complete_vars,
            'Vars with Data': vars_with_data,
            'Total Vars': len(data_cols),
        })

    return pd.DataFrame(results).sort_values('% Missing', ascending=False)


country_missing = analyze_country_missingness(cmaster)

print("=" * 80)
print("COUNTRY-LEVEL MISSINGNESS SUMMARY")
print("=" * 80)
print("\nTop 20 countries with MOST missing data:")
print(country_missing.head(20).to_string(index=False))
print("\n\nTop 20 countries with LEAST missing data:")
print(country_missing.tail(20).to_string(index=False))

# Coverage buckets
print("\n--- Country Coverage Buckets ---")
print(f"< 20% missing: {(country_missing['% Missing'] < 20).sum()}")
print(f"20-40% missing: {((country_missing['% Missing'] >= 20) & (country_missing['% Missing'] < 40)).sum()}")
print(f"> 40% missing: {(country_missing['% Missing'] >= 40).sum()}")

# ── Run diagnostics ──
var_missing = analyze_variable_missingness(cmaster)
country_missing = analyze_country_missingness(cmaster)


# ======================================================================
# NB2 cell 9
# ======================================================================
# ── Plot 1: Variable missingness (horizontal bar) ──
fig1, ax1 = plt.subplots(figsize=(12, 16))

var_sorted = var_missing.sort_values('% Missing', ascending=True)
colors = ['#d73027' if x > 40 else '#fc8d59' if x > 20 else '#91cf60'
          for x in var_sorted['% Missing']]

ax1.barh(range(len(var_sorted)), var_sorted['% Missing'], color=colors, height=0.7)
ax1.set_xlabel('% Missing', fontsize=12)
ax1.set_title('Missingness by Variable', fontsize=14, fontweight='bold', pad=20)
ax1.set_yticks(range(len(var_sorted)))
ax1.set_yticklabels(var_sorted['Variable'], fontsize=9)
ax1.axvline(x=20, color='orange', linestyle='--', alpha=0.7, linewidth=2)
ax1.axvline(x=40, color='red', linestyle='--', alpha=0.7, linewidth=2)

for i, (val, _) in enumerate(zip(var_sorted['% Missing'], var_sorted['Variable'])):
    ax1.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=8)

legend_elements = [
    Patch(facecolor='#91cf60', label='< 20% missing'),
    Patch(facecolor='#fc8d59', label='20-40% missing'),
    Patch(facecolor='#d73027', label='> 40% missing'),
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
ax1.set_xlim(0, max(var_sorted['% Missing']) + 10)
plt.tight_layout()
plt.savefig('Graphics/NB2/plot1_variable_missingness.png', dpi=150, bbox_inches='tight')
plt.show()

# ======================================================================
# NB2 cell 10
# ======================================================================
# ── Plot 2: Country missingness distribution ──
fig2, ax2 = plt.subplots(figsize=(12, 7))

n, bins, patches = ax2.hist(country_missing['% Missing'], bins=25, edgecolor='white', linewidth=1.2)
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i + 1]) / 2
    patch.set_facecolor('#d73027' if bin_center > 40 else '#fc8d59' if bin_center > 20 else '#91cf60')

ax2.axvline(country_missing['% Missing'].median(), color='black', linestyle='--',
            linewidth=2, label=f"Median: {country_missing['% Missing'].median():.1f}%")
ax2.axvline(country_missing['% Missing'].mean(), color='blue', linestyle=':',
            linewidth=2, label=f"Mean: {country_missing['% Missing'].mean():.1f}%")
ax2.set_xlabel('% Missing Data', fontsize=12)
ax2.set_ylabel('Number of Countries', fontsize=12)
ax2.set_title('Distribution of Country-Level Missingness', fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=11)
plt.tight_layout()
plt.savefig('Graphics/NB2/plot2_country_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ======================================================================
# NB2 cell 11
# ======================================================================
# ── Plot 3: All countries ranked ──
fig3, ax3 = plt.subplots(figsize=(14, 18))

country_sorted = country_missing.sort_values('% Missing', ascending=True)
colors = ['#d73027' if x > 40 else '#fc8d59' if x > 20 else '#91cf60'
          for x in country_sorted['% Missing']]

ax3.barh(range(len(country_sorted)), country_sorted['% Missing'], color=colors, height=0.7)
ax3.set_yticks(range(len(country_sorted)))
labels = [f"{row['Code']} - {str(row['Country'])[:25] if pd.notna(row['Country']) else 'N/A'}"
          for _, row in country_sorted.iterrows()]
ax3.set_yticklabels(labels, fontsize=9)
ax3.set_xlabel('% Missing', fontsize=12)
ax3.set_title('All Countries Ranked by Missingness', fontsize=14, fontweight='bold', pad=20)
ax3.axvline(x=20, color='orange', linestyle='--', alpha=0.7, linewidth=2)
ax3.axvline(x=40, color='red', linestyle='--', alpha=0.7, linewidth=2)

for i, val in enumerate(country_sorted['% Missing']):
    ax3.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=7)

ax3.set_xlim(0, max(country_sorted['% Missing']) + 8)
plt.tight_layout()
plt.savefig('Graphics/NB2/plot3_all_countries_ranked.png', dpi=150, bbox_inches='tight')
plt.show()

# ======================================================================
# NB2 cell 12
# ======================================================================
# ── Plot 4: Variable coverage (how many countries have data) ──
fig4, ax4 = plt.subplots(figsize=(14, 16))

var_sorted2 = var_missing.sort_values('Countries', ascending=True)
ax4.barh(range(len(var_sorted2)), var_sorted2['Countries'], color='#4575b4', height=0.7)
ax4.set_yticks(range(len(var_sorted2)))
ax4.set_yticklabels(var_sorted2['Variable'], fontsize=9)
ax4.set_xlabel('Number of Countries with Data', fontsize=12)
ax4.set_title('Variable Coverage: How Many Countries Have Data?', fontsize=14, fontweight='bold', pad=20)

total_countries = cmaster['Country Code'].nunique()
ax4.axvline(x=total_countries, color='red', linestyle='--', linewidth=2,
            label=f'Total countries: {total_countries}')

for i, (c, y) in enumerate(zip(var_sorted2['Countries'], var_sorted2['Years'])):
    ax4.text(c + 0.5, i, f'{c} countries, {y} years', va='center', fontsize=8)

ax4.legend(loc='lower right', fontsize=10)
ax4.set_xlim(0, total_countries + 15)
plt.tight_layout()
plt.savefig('Graphics/NB2/plot4_variable_coverage.png', dpi=150, bbox_inches='tight')
plt.show()

# ======================================================================
# NB2 cell 13
# ======================================================================
# ── Plot 5: Heatmap of missingness (top problem countries x top problem variables) ──
fig5, ax5 = plt.subplots(figsize=(16, 14))

data_cols = [c for c in cmaster.columns if c not in ['Country Code', 'Country Name', 'Year']]
missing_matrix = cmaster.groupby('Country Code')[data_cols].apply(lambda x: x.isna().mean() * 100)

top_countries = country_missing.head(25)['Code'].tolist()
top_vars = var_missing.head(20)['Variable'].tolist()
heatmap_data = missing_matrix.loc[top_countries, top_vars]

sns.heatmap(heatmap_data, cmap='RdYlGn_r', ax=ax5,
            cbar_kws={'label': '% Missing', 'shrink': 0.8},
            linewidths=0.5, linecolor='white', vmin=0, vmax=100)

ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax5.set_yticklabels(ax5.get_yticklabels(), fontsize=10)
ax5.set_title('Top 25 Problem Countries x Top 20 Problem Variables',
              fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('Graphics/NB2/plot5_heatmap_problem_areas.png', dpi=150, bbox_inches='tight')
plt.show()

# ======================================================================
# NB2 cell 14
# ======================================================================
# ── Plot 6: Missingness over time ──
fig6, ax6 = plt.subplots(figsize=(14, 8))

data_cols = [c for c in cmaster.columns if c not in ['Country Code', 'Country Name', 'Year']]
yearly_missing = cmaster.groupby('Year')[data_cols].apply(lambda x: x.isna().mean() * 100).mean(axis=1)
yearly_missing = yearly_missing.dropna()

ax6.plot(yearly_missing.index, yearly_missing.values, marker='o', linewidth=2,
         markersize=8, color='#4575b4')
ax6.fill_between(yearly_missing.index, yearly_missing.values, alpha=0.3, color='#4575b4')
ax6.set_xlabel('Year', fontsize=12)
ax6.set_ylabel('Average % Missing Across All Variables', fontsize=12)
ax6.set_title('Missingness Over Time', fontsize=14, fontweight='bold', pad=20)
ax6.grid(True, alpha=0.3)

min_year = yearly_missing.idxmin()
max_year = yearly_missing.idxmax()
ax6.annotate(f'Best: {min_year:.0f}\n({yearly_missing[min_year]:.1f}%)',
             xy=(min_year, yearly_missing[min_year]), fontsize=10,
             xytext=(min_year, yearly_missing[min_year] - 5),
             ha='center', color='green', fontweight='bold')
ax6.annotate(f'Worst: {max_year:.0f}\n({yearly_missing[max_year]:.1f}%)',
             xy=(max_year, yearly_missing[max_year]), fontsize=10,
             xytext=(max_year, yearly_missing[max_year] + 3),
             ha='center', color='red', fontweight='bold')
plt.tight_layout()
plt.savefig('Graphics/NB2/plot6_missingness_over_time.png', dpi=150, bbox_inches='tight')
plt.show()

# ======================================================================
# NB2 cell 16
# ======================================================================
fig = px.scatter(
    country_missing,
    x='Vars with Data', y='% Missing',
    color='% Missing', color_continuous_scale='RdYlGn_r',
    hover_name='Country',
    hover_data={'Code': True, 'Vars with Data': True, '% Missing': ':.1f',
                'Complete Vars': True, 'Years Covered': True, 'Rows': True},
    title='Country Data Profile: Coverage vs Completeness<br>'
          '<sup>Hover over points to see country details</sup>',
    labels={'Vars with Data': 'Number of Variables with Any Data',
            '% Missing': '% Missing Data Overall'},
)

fig.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.7))

median_vars = country_missing['Vars with Data'].median()
median_missing = country_missing['% Missing'].median()
fig.add_hline(y=median_missing, line_dash="dash", line_color="gray", opacity=0.5,
              annotation_text=f"Median: {median_missing:.1f}%", annotation_position="right")
fig.add_vline(x=median_vars, line_dash="dash", line_color="gray", opacity=0.5,
              annotation_text=f"Median: {median_vars:.0f} vars", annotation_position="top")

fig.update_layout(width=1000, height=700, plot_bgcolor='white')
fig.show()
fig.write_html('Graphics/NB2/plot7_interactive_country_profile.html')

# ======================================================================
# NB2 cell 18
# ======================================================================
vars_of_interest = [
    'GDP per capita (constant prices, PPP)',
    'Total natural resources rents (% of GDP)',
    'Rule of law index',
    'Employment in industry (% of total employment)',
    'Economic Complexity Index',
]

top5_missing_codes = country_missing.head(5)['Code'].tolist()
top5_df = cmaster[cmaster['Country Code'].isin(top5_missing_codes)]
rest_df = cmaster[~cmaster['Country Code'].isin(top5_missing_codes)]

comparison = []
for var in vars_of_interest:
    top5_mean = top5_df[var].mean()
    rest_mean = rest_df[var].mean()
    diff = top5_mean - rest_mean
    pct_diff = (diff / rest_mean * 100) if rest_mean != 0 else np.nan
    comparison.append({
        'Variable': var,
        'Top 5 Missing (Mean)': round(top5_mean, 2),
        'Rest of Countries (Mean)': round(rest_mean, 2),
        'Difference': round(diff, 2),
        '% Difference': round(pct_diff, 1),
    })

comparison_df = pd.DataFrame(comparison)
print("=" * 100)
print("SIDE-BY-SIDE MEAN COMPARISON: Top 5 most data-poor vs rest")
print("=" * 100)
print(comparison_df.to_string(index=False))

# ======================================================================
# NB2 cell 20
# ======================================================================
exclude_cols = ['Country Code', 'Country Name', 'Year']
indicator_cols = [c for c in cmaster.columns if c not in exclude_cols]
corr_matrix = cmaster[indicator_cols].corr(method='pearson')

# ── Static heatmap: full correlation matrix (lower triangle) ──
fig, ax = plt.subplots(figsize=(24, 20))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='RdYlGn', center=0,
            annot=False, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            vmin=-1, vmax=1, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
ax.set_title('Correlation Matrix: All Indicators (pairwise complete obs)',
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# ── Table: high-correlation pairs ──
print("\n" + "=" * 90)
print("HIGH CORRELATION PAIRS (|r| > 0.8)")
print("=" * 90)

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            n_obs = cmaster[[var1, var2]].dropna().shape[0]
            high_corr_pairs.append({
                'Variable 1': var1, 'Variable 2': var2,
                'Correlation': round(corr_val, 3), 'N (pairwise)': n_obs,
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
print(high_corr_df.to_string(index=False))

# Flatten upper triangle for summary
upper_tri_vals = corr_matrix.where(~np.triu(np.ones_like(corr_matrix, dtype=bool))).stack()
print(f"\nTotal variable pairs: {len(upper_tri_vals)}")
print(f"Pairs with |r| > 0.8: {len(high_corr_df)}")

# ======================================================================
# NB2 cell 22
# ======================================================================
# Build pairwise N matrix (vectorised: notna dot product)
notna = cmaster[indicator_cols].notna().astype(int)
n_matrix = pd.DataFrame(notna.T @ notna, index=indicator_cols, columns=indicator_cols)

# Mask weak correlations and upper triangle for display
corr_display = corr_matrix.copy()
corr_display[abs(corr_display) < 0.5] = np.nan
mask_upper = np.triu(np.ones_like(corr_display, dtype=bool), k=0)
corr_display = corr_display.mask(mask_upper)

# Build hover text
hover_text = []
for i, var1 in enumerate(indicator_cols):
    row_text = []
    for j, var2 in enumerate(indicator_cols):
        corr_val = corr_matrix.iloc[i, j]
        n_val = int(n_matrix.iloc[i, j])
        if i > j:
            strength = ("Very Strong" if abs(corr_val) >= 0.8
                        else "Strong" if abs(corr_val) >= 0.5
                        else "Weak")
            direction = "Positive" if corr_val > 0 else "Negative"
            text = (f"<b>{var1}</b> vs <b>{var2}</b><br>"
                    f"r = {corr_val:.3f} ({strength} {direction})<br>"
                    f"N = {n_val}")
        else:
            text = ""
        row_text.append(text)
    hover_text.append(row_text)

fig = go.Figure(data=go.Heatmap(
    z=corr_display.values, x=indicator_cols, y=indicator_cols,
    hovertext=hover_text, hoverinfo='text',
    colorscale='RdYlGn', zmid=0, zmin=-1, zmax=1,
    colorbar=dict(title='Correlation',
                  tickvals=[-1, -0.8, -0.5, 0, 0.5, 0.8, 1]),
    xgap=2, ygap=2,
))

fig.update_layout(
    title='Interactive Correlation Matrix (|r| > 0.5 shown)<br>'
          '<sup>Hover for details</sup>',
    width=1200, height=1100, plot_bgcolor='white',
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8), autorange='reversed'),
)
fig.show()
fig.write_html('Graphics/NB2/correlation_matrix_interactive.html')
print("Saved: correlation_matrix_interactive.html")