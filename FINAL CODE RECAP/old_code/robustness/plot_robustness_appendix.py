#!/usr/bin/env python3
"""
plot_robustness_appendix.py
===========================
Generates all robustness appendix figures from pre-computed CSVs.

Prerequisites (run first):
  python robustness/run_bootstrap.py          # phases 1-3
  python robustness/run_loco.py               # ML LOCO
  python robustness/run_loco_regression.py    # regression LOCO

Run from:  /Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP
    python robustness/plot_robustness_appendix.py

Output directory:  Final/appendix/

Figures produced
----------------
fig_A1_boot_reg_coefs.png/pdf     — Bootstrap CIs for OLS coefs (3a and 3b)
fig_A2_boot_ml_r2.png/pdf         — Bootstrap ML test-R² distributions
fig_A3_boot_rf_importance.png/pdf — RF feature importance with CIs
fig_A4_boot_lasso_selection.png/pdf — LASSO selection frequency
fig_A5_loco_ml_coef.png/pdf       — ML LOCO coefficient stability (EN + LASSO)
fig_A6_loco_ml_r2.png/pdf         — ML LOCO test-R² per excluded country
fig_A7_loco_reg_coef.png/pdf      — Regression LOCO coefficient stability
fig_A8_loco_reg_r2.png/pdf        — Regression LOCO R²
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BOOT_DIR = "intermediary/bootstrap"
LOCO_DIR = "intermediary/loco"
OUT_DIR  = os.path.join(os.path.expanduser('~'), 'Downloads', 'capstone_charts', 'appendix_robustness')
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
    'figure.dpi':        150,
})

BLUE  = '#4a6fa5'
GREEN = '#2e7d4a'
RED   = '#c23a3a'
TEAL  = '#2a8fa5'
GREY  = '#6b7280'

EXCLUDE_FROM_COEF_PLOT = {'Resource_HHI'}

MODEL_COLORS = {
    'LASSO':         BLUE,
    'Ridge':         '#7aa0c4',
    'Elastic Net':   GREEN,
    'Random Forest': RED,
    'EN':            GREEN,
    'RF':            RED,
}

# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAPS
# ─────────────────────────────────────────────────────────────────────────────

ML_NAMES = {
    'Total_Production_Value_Per_Capita':                                    'Prod Value p.c.',
    'Human capital index':                                                  'Human Capital',
    'Rule of law index':                                                    'Rule of Law',
    'Political stability — estimate':                                       'Political Stability',
    'Trade (% of GDP)':                                                     'Trade',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP':  'Capital Formation',
    'Share of investment in GDP':                                           'Investment Share',
    'Domestic credit to private sector (% of GDP)':                        'Domestic Credit',
    'Landlocked':                                                           'Landlocked',
    'Urban population (% of total population)':                            'Urban Population',
    'Government revenue':                                                   'Gov Revenue',
    'Capital depreciation rate':                                            'Depreciation',
    'Use of IMF credit (DOD, current US$)':                                'IMF Credit',
    'Real interest rate (%)':                                               'Real Interest Rate',
    'Inflation, consumer prices (annual %)':                               'Inflation (annual)',
    'Access to electricity (% of population)':                             'Electricity Access',
    'Adjusted savings: gross savings (% of GNI)':                          'Gross Savings',
    'L1_ECI':                                                              'Lagged ECI',
    'Forestry rents (% of GDP)':                                           'Forestry Rents',
    'Inflation_roll5':                                                      'Inflation (5yr avg)',
    'RealRate_roll5':                                                       'Real Rate (5yr avg)',
    'Resource_HHI':                                                         'Resource HHI',
    'HCI_x_ProductionValue':                                               'HC × Production',
    'RuleOfLaw_x_ProductionValue':                                         'RuleLaw × Production',
}

REG_NAMES = {
    'const':                                'Constant',
    'log_HCI':                              'Log HCI',
    'log_GFCF':                             'Log GFCF',
    'Political stability — estimate':       'Political Stability',
    'Rule of law index':                    'Rule of Law',
    'log_Production_Value':                 'Log Production',
    'Forestry rents (% of GDP)':            'Forestry Rents',
    'Trade (% of GDP)':                     'Trade',
    'log_HCI_x_log_Production':             'HCI × Production',
    'log_GFCF_x_log_Production':            'GFCF × Production',
    'log_HCI_x_forestry_rents':             'HCI × Forestry Rents',
    'log_GFCF_x_forestry_rents':            'GFCF × Forestry Rents',
    'ECI_lag1':                             'Lagged ECI',
}

def ml_sname(f):  return ML_NAMES.get(f, f[:22])
def reg_sname(v): return REG_NAMES.get(v, v[:22])


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(f"{path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{path}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}.png / .pdf")


def load(subdir, fname):
    path = os.path.join(subdir, fname)
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return None
    return pd.read_csv(path)


def ci_bar_chart(ax, y_pos, lo_vals, hi_vals, means, orig_vals, labels,
                 color=BLUE, orig_color='#333333'):
    """Horizontal CI bar chart: range whisker + mean dot + full-sample marker."""
    for j in range(len(y_pos)):
        ax.plot([lo_vals[j], hi_vals[j]], [y_pos[j], y_pos[j]],
                color=color, linewidth=1.6, alpha=0.7, solid_capstyle='round')

    dot_colors = [RED if m < 0 else color for m in means]
    ax.scatter(means, y_pos, c=dot_colors, s=55, zorder=5,
               edgecolors='white', linewidths=0.5)

    if orig_vals is not None:
        ax.scatter(orig_vals, y_pos, marker='|', c=orig_color,
                   s=90, zorder=6, linewidths=1.3)

    ax.axvline(0, color='#c9cfd6', linewidth=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()


# ─────────────────────────────────────────────────────────────────────────────
# FIG A1: Bootstrap CIs — OLS regression coefficients (3a and 3b)
# ─────────────────────────────────────────────────────────────────────────────

def fig_A1_boot_reg_coefs():
    summary = load(BOOT_DIR, "nb6_boot_summary.csv")
    if summary is None: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, model_id, title in zip(axes, ['3a', '3b'],
                                   ['Model 3a (no lag)', 'Model 3b (with lagged ECI)']):
        sub = summary[summary['Model'] == model_id].copy()
        sub = sub[sub['Variable'] != 'const'].reset_index(drop=True)
        sub['label'] = sub['Variable'].map(reg_sname)

        y_pos = np.arange(len(sub))

        # Colour: significant (CI excludes 0) vs not
        colors = []
        for _, row in sub.iterrows():
            if row['Boot_CI_lo'] > 0 or row['Boot_CI_hi'] < 0:
                colors.append(GREEN)
            else:
                colors.append(GREY)

        for j, (_, row) in enumerate(sub.iterrows()):
            ax.plot([row['Boot_CI_lo'], row['Boot_CI_hi']], [j, j],
                    color=colors[j], linewidth=2.0, alpha=0.7, solid_capstyle='round')

        dot_colors = [RED if m < 0 else c for m, c in zip(sub['Boot_Median'], colors)]
        ax.scatter(sub['Boot_Median'], y_pos, c=dot_colors, s=55,
                   zorder=5, edgecolors='white', linewidths=0.5)
        ax.scatter(sub['Original_Coef'], y_pos, marker='|',
                   c='#333333', s=90, zorder=6, linewidths=1.3)

        ax.axvline(0, color='#c9cfd6', linewidth=0.8, zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub['label'], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight='semibold', pad=8)
        ax.set_xlabel('OLS coefficient', fontsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN,
               markersize=7, label='Bootstrap median (sig.)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GREY,
               markersize=7, label='Bootstrap median (insig.)'),
        Line2D([0], [0], marker='|', color='#333333', markersize=9,
               markeredgewidth=1.3, linestyle='None', label='Original coef.'),
        Line2D([0], [0], color='#999', linewidth=1.6, label='95% bootstrap CI'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9,
                   frameon=True, framealpha=0.9, edgecolor='#e5e7eb')

    fig.suptitle('Bootstrap Coefficient Stability — OLS Regression (B = 200)',
                 fontsize=13, fontweight='semibold', y=1.01)
    savefig(fig, "fig_A1_boot_reg_coefs")


# ─────────────────────────────────────────────────────────────────────────────
# FIG A2: Bootstrap ML test-R² distributions
# ─────────────────────────────────────────────────────────────────────────────

def fig_A2_boot_ml_r2():
    metrics = load(BOOT_DIR, "nb5_boot_metrics.csv")
    if metrics is None: return

    models = ['LASSO', 'Ridge', 'Elastic Net', 'Random Forest']
    r2_cols = [f"{m}_test_r2" for m in models]
    data    = [metrics[c].dropna().values for c in r2_cols]

    fig, ax = plt.subplots(figsize=(9, 5))

    positions = np.arange(len(models))
    vp = ax.violinplot(data, positions=positions, showmedians=False,
                       showextrema=False, widths=0.6)

    for body, m in zip(vp['bodies'], models):
        body.set_facecolor(MODEL_COLORS[m])
        body.set_alpha(0.35)
        body.set_edgecolor(MODEL_COLORS[m])
        body.set_linewidth(1.0)

    # Box-plot stats overlay
    for j, (vals, m) in enumerate(zip(data, models)):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        lo95, hi95  = np.percentile(vals, [2.5, 97.5])
        clr = MODEL_COLORS[m]

        ax.vlines(j, lo95, hi95, color=clr, linewidth=1.5, alpha=0.7)
        ax.vlines(j, q1,   q3,   color=clr, linewidth=4.0)
        ax.scatter(j, med, color='white', s=40, zorder=5, edgecolors=clr, linewidths=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('Test R²', fontsize=11)
    ax.set_title('Bootstrap Distribution of Test R² — ML Models (B = 200)',
                 fontsize=13, fontweight='semibold', pad=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Annotation: median ± 95% CI
    for j, (vals, m) in enumerate(zip(data, models)):
        med   = np.median(vals)
        lo95  = np.percentile(vals, 2.5)
        hi95  = np.percentile(vals, 97.5)
        ax.text(j, lo95 - 0.015, f'{med:.3f}\n[{lo95:.3f}, {hi95:.3f}]',
                ha='center', va='top', fontsize=7.5, color=MODEL_COLORS[m])

    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig(fig, "fig_A2_boot_ml_r2")


# ─────────────────────────────────────────────────────────────────────────────
# FIG A3: Bootstrap RF feature importance with CIs
# ─────────────────────────────────────────────────────────────────────────────

def fig_A3_boot_rf_importance():
    imp = load(BOOT_DIR, "nb5_boot_importances.csv")
    if imp is None: return

    feat_cols = [c for c in imp.columns if c != 'b']
    medians   = imp[feat_cols].median().sort_values(ascending=False)
    top_feats = list(medians.head(12).index)

    lo95 = imp[top_feats].quantile(0.025)
    hi95 = imp[top_feats].quantile(0.975)
    med  = imp[top_feats].median()
    short = [ml_sname(f) for f in top_feats]

    y_pos = np.arange(len(top_feats))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for j, feat in enumerate(top_feats):
        ax.plot([lo95[feat], hi95[feat]], [j, j],
                color=RED, linewidth=1.6, alpha=0.6, solid_capstyle='round')

    ax.scatter(med.values, y_pos, c=RED, s=55, zorder=5,
               edgecolors='white', linewidths=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature importance (Gini)', fontsize=10)
    ax.set_title('Bootstrap RF Feature Importance — Top 12 (B = 200)',
                 fontsize=12, fontweight='semibold', pad=8)
    ax.tick_params(axis='x', labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RED,
               markersize=7, label='Bootstrap median'),
        Line2D([0], [0], color=RED, linewidth=1.6, alpha=0.6, label='95% CI'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              frameon=True, framealpha=0.9, edgecolor='#e5e7eb')

    plt.tight_layout()
    savefig(fig, "fig_A3_boot_rf_importance")


# ─────────────────────────────────────────────────────────────────────────────
# FIG A4: Bootstrap LASSO selection frequency
# ─────────────────────────────────────────────────────────────────────────────

def fig_A4_boot_lasso_selection():
    sel = load(BOOT_DIR, "nb5_boot_lasso_selection.csv")
    if sel is None: return

    feat_cols = [c for c in sel.columns if c != 'b' and c not in EXCLUDE_FROM_COEF_PLOT]
    freq = sel[feat_cols].mean().sort_values(ascending=False)
    short = [ml_sname(f) for f in freq.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(freq))

    bar_colors = []
    for v in freq.values:
        if v > 0.90:   bar_colors.append(GREEN)
        elif v > 0.50: bar_colors.append(BLUE)
        else:          bar_colors.append(GREY)

    ax.bar(x_pos, freq.values, color=bar_colors, edgecolor='white', linewidth=0.3)

    ax.axhline(0.90, color=GREEN, linewidth=1.0, linestyle='--', label='>90% (stable)')
    ax.axhline(0.50, color=GREY,  linewidth=1.0, linestyle=':', label='50% threshold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Selection frequency', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title('LASSO Selection Frequency Across Bootstrap Samples (B = 200)',
                 fontsize=12, fontweight='semibold', pad=8)
    ax.legend(fontsize=9, loc='upper right', frameon=True, framealpha=0.9,
              edgecolor='#e5e7eb')
    ax.tick_params(axis='y', labelsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    savefig(fig, "fig_A4_boot_lasso_selection")


# ─────────────────────────────────────────────────────────────────────────────
# FIG A4b: Bootstrap ML coefficient stability — LASSO, EN, Ridge (separate panels)
# ─────────────────────────────────────────────────────────────────────────────

def fig_A4b_boot_ml_coefs():
    coefs = load(BOOT_DIR, "nb5_boot_coefs.csv")
    if coefs is None: return

    ml_models = [('LASSO', BLUE), ('Elastic Net', GREEN), ('Ridge', '#7aa0c4')]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=False)
    fig.subplots_adjust(wspace=0.4)

    for ax, (model, clr) in zip(axes, ml_models):
        prefix = model + '__'
        feat_cols = [c for c in coefs.columns if c.startswith(prefix)]
        feat_names = [c[len(prefix):] for c in feat_cols]

        # Filter out excluded features
        feat_pairs = [(c, f) for c, f in zip(feat_cols, feat_names)
                      if f not in EXCLUDE_FROM_COEF_PLOT]
        feat_cols_f  = [p[0] for p in feat_pairs]
        feat_names_f = [p[1] for p in feat_pairs]

        med  = coefs[feat_cols_f].median()
        lo95 = coefs[feat_cols_f].quantile(0.025)
        hi95 = coefs[feat_cols_f].quantile(0.975)

        # Sort by absolute median descending
        order = med.abs().sort_values(ascending=False).index
        feat_cols_sorted  = list(order)
        feat_names_sorted = [feat_names_f[feat_cols_f.index(c)] for c in feat_cols_sorted]
        med_s  = med[feat_cols_sorted]
        lo_s   = lo95[feat_cols_sorted]
        hi_s   = hi95[feat_cols_sorted]
        y_pos  = np.arange(len(feat_cols_sorted))
        labels = [ml_sname(f) for f in feat_names_sorted]

        colors = [GREEN if (lo_s[c] > 0 or hi_s[c] < 0) else GREY for c in feat_cols_sorted]

        for j, c in enumerate(feat_cols_sorted):
            ax.plot([lo_s[c], hi_s[c]], [j, j],
                    color=colors[j], linewidth=1.6, alpha=0.7, solid_capstyle='round')

        dot_c = [RED if m < 0 else clr for m in med_s.values]
        ax.scatter(med_s.values, y_pos, c=dot_c, s=50, zorder=5,
                   edgecolors='white', linewidths=0.5)

        ax.axvline(0, color='#c9cfd6', linewidth=0.8, zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(model, fontsize=12, fontweight='semibold', pad=8)
        ax.set_xlabel('Standardised coefficient', fontsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN,
               markersize=7, label='Bootstrap median (sig.)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GREY,
               markersize=7, label='Bootstrap median (insig.)'),
        Line2D([0], [0], color='#999', linewidth=1.6, label='95% bootstrap CI'),
    ]
    axes[-1].legend(handles=legend_elements, loc='lower right', fontsize=9,
                    frameon=True, framealpha=0.9, edgecolor='#e5e7eb')

    fig.suptitle('Bootstrap Coefficient Stability — Penalised ML Models (B = 200)',
                 fontsize=13, fontweight='semibold', y=1.01)
    savefig(fig, "fig_A4b_boot_ml_coefs")


# ─────────────────────────────────────────────────────────────────────────────
# FIG A5: LOCO ML coefficient stability (EN + LASSO)
# ─────────────────────────────────────────────────────────────────────────────

def fig_A5_loco_ml_coef():
    en_df    = load(LOCO_DIR, "loco_coefs_en.csv")
    lasso_df = load(LOCO_DIR, "loco_coefs_lasso.csv")
    ridge_df = load(LOCO_DIR, "loco_coefs_ridge.csv")
    if en_df is None or lasso_df is None: return

    TOP_K = 10
    feat_cols = [c for c in en_df.columns if c != 'excluded_country']
    ranked    = en_df[feat_cols].abs().mean().sort_values(ascending=False)
    top_feats = list(ranked.head(TOP_K).index)

    panels = [
        ('ElasticNet', en_df,    GREEN),
        ('LASSO',      lasso_df, BLUE),
        ('Ridge',      ridge_df, '#7aa0c4'),
    ]
    panels = [(lbl, df, clr) for lbl, df, clr in panels if df is not None]

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6), sharey=False)
    if len(panels) == 1: axes = [axes]
    fig.subplots_adjust(wspace=0.35)

    for ax, (label, coef_df, clr) in zip(axes, panels):
        tf = [f for f in top_feats if f in coef_df.columns]
        means = coef_df[tf].mean()
        lo    = coef_df[tf].min()
        hi    = coef_df[tf].max()
        yp    = np.arange(len(tf))
        sh    = [ml_sname(f) for f in tf]

        for j, feat in enumerate(tf):
            ax.plot([lo[feat], hi[feat]], [j, j],
                    color=clr, linewidth=1.6, alpha=0.6, solid_capstyle='round')

        dot_colors = [RED if m < 0 else clr for m in means]
        ax.scatter(means.values, yp, c=dot_colors, s=55, zorder=5,
                   edgecolors='white', linewidths=0.5)

        ax.axvline(0, color='#c9cfd6', linewidth=0.8, zorder=0)
        ax.set_yticks(yp)
        ax.set_yticklabels(sh, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(label, fontsize=12, fontweight='semibold', pad=8)
        ax.set_xlabel('Standardised coefficient', fontsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555',
               markersize=7, label='LOCO mean'),
        Line2D([0], [0], color='#999', linewidth=1.6, label='LOCO min-max range'),
    ]
    axes[-1].legend(handles=legend_elements, loc='lower right', fontsize=9,
                    frameon=True, framealpha=0.9, edgecolor='#e5e7eb')

    fig.suptitle('LOCO Coefficient Stability — Penalised ML Models (N = 54 LOO runs)',
                 fontsize=13, fontweight='semibold', y=1.01)
    savefig(fig, "fig_A5_loco_ml_coef")


# ─────────────────────────────────────────────────────────────────────────────
# FIG A6: LOCO ML test-R² per excluded country
# ─────────────────────────────────────────────────────────────────────────────

def fig_A6_loco_ml_r2():
    r2_df = load(LOCO_DIR, "loco_r2.csv")
    if r2_df is None: return

    # Detect column names (old format has single 'test_r2', new has per-model)
    has_multi = 'LASSO_r2' in r2_df.columns

    if has_multi:
        model_cols = [('LASSO', 'LASSO_r2'), ('Ridge', 'Ridge_r2'),
                      ('EN', 'EN_r2'), ('RF', 'RF_r2')]
        r2_df = r2_df.copy()
        r2_df['mean_r2'] = r2_df[[c for _, c in model_cols]].mean(axis=1)
        r2_df = r2_df.sort_values('mean_r2').reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        x_pos = np.arange(len(r2_df))
        width = 0.2
        offsets = [-1.5, -0.5, 0.5, 1.5]

        for (label, col), off in zip(model_cols, offsets):
            ax.bar(x_pos + off * width, r2_df[col], width=width,
                   color=MODEL_COLORS[label], label=label,
                   edgecolor='white', linewidth=0.2)

    else:
        r2_df = r2_df.sort_values('test_r2').reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        x_pos = np.arange(len(r2_df))
        ax.bar(x_pos, r2_df['test_r2'], color=GREEN, edgecolor='white', linewidth=0.3,
               label='ElasticNet')

    ax.set_xticks(x_pos if not has_multi else np.arange(len(r2_df)))
    ax.set_xticklabels(r2_df['excluded_country'], rotation=90, fontsize=7)
    ax.set_ylabel('Test R²', fontsize=11)
    ax.set_xlabel('Excluded country', fontsize=11)
    ax.legend(fontsize=9, loc='lower left', frameon=True, framealpha=0.9, edgecolor='#e5e7eb')
    ax.tick_params(axis='y', labelsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('LOCO Test R² Stability — ML Models (N = 54 LOO runs)',
                 fontsize=13, fontweight='semibold', pad=10)

    plt.tight_layout()
    savefig(fig, "fig_A6_loco_ml_r2")


# ─────────────────────────────────────────────────────────────────────────────
# FIG A7: Regression LOCO coefficient stability
# ─────────────────────────────────────────────────────────────────────────────

def fig_A7_loco_reg_coef():
    df_3a = load(LOCO_DIR, "loco_reg_coefs_3a.csv")
    df_3b = load(LOCO_DIR, "loco_reg_coefs_3b.csv")
    if df_3a is None or df_3b is None: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    VARS_3A = ['log_HCI','log_GFCF','Political stability — estimate',
               'Rule of law index','log_Production_Value',
               'Forestry rents (% of GDP)','Trade (% of GDP)',
               'log_HCI_x_log_Production','log_GFCF_x_log_Production',
               'log_HCI_x_forestry_rents','log_GFCF_x_forestry_rents']
    VARS_3B = VARS_3A + ['ECI_lag1']

    for ax, (label, coef_df, var_list) in zip(axes, [
        ('Model 3a', df_3a, VARS_3A),
        ('Model 3b', df_3b, VARS_3B),
    ]):
        var_list_present = [v for v in var_list if v in coef_df.columns]
        y_pos  = np.arange(len(var_list_present))
        short  = [reg_sname(v) for v in var_list_present]
        means  = coef_df[var_list_present].mean()
        lo     = coef_df[var_list_present].min()
        hi     = coef_df[var_list_present].max()
        q025   = coef_df[var_list_present].quantile(0.025)
        q975   = coef_df[var_list_present].quantile(0.975)

        # Outer range (min-max)
        for j, v in enumerate(var_list_present):
            ax.plot([lo[v], hi[v]], [j, j],
                    color=TEAL, linewidth=1.0, alpha=0.35, solid_capstyle='round')
        # Inner CI (5-95 percentile)
        for j, v in enumerate(var_list_present):
            ax.plot([q025[v], q975[v]], [j, j],
                    color=TEAL, linewidth=2.5, alpha=0.65, solid_capstyle='round')

        dot_colors = [RED if m < 0 else TEAL for m in means.values]
        ax.scatter(means.values, y_pos, c=dot_colors, s=55, zorder=5,
                   edgecolors='white', linewidths=0.5)

        ax.axvline(0, color='#c9cfd6', linewidth=0.8, zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(label, fontsize=12, fontweight='semibold', pad=8)
        ax.set_xlabel('OLS coefficient', fontsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAL,
               markersize=7, label='LOCO mean'),
        Line2D([0], [0], color=TEAL, linewidth=2.5, alpha=0.65, label='5–95th pct range'),
        Line2D([0], [0], color=TEAL, linewidth=1.0, alpha=0.35, label='Min-max range'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9,
                   frameon=True, framealpha=0.9, edgecolor='#e5e7eb')

    fig.suptitle('LOCO Coefficient Stability — OLS Regression (N = 54 LOO runs)',
                 fontsize=13, fontweight='semibold', y=1.01)
    savefig(fig, "fig_A7_loco_reg_coef")


# ─────────────────────────────────────────────────────────────────────────────
# FIG A8: Regression LOCO R²
# ─────────────────────────────────────────────────────────────────────────────

def fig_A8_loco_reg_r2():
    r2_df = load(LOCO_DIR, "loco_reg_r2.csv")
    if r2_df is None: return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.subplots_adjust(hspace=0.45)

    for ax, (col, title, clr) in zip(axes, [
        ('r2_3a', 'Model 3a — No Lag', TEAL),
        ('r2_3b', 'Model 3b — With Lagged ECI', BLUE),
    ]):
        sub = r2_df[['excluded_country', col]].copy()
        sub = sub.sort_values(col).reset_index(drop=True)
        x_pos = np.arange(len(sub))

        ax.bar(x_pos, sub[col], color=clr, edgecolor='white', linewidth=0.3)
        ax.axhline(sub[col].mean(), color='#444', linewidth=1.0, linestyle='--',
                   label=f'Mean R² = {sub[col].mean():.3f}')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(sub['excluded_country'], rotation=90, fontsize=7)
        ax.set_ylabel('OLS R²', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='semibold', pad=6)
        ax.legend(fontsize=9, loc='lower left', frameon=True, framealpha=0.9,
                  edgecolor='#e5e7eb')
        ax.tick_params(axis='y', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('LOCO R² Stability — OLS Regression (N = 54 LOO runs)',
                 fontsize=13, fontweight='semibold', y=1.01)
    plt.tight_layout()
    savefig(fig, "fig_A8_loco_reg_r2")


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED SUMMARY TABLE  (text file for paper)
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_table():
    summary = load(BOOT_DIR, "nb6_boot_summary.csv")
    metrics = load(BOOT_DIR, "nb5_boot_metrics.csv")
    r2_ml   = load(LOCO_DIR, "loco_r2.csv")
    r2_reg  = load(LOCO_DIR, "loco_reg_r2.csv")

    lines = []

    if summary is not None:
        lines.append("=== BOOTSTRAP REGRESSION STABILITY (B=200) ===")
        se_col = 'Clustered_SE' if 'Clustered_SE' in summary.columns else 'DK_SE'
        lines.append(f"{'Model':<6}  {'Variable':<35}  {'Orig':>8}  {'Bt Med':>8}  "
                     f"{'Bt SE':>7}  {'CI lo':>8}  {'CI hi':>8}  {'Sign%':>6}")
        lines.append("-"*90)
        for _, row in summary.iterrows():
            if row['Variable'] == 'const': continue
            sig = "*" if (row['Boot_CI_lo'] > 0 or row['Boot_CI_hi'] < 0) else " "
            lines.append(
                f"  {row['Model']:<6}  {reg_sname(row['Variable']):<35}  "
                f"{row['Original_Coef']:>+8.4f}  {row['Boot_Median']:>+8.4f}  "
                f"{row['Boot_SE']:>7.4f}  {row['Boot_CI_lo']:>+8.4f}  "
                f"{row['Boot_CI_hi']:>+8.4f}  {row['Sign_Stability']:>5.0%} {sig}"
            )
        lines.append("")

    if metrics is not None:
        lines.append("=== BOOTSTRAP ML TEST-R² (B=200, 95% CI) ===")
        for m in ['LASSO', 'Ridge', 'Elastic Net', 'Random Forest']:
            vals = metrics[f"{m}_test_r2"].dropna()
            lines.append(f"  {m:<16}  median={vals.median():.4f}  "
                         f"CI=[{vals.quantile(.025):.4f}, {vals.quantile(.975):.4f}]  "
                         f"SD={vals.std():.4f}")
        lines.append("")

    if r2_ml is not None and 'EN_r2' in r2_ml.columns:
        lines.append("=== LOCO ML R² STABILITY ===")
        for label, col in [('LASSO','LASSO_r2'),('Ridge','Ridge_r2'),('EN','EN_r2'),('RF','RF_r2')]:
            vals = r2_ml[col]
            lines.append(f"  {label:<8}  mean={vals.mean():.4f}  "
                         f"range=[{vals.min():.4f}, {vals.max():.4f}]  SD={vals.std():.4f}")
        lines.append("")

    if r2_reg is not None:
        lines.append("=== LOCO REGRESSION R² STABILITY ===")
        for label, col in [('Model 3a','r2_3a'),('Model 3b','r2_3b')]:
            vals = r2_reg[col]
            lines.append(f"  {label}  mean={vals.mean():.4f}  "
                         f"range=[{vals.min():.4f}, {vals.max():.4f}]  SD={vals.std():.4f}")

    path = os.path.join(OUT_DIR, "robustness_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir).lower() in ("robustness", "scripts"):
        os.chdir(os.path.dirname(script_dir))

    print("="*70)
    print("  ROBUSTNESS APPENDIX FIGURES")
    print(f"  Output: {OUT_DIR}/")
    print("="*70)

    print("\n[A1] Bootstrap OLS coefficient CIs...")
    fig_A1_boot_reg_coefs()

    print("[A2] Bootstrap ML R² distributions...")
    fig_A2_boot_ml_r2()

    print("[A3] Bootstrap RF feature importance...")
    fig_A3_boot_rf_importance()

    print("[A4] Bootstrap LASSO selection frequency...")
    fig_A4_boot_lasso_selection()

    print("[A4b] Bootstrap ML coefficient stability (LASSO/EN/Ridge)...")
    fig_A4b_boot_ml_coefs()

    print("[A5] LOCO ML coefficient stability...")
    fig_A5_loco_ml_coef()

    print("[A6] LOCO ML R² stability...")
    fig_A6_loco_ml_r2()

    print("[A7] LOCO regression coefficient stability...")
    fig_A7_loco_reg_coef()

    print("[A8] LOCO regression R²...")
    fig_A8_loco_reg_r2()

    print("\nSummary table...")
    write_summary_table()

    print("\n" + "="*70)
    print("  ALL DONE — figures saved to Final/appendix/")
    print("="*70)
