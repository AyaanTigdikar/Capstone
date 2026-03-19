"""
viz_updates.py — Apply all chart modifications from user feedback.
Run from: /Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP/
"""
import os, warnings
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error

from viz_utils import (
    CLUSTER_LABELS, CLUSTER_COLORS, INCLUDE_LIST,
    WRITE_CONFIG, BG, GRID, FONT, NAVY, PALETTE,
    load_master, load_master_wide, load_clusters, load_nr,
    shorten_feat, base_layout, save,
    analyze_country_missingness,
)

OUT = 'Final/charts'
NB5 = 'Final/NB5'

os.makedirs(OUT, exist_ok=True)

LABEL_EXCL = ['L1_ECI', 'Inflation_roll5', 'RealRate_roll5', 'Resource_HHI']



# =============================================================================
# CHART 07 — Feature Importance Consensus (LASSO / Ridge / Elastic Net)
# =============================================================================
print("\n[07] Feature importance consensus...")

imp = pd.read_csv(os.path.join(NB5, 'all_importance.csv'))
imp = imp[~imp['Feature'].apply(lambda f: any(e in f for e in LABEL_EXCL))]

lin_cols = [c for c in ['LASSO', 'Ridge', 'Elastic Net'] if c in imp.columns]
imp = (imp.sort_values('Elastic Net' if 'Elastic Net' in imp.columns else lin_cols[0],
                       ascending=False)
         .head(12).reset_index(drop=True))
imp = imp.iloc[::-1].reset_index(drop=True)
imp['Label'] = imp['Feature'].apply(shorten_feat)

fig = go.Figure()
for _, row in imp.iterrows():
    vals = [row[c] for c in lin_cols if not pd.isna(row[c])]
    if len(vals) >= 2:
        fig.add_trace(go.Scatter(
            x=[min(vals), max(vals)], y=[row['Label'], row['Label']],
            mode='lines', line=dict(color='#c0c8d4', width=3),
            showlegend=False, hoverinfo='skip',
        ))

model_cfg07 = [
    ('LASSO',       'circle',      PALETTE['lasso']),
    ('Ridge',       'square',      PALETTE['ridge']),
    ('Elastic Net', 'triangle-up', PALETTE['en']),
]
for mname, sym, col in model_cfg07:
    if mname not in imp.columns:
        continue
    fig.add_trace(go.Scatter(
        x=imp[mname], y=imp['Label'],
        mode='markers',
        marker=dict(symbol=sym, size=13, color=col, line=dict(color='white', width=1.5)),
        name=mname,
        hovertemplate=f'%{{y}}: %{{x:.3f}}<extra>{mname}</extra>',
    ))

x_max = imp[lin_cols].max().max()
fig.update_layout(**base_layout(
    height=560,
    margin=dict(l=200, r=80, t=70, b=80),
    xaxis=dict(title=dict(text='Normalised Feature Importance (min-max, 0–1)', font=dict(size=13)),
               range=[-0.02, x_max + 0.1], gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(tickfont=dict(size=11)),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=11)),
))
save(fig, '07_ml__feature_importance_consensus_three_models', OUT, w=1100, h=560)


# =============================================================================
# CHART 08 — Standardised Coefficients (LASSO / Ridge / Elastic Net)
# =============================================================================
print("\n[08] Standardised coefficients...")

tbl = pd.read_csv(os.path.join(NB5, 'coefficient_summary_table.csv'))
tbl = tbl[~tbl['Feature'].apply(lambda f: any(e in f for e in LABEL_EXCL))]
tbl['abs_en'] = tbl['Elastic Net'].abs()
top = tbl.nlargest(12, 'abs_en').sort_values('abs_en', ascending=True).reset_index(drop=True)

fig = go.Figure()
fig.add_vline(x=0, line=dict(color='#444', width=1.5))

model_cfg08 = [
    ('LASSO',       PALETTE['lasso']),
    ('Ridge',       PALETTE['ridge']),
    ('Elastic Net', PALETTE['en']),
]
for mname, col in model_cfg08:
    if mname not in top.columns:
        continue
    fig.add_trace(go.Bar(
        y=top['Feature'], x=top[mname], orientation='h',
        name=mname,
        marker=dict(color=col, opacity=0.88, line=dict(color='white', width=0.5)),
        hovertemplate=f'%{{y}}: %{{x:+.3f}}<extra>{mname}</extra>',
    ))

fig.update_layout(**base_layout(
    barmode='group', height=620,
    margin=dict(l=200, r=80, t=70, b=60),
    xaxis=dict(title=dict(text='Coefficient (standardised inputs)', font=dict(size=13)),
               gridcolor=GRID, gridwidth=0.5, zeroline=False),
    yaxis=dict(tickfont=dict(size=11)),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=11)),
))
save(fig, '08_ml__standardised_coefficients_lasso_ridge_en', OUT, w=1100, h=620)

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# =============================================================================
# CHART 09 — Train vs Test R² (XGBoost removed)
# =============================================================================
print("\n[09] Train vs Test R²...")

perf_l = pd.read_csv(os.path.join(NB5, 'model_performance_level.csv'))
perf_d = pd.read_csv(os.path.join(NB5, 'model_performance_delta.csv'))
perf_l = perf_l[perf_l['Model'] != 'XGBoost'].reset_index(drop=True)
perf_d = perf_d[perf_d['Model'] != 'XGBoost'].reset_index(drop=True)

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14)

for col_idx, (perf, panel) in enumerate([(perf_l, 'ECI Level'), (perf_d, 'ΔECI')], 1):
    models   = perf['Model'].tolist()
    train_r2 = perf['Train R²'].tolist()
    test_r2  = perf['Test R²'].tolist()

    for m, tr, te in zip(models, train_r2, test_r2):
        fig.add_trace(go.Scatter(
            x=[tr, te], y=[m, m], mode='lines',
            line=dict(color='#c0c8d4', width=2.5),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=col_idx)

    fig.add_trace(go.Scatter(
        x=train_r2, y=models, mode='markers',
        marker=dict(symbol='circle', size=13, color=PALETTE['blue'],
                    line=dict(color='white', width=1.5)),
        name='Train R²', showlegend=(col_idx == 1), legendgroup='train',
        hovertemplate='%{y} Train: %{x:.3f}<extra></extra>',
    ), row=1, col=col_idx)

    fig.add_trace(go.Scatter(
        x=test_r2, y=models, mode='markers',
        marker=dict(symbol='diamond', size=13, color=PALETTE['red'],
                    line=dict(color='white', width=1.5)),
        name='Test R²', showlegend=(col_idx == 1), legendgroup='test',
        hovertemplate='%{y} Test: %{x:.3f}<extra></extra>',
    ), row=1, col=col_idx)

    fig.update_xaxes(title_text='R²', gridcolor=GRID, gridwidth=0.5, row=1, col=col_idx)
    fig.update_yaxes(tickfont=dict(size=11), row=1, col=col_idx)

for x_paper, label in [(0.23, 'ECI Level'), (0.77, 'ΔECI')]:
    fig.add_annotation(
        x=x_paper, y=1.04, xref='paper', yref='paper',
        text=f'<b>{label}</b>', showarrow=False,
        font=dict(size=12, color=NAVY, family=FONT),
        xanchor='center', yanchor='bottom',
    )

fig.update_layout(**base_layout(
    height=440, margin=dict(l=130, r=60, t=80, b=60),
    legend=dict(orientation='h', yanchor='bottom', y=1.06,
                xanchor='center', x=0.5, font=dict(size=11)),
))
save(fig, '09_ml__train_vs_test_r2_all_models', OUT)


# =============================================================================
# CHART 10 — Actual vs Predicted (ECI level + ΔECI side by side)
# =============================================================================
print("\n[10] Actual vs Predicted — ECI + ΔECI...")

preds = pd.read_csv(os.path.join(NB5, 'test_predictions.csv'))

fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.12,
)

for col_idx, (actual_col, pred_col, label) in enumerate([
    ('Actual_ECI',   'Predicted_ECI',   'ECI'),
    ('Actual_Delta', 'Predicted_Delta', 'ΔECI'),
], 1):
    actual = preds[actual_col].dropna().values
    pred   = preds.loc[preds[actual_col].notna(), pred_col].values
    codes  = preds.loc[preds[actual_col].notna(), 'Country Code'].values
    names  = preds.loc[preds[actual_col].notna(), 'Country Name'].values

    lims = [min(actual.min(), pred.min()) - 0.1, max(actual.max(), pred.max()) + 0.1]
    mid  = 0.0

    for x0, x1, y0, y1, fc in [
        (lims[0], mid,     lims[0], mid,     'rgba(46,125,74,0.07)'),
        (mid,     lims[1], mid,     lims[1], 'rgba(46,125,74,0.07)'),
        (lims[0], mid,     mid,     lims[1], 'rgba(194,58,58,0.07)'),
        (mid,     lims[1], lims[0], mid,     'rgba(194,58,58,0.07)'),
    ]:
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=fc, line=dict(width=0), layer='below',
                      row=1, col=col_idx)

    fig.add_trace(go.Scatter(
        x=[lims[0], lims[1]], y=[lims[0], lims[1]],
        mode='lines', line=dict(color=PALETTE['red'], width=1.5, dash='dash'),
        name='45° line', showlegend=(col_idx == 1), legendgroup='line45',
    ), row=1, col=col_idx)

    resid   = np.abs(actual - pred)
    top_idx = set(np.argsort(resid)[::-1][:5])
    mask_n  = np.array([i not in top_idx for i in range(len(actual))])

    fig.add_trace(go.Scatter(
        x=actual[mask_n], y=pred[mask_n], mode='markers',
        marker=dict(size=6, color=PALETTE['blue'], opacity=0.65,
                    line=dict(color='white', width=0.5)),
        name='Test obs.', showlegend=(col_idx == 1), legendgroup='obs',
        customdata=np.stack([codes[mask_n], names[mask_n]], axis=1),
        hovertemplate='<b>%{customdata[1]}</b><br>'
                      'Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>',
    ), row=1, col=col_idx)

    out_idx = list(top_idx)
    fig.add_trace(go.Scatter(
        x=actual[out_idx], y=pred[out_idx], mode='markers+text',
        marker=dict(size=9, color=PALETTE['orange'], opacity=0.9,
                    line=dict(color='white', width=1)),
        text=codes[out_idx], textposition='top center', textfont=dict(size=9),
        name='Largest residuals', showlegend=(col_idx == 1), legendgroup='outliers',
        customdata=np.stack([codes[out_idx], names[out_idx]], axis=1),
        hovertemplate='<b>%{customdata[1]}</b><br>'
                      'Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>',
    ), row=1, col=col_idx)

    fig.add_hline(y=0, line=dict(color=GRID, width=1), row=1, col=col_idx)
    fig.add_vline(x=0, line=dict(color=GRID, width=1), row=1, col=col_idx)

    fig.update_xaxes(title_text=f'Actual {label} (test set)', range=lims,
                     gridcolor=GRID, gridwidth=0.5, row=1, col=col_idx)
    fig.update_yaxes(title_text=f'Predicted {label}', range=lims,
                     gridcolor=GRID, gridwidth=0.5, row=1, col=col_idx)

fig.update_layout(**base_layout(
    height=560, margin=dict(l=70, r=50, t=70, b=60),
    legend=dict(orientation='h', yanchor='bottom', y=1.04,
                xanchor='center', x=0.5, font=dict(size=10)),
))
save(fig, '10_ml__actual_vs_predicted_eci_test_set', OUT)


# =============================================================================
# CHART 11 — ECI Forecast: two panels (best performers | worst performers)
#            Each panel shows all 54 grey + case studies + top3 or bottom3
# =============================================================================
print("\n[11] ECI Forecast — split into best / worst panels...")

fc   = pd.read_csv(os.path.join(NB5, 'ECI_Forecast_2020_2030.csv'))
rank = pd.read_csv(os.path.join(NB5, 'Country_Ranking_2020_2030.csv'))
perf = pd.read_csv(os.path.join(NB5, 'model_performance_level.csv'))
perf = perf[perf['Model'] != 'XGBoost']
best_rmse = perf.iloc[0]['Test RMSE'] if 'Test RMSE' in perf.columns else 0.08

master = load_master()
hist   = master[['Country Code', 'Country Name', 'Year',
                  'Economic Complexity Index']].dropna()

rank_sorted  = rank.sort_values('Total_Change', ascending=False).reset_index(drop=True)
CASE_STUDIES = ['COG', 'AZE', 'CHL']
top3    = [cc for cc in rank_sorted['Country Code'].tolist() if cc not in CASE_STUDIES][:3]
bottom3 = [cc for cc in rank_sorted['Country Code'].tolist()[::-1] if cc not in CASE_STUDIES][:3]

CASE_COL = '#4a6fa5'
TOP_COL  = '#2e7d4a'
BOT_COL  = '#c23a3a'
GREY     = '#b0b8c4'

all_eci = pd.concat([
    hist[hist['Country Code'].isin(INCLUDE_LIST)]['Economic Complexity Index'],
    fc[fc['Country Code'].isin(INCLUDE_LIST)]['Ensemble'],
]).dropna()
Y_RANGE = [all_eci.min() - 0.15, all_eci.max() + 0.15]

def _add_country_traces(fig, cc, cname, col, lw, opacity, legendgroup,
                        row_n, col_n, show_legend=False):
    """Draw historical solid + forecast dashed (+ confidence band for highlights)."""
    h = hist[hist['Country Code'] == cc].sort_values('Year')
    f = fc[fc['Country Code'] == cc].sort_values('Year')
    if h.empty or f.empty:
        return None
    ens = f['Ensemble'].values
    yrs = f['Year'].values

    fig.add_trace(go.Scatter(
        x=h['Year'], y=h['Economic Complexity Index'],
        mode='lines', line=dict(color=col, width=lw), opacity=opacity,
        legendgroup=legendgroup, showlegend=False,
        hovertemplate=f'<b>{cname} ({cc})</b><br>%{{x}}: %{{y:.3f}}<extra>Historical</extra>',
    ), row=row_n, col=col_n)

    if col != GREY:
        rgb = ','.join(str(v) for v in _hex_to_rgb(col))
        fig.add_trace(go.Scatter(
            x=np.concatenate([yrs, yrs[::-1]]).tolist(),
            y=np.concatenate([ens + best_rmse, (ens - best_rmse)[::-1]]).tolist(),
            fill='toself', fillcolor=f'rgba({rgb},0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='skip', legendgroup=legendgroup,
        ), row=row_n, col=col_n)

    last_yr  = int(h['Year'].iloc[-1])
    last_eci = float(h['Economic Complexity Index'].iloc[-1])
    fig.add_trace(go.Scatter(
        x=[last_yr] + yrs.tolist(), y=[last_eci] + ens.tolist(),
        mode='lines', line=dict(color=col, width=lw, dash='dash'), opacity=opacity,
        legendgroup=legendgroup, showlegend=False,
        hovertemplate=f'<b>{cname} ({cc})</b><br>%{{x}}: %{{y:.3f}}<extra>Forecast</extra>',
    ), row=row_n, col=col_n)
    return float(ens[-1])


def _deconflict(label_info, min_gap=0.22):
    sorted_lbl = sorted(label_info.items(), key=lambda x: x[1][0])
    adjusted   = {}
    floor_y    = None
    for cc, (y_act, _) in sorted_lbl:
        y_place = y_act if floor_y is None else max(y_act, floor_y + min_gap)
        adjusted[cc] = y_place
        floor_y = y_place
    return adjusted


def _add_labels(fig, label_info, adjusted_y, x_anchor=2031):
    for cc, (y_act, col) in label_info.items():
        fig.add_annotation(
            x=2030, y=y_act,
            ax=x_anchor, ay=adjusted_y[cc],
            axref='x', ayref='y',
            text=f'<b>{cc}</b>',
            showarrow=True,
            arrowhead=2, arrowwidth=1, arrowsize=0.8, arrowcolor=col,
            font=dict(size=9.5, color=col, family=FONT),
            xanchor='left', yanchor='middle',
        )


fig = make_subplots(
    rows=1, cols=2,
    horizontal_spacing=0.07,
)

for panel_col, highlight_group, highlight_col, grp_name in [
    (1, top3,    TOP_COL, 'top3'),
    (2, bottom3, BOT_COL, 'bottom3'),
]:
    # shared x-axis decorations
    fig.add_vrect(x0=2019.5, x1=2030.5, fillcolor='rgba(200,210,225,0.18)',
                  line=dict(width=0), layer='below', row=1, col=panel_col)
    fig.add_vline(x=2019.5, line=dict(color='#aaa', width=1.5, dash='dot'),
                  row=1, col=panel_col)

    highlighted_here = set(CASE_STUDIES + highlight_group)

    # 1. Grey background — all non-highlighted
    for cc in INCLUDE_LIST:
        if cc in highlighted_here:
            continue
        row_   = rank[rank['Country Code'] == cc]
        cname_ = row_['Country'].values[0] if len(row_) else cc
        _add_country_traces(fig, cc, cname_, GREY, lw=0.7, opacity=0.3,
                            legendgroup='others', row_n=1, col_n=panel_col)

    label_info = {}

    # 2. Highlight group (top3 or bottom3)
    for cc in highlight_group:
        row_   = rank[rank['Country Code'] == cc]
        cname_ = row_['Country'].values[0] if len(row_) else cc
        y_end  = _add_country_traces(fig, cc, cname_, highlight_col, lw=2.2, opacity=1.0,
                                     legendgroup=grp_name, row_n=1, col_n=panel_col)
        if y_end is not None:
            label_info[cc] = (y_end, highlight_col)

    # 3. Case studies
    for cc in CASE_STUDIES:
        row_   = rank[rank['Country Code'] == cc]
        cname_ = row_['Country'].values[0] if len(row_) else cc
        y_end  = _add_country_traces(fig, cc, cname_, CASE_COL, lw=2.5, opacity=1.0,
                                     legendgroup='cases', row_n=1, col_n=panel_col)
        if y_end is not None:
            label_info[cc] = (y_end, CASE_COL)

    adjusted_y = _deconflict(label_info)

    # Annotations use panel-specific xref/yref
    xref = 'x' if panel_col == 1 else 'x2'
    yref = 'y' if panel_col == 1 else 'y2'
    for cc, (y_act, col) in label_info.items():
        fig.add_annotation(
            x=2030, y=y_act,
            ax=2031, ay=adjusted_y[cc],
            axref=xref, ayref=yref,
            xref=xref, yref=yref,
            text=f'<b>{cc}</b>',
            showarrow=True,
            arrowhead=2, arrowwidth=1, arrowsize=0.8, arrowcolor=col,
            font=dict(size=9.5, color=col, family=FONT),
            xanchor='left', yanchor='middle',
        )

    fig.update_xaxes(title_text='Year', gridcolor=GRID, gridwidth=0.5,
                     dtick=5, range=[1994, 2033], row=1, col=panel_col)
    fig.update_yaxes(title_text='Economic Complexity Index' if panel_col == 1 else '',
                     range=Y_RANGE, gridcolor=GRID, gridwidth=0.5, row=1, col=panel_col)

# Shared legend entries
for lbl, col, rk, grp in [
    ('Case studies — COG · AZE · CHL',       CASE_COL, 1, 'cases'),
    ('Top 3 improvers — GNQ · MNG · ECU',    TOP_COL,  2, 'top3'),
    ('Bottom 3 decliners — ZWE · SAU · KAZ', BOT_COL,  3, 'bottom3'),
    ('Other countries',                       GREY,     4, 'others'),
]:
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color=col, width=2.5), name=lbl,
        legendgroup=grp, showlegend=True, legendrank=rk))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
    line=dict(color='#888', width=1.5), name='── Historical',
    showlegend=True, legendrank=10))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
    line=dict(color='#888', width=1.5, dash='dash'), name='- - Forecast',
    showlegend=True, legendrank=11))

fig.update_layout(**base_layout(
    height=620, margin=dict(l=70, r=20, t=70, b=110),
    legend=dict(
        orientation='h', font=dict(size=9.5),
        bgcolor='rgba(255,255,255,0.92)', bordercolor=GRID, borderwidth=1,
        x=0.0, y=-0.15, xanchor='left', yanchor='top', tracegroupgap=0,
    ),
))
save(fig, '11_ml__eci_forecast_top_improvers_2020_2030', OUT)


# =============================================================================
# CHART 15 — ECI Cluster Trajectories (cluster names instead of numbers)
# =============================================================================
print("\n[15] ECI Cluster Trajectories — cluster names...")

master   = load_master()
clusters = load_clusters('1995')[['Country Code', 'Cluster']].drop_duplicates()
df       = master[master['Country Code'].isin(INCLUDE_LIST)].copy()
df       = df.merge(clusters, on='Country Code', how='left')

traj = (df.groupby(['Year', 'Cluster'])['Economic Complexity Index']
          .median().reset_index())

fig = go.Figure()
for cl in sorted(traj['Cluster'].dropna().unique()):
    sub = traj[traj['Cluster'] == cl]
    fig.add_trace(go.Scatter(
        x=sub['Year'], y=sub['Economic Complexity Index'],
        mode='lines+markers',
        name=CLUSTER_LABELS.get(int(cl), f'Cluster {int(cl)}'),
        line=dict(color=CLUSTER_COLORS.get(int(cl), '#999'), width=2.2),
        marker=dict(size=5),
        hovertemplate='%{x}: %{y:.3f}<extra>' +
                      CLUSTER_LABELS.get(int(cl), '') + '</extra>',
    ))

fig.update_layout(**base_layout(
    height=480,
    xaxis=dict(title='Year', gridcolor=GRID, gridwidth=0.5, dtick=5),
    yaxis=dict(title='Median ECI', gridcolor=GRID, gridwidth=0.5,
               zeroline=True, zerolinecolor='#ddd', zerolinewidth=1),
    legend=dict(font=dict(size=10), bgcolor='rgba(255,255,255,0.9)',
                bordercolor=GRID, borderwidth=1),
    hovermode='x unified',
))
save(fig, '15_reg__eci_mean_trajectory_by_cluster', OUT)


# =============================================================================
# CHART 16 — Coefficients 3a vs 3b (remove Driscoll-Kraay from axis title)
# =============================================================================
print("\n[16] Coefficients 3a vs 3b — removing Driscoll-Kraay...")

chart16_path = os.path.join(OUT, '16_reg__coefficients_model3a_vs_model3b.html')
if os.path.exists(chart16_path):
    html = open(chart16_path, encoding='utf-8').read()
    html = html.replace('95 % CI, Driscoll-Kraay', '95% CI')
    html = html.replace('95% CI, Driscoll-Kraay', '95% CI')
    html = html.replace('Driscoll-Kraay', '')
    open(chart16_path, 'w', encoding='utf-8').write(html)
    print(f"  Updated: 16_reg__coefficients_model3a_vs_model3b.html")


# =============================================================================
# CHART 17 — HCI × Production interaction (simplified: country means)
# =============================================================================
print("\n[17] HCI × Production interaction — simplifying...")

master = load_master()
df     = master[master['Country Code'].isin(INCLUDE_LIST)].copy()
df['prod_pc']    = df['Total_Production_Value'] / df['Population'].replace(0, np.nan)
df['log_HCI']    = np.log1p(df['Human capital index'])
df['log_prod_pc']= np.log1p(df['prod_pc'])

# Aggregate to country means — one point per country, much cleaner than all obs
country_avg = (
    df[['Country Code', 'log_HCI', 'Economic Complexity Index', 'log_prod_pc']]
    .dropna()
    .groupby('Country Code')
    .mean()
    .reset_index()
)

# Assign production quartile based on country average
country_avg['Prod_quartile'] = pd.qcut(
    country_avg['log_prod_pc'], q=4,
    labels=['Q1 — Low production', 'Q2', 'Q3', 'Q4 — High production']
)

q_colors = [PALETTE['light_blue'], PALETTE['blue'], PALETTE['orange'], PALETTE['red']]

fig = go.Figure()
for q, col in zip(['Q1 — Low production', 'Q2', 'Q3', 'Q4 — High production'], q_colors):
    sub = country_avg[country_avg['Prod_quartile'] == q]
    fig.add_trace(go.Scatter(
        x=sub['log_HCI'], y=sub['Economic Complexity Index'],
        mode='markers+text',
        text=sub['Country Code'],
        textposition='top center',
        textfont=dict(size=8, color='#555'),
        marker=dict(color=col, size=9, opacity=0.85,
                    line=dict(width=0.8, color='white')),
        name=f'Production {q}',
        hovertemplate='<b>%{text}</b><br>log(HCI): %{x:.2f}<br>ECI: %{y:.2f}<extra></extra>',
    ))

fig.update_layout(**base_layout(
    height=540,
    xaxis=dict(title='log(Human Capital Index)', gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(title='Economic Complexity Index', gridcolor=GRID, gridwidth=0.5),
    legend=dict(title=dict(text='Avg. Production p.c. Quartile'),
                font=dict(size=10), bgcolor='rgba(255,255,255,0.9)',
                bordercolor=GRID, borderwidth=1),
))
save(fig, '17_reg__hci_production_interaction_effect_on_eci', OUT)


# =============================================================================
# CHART 26 — PCA Resource Loadings Heatmap
#   - Red-white-blue colorscale (was yellow-white-blue)
#   - PC1 row at top
#   - Larger colorbar and chart
# =============================================================================
print("\n[26] PCA Loadings heatmap — fixing colours, PC1 first, larger...")

nr        = load_nr()
nr_sample = nr[nr['Country Code'].isin(INCLUDE_LIST)]
nr_1995   = nr_sample[nr_sample['Year'] == 1995]

pivot = nr_1995.pivot_table(
    index=['Country', 'Country Code', 'Year', 'Population'],
    columns='Resource', values='Production_TotalValue',
).reset_index()

resource_cols = [c for c in pivot.columns
                 if c not in ['Country', 'Country Code', 'Year', 'Population']]
pivot[resource_cols] = pivot[resource_cols].div(pivot['Population'], axis=0)
pivot = pivot.fillna(0)

X   = np.log1p(pivot[resource_cols].fillna(0))
pca = PCA(n_components=2, random_state=42)
pca.fit(X)

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=resource_cols)
top20    = loadings.abs().sum(axis=1).nlargest(20).index
plot_df  = loadings.loc[top20]
plot_df  = (plot_df
            .assign(_s=plot_df['PC1'].abs() + plot_df['PC2'].abs())
            .sort_values('_s', ascending=False)
            .drop(columns='_s'))

# PC labels — PC1 first = top row (reversed y in Plotly)
# PC1 loads on oil & gas (hydrocarbons); PC2 loads on copper, gold & coal
pc_labels_ordered = [
    f'PC1 ({var1:.1f}%)<br><i>↑ Oil & Gas</i>',
    f'PC2 ({var2:.1f}%)<br><i>↑ Copper, Gold & Coal</i>',
]
# Plotly heatmap y goes bottom→top, so put PC2 first in the list → PC1 on top
y_labels = pc_labels_ordered[::-1]          # [PC2, PC1] → displayed bottom→top = PC2 bottom, PC1 top
z_values = plot_df[['PC2', 'PC1']].T.values  # rows match y_labels order

fig = go.Figure(go.Heatmap(
    z=z_values,
    x=plot_df.index.tolist(),
    y=y_labels,
    colorscale=[
        [0.0, '#c23a3a'],
        [0.5, '#ffffff'],
        [1.0, '#1a4a8a'],
    ],
    zmid=0, zmin=-1, zmax=1,
    text=z_values.round(2),
    texttemplate='%{text:.2f}',
    textfont=dict(size=8, family=FONT),
    hovertemplate='%{x} / %{y}: %{z:.3f}<extra></extra>',
    colorbar=dict(
        title=dict(text='Loading', font=dict(size=12)),
        thickness=20, len=1.0,
        tickvals=[-1, -0.5, 0, 0.5, 1],
        tickfont=dict(size=11),
    ),
))

fig.update_xaxes(title_text='Resource/Feature', tickangle=-40, tickfont=dict(size=10, family=FONT), showgrid=False)
fig.update_yaxes(title_text='Principal Component', tickfont=dict(size=12, family=FONT), showgrid=False)
fig.update_layout(**base_layout(
    height=420,
    margin=dict(l=260, r=120, t=60, b=160),
))
save(fig, '26_diag__pca_resource_loadings_heatmap', OUT, w=1300, h=420)


# =============================================================================
# CHART 31 — ML Prediction Intervals (aggregated by country)
# =============================================================================
print("\n[31] Prediction intervals — aggregating by country...")

preds = pd.read_csv(os.path.join(NB5, 'test_predictions.csv'))

# Country-level: mean actual, mean predicted, std actual over test years
country_stats = (
    preds.groupby(['Country Code', 'Country Name'])
    .agg(
        Actual_mean   = ('Actual_ECI', 'mean'),
        Predicted_mean= ('Predicted_ECI', 'mean'),
        Actual_std    = ('Actual_ECI', 'std'),
        n_years       = ('Year', 'count'),
    )
    .reset_index()
    .sort_values('Actual_mean')
    .reset_index(drop=True)
)
country_stats['Actual_std'] = country_stats['Actual_std'].fillna(0)

# In-band: |actual_mean - predicted_mean| < 1 std dev
country_stats['In_band'] = (
    (country_stats['Actual_mean'] - country_stats['Predicted_mean']).abs()
    < country_stats['Actual_std']
)

fig = go.Figure()

# Std-dev band per country (horizontal error bar on actual)
fig.add_trace(go.Scatter(
    x=list(range(len(country_stats))) * 2 + list(range(len(country_stats)))[::-1] * 2,
    y=(country_stats['Actual_mean'] + country_stats['Actual_std']).tolist() +
      (country_stats['Actual_mean'] - country_stats['Actual_std']).iloc[::-1].tolist(),
    fill='toself', fillcolor='rgba(74,111,165,0.15)',
    line=dict(color='rgba(0,0,0,0)'),
    hoverinfo='skip', name='±1 SD (actual ECI in test years)',
))

fig.add_trace(go.Scatter(
    x=list(range(len(country_stats))),
    y=country_stats['Actual_mean'],
    mode='lines', line=dict(color=PALETTE['blue'], width=2),
    name='Mean Actual ECI',
))

for in_band, color, sym, lbl in [
    (True,  PALETTE['green'], 'circle',  'Predicted ≈ Actual (within ±1 SD)'),
    (False, PALETTE['red'],   'diamond', 'Predicted outside ±1 SD'),
]:
    mask = country_stats['In_band'] == in_band
    sub  = country_stats[mask]
    fig.add_trace(go.Scatter(
        x=sub.index.tolist(),
        y=sub['Predicted_mean'],
        mode='markers',
        marker=dict(color=color, size=8 if in_band else 10, symbol=sym, opacity=0.85),
        name=lbl,
        customdata=sub[['Country Code', 'Country Name']].values,
        hovertemplate='<b>%{customdata[1]}</b> (%{customdata[0]})<br>'
                      'Avg Actual: %{text}<br>Avg Predicted: %{y:.3f}<extra></extra>',
        text=[f'{v:.3f}' for v in sub['Actual_mean']],
    ))

fig.update_layout(**base_layout(
    height=500,
    xaxis=dict(
        title='Countries (sorted by mean actual ECI)',
        tickvals=list(range(len(country_stats))),
        ticktext=country_stats['Country Code'].tolist(),
        tickangle=-60, tickfont=dict(size=8),
        gridcolor=GRID,
    ),
    yaxis=dict(title='ECI (test set mean)', gridcolor=GRID, gridwidth=0.5),
    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                xanchor='center', x=0.5, font=dict(size=10)),
))
save(fig, '31_diag__ml_prediction_intervals', OUT)


# =============================================================================
# CHART 32 — Country Data Coverage (highlight high-missingness countries)
# =============================================================================
print("\n[32] Country data coverage — highlighting high-missingness countries...")

raw = load_master_wide()
sample_df = raw[raw['Country Code'].isin(INCLUDE_LIST)].copy()
country_missing = analyze_country_missingness(sample_df)

LABEL_THRESHOLD = 20.0   # label countries above this % missing

fig = go.Figure()

for above in [True, False]:
    mask = country_missing['% Missing'] >= LABEL_THRESHOLD if above else \
           country_missing['% Missing'] < LABEL_THRESHOLD
    sub  = country_missing[mask]
    color= PALETTE['red'] if above else PALETTE['blue']
    size = 12 if above else 8
    mode = 'markers+text' if above else 'markers'

    fig.add_trace(go.Scatter(
        x=sub['Vars with Data'],
        y=sub['% Missing'],
        mode=mode,
        text=sub['Code'] if above else None,
        textposition='top center',
        textfont=dict(size=9, color=PALETTE['red']),
        marker=dict(color=color, size=size, opacity=0.75 if above else 0.55,
                    line=dict(color='white', width=0.8)),
        name=f'>= {LABEL_THRESHOLD}% missing' if above else f'< {LABEL_THRESHOLD}% missing',
        customdata=sub[['Code', 'Country', 'Complete Vars', 'Years Covered', 'Rows']].values,
        hovertemplate=(
            '<b>%{customdata[1]}</b> (%{customdata[0]})<br>'
            'Vars with data: %{x}<br>'
            '% Missing: %{y:.1f}%<br>'
            'Complete vars: %{customdata[2]}<br>'
            'Years covered: %{customdata[3]}<extra></extra>'
        ),
    ))

med_vars    = country_missing['Vars with Data'].median()
med_missing = country_missing['% Missing'].median()
fig.add_hline(y=med_missing, line_dash='dash', line_color='#aaa', opacity=0.6,
              annotation_text=f'Median {med_missing:.1f}%', annotation_position='right')
fig.add_vline(x=med_vars, line_dash='dash', line_color='#aaa', opacity=0.6,
              annotation_text=f'Median {med_vars:.0f} vars', annotation_position='top')

fig.update_layout(**base_layout(
    height=520,
    xaxis=dict(title='Variables with Any Data', gridcolor=GRID, gridwidth=0.5),
    yaxis=dict(title='% Missing Data Overall', gridcolor=GRID, gridwidth=0.5),
    legend=dict(font=dict(size=10), bgcolor='rgba(255,255,255,0.9)',
                bordercolor=GRID, borderwidth=1),
))
save(fig, '32_diag__country_data_coverage_scatter', OUT)


print("\n✓ All updates complete.")
