"""
save_for_viz.py
===============
Paste the relevant section at the END of NB4 / NB5 (as a new cell).
These save the intermediate CSVs that generate_report_charts.py and
extra_viz.py need to produce charts.

Structure:
    Section A  →  paste at end of NB4 (after the cluster export cell)
    Section B  →  paste at end of NB5 (after the forecast cell, before summary)
"""


# ═════════════════════════════════════════════════════════════════════════════
#
#   SECTION A — paste at END of NB4
#
#   Creates aliases that viz scripts expect:
#     intermediary/clusters1995.csv   (copy of clusters_k4_1995.csv)
#     intermediary/clusters2019.csv   (copy of clusters_k4_2019.csv)
#     intermediary/clustersagg.csv    (copy of clusters_k4_agg.csv)
#
# ═════════════════════════════════════════════════════════════════════════════

# --- START: paste this cell at end of NB4 ---

import shutil

for label in ['1995', '2019', 'agg']:
    src = f'intermediary/clusters_k4_{label}.csv'
    dst = f'intermediary/clusters{label}.csv'
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f'  {src} -> {dst}')

print('Cluster aliases saved for viz scripts.')

# --- END: NB4 cell ---


# ═════════════════════════════════════════════════════════════════════════════
#
#   SECTION B — paste at END of NB5
#
#   Saves the following CSVs to Graphics/NB5/:
#     all_importance.csv            (feature importance: LASSO, Ridge, EN, RF)
#     model_performance_level.csv   (train/test R2, RMSE, MAE for ECI)
#     model_performance_delta.csv   (same for delta-ECI)
#     test_predictions.csv          (actual vs predicted, per country-year)
#     vif_table.csv                 (variance inflation factors)
#     ECI_Forecast_2020_2030.csv    (full forecast timeseries)
#
#   Variables assumed to exist in the NB5 kernel at this point:
#     all_importance   (DataFrame)     — from Cell 10
#     perf_level       (DataFrame)     — from Cell 10
#     perf_delta       (DataFrame)     — from Cell 10
#     vif_data         (DataFrame)     — from Cell 10
#     test_df          (DataFrame)     — from Cell 6
#     X_test           (ndarray)       — from Cell 6
#     y_test_level     (ndarray)       — from Cell 10
#     y_test_delta     (ndarray)       — from Cell 10
#     models_level     (dict)          — from Cell 8
#     all_features     (list)          — from Cell 4
#     scaler           (StandardScaler)— from Cell 6
#     forecast_df      (DataFrame)     — from Cell 18
#     OUT              (str)           — 'Graphics/NB5'
#
# ═════════════════════════════════════════════════════════════════════════════

# --- START: paste this cell at end of NB5 (before the summary cell) ---

print('Saving intermediate CSVs for viz scripts...')

# 1. Feature importance (all models)
all_importance.to_csv(os.path.join(OUT, 'all_importance.csv'), index=False)
print(f'  all_importance.csv ({len(all_importance)} features)')

# 2. Model performance tables
perf_level.to_csv(os.path.join(OUT, 'model_performance_level.csv'), index=False)
perf_delta.to_csv(os.path.join(OUT, 'model_performance_delta.csv'), index=False)
print(f'  model_performance_level.csv ({len(perf_level)} models)')
print(f'  model_performance_delta.csv ({len(perf_delta)} models)')

# 3. VIF table
vif_data.to_csv(os.path.join(OUT, 'vif_table.csv'), index=False)
print(f'  vif_table.csv ({len(vif_data)} features)')

# 4. Test predictions (actual vs predicted, both targets)
_pred_level = models_level['Elastic Net'].predict(X_test)
_pred_delta = models_delta['Elastic Net'].predict(X_test)

test_preds = test_df[['Country Code', 'Country Name', 'Year']].copy().reset_index(drop=True)
test_preds['Actual_ECI']     = y_test_level
test_preds['Predicted_ECI']  = _pred_level
test_preds['Actual_Delta']   = y_test_delta
test_preds['Predicted_Delta'] = _pred_delta
test_preds.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)
print(f'  test_predictions.csv ({len(test_preds)} obs)')

# 5. Full forecast timeseries (if forecast_df exists from Cell 18)
try:
    forecast_df.to_csv(os.path.join(OUT, 'ECI_Forecast_2020_2030.csv'), index=False)
    print(f'  ECI_Forecast_2020_2030.csv ({len(forecast_df)} rows)')
except NameError:
    print('  SKIPPED: forecast_df not found (run Cell 18 first)')

print(f'\nAll viz CSVs saved to {OUT}/')

# --- END: NB5 cell ---
