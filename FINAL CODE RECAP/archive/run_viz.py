"""
Run all visualisation functions from viz_diagnostic.py and viz_analysis.py.
Output goes to Final/NB2, Final/NB4, Final/NB5, Final/NB6.
Errors are caught per-function so a single failure does not halt everything.
"""
import os, sys, traceback
import matplotlib
matplotlib.use('Agg')   # non-interactive — no GUI windows, plt.show() is a no-op

# ── Make sure we run from the notebook directory ──────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

for d in ['Final/NB2', 'Final/NB4', 'Final/NB5', 'Final/NB6']:
    os.makedirs(d, exist_ok=True)

results = {}

# ── Load modules ─────────────────────────────────────────────────────────
print("=" * 60)
print("Loading viz_diagnostic...")
try:
    import viz_diagnostic as vd
    print("  OK")
except Exception as e:
    print(f"  FAILED TO IMPORT: {e}")
    vd = None

print("Loading viz_analysis...")
try:
    import viz_analysis as va
    print("  OK")
except Exception as e:
    print(f"  FAILED TO IMPORT: {e}")
    va = None

# ── Diagnostic functions (NB2) ────────────────────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSTIC (NB2)")
print("=" * 60)

diag_fns = [
    'plot_variable_missingness',
    'plot_country_missingness_distribution',
    'plot_all_countries_ranked',
    'plot_variable_coverage',
    'plot_heatmap_problem_areas',
    'plot_missingness_over_time',
    'plot_interactive_country_profile',
    'plot_correlation_matrix_static',
    'plot_interactive_correlation_matrix',
]

if vd:
    for fn_name in diag_fns:
        fn = getattr(vd, fn_name, None)
        if fn is None:
            print(f"  [{fn_name}] NOT FOUND")
            results[fn_name] = 'missing'
            continue
        try:
            print(f"  [{fn_name}] running...", end=' ', flush=True)
            fn()
            print("OK")
            results[fn_name] = 'ok'
        except Exception as e:
            print(f"FAIL — {e}")
            traceback.print_exc()
            results[fn_name] = f'fail: {e}'

# ── Analysis functions (NB4, NB5, NB6) ───────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS (NB4 Clustering)")
print("=" * 60)

cluster_fns = [
    'plot_silhouette_validation',
    'plot_pca_loadings',
    'plot_pca_biplot',
    'plot_cluster_choropleth',
    'plot_rosling_eci_gdp',
]

ml_fns = [
    'plot_oos_r2_rmse_static',
    'plot_predicted_vs_actual',
    'plot_shap_importance',
    'plot_prediction_intervals',
    'plot_vif',
    'plot_coefficient_comparison_3panel',
    'plot_model_agreement',
    'plot_random_forest_importance',
    'plot_coefficient_summary_table',
    'plot_forecast_country_ranking',
    'plot_forecast_top_improvers_trajectory',
    'plot_forecast_heatmap_all_countries',
    'plot_avg_delta_eci_actual_vs_pred',
]

reg_fns = [
    'plot_eci_distribution_comparison',
    'plot_eci_cluster_trajectories',
    'plot_eci_correlation_heatmap',
    'plot_residual_qq',
    'plot_coef_comparison_3a_3b',
    'plot_eci_hci_production_interaction',
]

if va:
    for label, fn_list in [
        ('NB4 Clustering', cluster_fns),
        ('NB5 ML',         ml_fns),
        ('NB6 Regressions', reg_fns),
    ]:
        print(f"\n--- {label} ---")
        for fn_name in fn_list:
            fn = getattr(va, fn_name, None)
            if fn is None:
                print(f"  [{fn_name}] NOT FOUND")
                results[fn_name] = 'missing'
                continue
            try:
                print(f"  [{fn_name}] running...", end=' ', flush=True)
                fn()
                print("OK")
                results[fn_name] = 'ok'
            except Exception as e:
                print(f"FAIL — {e}")
                results[fn_name] = f'fail: {e}'

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
ok    = [k for k, v in results.items() if v == 'ok']
fail  = {k: v for k, v in results.items() if v.startswith('fail')}
miss  = [k for k, v in results.items() if v == 'missing']

print(f"  OK:      {len(ok)}")
print(f"  FAILED:  {len(fail)}")
print(f"  MISSING: {len(miss)}")
if fail:
    print("\nFailed functions:")
    for k, v in fail.items():
        print(f"  {k}: {v}")
if miss:
    print("\nMissing functions:")
    for k in miss:
        print(f"  {k}")

print("\nOutput written to Final/NB2, Final/NB4, Final/NB5, Final/NB6")
