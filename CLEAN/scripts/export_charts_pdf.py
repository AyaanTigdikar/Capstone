"""
export_charts_pdf.py — Screenshot all 34 charts at 2x DPI and combine into PDF.
Run from: /Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP/
Output:   Final/charts_overview.pdf
"""
import os, shutil
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

CHARTS_DIR = Path('Final/charts')
OUT_PDF    = Path('Final/charts_overview.pdf')
SHOT_DIR   = Path('Final/_screenshot_tmp')
SHOT_DIR.mkdir(exist_ok=True)

VIEWPORT_W   = 1400
VIEWPORT_H   = 800
SCALE_FACTOR = 4          # 4x pixel density
MAP_WAIT_MS  = 3000       # extra wait for choropleth/geo maps
STD_WAIT_MS  = 1000       # standard wait for non-map charts

# Charts that contain a geo map and need extra render time
MAP_STEMS = {
    '01_sample__54_resource_dependent_countries_map',
    '04_cluster__world_map_four_resource_profiles',
}

# Charts already used as-is in the thesis — get a red ● badge in top-right
ALREADY_USED = {
    # ── Confirmed in paper body (explicit Figure references found in PDF text) ──
    '01_sample__54_resource_dependent_countries_map',       # intro / sample section
    '02_data__variable_correlations_with_eci',              # Figure 3 (p.15) — correlation
    '03_cluster__pca_biplot_country_resource_groups',       # Figure 4 (p.18) — PCA biplot
    '04_cluster__world_map_four_resource_profiles',         # Figure 5 (p.18) — cluster map
    '05_cluster__eci_vs_gdp_animated_1995_to_2019',         # Figure 6 (p.20) — ECI vs GDP
    '07_ml__feature_importance_consensus_three_models',     # Figure 7 (p.30) — model agreement
    '08_ml__standardised_coefficients_lasso_ridge_en',      # Figure 8 (p.31) — coefficients
    '09_ml__train_vs_test_r2_all_models',                   # Figure 9 (p.34) + Figure [X] (p.36)
    '10_ml__actual_vs_predicted_eci_test_set',              # Figure 10 (p.35) + Figure [Y] (p.37)
    '11_ml__eci_forecast_top_improvers_2020_2030',          # Figure [Z] (p.38) — top improvers
    '26_diag__pca_resource_loadings_heatmap',               # Figure 3 (p.17) — PCA loadings
    '27_diag__random_forest_feature_importance',            # Figure 11 (p.32) — RF importance
    '28_diag__variance_inflation_factors',                  # Appendix Figure 6 (p.85) — VIF
    '29_diag__bootstrap_r2_and_coefficient_stability',      # Appendix Figures 3–5 (p.81–84)

    # ── Case study section (embedded in Word, not text-referenced by number) ──
    '21_case__congo_macro_indicators_vs_sample',
    '22_case__congo_eci_trajectory_vs_cluster_peers',
    '23_case__azerbaijan_eci_trajectory_vs_cluster_peers',
    '24_case__eci_forecast_2020_2030_three_countries',
}

html_files = sorted(CHARTS_DIR.glob('*.html'))
print(f"Found {len(html_files)} charts  [device_scale_factor={SCALE_FACTOR}]\n")

shot_paths = []

with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx     = browser.new_context(
        viewport={'width': VIEWPORT_W, 'height': VIEWPORT_H},
        device_scale_factor=SCALE_FACTOR,
    )
    page = ctx.new_page()

    for i, html in enumerate(html_files, 1):
        url    = html.resolve().as_uri()
        name   = html.stem
        out    = SHOT_DIR / f'{name}.png'
        is_map = name in MAP_STEMS

        print(f"  [{i:02d}/{len(html_files)}] {name}{'  [map]' if is_map else ''}")
        page.goto(url, wait_until='networkidle', timeout=45_000)

        # Give geo/WebGL traces time to fully paint
        wait_ms = MAP_WAIT_MS if is_map else STD_WAIT_MS
        page.wait_for_timeout(wait_ms)

        # Measure the Plotly div's actual rendered height — avoids capturing blank body
        chart_h = page.evaluate("""() => {
            const el = document.querySelector('.js-plotly-plot')
                     || document.querySelector('.plotly-graph-div');
            if (el) return Math.ceil(el.getBoundingClientRect().height);
            return document.body.scrollHeight;
        }""")
        snap_h = max(chart_h, 200)
        page.set_viewport_size({'width': VIEWPORT_W, 'height': snap_h})
        page.wait_for_timeout(300)

        # Clip to the chart area only — no extra whitespace below
        page.screenshot(path=str(out), clip={
            'x': 0, 'y': 0, 'width': VIEWPORT_W, 'height': snap_h,
        })
        shot_paths.append(out)

    browser.close()

print(f"\nScreenshots done. Composing PDF …")

# ── Build PDF (A4 landscape) ──────────────────────────────────────────────────
# At scale_factor=2, playwright renders at 2x CSS pixels, so effective DPI = 2*96 = 192
RENDER_DPI  = 96 * SCALE_FACTOR   # 192
A4_W_MM, A4_H_MM = 297, 210       # landscape
MM_PER_IN   = 25.4
A4_W_PX = int(A4_W_MM / MM_PER_IN * RENDER_DPI)   # pixels at render DPI
A4_H_PX = int(A4_H_MM / MM_PER_IN * RENDER_DPI)

def stamp_used(canvas: Image.Image) -> None:
    """Draw a red filled circle with a white ● in the top-right corner."""
    draw   = ImageDraw.Draw(canvas)
    r      = 38               # circle radius in pixels (at 4x scale looks ~10px on page)
    pad    = 24
    cx     = canvas.width  - pad - r
    cy     = pad + r
    RED    = (200, 30, 30)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED)
    # inner white dot
    ri = r // 3
    draw.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=(255, 255, 255))


images = []
for sp, html in zip(shot_paths, html_files):
    img = Image.open(sp).convert('RGB')

    # Fit into A4 canvas preserving aspect ratio (no upscale)
    scale_fit = min(A4_W_PX / img.width, A4_H_PX / img.height, 1.0)
    fit_w = int(img.width  * scale_fit)
    fit_h = int(img.height * scale_fit)
    if scale_fit < 1.0:
        img = img.resize((fit_w, fit_h), Image.LANCZOS)

    canvas = Image.new('RGB', (A4_W_PX, A4_H_PX), 'white')
    x_off  = (A4_W_PX - fit_w) // 2
    y_off  = (A4_H_PX - fit_h) // 2
    canvas.paste(img, (x_off, y_off))

    if html.stem in ALREADY_USED:
        stamp_used(canvas)

    images.append(canvas)

images[0].save(
    str(OUT_PDF),
    save_all=True,
    append_images=images[1:],
    resolution=RENDER_DPI,
)
print(f"Saved: {OUT_PDF}  ({len(images)} pages)")

shutil.rmtree(SHOT_DIR)
print("Temp screenshots removed.")
