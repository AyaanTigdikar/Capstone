"""
export_charts_png.py — Screenshot all charts as PNG files.
Run from: /Users/leoss/Desktop/GitHub/Capstone/FINAL CODE RECAP/
Output:   Final/charts_png/<name>.png
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#os.chdir(r'C:\Users\Usuario\Github\Capstone\FINAL CODE RECAP')

from pathlib import Path

from playwright.sync_api import sync_playwright

CHARTS_DIR = Path('Final/charts')
OUT_DIR    = Path('Final/charts_png')
OUT_DIR.mkdir(exist_ok=True)

print("Working dir:", os.getcwd())
print("CHARTS_DIR absolute:", CHARTS_DIR.resolve())
print("CHARTS_DIR exists:", CHARTS_DIR.exists())
print("Files in CHARTS_DIR:", list(CHARTS_DIR.glob('*.html'))[:5])


from pathlib import Path

VIEWPORT_W   = 1400
SCALE_FACTOR = 2          # 2× pixel density → ~192 DPI, crisp on retina/print

# Charts containing a geo map — need extra render time
MAP_STEMS = {
    '01_sample__54_resource_dependent_countries_map',
    '04_cluster__world_map_four_resource_profiles',
}

html_files = sorted(CHARTS_DIR.glob('*.html'))
print(f"Found {len(html_files)} charts  [device_scale_factor={SCALE_FACTOR}]\n")

with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx     = browser.new_context(
        viewport={'width': VIEWPORT_W, 'height': 800},
        device_scale_factor=SCALE_FACTOR,
    )
    page = ctx.new_page()

    for i, html in enumerate(html_files, 1):
        url    = html.resolve().as_uri()
        name   = html.stem
        out    = OUT_DIR / f'{name}.png'
        is_map = name in MAP_STEMS

        print(f"  [{i:02d}/{len(html_files)}] {name}{'  [map]' if is_map else ''}")
        page.goto(url, wait_until='networkidle', timeout=45_000)
        page.wait_for_timeout(3000 if is_map else 1000)

        chart_h = page.evaluate("""() => {
            const el = document.querySelector('.js-plotly-plot')
                     || document.querySelector('.plotly-graph-div');
            if (el) return Math.ceil(el.getBoundingClientRect().height);
            return document.body.scrollHeight;
        }""")
        snap_h = max(chart_h, 200)
        page.set_viewport_size({'width': VIEWPORT_W, 'height': snap_h})
        page.wait_for_timeout(300)

        page.screenshot(path=str(out), clip={
            'x': 0, 'y': 0, 'width': VIEWPORT_W, 'height': snap_h,
        })

    browser.close()

print(f"\nDone. PNGs saved to {OUT_DIR}/")
