"""
scripts/generate_preview.py

CI-only helper -- NOT part of the numbered pipeline (src/01 through
src/06). Regenerates the README's static assets/waterfall_preview.png by
opening the live outputs/waterfall_chart.html (which src/06_visualize.py
just built) in a real headless browser and screenshotting it, rather than
leaving that image as a one-time manual capture that silently drifts out
of sync with the data once the pipeline is automated.

Requires Playwright + a Chromium install, which is deliberately NOT in
requirements.txt -- ordinary local pipeline runs and the deployed
Streamlit app never need a browser, only the CI workflow that runs this
script does. See .github/workflows/refresh_pluto.yml.

Usage:
    pip install playwright
    playwright install --with-deps chromium
    python scripts/generate_preview.py

Input:
    outputs/waterfall_chart.html   (src/06_visualize.py output)

Output:
    assets/waterfall_preview.png
"""

import os
import sys

from playwright.sync_api import sync_playwright

SRC_PATH = "outputs/waterfall_chart.html"
OUT_PATH = "assets/waterfall_preview.png"
VIEWPORT = {"width": 1400, "height": 620}


def main():
    if not os.path.exists(SRC_PATH):
        sys.exit(f"\n[ERROR] Couldn't find {SRC_PATH}. Run src/06_visualize.py first.\n")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(f"file://{os.path.abspath(SRC_PATH)}")
        # Let Plotly finish its render/animation pass before capturing --
        # screenshotting immediately on page load can catch it mid-draw.
        page.wait_for_timeout(500)
        page.screenshot(path=OUT_PATH)
        browser.close()

    print(f"[generate_preview] Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
