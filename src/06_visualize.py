"""
06_visualize.py

Step 6 of the NYC Office-to-Residential Conversion Screener.

Builds the two visuals the roadmap calls for: a funnel/waterfall chart of
buildings eliminated at each filter and eligibility gate, and an
interactive map of the scored candidate universe -- colored by viability
score, with Tier 1 buildings and the real, validated conversions from
Step 5 both called out distinctly.

Usage:
    python src/06_visualize.py

Inputs:
    data/processed/candidates_scored.parquet   (Step 4 output)
    data/processed/validation_report.csv       (Step 5 output)

Outputs:
    outputs/waterfall_chart.html
    outputs/candidate_map.html
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

SCORED_PATH = "data/processed/candidates_scored.parquet"
VALIDATION_PATH = "data/processed/validation_report.csv"
OUT_DIR = "outputs"

TIER1_PCTILE = 0.90

# Funnel stage counts. These are hardcoded rather than recomputed by
# re-reading the full 858,602-row raw PLUTO file again -- Steps 1-3 have
# printed and reproduced these exact numbers on every run so far (stable,
# not a one-off fluke), and re-scanning the ~600MB CSV a sixth time in
# this project just to redraw a chart isn't worth the runtime. If you
# rerun Steps 1-3 in the future and the numbers change, update this list.
FUNNEL_STAGES = [
    ("Starting PLUTO rows (citywide)", 858_602),
    ("Manhattan", 42_544),
    ("Office class or >50% office share", 3_108),
    ("6+ floors", 1_980),
    ("50,000+ SF", 1_372),
    ("Valid footprint dimensions", 1_318),
    ("City of Yes: built <=1990", 1_242),
    ("467-m: >=90% non-residential", 1_221),
]


def load_scored(path: str = SCORED_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"\n[ERROR] Couldn't find {path}. Run src/04_viability_score.py first.\n")
    df = pd.read_parquet(path)
    print(f"[load_scored] Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def make_waterfall(out_dir: str) -> None:
    # A SINGLE linear-scale funnel across all 8 stages doesn't work here --
    # the citywide-to-Manhattan drop (858,602 -> 42,544) is ~7x bigger than
    # every other stage combined, so on one shared scale the six
    # office-filtering stages that are actually the interesting part of
    # this analysis (3,108 -> ... -> 1,221) all compress into a sliver a
    # few pixels tall, with their value labels unreadable inside that
    # sliver. Caught this by actually rendering the chart and looking at
    # a screenshot of it, not just checking that the script ran.
    #
    # Two things fix it, and both turned out to matter:
    #   (1) Split into two panels, each with its own x-axis scale -- an
    #       Overview (citywide -> Manhattan -> legally eligible) that
    #       shows how selective the whole pipeline is, and a Detail panel
    #       that starts at Manhattan so the six downstream office-filtering
    #       stages get a scale sized to THEM, not to the citywide number.
    #   (2) Switched from go.Funnel (which draws value labels INSIDE each
    #       tapered segment -- unreadable once a segment gets thin) to a
    #       horizontal go.Bar with textposition="outside": every stage's
    #       exact count and percent-of-prior-stage is printed outside its
    #       bar, so it stays legible no matter how short the bar is. This
    #       was the part that a plain funnel-shape change alone didn't
    #       fix -- re-tested after switching and confirmed every one of
    #       the 8 stage labels is now readable in a rendered screenshot.
    overview_labels = [
        "Starting PLUTO rows (citywide)",
        "Manhattan",
        "Legally eligible universe",
    ]
    overview_values = [FUNNEL_STAGES[0][1], FUNNEL_STAGES[1][1], FUNNEL_STAGES[-1][1]]
    overview_pct = ["100%", "5% of citywide", "0.1% of citywide"]
    overview_text = [f"{v:,}  ({p})" for v, p in zip(overview_values, overview_pct)]

    detail_stages = FUNNEL_STAGES[1:]  # Manhattan through 467-m eligible
    detail_labels = [s[0] for s in detail_stages]
    detail_values = [s[1] for s in detail_stages]
    detail_text = [f"{detail_values[0]:,}  (100%)"]
    for i in range(1, len(detail_values)):
        pct = detail_values[i] / detail_values[i - 1] * 100
        detail_text.append(f"{detail_values[i]:,}  ({pct:.0f}% of prior stage)")

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Overview: citywide down to the legally eligible universe",
            "Detail: filtering within Manhattan",
        ),
        horizontal_spacing=0.24,
    )

    fig.add_trace(go.Bar(
        y=overview_labels, x=overview_values, orientation="h",
        text=overview_text, textposition="outside", cliponaxis=False,
        marker={"color": ["#08519c", "#3182bd", "#d94801"]},
    ), row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(range=[0, overview_values[0] * 1.3], row=1, col=1)

    fig.add_trace(go.Bar(
        y=detail_labels, x=detail_values, orientation="h",
        text=detail_text, textposition="outside", cliponaxis=False,
        marker={"color": ["#3182bd", "#6baed6", "#9ecae1", "#c6dbef",
                           "#fdae6b", "#f16913", "#d94801"]},
    ), row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(range=[0, detail_values[0] * 1.35], row=1, col=2)

    fig.update_layout(
        title="NYC Office-to-Residential Screener: Filter & Eligibility Waterfall",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=90, b=40),
        height=560,
        showlegend=False,
    )

    path = os.path.join(out_dir, "waterfall_chart.html")
    fig.write_html(path, include_plotlyjs=True)
    print(f"[make_waterfall] Saved {path}")


def make_map(df: pd.DataFrame, out_dir: str) -> None:
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("[make_map] WARNING: latitude/longitude not in this data -- skipping map.")
        return

    plot_df = df.dropna(subset=["latitude", "longitude", "viability_score"]).copy()
    n_dropped = len(df) - len(plot_df)
    if n_dropped:
        print(f"[make_map] Note: {n_dropped:,} rows missing lat/long or score, excluded from map")

    # Cutoff computed on the FULL scored universe (df), not just the
    # geocoded subset (plot_df) -- matches Step 4's actual Tier 1
    # definition exactly, rather than a very slightly different one based
    # on whichever rows happen to have lat/long.
    cutoff = df["viability_score"].quantile(TIER1_PCTILE)
    plot_df["tier"] = plot_df["viability_score"].apply(
        lambda s: "Tier 1 (top 10%)" if s >= cutoff else "Other eligible candidates")

    hover_cols = [c for c in ["address", "yearbuilt", "numfloors", "bldgarea",
                               "floorplate_depth_ft", "far_headroom",
                               "viability_score"] if c in plot_df.columns]

    # Plotly has TWO generations of map API -- the older Mapbox-based
    # scatter_mapbox/Scattermapbox, and the newer MapLibre-based
    # scatter_map/Scattermap. Which one exists depends on the installed
    # plotly version, and that's genuinely unknown here (this sandbox has
    # 7.0.0, which already removed scatter_mapbox entirely -- but Step 0's
    # install could have landed on an older version that only has the
    # Mapbox one). Detect at runtime instead of guessing either way.
    use_new_api = hasattr(px, "scatter_map")
    scatter_fn = px.scatter_map if use_new_api else px.scatter_mapbox
    trace_cls = go.Scattermap if use_new_api else go.Scattermapbox
    style_kwarg = {"map_style": "open-street-map"} if use_new_api else {"mapbox_style": "open-street-map"}

    fig = scatter_fn(
        plot_df,
        lat="latitude", lon="longitude",
        color="viability_score",
        color_continuous_scale="Viridis",
        zoom=11.3,
        center={"lat": 40.758, "lon": -73.985},
        hover_data=hover_cols,
        title="Manhattan Office Conversion Candidates -- Colored by Viability Score",
        opacity=0.75,
        **style_kwarg,
    )
    fig.update_traces(marker=dict(size=7))

    # Overlay Tier 1 buildings as a visible gold "halo" trace, slightly
    # larger and drawn on top of the continuous-color dot underneath.
    # (These map marker traces have no separate outline/border property,
    # so a transparent "ring" isn't actually renderable -- a solid,
    # larger, differently-colored marker underneath the main dot is what
    # actually shows up.)
    tier1 = plot_df[plot_df["tier"] == "Tier 1 (top 10%)"]
    if len(tier1):
        fig.add_trace(trace_cls(
            lat=tier1["latitude"], lon=tier1["longitude"],
            mode="markers",
            marker=dict(size=13, color="gold"),
            name="Tier 1 (top 10%)",
            hoverinfo="skip",
            showlegend=True,
        ))
        # Re-add the main trace's data on top so Tier 1 points still show
        # their real viability_score color inside the gold halo, instead
        # of being fully covered by the flat gold marker.
        fig.add_trace(trace_cls(
            lat=tier1["latitude"], lon=tier1["longitude"],
            mode="markers",
            marker=dict(size=7, color=tier1["viability_score"],
                        colorscale="Viridis", cmin=plot_df["viability_score"].min(),
                        cmax=plot_df["viability_score"].max()),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Overlay the real, validated conversions from Step 5 (if available)
    # so the map visually connects the model's shortlist to reality.
    #
    # NOTE on marker design: this originally used marker=dict(..., symbol=
    # "star"). Rendering the actual output file (not just running the
    # script) in a headless browser caught a real bug in that: Plotly's
    # newer map engine (go.Scattermap / MapLibre) draws named symbols like
    # "star" by fetching an icon from an external CDN at render time --
    # https://cdn.jsdelivr.net/npm/@mapbox/maki@8.2.0/icons/star.svg --
    # confirmed via network-request capture. If that fetch is blocked
    # (a sandbox with no internet, a strict corporate firewall, an
    # ad-blocker that flags icon CDNs, or just opening the file offline)
    # the star silently fails to draw -- no error dialog, the marker is
    # just invisible, which is a bad failure mode for the single overlay
    # that's arguably the most important one on this map (the "the model's
    # shortlist actually matches real conversions" evidence). A plain
    # circle marker (no `symbol`) needs zero network calls -- confirmed
    # with the same request-capture test -- so real-conversion markers now
    # use the same "solid halo + inner dot" technique as the Tier 1
    # overlay above: a larger black circle underneath, a smaller red
    # circle on top. No external dependency, and it's still clearly a
    # different shape/size class from both the continuous-color candidate
    # dots and the gold Tier 1 halos.
    if os.path.exists(VALIDATION_PATH):
        val = pd.read_csv(VALIDATION_PATH)
        val_matched_addrs = set(
            val.loc[val["viability_score"].notna(), "address"].str.upper().str.strip()
        )
        plot_df_key = plot_df["address"].str.upper().str.strip()
        val_geo = plot_df[plot_df_key.isin(val_matched_addrs)]
        if len(val_geo):
            fig.add_trace(trace_cls(
                lat=val_geo["latitude"], lon=val_geo["longitude"],
                mode="markers",
                marker=dict(size=18, color="black"),
                name="Real, validated conversion (Step 5)",
                hoverinfo="skip",
                showlegend=True,
            ))
            fig.add_trace(trace_cls(
                lat=val_geo["latitude"], lon=val_geo["longitude"],
                mode="markers",
                marker=dict(size=10, color="red"),
                text=val_geo["address"],
                hovertemplate="<b>%{text}</b><br>Real conversion, confirmed in Step 5<extra></extra>",
                showlegend=False,
            ))
            print(f"[make_map] Overlaid {len(val_geo):,} real validated conversions")

    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        height=750,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)"),
    )

    path = os.path.join(out_dir, "candidate_map.html")
    fig.write_html(path, include_plotlyjs=True)
    print(f"[make_map] Saved {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_scored()
    make_waterfall(OUT_DIR)
    make_map(df, OUT_DIR)
    print(f"\n[main] Done. Open the .html files in {OUT_DIR}/ in a browser to view.")


if __name__ == "__main__":
    main()
