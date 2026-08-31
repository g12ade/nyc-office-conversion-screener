"""
app.py

Interactive Streamlit deployment of the NYC Office-to-Residential Conversion
Screener. Loads the pipeline's final outputs (data/processed/candidates_scored
.parquet, data/processed/validation_report.csv -- both committed to the repo
since Streamlit Cloud can't regenerate them from the 450MB raw PLUTO file at
deploy time) and lets a visitor interactively explore the two assumptions
Step 4 flagged as provisional: the viability score's component weights, and
the floor-plate depth cutoff used for the hard-filter sensitivity test.

Run locally:
    streamlit run app.py

Deployed via Streamlit Community Cloud, pointed at this repo.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

SCORED_PATH = "data/processed/candidates_scored.parquet"
VALIDATION_PATH = "data/processed/validation_report.csv"

BASE_WEIGHTS = {"depth": 0.50, "far": 0.25, "value": 0.15, "age": 0.10}
DEPTH_CUTOFFS_FT = [50, 55, 60, 65, 70, 75, 80, 90]

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

st.set_page_config(
    page_title="NYC Office-to-Residential Conversion Screener",
    page_icon="\U0001F3D9️",
    layout="wide",
)


# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(SCORED_PATH):
        st.error(
            f"Couldn't find {SCORED_PATH}. This app expects the repo's "
            "committed data/processed/candidates_scored.parquet -- see the "
            "README's 'Getting the data' section."
        )
        st.stop()
    df = pd.read_parquet(SCORED_PATH)

    val = None
    if os.path.exists(VALIDATION_PATH):
        val = pd.read_csv(VALIDATION_PATH)
    return df, val


# --------------------------------------------------------------------------
# SCORING (mirrors src/04_viability_score.py exactly, reweighted live)
# --------------------------------------------------------------------------
def compute_weighted_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted average of the four PRECOMPUTED percentile-rank component
    scores (score_depth/far/value/age -- already NaN'd for extreme_lot_flag
    rows by Step 4), renormalized per row over whichever components have
    data. Weights don't need to sum to 1 -- this ratio does that
    automatically, same as the pipeline script."""
    score_cols = {
        "depth": "score_depth", "far": "score_far",
        "value": "score_value", "age": "score_age",
    }
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_total = pd.Series(0.0, index=df.index)
    for key, col in score_cols.items():
        w = weights[key]
        available = df[col].notna()
        weighted_sum += df[col].fillna(0) * w * available
        weight_total += w * available
    return weighted_sum / weight_total.replace(0, np.nan)


def depth_cutoff_sensitivity(df: pd.DataFrame, cutoffs: list) -> pd.DataFrame:
    reliable = df[~df.get("extreme_lot_flag", pd.Series(False, index=df.index))]
    n_reliable = len(reliable)
    rows = []
    for c in cutoffs:
        n_pass = int((reliable["floorplate_depth_ft"] <= c).sum())
        pct = n_pass / n_reliable * 100 if n_reliable else 0.0
        rows.append({"cutoff_ft": c, "n_pass": n_pass, "pct_pass": pct})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# CHARTS
# --------------------------------------------------------------------------
def waterfall_figure():
    overview_labels = ["Starting PLUTO rows (citywide)", "Manhattan", "Legally eligible universe"]
    overview_values = [FUNNEL_STAGES[0][1], FUNNEL_STAGES[1][1], FUNNEL_STAGES[-1][1]]
    overview_text = [f"{overview_values[0]:,}  (100%)",
                      f"{overview_values[1]:,}  (5% of citywide)",
                      f"{overview_values[2]:,}  (0.1% of citywide)"]

    detail_stages = FUNNEL_STAGES[1:]
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
        subplot_titles=("Overview: citywide down to the legally eligible universe",
                         "Detail: filtering within Manhattan"),
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

    fig.update_layout(margin=dict(l=20, r=20, t=60, b=40), height=480, showlegend=False)
    return fig


def sensitivity_figure(sens_df: pd.DataFrame, chosen_cutoff: int):
    colors = ["#d94801" if c == chosen_cutoff else "#6baed6" for c in sens_df["cutoff_ft"]]
    fig = go.Figure(go.Bar(
        x=sens_df["cutoff_ft"], y=sens_df["pct_pass"],
        text=[f"{p:.1f}%" for p in sens_df["pct_pass"]],
        textposition="outside",
        marker_color=colors,
    ))
    fig.update_layout(
        xaxis_title="Hard depth cutoff (ft)",
        yaxis_title="% of reliable-depth buildings passing",
        margin=dict(l=40, r=20, t=20, b=40), height=340,
    )
    return fig


def map_figure(plot_df: pd.DataFrame, val: pd.DataFrame, tier1_pctile: float):
    cutoff = plot_df["viability_score"].quantile(tier1_pctile)
    plot_df = plot_df.copy()
    plot_df["tier"] = np.where(plot_df["viability_score"] >= cutoff,
                                "Tier 1", "Other eligible candidates")

    hover_cols = [c for c in ["address", "yearbuilt", "numfloors", "bldgarea",
                               "floorplate_depth_ft", "far_headroom",
                               "viability_score"] if c in plot_df.columns]

    use_new_api = hasattr(px, "scatter_map")
    scatter_fn = px.scatter_map if use_new_api else px.scatter_mapbox
    trace_cls = go.Scattermap if use_new_api else go.Scattermapbox
    style_kwarg = {"map_style": "open-street-map"} if use_new_api else {"mapbox_style": "open-street-map"}

    fig = scatter_fn(
        plot_df, lat="latitude", lon="longitude",
        color="viability_score", color_continuous_scale="Viridis",
        zoom=11.3, center={"lat": 40.758, "lon": -73.985},
        hover_data=hover_cols, opacity=0.75, **style_kwarg,
    )
    fig.update_traces(marker=dict(size=7))

    tier1 = plot_df[plot_df["tier"] == "Tier 1"]
    if len(tier1):
        fig.add_trace(trace_cls(
            lat=tier1["latitude"], lon=tier1["longitude"], mode="markers",
            marker=dict(size=13, color="gold"), name=f"Tier 1 (top {(1 - tier1_pctile):.0%})",
            hoverinfo="skip", showlegend=True,
        ))
        fig.add_trace(trace_cls(
            lat=tier1["latitude"], lon=tier1["longitude"], mode="markers",
            marker=dict(size=7, color=tier1["viability_score"], colorscale="Viridis",
                        cmin=plot_df["viability_score"].min(), cmax=plot_df["viability_score"].max()),
            showlegend=False, hoverinfo="skip",
        ))

    # Real-conversion overlay uses plain circles, not a named "symbol" marker
    # -- Plotly's map engine fetches named-symbol icons from an external CDN
    # at render time, which silently fails to draw (and, even when it does
    # load, doesn't reliably apply the configured color) if that fetch is
    # blocked. Caught and fixed during Step 6; kept consistent here.
    if val is not None and "address" in plot_df.columns:
        matched = set(val.loc[val["viability_score"].notna(), "address"].str.upper().str.strip())
        key = plot_df["address"].str.upper().str.strip()
        val_geo = plot_df[key.isin(matched)]
        if len(val_geo):
            fig.add_trace(trace_cls(
                lat=val_geo["latitude"], lon=val_geo["longitude"], mode="markers",
                marker=dict(size=18, color="black"), name="Real, validated conversion",
                hoverinfo="skip", showlegend=True,
            ))
            fig.add_trace(trace_cls(
                lat=val_geo["latitude"], lon=val_geo["longitude"], mode="markers",
                marker=dict(size=10, color="red"), text=val_geo["address"],
                hovertemplate="<b>%{text}</b><br>Real conversion, confirmed in Step 5<extra></extra>",
                showlegend=False,
            ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=650,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


# --------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------
def main():
    df, val = load_data()

    st.title("NYC Office-to-Residential Conversion Screener")
    st.markdown(
        "Screening Manhattan office buildings for conversion viability under "
        "**RPTL 467-m** and **City of Yes for Housing Opportunity**. "
        "Policy solved the *legal* barrier to conversion -- the real "
        "remaining constraint is physical building geometry, and how much "
        "weight you put on it changes the answer. Adjust the sliders below "
        "to see how sensitive the shortlist actually is."
    )

    st.sidebar.header("Viability score weights")
    st.sidebar.caption("Don't need to sum to 1 -- renormalized automatically, same as the pipeline script.")
    w_depth = st.sidebar.slider("Floor-plate depth", 0.0, 1.0, BASE_WEIGHTS["depth"], 0.05)
    w_far = st.sidebar.slider("FAR headroom", 0.0, 1.0, BASE_WEIGHTS["far"], 0.05)
    w_value = st.sidebar.slider("Assessed value / SF", 0.0, 1.0, BASE_WEIGHTS["value"], 0.05)
    w_age = st.sidebar.slider("Building age", 0.0, 1.0, BASE_WEIGHTS["age"], 0.05)
    weights = {"depth": w_depth, "far": w_far, "value": w_value, "age": w_age}

    st.sidebar.header("Physical feasibility")
    depth_cutoff = st.sidebar.select_slider(
        "Hard floor-plate depth cutoff (ft) -- for the sensitivity chart only, "
        "does not affect the map/score",
        options=DEPTH_CUTOFFS_FT, value=65,
    )

    st.sidebar.header("Shortlist threshold")
    tier1_pctile = st.sidebar.slider("Tier 1 percentile cutoff", 0.75, 0.99, 0.90, 0.01)

    df = df.copy()
    df["viability_score"] = compute_weighted_score(df, weights)

    tier1_cutoff = df["viability_score"].quantile(tier1_pctile)
    tier1_mask = df["viability_score"] >= tier1_cutoff
    n_tier1 = int(tier1_mask.sum())

    # Tier 1 COUNT is always ~top-N% by construction (a quantile cutoff),
    # so it barely moves as weights change -- not a useful thing to watch.
    # What's actually informative, and what Step 4's real sensitivity test
    # reported, is how much the Tier 1 SET itself reshuffles: compare
    # against the base-weight Tier 1 set and report the overlap.
    base_score = compute_weighted_score(df, BASE_WEIGHTS)
    base_tier1_mask = base_score >= base_score.quantile(tier1_pctile)
    overlap = int((tier1_mask & base_tier1_mask).sum())
    base_n = int(base_tier1_mask.sum())
    overlap_pct = overlap / base_n * 100 if base_n else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Legally eligible buildings", f"{len(df):,}")
    col2.metric(f"Tier 1 (top {(1 - tier1_pctile):.0%})", f"{n_tier1:,}")
    col3.metric(
        "Overlap vs. base weights", f"{overlap_pct:.0f}%",
        help="Share of the CURRENT Tier 1 shortlist that would also be Tier 1 under the pipeline's original base weights (depth 0.50 / FAR 0.25 / value 0.15 / age 0.10). This is what actually moves as you adjust the sliders -- Tier 1's SIZE is ~fixed by definition (a top-decile cutoff), but WHICH buildings are in it can shift.",
    )
    n_extreme = int(df.get("extreme_lot_flag", pd.Series(False, index=df.index)).sum())
    col4.metric("Excluded from depth score", f"{n_extreme:,}", help="Superblock/full-block lots where PLUTO's bounding-box depth measurement is unreliable (Step 2).")

    tab_map, tab_waterfall, tab_sensitivity, tab_table = st.tabs(
        ["Candidate map", "Filter waterfall", "Depth-cutoff sensitivity", "Top candidates"]
    )

    with tab_map:
        plot_df = df.dropna(subset=["latitude", "longitude", "viability_score"])
        n_dropped = len(df) - len(plot_df)
        if n_dropped:
            st.caption(f"{n_dropped:,} rows missing lat/long or score, excluded from map.")
        st.plotly_chart(map_figure(plot_df, val, tier1_pctile), use_container_width=True)

    with tab_waterfall:
        st.plotly_chart(waterfall_figure(), use_container_width=True)
        st.caption(
            "Fixed pipeline stage counts (Steps 1-3) -- not affected by the "
            "sliders above, which only apply to physical viability scoring "
            "within the legally eligible universe."
        )

    with tab_sensitivity:
        sens_df = depth_cutoff_sensitivity(df, DEPTH_CUTOFFS_FT)
        st.plotly_chart(sensitivity_figure(sens_df, depth_cutoff), use_container_width=True)
        st.caption(
            "Share of legally-eligible buildings (excluding unreliable-depth "
            "superblock lots) that would pass a HARD floor-plate depth "
            "cutoff, at each candidate threshold. This swings roughly 40x "
            "across a defensible range (50-90 ft) -- there's no legal basis "
            "for exactly where this line sits."
        )

    with tab_table:
        cols = [c for c in ["address", "yearbuilt", "numfloors", "bldgarea",
                             "floorplate_depth_ft", "far_headroom",
                             "assessed_val_per_sf", "viability_score"] if c in df.columns]
        top_n = st.slider("Show top N by viability score", 5, 100, 20, 5)
        st.dataframe(
            df.sort_values("viability_score", ascending=False)[cols].head(top_n),
            use_container_width=True, hide_index=True,
        )

    if val is not None:
        n_matched = int(val["viability_score"].notna().sum())
        st.markdown(
            f"**Validated against reality:** of {len(val)} real, documented NYC "
            f"office conversions, {n_matched} matched a building in this "
            "pipeline's candidate universe and are overlaid on the map above "
            "as black-ringed red markers."
        )

    st.markdown("---")
    st.caption(
        "Known limitations: zoning-district-specific eligibility and the "
        "Special Mixed Use District exception aren't gated (documented, not "
        "computed); floor-plate depth is a PLUTO bounding-box proxy, not a "
        "measured value; the validation sample is 13 buildings with "
        "exact-string address matching. Full write-up in the repo README."
    )


if __name__ == "__main__":
    main()
