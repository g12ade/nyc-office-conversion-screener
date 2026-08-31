"""
04_viability_score.py

Step 4 of the NYC Office-to-Residential Conversion Screener.

Builds a weighted PHYSICAL/ECONOMIC viability score across the legally
eligible universe (Step 3's output), then runs an explicit sensitivity
test on both the score's weights and the floor-plate depth cutoff that
Steps 2-3 deliberately left provisional. This is the step where the
"only ~1.4% of eligible buildings look physically viable" headline number
either holds up under scrutiny or turns out to be an artifact of one
arbitrary threshold choice.

Usage:
    python src/04_viability_score.py

Input:
    data/processed/candidates_eligible.parquet   (output of Step 3)

Output:
    data/processed/candidates_scored.parquet
"""

import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
IN_PATH = "data/processed/candidates_eligible.parquet"
OUT_PATH = "data/processed/candidates_scored.parquet"

# Base weights for the composite viability score. These are a starting
# assumption, NOT a verified fact -- that's exactly why this script runs a
# sensitivity test on them below rather than just reporting one score and
# moving on.
#
#   depth : floor-plate depth is the thesis's central physical constraint
#           -- weighted heaviest.
#   far   : zoning bulk headroom -- a real but secondary structural factor.
#   value : assessed value per SF, lower = more "obsolete"/economically
#           primed to convert. DIRECTIONAL ASSUMPTION: a cheaper building
#           is easier to justify converting than a trophy asset. Arguable
#           either way -- included in the sensitivity test.
#   age   : building age, older = more "obsolete." Same caveat as value.
BASE_WEIGHTS = {
    "depth": 0.50,
    "far": 0.25,
    "value": 0.15,
    "age": 0.10,
}

TOP_TIER_PCTILE = 0.90  # top-decile buildings get called out as Tier 1

# The cutoffs Step 2 tested a single 65-ft threshold against. Re-running
# the eligible-universe count across this whole range shows how much the
# "1.4%" headline number actually depends on where that line gets drawn.
DEPTH_CUTOFFS_FT = [50, 55, 60, 65, 70, 75, 80, 90]

# How far to push the depth weight in the weight-sensitivity test, holding
# the other three weights' RELATIVE proportions constant.
DEPTH_WEIGHT_RANGE = [0.30, 0.40, 0.50, 0.60, 0.70]


# --------------------------------------------------------------------------
# LOAD
# --------------------------------------------------------------------------
def load_eligible(path: str = IN_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(
            f"\n[ERROR] Couldn't find {path}.\n"
            "Run src/03_eligibility_gates.py first.\n"
        )
    df = pd.read_parquet(path)
    print(f"[load_eligible] Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# --------------------------------------------------------------------------
# SCORING
# --------------------------------------------------------------------------
def _pct_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Percentile-rank a column to [0, 1], NaNs preserved as NaN. Percentile
    rank (rather than min-max scaling) is deliberately robust to the small
    number of extreme outliers already identified in Step 2 (e.g. the
    800-ft superblock lots) -- one wild value can't blow up the whole
    scale, it just occupies the top percentile slot."""
    r = series.rank(pct=True, na_option="keep")
    return r if higher_is_better else (1 - r)


def compute_component_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["score_depth"] = _pct_rank(df["floorplate_depth_ft"], higher_is_better=False)
    # Depth readings on extreme/superblock lots are known-unreliable (Step 2)
    # -- don't score on a number we've already said not to trust at face
    # value. These rows fall back to the other three components instead.
    if "extreme_lot_flag" in df.columns:
        df.loc[df["extreme_lot_flag"], "score_depth"] = np.nan

    df["score_far"] = _pct_rank(df["far_headroom"], higher_is_better=True)
    df["score_value"] = _pct_rank(df["assessed_val_per_sf"], higher_is_better=False)
    df["score_age"] = _pct_rank(df["building_age_yrs"], higher_is_better=True)

    print("[compute_component_scores] Computed percentile-rank scores for "
          "depth, FAR headroom, assessed value, and age")
    return df


def compute_weighted_score(df: pd.DataFrame, weights: dict, out_col: str = "viability_score") -> pd.Series:
    """Weighted average of the four component scores, RENORMALIZED per row
    over whichever components actually have data -- so a row missing one
    input (e.g. an extreme-lot row with no depth score) doesn't get
    unfairly zeroed out, it just leans on the remaining three."""
    score_cols = {
        "depth": "score_depth",
        "far": "score_far",
        "value": "score_value",
        "age": "score_age",
    }
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_total = pd.Series(0.0, index=df.index)

    for key, col in score_cols.items():
        w = weights[key]
        available = df[col].notna()
        weighted_sum += df[col].fillna(0) * w * available
        weight_total += w * available

    return (weighted_sum / weight_total.replace(0, np.nan)).rename(out_col)


# --------------------------------------------------------------------------
# PROFILE
# --------------------------------------------------------------------------
def profile_score(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("VIABILITY SCORE -- BASE WEIGHTS")
    print("=" * 60)
    print(f"Weights: {BASE_WEIGHTS}")

    print("\n-- describe() on viability_score --")
    print(df["viability_score"].describe())

    cutoff = df["viability_score"].quantile(TOP_TIER_PCTILE)
    n_tier1 = int((df["viability_score"] >= cutoff).sum())
    print(f"\n-- Tier 1 (top {(1 - TOP_TIER_PCTILE):.0%}, score >= {cutoff:.3f}) --")
    print(f"{n_tier1:,} of {len(df):,} buildings")

    print("\n-- Top 15 buildings by viability_score --")
    cols = [c for c in ["address", "yearbuilt", "numfloors", "bldgarea",
                         "floorplate_depth_ft", "far_headroom",
                         "assessed_val_per_sf", "viability_score"]
            if c in df.columns]
    print(df.sort_values("viability_score", ascending=False)[cols].head(15).to_string(index=False))
    print("\n  ACTION: spot-check a few of these addresses -- this is the "
          "shortlist Step 5's real-world validation will be checked against.")

    print("=" * 60 + "\n")


def sensitivity_test(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("SENSITIVITY TEST 1 -- DEPTH CUTOFF (the number Steps 2-3 deferred)")
    print("=" * 60)
    print("How many legally-eligible buildings would pass a HARD floor-plate\n"
          "depth cutoff, at each candidate threshold? (excludes extreme_lot_flag\n"
          "rows -- their depth reading isn't trustworthy at any cutoff)\n")

    reliable = df[~df.get("extreme_lot_flag", pd.Series(False, index=df.index))]
    n_reliable = len(reliable)
    for cutoff in DEPTH_CUTOFFS_FT:
        n_pass = int((reliable["floorplate_depth_ft"] <= cutoff).sum())
        pct = n_pass / n_reliable * 100 if n_reliable else 0.0
        print(f"  depth <= {cutoff:>3} ft: {n_pass:>4,} of {n_reliable:,} "
              f"({pct:5.1f}%) pass")
    print(
        "\n  TAKEAWAY: if this number swings by 2-3x between 55 ft and 80 ft,\n"
        "  the earlier '1.4%' headline is highly sensitive to an assumption\n"
        "  with no legal force behind it -- worth stating that plainly in\n"
        "  the README rather than presenting one number as settled fact."
    )

    print("\n" + "=" * 60)
    print("SENSITIVITY TEST 2 -- SCORE WEIGHT ON DEPTH")
    print("=" * 60)
    print("Re-running the composite score with the depth weight pushed up/down,\n"
          "rescaling the other three weights proportionally to still sum to 1.0.\n"
          "Reports how much the Tier-1 (top 10%) SET actually changes -- a\n"
          "robust finding shouldn't reshuffle wildly from a modest weight change.\n")

    base_score = compute_weighted_score(df, BASE_WEIGHTS)
    base_cutoff = base_score.quantile(TOP_TIER_PCTILE)
    base_tier1 = set(df.index[base_score >= base_cutoff])

    other_keys = [k for k in BASE_WEIGHTS if k != "depth"]
    other_total_base = sum(BASE_WEIGHTS[k] for k in other_keys)

    for depth_w in DEPTH_WEIGHT_RANGE:
        remaining = 1.0 - depth_w
        trial_weights = {"depth": depth_w}
        for k in other_keys:
            trial_weights[k] = remaining * (BASE_WEIGHTS[k] / other_total_base)

        trial_score = compute_weighted_score(df, trial_weights)
        trial_cutoff = trial_score.quantile(TOP_TIER_PCTILE)
        trial_tier1 = set(df.index[trial_score >= trial_cutoff])

        overlap = len(base_tier1 & trial_tier1)
        pct_overlap = overlap / len(base_tier1) * 100 if base_tier1 else 0.0
        marker = "  <- base" if abs(depth_w - BASE_WEIGHTS["depth"]) < 1e-9 else ""
        print(f"  depth weight = {depth_w:.2f}: Tier-1 overlap with base "
              f"weights = {overlap:,}/{len(base_tier1):,} ({pct_overlap:5.1f}%){marker}")

    print(
        "\n  TAKEAWAY: high overlap across the whole range means the Tier-1\n"
        "  shortlist is driven by buildings that are strong on multiple\n"
        "  dimensions at once, not an artifact of exactly how depth is\n"
        "  weighted. Low overlap would mean the ranking is fragile and\n"
        "  shouldn't be presented as a confident shortlist without more work."
    )
    print("=" * 60 + "\n")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    df = load_eligible()
    df = compute_component_scores(df)
    df["viability_score"] = compute_weighted_score(df, BASE_WEIGHTS)

    profile_score(df)
    sensitivity_test(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[main] Saved {len(df):,} scored rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
