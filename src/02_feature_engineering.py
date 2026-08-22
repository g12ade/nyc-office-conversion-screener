"""
02_feature_engineering.py

Step 2 of the NYC Office-to-Residential Conversion Screener.

Loads the Step 1 candidate universe (data/processed/candidates.parquet)
and engineers the features that drive Steps 3-4:
  - physical floor-plate geometry (the depth/window-access constraint)
  - FAR headroom under residential zoning
  - assessed value per square foot
  - building age relative to City of Yes's 1990 eligibility cutoff

Usage:
    python src/02_feature_engineering.py

Input:
    data/processed/candidates.parquet   (output of Step 1)

Output:
    data/processed/candidates_features.parquet
"""

import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
IN_PATH = "data/processed/candidates.parquet"
OUT_PATH = "data/processed/candidates_features.parquet"

CURRENT_YEAR = 2026
CITY_OF_YES_CUTOFF_YEAR = 1990  # City of Yes: legal eligibility = offices
                                 # built before 1990

# Rule-of-thumb max distance from a window to still count as usable,
# daylight-served space is roughly 30 ft; double-loaded-corridor layouts
# (units/rooms on both sides of a central hallway) put a core/hallway in
# the middle, so a building can go roughly twice that -- ~60-65 ft --
# before the interior stops being convertible to livable space. This is
# a widely-cited adaptive-reuse rule of thumb, NOT a legal or engineering
# hard number.
#
# IMPORTANT: this is kept as a PREVIEW threshold only, for the sanity-check
# cross-tab in profile_features() below. It is NOT used to drop or hard-gate
# any rows here -- Step 1's own size filter (50k+ SF, 6+ floors) already
# selects toward larger floor plates almost by construction (a genuinely
# shallow, ~65-ft-deep building can't hit 50k SF across only 6 floors
# without an unrealistic footprint), so nearly the entire candidate set
# reads as "deep" against a flat 65-ft cutoff. That's a real, worth-noting
# tension in the data, not a bug -- but it also means a hard cutoff here
# would leave nothing left to differentiate later. The real threshold
# (and how sensitive the results are to it) belongs in Step 4's weighted
# viability score + sensitivity test, per the roadmap -- not baked in as
# a binary flag this early.
DEEP_FLOORPLATE_THRESHOLD_FT = 65

# Typical NYC block depth from street to street (not avenue to avenue) is
# roughly 200-260 ft. Beyond that, a lot's front/depth bounding box is very
# likely spanning a full block or an assembled superblock (avenue-to-avenue
# runs can hit 800-920 ft) rather than measuring anything close to a real
# single-facade-to-core depth. Flag these separately so they don't get
# silently scored as if the proxy were reliable for them.
EXTREME_LOT_DEPTH_FT = 300


# --------------------------------------------------------------------------
# LOAD
# --------------------------------------------------------------------------
def load_candidates(path: str = IN_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(
            f"\n[ERROR] Couldn't find {path}.\n"
            "Run src/01_load_and_filter.py first to build the candidate "
            "universe.\n"
        )
    df = pd.read_parquet(path)
    print(f"[load_candidates] Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# --------------------------------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the Step 2 feature columns. All engineered columns are additive
    (nothing here filters rows out -- that's Step 3's job)."""
    df = df.copy()

    # ---- Physical floor-plate geometry -----------------------------------
    # PLUTO has no facade-to-core depth field (no public dataset does, at
    # scale) -- bldgdepth (the building footprint's depth, in feet) is the
    # closest available proxy, and the standard workaround in adaptive-reuse
    # feasibility screens. It's an approximation, not a survey measurement:
    # note that explicitly wherever this feature gets used downstream.
    df["floorplate_depth_ft"] = df["bldgdepth"]
    df["floorplate_aspect_ratio"] = df["bldgdepth"] / df["bldgfront"]
    df["avg_floor_plate_sf"] = df["bldgarea"] / df["numfloors"]

    # Preview-only flag -- see the long comment on DEEP_FLOORPLATE_THRESHOLD_FT
    # above. Not used to filter anything in this script.
    df["deep_floorplate_flag"] = df["floorplate_depth_ft"] > DEEP_FLOORPLATE_THRESHOLD_FT

    # Separately: lots whose depth is large enough that the front/depth
    # bounding box almost certainly isn't measuring a real single-building
    # facade-to-core depth (full-block or assembled superblock sites).
    # These need a different treatment later (manual review / a different
    # geometry proxy) rather than being scored at face value.
    df["extreme_lot_flag"] = df["floorplate_depth_ft"] > EXTREME_LOT_DEPTH_FT

    # ---- FAR headroom -----------------------------------------------------
    # residfar = max residential floor-area-ratio the zoning lot allows.
    # builtfar = FAR the existing building actually uses today.
    # Positive headroom -> the building's existing bulk already fits inside
    # what residential zoning permits, so nothing in the zoning code blocks
    # an as-of-right conversion on a bulk/FAR basis.
    # Negative headroom -> the building is built denser than resi zoning
    # would allow -> conversion is zoning-constrained (needs a variance,
    # bulk waiver, or a program like City of Yes to unlock it).
    df["far_headroom"] = df["residfar"] - df["builtfar"]

    # ---- Assessed value per SF --------------------------------------------
    # A rough proxy for how much the market/city already prices the asset
    # -- useful later as a covariate (e.g. very high-value trophy office
    # towers are less likely to convert regardless of physical feasibility,
    # because the office rent roll is still worth more than resi upside).
    df["assessed_val_per_sf"] = df["assesstot"] / df["bldgarea"]
    df["assessed_land_per_sf"] = df["assessland"] / df["bldgarea"]

    # ---- Building age vs. City of Yes cutoff -------------------------------
    df["building_age_yrs"] = CURRENT_YEAR - df["yearbuilt"]
    df["pre_1990_eligible"] = df["yearbuilt"] < CITY_OF_YES_CUTOFF_YEAR

    print("[engineer_features] Added floor-plate, FAR headroom, assessed "
          "value, and building-age features")
    return df


# --------------------------------------------------------------------------
# PROFILE / SANITY CHECK
# --------------------------------------------------------------------------
def profile_features(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("FEATURE PROFILE / SANITY CHECKS")
    print("=" * 60)

    feature_cols = [c for c in [
        "floorplate_depth_ft", "floorplate_aspect_ratio", "avg_floor_plate_sf",
        "far_headroom", "assessed_val_per_sf", "assessed_land_per_sf",
        "building_age_yrs",
    ] if c in df.columns]

    print("\n-- describe() on engineered features --")
    print(df[feature_cols].describe())

    print("\n-- null percentage on engineered features --")
    null_pct = (df[feature_cols].isna().mean() * 100).round(1).sort_values(ascending=False)
    nonzero_nulls = null_pct[null_pct > 0]
    print(nonzero_nulls if len(nonzero_nulls) else "No nulls in engineered columns.")

    if "extreme_lot_flag" in df.columns:
        n_extreme = int(df["extreme_lot_flag"].sum())
        n_valid = int(df["floorplate_depth_ft"].notna().sum())
        print(f"\n-- extreme lot flag (depth > {EXTREME_LOT_DEPTH_FT} ft -- proxy likely unreliable) --")
        print(f"{n_extreme:,} of {n_valid:,} buildings ({n_extreme / n_valid * 100:.1f}%) "
              f"have a depth reading that's almost certainly a full-block/assembled-lot "
              f"artifact, not a real facade-to-core measurement -- needs manual review, "
              f"don't trust the depth feature at face value for these")

    if "deep_floorplate_flag" in df.columns:
        n_deep = int(df["deep_floorplate_flag"].sum())
        n_valid = int(df["floorplate_depth_ft"].notna().sum())
        pct_deep = (n_deep / n_valid * 100) if n_valid else 0.0
        print(f"\n-- PREVIEW ONLY: deep floor-plate flag (> {DEEP_FLOORPLATE_THRESHOLD_FT} ft) --")
        print(f"{n_deep:,} of {n_valid:,} buildings with valid depth data "
              f"({pct_deep:.1f}%) sit above the {DEEP_FLOORPLATE_THRESHOLD_FT}-ft rule of thumb.")
        print("  NOTE: this rate reads very high largely because Step 1's own size filter\n"
              "  (50k+ SF, 6+ floors) already selects toward deep floor plates -- a building\n"
              "  that shallow can't hit 50k SF across only 6 floors without an unusually\n"
              "  large footprint. Not a bug, but a real reason not to hard-gate on this\n"
              "  threshold yet. The actual cutoff (and how sensitive results are to it)\n"
              "  gets decided in Step 4's weighted score + sensitivity test.")

    if "pre_1990_eligible" in df.columns:
        n_elig = int(df["pre_1990_eligible"].sum())
        n_valid = int(df["yearbuilt"].notna().sum())
        pct_elig = (n_elig / n_valid * 100) if n_valid else 0.0
        print("\n-- pre-1990 (City of Yes legally eligible) --")
        print(f"{n_elig:,} of {n_valid:,} buildings with known year built "
              f"({pct_elig:.1f}%) were built before 1990")

    if "far_headroom" in df.columns:
        n_valid = int(df["far_headroom"].notna().sum())
        n_pos = int((df["far_headroom"] > 0).sum())
        pct_pos = (n_pos / n_valid * 100) if n_valid else 0.0
        print("\n-- positive FAR headroom (residfar > builtfar) --")
        print(f"{n_pos:,} of {n_valid:,} buildings with valid FAR data "
              f"({pct_pos:.1f}%) have zoning headroom to convert")

    # The headline cross-tab the whole project is built around: legally
    # eligible (pre-1990) vs. physically feasible (not flagged deep). This
    # is a preview only, using the provisional 65-ft threshold above --
    # Step 3 formalizes the real LEGAL eligibility gates (467-m + City of
    # Yes rules), and Step 4 formalizes the PHYSICAL side as a weighted,
    # sensitivity-tested score rather than a single hard cutoff.
    if "pre_1990_eligible" in df.columns and "deep_floorplate_flag" in df.columns:
        valid = df[df["yearbuilt"].notna() & df["floorplate_depth_ft"].notna()]
        eligible = valid[valid["pre_1990_eligible"]]
        if len(eligible):
            clears_both = int((~eligible["deep_floorplate_flag"]).sum())
            pct_clears = clears_both / len(eligible) * 100
            print("\n-- preview: legally eligible AND clears floor-plate depth test --")
            print(f"{clears_both:,} of {len(eligible):,} legally-eligible buildings "
                  f"({pct_clears:.1f}%) also clear the "
                  f"{DEEP_FLOORPLATE_THRESHOLD_FT}-ft depth threshold")
            print("  Directionally this is the number the project thesis is built on, but\n"
                  "  treat it as provisional -- it moves a lot depending on where the depth\n"
                  "  cutoff is set. Step 4 formalizes this properly with a real sensitivity\n"
                  "  test across threshold values instead of one fixed number.")

    print("=" * 60 + "\n")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    df = load_candidates()
    df = engineer_features(df)
    profile_features(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[main] Saved {len(df):,} rows with engineered features to {OUT_PATH}")


if __name__ == "__main__":
    main()
