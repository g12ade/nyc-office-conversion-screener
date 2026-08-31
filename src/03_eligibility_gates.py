"""
03_eligibility_gates.py

Step 3 of the NYC Office-to-Residential Conversion Screener.

Encodes the LEGAL eligibility rules for City of Yes for Housing Opportunity
and RPTL 467-m as explicit, sourced gates -- turning statute into code.

Physical/geometric feasibility (Step 2's floor-plate depth features) is
deliberately NOT gated here. Legal eligibility is a real yes/no under the
statute; physical feasibility is an interpretive judgment call (what depth
is "too deep"?) that belongs in Step 4's weighted, sensitivity-tested
score, not a hard cutoff this early.

Usage:
    python src/03_eligibility_gates.py

Input:
    data/processed/candidates_features.parquet   (output of Step 2)

Output:
    data/processed/candidates_eligible.parquet

--------------------------------------------------------------------------
SOURCES (verified against official NYC.gov/DCP/HPD documentation, Aug 2026)
--------------------------------------------------------------------------

City of Yes for Housing Opportunity (adopted Dec 2024):
  "allows any office building constructed prior to Dec. 31, 1990, to be
  converted to residences" -- operationalized as yearbuilt <= 1990 (see
  Step 2). Buildings in Special Mixed Use Districts keep a later cutoff
  (12/10/1997) -- NOT modeled here; PLUTO's zonedist1 doesn't cleanly
  identify SMU overlays in this dataset. Known gap, flagged below.
  Source: https://www.nyc.gov/assets/planning/downloads/pdf/our-work/plans/citywide/city-of-yes-housing-opportunity/housing-opportunity-guide-conversions.pdf

  Zoning-district-specific eligibility ("expands eligibility to anywhere
  residential uses are allowed") is NOT independently gated here -- I
  could not verify a definitive, current list of which commercial (C1-C8)
  and manufacturing (M1/M2/M3) districts are included/excluded from
  primary sources in the time available. Rather than guess at a legal
  rule, this is left as an explicit open question (see KNOWN GAPS in the
  profile output) for manual confirmation against the DCP zoning text
  before any final numbers are presented as authoritative.

RPTL 467-m (enacted April 2024):
  - Non-residential status: building must have had a certificate of
    occupancy for commercial, manufacturing, or other non-residential use
    for >= 90% of aggregate floor area. Operationalized as
    (bldgarea - resarea) / bldgarea >= 0.90, using PLUTO's resarea
    (existing residential floor area) as the residential-use proxy.
  - Minimum 6 dwelling units in the converted building -- NOT gated here.
    This describes the POST-conversion building, not something PLUTO's
    pre-conversion data can measure. Every candidate here already clears
    Step 1's 50,000+ SF minimum, so 6 units is essentially a non-binding
    floor for this universe -- documented, not computed.
  - Commencement window (after 12/31/2022, on/before 6/30/2031),
    completion window (by 12/31/2039), 25%-affordable-at-80%-AMI
    requirement, and the Manhattan Prime Development Area exemption-tier
    boundary are all PROJECT- or DEAL-level terms, not properties of the
    building itself -- not something PLUTO can gate on. Documented for
    the README, not computed here.
  Source: https://www.nyc.gov/assets/hpd/downloads/pdfs/services/467m-requirements-faq.pdf
"""

import os
import sys

import numpy as np
import pandas as pd

IN_PATH = "data/processed/candidates_features.parquet"
OUT_PATH = "data/processed/candidates_eligible.parquet"

MIN_NONRESIDENTIAL_SHARE = 0.90


# --------------------------------------------------------------------------
# LOAD
# --------------------------------------------------------------------------
def load_features(path: str = IN_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(
            f"\n[ERROR] Couldn't find {path}.\n"
            "Run src/02_feature_engineering.py first.\n"
        )
    df = pd.read_parquet(path)
    if "pre_1990_eligible" not in df.columns:
        sys.exit(
            "[ERROR] pre_1990_eligible column missing -- re-run "
            "src/02_feature_engineering.py with the current version of the "
            "script (it was updated alongside this one) before continuing."
        )
    print(f"[load_features] Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# --------------------------------------------------------------------------
# ELIGIBILITY GATES
# --------------------------------------------------------------------------
def build_eligible_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the legal eligibility gates as an actual filter waterfall,
    same style as Step 1 -- these are real yes/no statutory tests, not
    judgment calls, so unlike Step 2's physical features they're
    appropriate to hard-gate on."""
    df = df.copy()

    # ---- 467-m: >= 90% non-residential floor area -------------------------
    if "resarea" in df.columns:
        nonres_area = df["bldgarea"] - df["resarea"].fillna(0)
        df["nonresidential_share"] = nonres_area / df["bldgarea"]
        df["rptl467m_nonres_eligible"] = df["nonresidential_share"] >= MIN_NONRESIDENTIAL_SHARE
    else:
        print("[build_eligible_universe] WARNING: 'resarea' not in this "
              "PLUTO version -- skipping the 467-m non-residential-share "
              "gate, treating all rows as passing it by default.")
        df["nonresidential_share"] = np.nan
        df["rptl467m_nonres_eligible"] = True

    print("\n" + "=" * 60)
    print("ELIGIBILITY GATE WATERFALL")
    print("=" * 60)

    n0 = len(df)
    print(f"{'Starting rows (Step 2 output)':<50}{n0:>10,}")

    df = df[df["pre_1990_eligible"]]
    print(f"{'After City of Yes: built <= 1990':<50}{len(df):>10,}")

    df = df[df["rptl467m_nonres_eligible"]]
    print(f"{'After 467-m: >= ' + f'{MIN_NONRESIDENTIAL_SHARE:.0%}' + ' non-residential':<50}{len(df):>10,}")

    print("=" * 60)
    print(f"{'LEGALLY ELIGIBLE UNIVERSE':<50}{len(df):>10,}")
    print("=" * 60 + "\n")

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# PROFILE / SANITY CHECK
# --------------------------------------------------------------------------
def profile_eligible(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("LEGALLY ELIGIBLE UNIVERSE -- PHYSICAL FEASIBILITY PREVIEW")
    print("=" * 60)

    if "extreme_lot_flag" in df.columns:
        n_extreme = int(df["extreme_lot_flag"].sum())
        print(f"\n{n_extreme:,} of {len(df):,} legally-eligible buildings are "
              f"flagged extreme_lot_flag (depth proxy likely unreliable) -- "
              f"exclude these from any physical-feasibility read until "
              f"manually reviewed.")

    if "deep_floorplate_flag" in df.columns:
        reliable = df[~df.get("extreme_lot_flag", False)]
        n_clears = int((~reliable["deep_floorplate_flag"]).sum())
        pct = (n_clears / len(reliable) * 100) if len(reliable) else 0.0
        print(f"\nOf the {len(reliable):,} legally-eligible buildings with a "
              f"trustworthy depth reading, {n_clears:,} ({pct:.1f}%) also "
              f"clear the preview 65-ft floor-plate threshold from Step 2.")
        print("  Still provisional -- Step 4 formalizes the physical side "
              "with a real sensitivity test instead of one fixed cutoff.")

    print(
        "\nKNOWN GAPS -- not gated in this script, documented instead:\n"
        "  - Zoning-district-specific eligibility (which C/M districts\n"
        "    qualify) -- could not verify a definitive current list from\n"
        "    primary sources; needs manual DCP zoning-text confirmation\n"
        "    before these numbers are presented as final.\n"
        "  - Special Mixed Use District 1997 cutoff exception (vs. the\n"
        "    standard 1990 cutoff) -- not identifiable from this dataset.\n"
        "  - RPTL 467-m's 6-unit minimum, commencement/completion window,\n"
        "    and 25%-affordable/80%-AMI requirement -- these are post-\n"
        "    conversion, project-level, or deal-level terms, not\n"
        "    pre-conversion building characteristics PLUTO can gate on."
    )
    print("=" * 60 + "\n")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    df = load_features()
    df = build_eligible_universe(df)
    profile_eligible(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[main] Saved {len(df):,} legally-eligible rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
