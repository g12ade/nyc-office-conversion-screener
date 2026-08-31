"""
01_load_and_filter.py

Step 1 of the NYC Office-to-Residential Conversion Screener.

Loads NYC PLUTO (Primary Land Use Tax Lot Output), filters down to a
Manhattan office-building candidate universe, runs sanity checks on
data quality, and saves the result to data/processed/candidates.parquet.

Usage:
    python src/01_load_and_filter.py

Input:
    data/raw/pluto.csv   (download from NYC Open Data — see PROJECT_ROADMAP.md)

Output:
    data/processed/candidates.parquet
"""

import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG — adjust these to widen/narrow the candidate universe
# --------------------------------------------------------------------------
RAW_PATH = "data/raw/pluto.csv"
OUT_PATH = "data/processed/candidates.parquet"

BOROUGH = "MN"          # Manhattan. (Future work: extend to "BK"/"QN" for
                         # Downtown Brooklyn / Long Island City.)
MIN_FLOORS = 6
MIN_BLDG_SF = 50_000
OFFICE_AREA_SHARE = 0.5  # officearea must exceed this share of bldgarea
                          # to catch mixed-use towers that bldgclass alone
                          # would miss

# Candidate columns to pull from PLUTO. Not every version of PLUTO has
# every one of these — load_pluto() only requests the ones that actually
# exist in the file's header, so this list is safe to keep broad.
WANT = [
    "borough", "block", "lot", "bbl", "address", "zipcode", "ownername",
    "bldgclass", "landuse", "numbldgs", "numfloors", "yearbuilt",
    "yearalter1", "yearalter2", "lotarea", "bldgarea", "officearea",
    "retailarea", "comarea", "resarea", "bldgfront", "bldgdepth",
    "lotfront", "lotdepth", "unitsres", "unitstotal", "zonedist1",
    "builtfar", "residfar", "commfar", "assessland", "assesstot",
    "histdist", "landmark", "latitude", "longitude",
]


# --------------------------------------------------------------------------
# LOAD
# --------------------------------------------------------------------------
def load_pluto(path: str = RAW_PATH) -> pd.DataFrame:
    """Load PLUTO, requesting only the columns in WANT that are actually
    present in this version's header (PLUTO has renamed/dropped fields
    across releases, so we peek first rather than assuming)."""
    if not os.path.exists(path):
        sys.exit(
            f"\n[ERROR] Couldn't find {path}.\n"
            "Download PLUTO 26v1 CSV from NYC Open Data and save it there:\n"
            "https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks\n"
        )

    header = pd.read_csv(path, nrows=0)
    # Map lowercase name -> the column's REAL casing in this file (PLUTO
    # keeps some columns like "BBL" capitalized as an acronym). We need the
    # real casing to pass to usecols below — pandas matches usecols exactly,
    # it doesn't ignore case.
    lower_to_original = {c.lower(): c for c in header.columns}

    available_lower = [c for c in WANT if c in lower_to_original]
    missing = sorted(set(WANT) - set(available_lower))
    if missing:
        print(f"[load_pluto] Note: {len(missing)} requested columns not in "
              f"this PLUTO version, skipping: {missing}")

    usecols_original = [lower_to_original[c] for c in available_lower]
    df = pd.read_csv(path, usecols=usecols_original, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    print(f"[load_pluto] Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# --------------------------------------------------------------------------
# CLEAN
# --------------------------------------------------------------------------
def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to numeric, first stripping thousands-separator
    commas. NYC Open Data's CSV export bakes literal commas into large
    number fields (e.g., "125,000" instead of "125000") for columns
    configured with human-readable display formatting. pandas.to_numeric
    can't parse a string with a comma in it and silently turns the whole
    value into NaN — which is dangerous specifically because it doesn't
    error, it just quietly drops most of your data. Small values (under
    1,000) never get a comma, so this bug hides behind any column whose
    values happen to stay small (like numfloors) and only bites you on
    columns with genuinely large numbers (bldgarea, lotarea, assesstot).
    """
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix PLUTO's known missing-data encodings before any filtering."""
    df = df.copy()

    # PLUTO codes an unknown construction year as 0, NOT null. If left
    # unhandled, a "built before 1991" filter downstream will silently
    # treat these as pre-1991 and pass junk rows through. This is the
    # single most important gotcha in this script.
    for col in ["yearbuilt", "yearalter1", "yearalter2"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
            df.loc[df[col] <= 0, col] = np.nan

    # Zero-valued floors/areas/dimensions/assessed-values are also
    # missing-data codes, not real zeros — scrub them so Step 2's ratio
    # calculations don't divide by zero and produce infinities. This list
    # also covers assesstot/assessland and the FAR fields up front, even
    # though today's script doesn't filter on them yet, so Step 2 doesn't
    # walk into the same comma-parsing trap on those columns later.
    for col in ["numfloors", "bldgarea", "officearea", "lotarea",
                "bldgfront", "bldgdepth", "lotfront", "lotdepth",
                "assessland", "assesstot", "builtfar", "residfar", "commfar",
                "numbldgs"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
            df.loc[df[col] <= 0, col] = np.nan

    # Floor-area BREAKDOWN columns: resarea (residential), comarea
    # (commercial), retailarea. These legitimately CAN be zero -- an
    # all-office building genuinely has resarea == 0, that's real data,
    # not a missing-data code -- so they get comma-stripped/coerced to
    # numeric like everything else, but WITHOUT the <=0 -> NaN treatment
    # above (that would wrongly wipe out real zeros). Needed for Step 3's
    # RPTL 467-m non-residential-share eligibility check.
    for col in ["resarea", "comarea", "retailarea"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    # Normalize historic-district / landmark blanks to a clean boolean-ish
    # flag rather than leaving mixed NaN/empty-string representations.
    for col in ["histdist", "landmark"]:
        if col in df.columns:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    # Defensive uppercasing — PLUTO borough/building-class codes are
    # usually already upper, but don't trust it silently.
    if "borough" in df.columns:
        df["borough"] = df["borough"].astype(str).str.upper().str.strip()
    if "bldgclass" in df.columns:
        df["bldgclass"] = df["bldgclass"].astype(str).str.upper().str.strip()

    print(f"[clean] Cleaned {len(df):,} rows")
    return df


# --------------------------------------------------------------------------
# FILTER
# --------------------------------------------------------------------------
def build_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the filter waterfall that defines the candidate universe,
    printing a before -> after count at every cut."""
    print("\n" + "=" * 60)
    print("FILTER WATERFALL")
    print("=" * 60)

    n0 = len(df)
    print(f"{'Starting rows':<45}{n0:>10,}")

    df = df[df["borough"] == BOROUGH]
    print(f"{'After borough == ' + BOROUGH:<45}{len(df):>10,}")

    is_office_class = df["bldgclass"].str.startswith("O", na=False)
    office_share = df["officearea"] / df["bldgarea"]
    is_office_share = office_share > OFFICE_AREA_SHARE
    df = df[is_office_class | is_office_share]
    print(f"{'After office-class OR office-share filter':<45}{len(df):>10,}")

    df = df[df["numfloors"] >= MIN_FLOORS]
    print(f"{'After numfloors >= ' + str(MIN_FLOORS):<45}{len(df):>10,}")

    df = df[df["bldgarea"] >= MIN_BLDG_SF]
    print(f"{'After bldgarea >= ' + f'{MIN_BLDG_SF:,}':<45}{len(df):>10,}")

    footprint_cols = ["bldgfront", "bldgdepth", "lotfront", "lotdepth"]
    df = df.dropna(subset=[c for c in footprint_cols if c in df.columns])
    print(f"{'After valid footprint dimensions':<45}{len(df):>10,}")

    print("=" * 60)
    print(f"{'FINAL CANDIDATE UNIVERSE':<45}{len(df):>10,}")
    print("=" * 60 + "\n")

    if len(df) < 200:
        print("[build_universe] WARNING: candidate count looks very low "
              "(<200). Filters may be too tight — double check "
              "bldgclass/office-share logic.")
    elif len(df) > 40_000:
        print("[build_universe] WARNING: candidate count looks very high "
              "(>40,000). Filters may be too loose — you're probably "
              "catching irrelevant buildings.")

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# PROFILE / SANITY CHECK
# --------------------------------------------------------------------------
def profile(df: pd.DataFrame) -> None:
    """Print sanity-check output: numeric summary stats, null rates,
    construction-era breakdown, top owners, and a random spot-check
    sample to manually verify against Google."""
    print("\n" + "=" * 60)
    print("PROFILE / SANITY CHECKS")
    print("=" * 60)

    numeric_cols = [c for c in [
        "numfloors", "yearbuilt", "lotarea", "bldgarea", "officearea",
        "bldgfront", "bldgdepth", "builtfar", "residfar", "assesstot",
    ] if c in df.columns]

    print("\n-- describe() on key numeric columns --")
    print(df[numeric_cols].describe())

    print("\n-- null percentage by column --")
    null_pct = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    print(null_pct[null_pct > 0])

    if "yearbuilt" in df.columns:
        print("\n-- construction-era breakdown --")
        bins = [0, 1930, 1961, 1975, 1991, 2010, 3000]
        labels = ["pre-1930", "1930-60", "1961-74", "1975-90",
                  "1991-2009", "2010+"]
        era = pd.cut(df["yearbuilt"], bins=bins, labels=labels, right=False)
        print(era.value_counts().sort_index())
        print(
            "\n  NOTE: 1991+ buildings are likely excluded from City of Yes "
            "conversion eligibility (pre-1990 cutoff) — this needs to be "
            "verified against the exact DCP zoning text before it's used "
            "as a hard filter in Step 3."
        )

    if "ownername" in df.columns:
        print("\n-- top 10 owners by building count (sanity check) --")
        print(df["ownername"].value_counts().head(10))

    print("\n-- 5-row random spot-check sample --")
    sample_cols = [c for c in ["address", "zipcode", "bldgclass",
                                "numfloors", "yearbuilt", "bldgarea"]
                   if c in df.columns]
    sample = df[sample_cols].sample(min(5, len(df)), random_state=42)
    print(sample.to_string(index=False))
    print(
        "\n  ACTION: manually Google these 5 addresses and confirm they're "
        "actually office buildings before trusting the filter logic."
    )
    print("=" * 60 + "\n")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    df = load_pluto()
    df = clean(df)
    df = build_universe(df)
    profile(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[main] Saved {len(df):,} candidate rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
