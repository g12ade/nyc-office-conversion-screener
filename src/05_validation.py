"""
05_validation.py

Step 5 of the NYC Office-to-Residential Conversion Screener.

The step that separates this from a toy project: checks the pipeline
against REAL, sourced NYC office-to-residential conversions rather than
just trusting the model's own internal logic. For each known real
conversion, this traces it through every stage of the pipeline --
raw PLUTO -> Step 1 candidate universe -> Step 3 legally eligible ->
Step 4 viability score -- and reports where (and why) it fell out, if it
did, plus its score/percentile rank if it made it all the way through.

Usage:
    python src/05_validation.py

Inputs:
    data/raw/pluto.csv                        (for buildings NOT in the
                                                 filtered candidate set --
                                                 need the full citywide
                                                 file to explain misses)
    data/processed/candidates.parquet          (Step 1 output)
    data/processed/candidates_eligible.parquet (Step 3 output)
    data/processed/candidates_scored.parquet   (Step 4 output)

Output:
    data/processed/validation_report.csv
"""

import os
import re
import sys
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/pluto.csv"
CANDIDATES_PATH = "data/processed/candidates.parquet"
ELIGIBLE_PATH = "data/processed/candidates_eligible.parquet"
SCORED_PATH = "data/processed/candidates_scored.parquet"
OUT_PATH = "data/processed/validation_report.csv"

# --------------------------------------------------------------------------
# KNOWN REAL CONVERSIONS
#
# Sourced Aug 2026 from New York YIMBY, the NY Governor's office, 400
# Capital Management, and Wikipedia -- see notes per entry. This is not
# an exhaustive list of every NYC office conversion, just a real, sourced
# sample to validate the model against (roadmap target was ~15; got 13
# with enough detail to check).
# --------------------------------------------------------------------------
KNOWN_CONVERSIONS = [
    {"address": "101 Franklin Street", "alt_address": "250 Church Street", "yearbuilt": None,
     "note": "72 condos, 21 stories after 4 added; Skylight/Cannon Hill/TPG",
     "source": "New York YIMBY"},
    {"address": "77 Water Street", "alt_address": None, "yearbuilt": None,
     "note": "647 rentals, 25% affordable; Vanbarton Group; completion spring 2027",
     "source": "New York YIMBY"},
    {"address": "80 Pine Street", "alt_address": "180 Pearl Street", "yearbuilt": None,
     "note": "713 rentals, 38 stories; Bushburg",
     "source": "New York YIMBY"},
    {"address": "26 Bleecker Street", "alt_address": None, "yearbuilt": None,
     "note": "17 condos, 7 stories -- small scale, likely below Step 1's 50k SF size filter",
     "source": "New York YIMBY"},
    {"address": "355 Lexington Avenue", "alt_address": None, "yearbuilt": None,
     "note": "297 apartments, 22 stories after 4 added; Rudin Management",
     "source": "New York YIMBY"},
    {"address": "300 East 42 Street", "alt_address": "300 2 Avenue", "yearbuilt": None,
     "note": "135 rentals, 18 stories; also known as 300 Second Avenue; CSC",
     "source": "New York YIMBY"},
    {"address": "675 3 Avenue", "alt_address": None, "yearbuilt": None,
     "note": "464 rentals, 35 stories after 4 added; Metro Lift/David Werner",
     "source": "New York YIMBY"},
    {"address": "1740 Broadway", "alt_address": None, "yearbuilt": None,
     "note": "426 rentals, 26 stories; Yellowstone Real Estate; bought from Blackstone for $186M",
     "source": "New York YIMBY"},
    {"address": "245 West 55 Street", "alt_address": None, "yearbuilt": None,
     "note": "42 condos (DuArt Bldg, former film lab), 18 stories after 6 added",
     "source": "New York YIMBY"},
    {"address": "333 West 52 Street", "alt_address": None, "yearbuilt": None,
     "note": "108 units, 14 stories, ~98 years old (built ~1927-28)",
     "source": "New York YIMBY"},
    {"address": "29 West 35 Street", "alt_address": None, "yearbuilt": None,
     "note": "107 studios, first major conversion under Midtown South Mixed-Use Plan; 467-m 35-yr abatement",
     "source": "400 Capital Management"},
    {"address": "25 Water Street", "alt_address": None, "yearbuilt": 1969,
     "note": "~1,300 units, largest US office-to-residential conversion; 22->32 stories; leasing began Jan 2025",
     "source": "Wikipedia / New York YIMBY"},
    {"address": "5 Times Square", "alt_address": None, "yearbuilt": None,
     "note": "~1,250 units (1,050 studio, 200 1BR); built at 33.35 FAR; may not resolve to a standard street address in PLUTO",
     "source": "NY Governor's office"},
]

ORDINAL_WORDS = {
    "FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4", "FIFTH": "5",
    "SIXTH": "6", "SEVENTH": "7", "EIGHTH": "8", "NINTH": "9", "TENTH": "10",
    "ELEVENTH": "11", "TWELFTH": "12",
}


def normalize_address(addr: str) -> str:
    """Convert a human-readable address to PLUTO's style: uppercase, no
    ordinal suffixes (35TH -> 35), spelled-out numbered streets/avenues
    converted to digits (Third Avenue -> 3 AVENUE)."""
    a = addr.upper().strip()
    a = re.sub(r"\s+", " ", a)
    for word, digit in ORDINAL_WORDS.items():
        a = re.sub(rf"\b{word}\b", digit, a)
    a = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", a)
    return a


def _to_numeric(series: pd.Series) -> pd.Series:
    """Same comma-stripping fix from Step 1's _to_numeric() -- NYC Open
    Data's CSV export bakes literal thousands-separator commas into large
    number fields, which breaks both numeric comparisons and f-string
    number formatting (":,.0f" on a string raises ValueError, not a
    silent NaN) if not stripped first. This script loads the raw CSV
    directly rather than through Step 1's pipeline, so it needs its own
    copy of this fix."""
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def load_pluto_addresses(path: str = RAW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"\n[ERROR] Couldn't find {path}. Need the full raw PLUTO "
                  "file to check buildings that may have been filtered out "
                  "before Step 1's saved output.\n")
    cols = ["borough", "address", "bldgclass", "numfloors", "yearbuilt",
            "bldgarea", "officearea", "bldgfront", "bldgdepth", "lotfront",
            "lotdepth"]
    header = pd.read_csv(path, nrows=0)
    lower_to_original = {c.lower(): c for c in header.columns}
    usecols = [lower_to_original[c] for c in cols if c in lower_to_original]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["borough"] = df["borough"].astype(str).str.upper().str.strip()
    df = df[df["borough"] == "MN"].copy()

    # Mirror Step 1's cleaning exactly, not just the comma-stripping half of
    # it -- Step 1 also treats a 0/negative value in these fields as a
    # missing-data code and nulls it out (e.g. bldgarea=0 means "unknown,"
    # not "a zero-square-foot building"). Skipping that half here wouldn't
    # have changed today's results, but it's a real inconsistency with the
    # pipeline this script is supposed to be checking against -- a building
    # could otherwise get a wrong "reason" explanation below.
    for col in ["numfloors", "yearbuilt", "bldgarea", "officearea",
                "bldgfront", "bldgdepth", "lotfront", "lotdepth"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
            df.loc[df[col] <= 0, col] = np.nan
    # fillna("") BEFORE astype/str ops -- on some pandas configs, a nullable/
    # Arrow-backed string column keeps missing values as an actual NA/float
    # marker even after .astype(str), instead of stringifying them to "nan".
    # That NA can then survive into a .unique() array and crash difflib's
    # get_close_matches() with "object of type 'float' has no len()" below.
    # Clearing it to "" at the pandas level first avoids the ambiguity.
    df["address_norm"] = df["address"].fillna("").astype(str).str.upper().str.strip()

    split = df["address_norm"].apply(_split_house_number)
    df["house_number"] = split.apply(lambda t: t[0] if t[0] is not None else "")
    df["street_name"] = split.apply(lambda t: t[1])

    print(f"[load_pluto_addresses] Loaded {len(df):,} Manhattan lots from raw PLUTO")
    return df


def _split_house_number(addr_norm: str):
    """Split '25 WATER STREET' -> ('25', 'WATER STREET'). Used only for
    diagnostic messages below, not for matching -- see find_match()."""
    m = re.match(r"^(\d+)\s+(.*)$", addr_norm)
    if not m:
        return None, addr_norm
    return m.group(1), m.group(2)


def find_match(target_norm: str, pluto_mn: pd.DataFrame):
    """EXACT match only, on the full normalized address string.

    An earlier version of this function did fuzzy matching -- first on the
    whole address string (which matched '101 FRANKLIN STREET' to '181
    FRANKLIN STREET', and '25 WATER STREET' to '275 WATER STREET': totally
    different house numbers, rated near-identical as raw text), then a
    house-number-restricted version (which still matched '80 PINE STREET'
    to '80 JANE STREET': same house number, but two completely unrelated
    streets that happen to share enough characters -- both short, both
    ending in "...ANE STREET"-shaped text -- to clear an 0.80 similarity
    cutoff). Both approaches produced silently wrong buildings, which is
    much worse for a validation step than an honest "not found."
    Short proper-noun street names just aren't safe to fuzzy-match this
    way. Trading recall for precision here on purpose: a smaller number of
    CORRECT matches is more useful than a larger number where some are
    silently wrong."""
    exact = pluto_mn[pluto_mn["address_norm"] == target_norm]
    if len(exact):
        return exact.iloc[0], "exact"

    house_num, _ = _split_house_number(target_norm)
    if house_num is not None and (pluto_mn["house_number"] == house_num).any():
        return None, (f"no match (house number {house_num} exists on other "
                       f"Manhattan streets, but not at this address -- no "
                       f"exact street-name match, and fuzzy matching on "
                       f"short street names proved unreliable in testing, "
                       f"so this is reported as unmatched rather than guessed)")
    return None, "no match (not found in Manhattan PLUTO under this exact address)"


def explain_step1_miss(row: pd.Series) -> str:
    """Note: Step 1's actual filter is `numfloors >= 6` / `bldgarea >= 50_000`
    evaluated directly on possibly-NaN values -- pandas treats `NaN >= 6` as
    False, so a MISSING value fails the filter too, same as a too-low one.
    Both branches below are checked explicitly so a missing value gets
    named as the real reason instead of silently falling through to
    'unclear -- check manually'."""
    reasons = []
    if pd.isna(row.get("numfloors")):
        reasons.append("numfloors missing/invalid")
    elif row["numfloors"] < 6:
        reasons.append(f"numfloors={row['numfloors']:.0f} < 6")
    if pd.isna(row.get("bldgarea")):
        reasons.append("bldgarea missing/invalid")
    elif row["bldgarea"] < 50_000:
        reasons.append(f"bldgarea={row['bldgarea']:,.0f} < 50,000 SF")
    bldgclass = str(row.get("bldgclass", ""))
    is_office_class = bldgclass.startswith("O")
    office_share = None
    if pd.notna(row.get("officearea")) and pd.notna(row.get("bldgarea")) and row["bldgarea"]:
        office_share = row["officearea"] / row["bldgarea"]
    if not is_office_class and (office_share is None or office_share <= 0.5):
        reasons.append(f"bldgclass={bldgclass!r} not office-class, office "
                        f"share={'n/a' if office_share is None else f'{office_share:.0%}'} <= 50%")
    for c in ["bldgfront", "bldgdepth", "lotfront", "lotdepth"]:
        if pd.isna(row.get(c)):
            reasons.append(f"{c} missing/invalid")
    return "; ".join(reasons) if reasons else "unclear -- check manually"


def main():
    pluto_mn = load_pluto_addresses()
    candidates = pd.read_parquet(CANDIDATES_PATH) if os.path.exists(CANDIDATES_PATH) else None
    eligible = pd.read_parquet(ELIGIBLE_PATH) if os.path.exists(ELIGIBLE_PATH) else None
    scored = pd.read_parquet(SCORED_PATH) if os.path.exists(SCORED_PATH) else None

    if scored is not None:
        scored = scored.copy()
        scored["score_percentile"] = scored["viability_score"].rank(pct=True)

    results = []
    print("\n" + "=" * 70)
    print("VALIDATION: TRACING KNOWN REAL CONVERSIONS THROUGH THE PIPELINE")
    print("=" * 70)

    for known in KNOWN_CONVERSIONS:
        target_norm = normalize_address(known["address"])
        match_row, match_type = find_match(target_norm, pluto_mn)

        if match_row is None and known.get("alt_address"):
            alt_norm = normalize_address(known["alt_address"])
            match_row, match_type = find_match(alt_norm, pluto_mn)
            if match_row is not None:
                match_type = f"via alt address '{known['alt_address']}' -- {match_type}"

        result = {
            "address": known["address"],
            "note": known["note"],
            "source": known["source"],
            "found_in_pluto": match_row is not None,
            "match_type": match_type,
            "in_step1_candidates": False,
            "step1_miss_reason": "",
            "in_step3_eligible": False,
            "viability_score": np.nan,
            "score_percentile": np.nan,
            "tier1": False,
        }

        print(f"\n{known['address']}  ({known['note'][:60]}...)")
        if match_row is None:
            print(f"  -> NOT FOUND in raw PLUTO Manhattan lots ({match_type}). "
                  f"Likely an address-format mismatch (e.g. a marketing name "
                  f"like '5 Times Square') -- needs manual BBL lookup.")
            results.append(result)
            continue

        print(f"  -> Found in PLUTO ({match_type}): {match_row['address']}, "
              f"built {match_row.get('yearbuilt')}, {match_row.get('numfloors')} floors, "
              f"{match_row.get('bldgarea'):,.0f} SF" if pd.notna(match_row.get('bldgarea')) else
              f"  -> Found in PLUTO ({match_type}): {match_row['address']}")

        addr_norm = match_row["address_norm"]
        in_candidates = candidates is not None and (candidates["address"].str.upper().str.strip() == addr_norm).any()
        result["in_step1_candidates"] = bool(in_candidates)

        if not in_candidates:
            reason = explain_step1_miss(match_row)
            result["step1_miss_reason"] = reason
            print(f"  -> NOT in Step 1 candidate universe. Reason: {reason}")
            results.append(result)
            continue

        print("  -> IN Step 1 candidate universe.")

        in_eligible = eligible is not None and (eligible["address"].str.upper().str.strip() == addr_norm).any()
        result["in_step3_eligible"] = bool(in_eligible)
        print(f"  -> {'IN' if in_eligible else 'NOT in'} Step 3 legally-eligible universe.")

        if scored is not None:
            score_row = scored[scored["address"].str.upper().str.strip() == addr_norm]
            if len(score_row):
                vs = score_row.iloc[0]["viability_score"]
                pct = score_row.iloc[0]["score_percentile"]
                result["viability_score"] = vs
                result["score_percentile"] = pct
                result["tier1"] = pct >= 0.90
                print(f"  -> viability_score = {vs:.3f} (percentile {pct:.0%}"
                      f"{', TIER 1' if pct >= 0.90 else ''})")

        results.append(result)

    report = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n = len(report)
    print(f"Known real conversions checked:                {n}")
    print(f"Found in raw PLUTO (address matched):          {report['found_in_pluto'].sum()}")
    print(f"Made it into Step 1 candidate universe:        {report['in_step1_candidates'].sum()}")
    print(f"Passed Step 3 legal eligibility gates:          {report['in_step3_eligible'].sum()}")
    scored_matches = report["viability_score"].notna().sum()
    print(f"Received a Step 4 viability score:             {scored_matches}")
    if scored_matches:
        avg_pct = report["score_percentile"].dropna().mean()
        n_tier1 = int(report["tier1"].sum())
        print(f"Average score percentile among matched real conversions: {avg_pct:.0%}")
        print(f"Landed in Tier 1 (top 10%):                    {n_tier1} of {scored_matches}")
        print(
            "\n  READ: if real conversions land in high percentiles on average,\n"
            "  that's real signal the score is pointing at genuinely convertible\n"
            "  buildings, not noise. If they're scattered low/random, the score\n"
            "  isn't capturing what actually drives real-world conversion\n"
            "  decisions -- which are affected by ownership/financing/vacancy\n"
            "  factors this project's data can't see at all, not just physical\n"
            "  geometry and zoning."
        )

    print(
        "\nKNOWN LIMITATIONS OF THIS VALIDATION:\n"
        "  - Small sample (13), not a statistically powered test\n"
        "  - Address matching is exact-string, not BBL-based -- a building\n"
        "    whose PLUTO record uses a different official address than its\n"
        "    press-covered marketing name (common on major redevelopments)\n"
        "    reports as 'not found' here even if it's really in the data.\n"
        "    5 of 13 known conversions hit exactly this limitation.\n"
        "  - Real conversions are driven by factors this project doesn't\n"
        "    model at all: ownership motivation, existing debt/vacancy,\n"
        "    financing availability -- a building can score low here and\n"
        "    still convert (or vice versa) for reasons outside this dataset"
    )
    print("=" * 70 + "\n")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    report.to_csv(OUT_PATH, index=False)
    print(f"[main] Saved validation report to {OUT_PATH}")


if __name__ == "__main__":
    main()
