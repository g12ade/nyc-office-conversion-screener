"""
diagnose_floorplate.py

One-off diagnostic (not part of the pipeline) to check whether the
floorplate_depth_ft outliers/high deep-flag rate from Step 2 are a real
signal or an artifact of multi-building tax lots. PLUTO's bldgfront/
bldgdepth are bounding-box measurements of the WHOLE LOT, not of a single
building -- on a lot with more than one building (numbldgs > 1), that
bounding box can span far more than any individual building's footprint.

Usage:
    python diagnose_floorplate.py
"""

import pandas as pd

df = pd.read_parquet("data/processed/candidates_features.parquet")

print("\n" + "=" * 70)
print("TOP 15 BY floorplate_depth_ft")
print("=" * 70)
cols = [c for c in ["address", "numbldgs", "bldgfront", "bldgdepth",
                     "bldgarea", "numfloors", "floorplate_depth_ft"]
        if c in df.columns]
print(df[cols].sort_values("floorplate_depth_ft", ascending=False).head(15).to_string(index=False))

if "numbldgs" in df.columns:
    # numbldgs was never numeric-coerced in Step 1 (nothing filtered on it
    # yet, so the gap went unnoticed) -- it loaded as text, which breaks
    # ">" comparisons. Fix it here locally rather than editing Step 1 yet.
    df["numbldgs"] = pd.to_numeric(df["numbldgs"], errors="coerce")

    print("\n" + "=" * 70)
    print("numbldgs VALUE COUNTS")
    print("=" * 70)
    print(df["numbldgs"].value_counts().sort_index())

    single = df[df["numbldgs"] == 1]
    multi = df[df["numbldgs"] > 1]

    print("\n" + "=" * 70)
    print("DEEP-FLAG RATE: single-building lots vs multi-building lots")
    print("=" * 70)
    for label, subset in [("numbldgs == 1", single), ("numbldgs > 1", multi)]:
        n_valid = subset["floorplate_depth_ft"].notna().sum()
        n_deep = subset["deep_floorplate_flag"].sum()
        pct = (n_deep / n_valid * 100) if n_valid else 0.0
        print(f"{label:<16} n={len(subset):>5,}  "
              f"deep={n_deep:>5,} of {n_valid:>5,} valid  ({pct:.1f}%)")
        print(f"{'':16}  mean depth = {subset['floorplate_depth_ft'].mean():.1f} ft, "
              f"median = {subset['floorplate_depth_ft'].median():.1f} ft, "
              f"max = {subset['floorplate_depth_ft'].max():.1f} ft")

print("\n" + "=" * 70)
print("OVERALL floorplate_depth_ft PERCENTILES")
print("=" * 70)
print(df["floorplate_depth_ft"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

print("\n" + "=" * 70)
print("DEPTH BUCKETS (how many rows are driving the deep-flag rate)")
print("=" * 70)
bins = [0, 65, 100, 150, 200, 300, 10000]
labels = ["<=65 (not flagged)", "65-100", "100-150", "150-200", "200-300", "300+"]
bucket = pd.cut(df["floorplate_depth_ft"], bins=bins, labels=labels, right=True)
print(bucket.value_counts().sort_index())
