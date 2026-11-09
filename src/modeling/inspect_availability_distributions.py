
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/modeling/ros/splits/train.csv"
OUTPUT_DIR = ROOT / "models/ros_availability"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "days_since_last_game",
    "games_in_last_5",
    "games_in_last_10",
    "games_in_last_20",
    "games_in_last_10_days",
    "time_on_ice_seconds_last_5_avg",
    "time_on_ice_seconds_last_10_avg",
    "time_on_ice_seconds_last_20_avg",
    "time_on_ice_seconds_trend_5_vs_20",
    "season_games_played",
]

TARGET = "remaining_games"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training dataset...")

df = pd.read_csv(TRAIN_PATH)

print(f"Training records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

available = [f for f in FEATURES if f in df.columns]
missing = [f for f in FEATURES if f not in df.columns]

if missing:
    print("\nMissing features:")
    for feature in missing:
        print(f" - {feature}")


# ============================================================
# 1. GENERAL DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 75)
print("1. FEATURE DISTRIBUTIONS")
print("=" * 75)

summary_rows = []

for feature in available:

    series = pd.to_numeric(df[feature], errors="coerce")
    valid = series.dropna()

    if valid.empty:
        continue

    summary = {
        "feature": feature,
        "count": len(valid),
        "missing_pct": round(series.isna().mean() * 100, 2),
        "unique_values": valid.nunique(),
        "min": valid.min(),
        "p05": valid.quantile(0.05),
        "p25": valid.quantile(0.25),
        "median": valid.median(),
        "mean": valid.mean(),
        "p75": valid.quantile(0.75),
        "p95": valid.quantile(0.95),
        "max": valid.max(),
        "std": valid.std(),
        "zero_pct": round((valid == 0).mean() * 100, 2),
    }

    summary_rows.append(summary)

    print(f"\n{feature}")
    print(f"  Count:        {len(valid):,}")
    print(f"  Missing:      {series.isna().mean() * 100:.2f}%")
    print(f"  Unique:       {valid.nunique():,}")
    print(f"  Min / Max:    {valid.min():.4f} / {valid.max():.4f}")
    print(f"  Mean:         {valid.mean():.4f}")
    print(f"  Median:       {valid.median():.4f}")
    print(f"  Std:          {valid.std():.4f}")
    print(f"  Zero values:  {(valid == 0).mean() * 100:.2f}%")


summary_df = pd.DataFrame(summary_rows)

summary_path = OUTPUT_DIR / "availability_distribution_summary.csv"
summary_df.to_csv(summary_path, index=False)


# ============================================================
# 2. FREQUENCY DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 75)
print("2. FREQUENCY DISTRIBUTIONS")
print("=" * 75)

frequency_features = [
    "days_since_last_game",
    "games_in_last_5",
    "games_in_last_10",
    "games_in_last_20",
    "games_in_last_10_days",
]

for feature in frequency_features:

    if feature not in df.columns:
        continue

    print(f"\n{feature}")

    counts = (
        df[feature]
        .value_counts(dropna=False)
        .sort_index()
    )

    print(counts.head(30).to_string())

    if len(counts) > 30:
        print(f"... {len(counts) - 30} additional values")


# ============================================================
# 3. GAMES IN LAST 10: INVESTIGATE THE CONSTANT VALUE
# ============================================================

print("\n" + "=" * 75)
print("3. INVESTIGATING GAMES IN LAST 10")
print("=" * 75)

if "games_in_last_10" in df.columns:

    values = df["games_in_last_10"]

    print(f"Unique values: {values.nunique()}")
    print(f"Minimum: {values.min()}")
    print(f"Maximum: {values.max()}")
    print(f"Percentage equal to 10: {(values == 10).mean() * 100:.2f}%")
    print(f"Percentage below 10: {(values < 10).mean() * 100:.2f}%")

    print("\nValue counts:")
    print(values.value_counts(dropna=False).sort_index().to_string())


# ============================================================
# 4. ICE TIME TREND: INVESTIGATE DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("4. INVESTIGATING ICE TIME TREND")
print("=" * 75)

trend_feature = "time_on_ice_seconds_trend_5_vs_20"

if trend_feature in df.columns:

    trend = df[trend_feature].dropna()

    print("\nQuantiles:")
    print(
        trend.quantile(
            [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]
        ).to_string()
    )

    print("\nPercentage by range:")

    ranges = {
        "Below -0.50": trend < -0.50,
        "-0.50 to -0.15": (trend >= -0.50) & (trend < -0.15),
        "-0.15 to -0.05": (trend >= -0.15) & (trend < -0.05),
        "-0.05 to 0.05": (trend >= -0.05) & (trend <= 0.05),
        "0.05 to 0.15": (trend > 0.05) & (trend <= 0.15),
        "0.15 to 0.50": (trend > 0.15) & (trend <= 0.50),
        "Above 0.50": trend > 0.50,
    }

    for label, mask in ranges.items():
        print(f"{label:20s}: {mask.mean() * 100:6.2f}%")


# ============================================================
# 5. DISTRIBUTIONS BY CUTOFF
# ============================================================

print("\n" + "=" * 75)
print("5. DISTRIBUTIONS BY CUTOFF")
print("=" * 75)

cutoff_column = "cutoff_percentage"

if cutoff_column in df.columns:

    cutoff_features = [
        "days_since_last_game",
        "games_in_last_10",
        "games_in_last_10_days",
        "time_on_ice_seconds_last_10_avg",
        "time_on_ice_seconds_trend_5_vs_20",
    ]

    for feature in cutoff_features:

        if feature not in df.columns:
            continue

        print(f"\n{feature}")

        result = (
            df.groupby(cutoff_column)[feature]
            .agg(["count", "mean", "median", "min", "max", "std"])
            .round(3)
        )

        print(result.to_string())


# ============================================================
# 6. CORRELATIONS WITH REMAINING GAMES
# ============================================================

print("\n" + "=" * 75)
print("6. CORRELATIONS WITH REMAINING GAMES")
print("=" * 75)

if TARGET in df.columns:

    correlations = []

    for feature in available:

        data = df[[feature, TARGET]].dropna()

        if len(data) < 10 or data[feature].nunique() <= 1:
            continue

        correlations.append({
            "feature": feature,
            "pearson": data[feature].corr(data[TARGET]),
            "spearman": data[feature].corr(data[TARGET], method="spearman"),
        })

    corr_df = pd.DataFrame(correlations)

    if not corr_df.empty:
        corr_df = corr_df.sort_values(
            "spearman",
            key=lambda x: x.abs(),
            ascending=False,
        )

        print(corr_df.round(4).to_string(index=False))

        corr_df.to_csv(
            OUTPUT_DIR / "availability_distribution_correlations.csv",
            index=False,
        )


# ============================================================
# 7. SAVE SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("INSPECTION COMPLETED")
print("=" * 75)

print(f"Summary: {summary_path}")
print("Only training data was used.")
print("No model was trained or modified.")
