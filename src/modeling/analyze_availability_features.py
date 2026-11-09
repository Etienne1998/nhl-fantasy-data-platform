from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "modeling" / "ros" / "splits"
OUTPUT_DIR = ROOT / "models" / "ros_availability"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"
METADATA_PATH = SPLIT_DIR / "metadata.json"

TARGET = "remaining_games"

KEYWORDS = [
    "game",
    "availability",
    "miss",
    "toi",
    "time_on_ice",
    "streak",
    "last_",
    "recent",
    "participation",
    "season_games_played",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading ROS datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

features = metadata["features"]

print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print(f"Training rows:   {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Test rows:       {len(test):,}")
print(f"Model features:  {len(features):,}")


# ============================================================
# IDENTIFY AVAILABILITY-RELATED FEATURES
# ============================================================

candidates = []

for feature in features:
    name = feature.lower()

    if any(keyword in name for keyword in KEYWORDS):
        candidates.append(feature)

print("\n" + "=" * 70)
print("AVAILABILITY-RELATED FEATURES")
print("=" * 70)

print(f"\nPotential features found: {len(candidates)}")

for feature in candidates:
    print(f" - {feature}")


# ============================================================
# CHECK FEATURE COVERAGE ACROSS SPLITS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE COVERAGE BY SPLIT")
print("=" * 70)

coverage_rows = []

splits = {
    "train": train,
    "validation": validation,
    "test": test,
}

for feature in candidates:
    for split_name, df in splits.items():

        if feature not in df.columns:
            coverage_rows.append({
                "feature": feature,
                "split": split_name,
                "present": False,
                "missing_pct": 100.0,
                "unique_values": 0,
            })
            continue

        series = df[feature]

        coverage_rows.append({
            "feature": feature,
            "split": split_name,
            "present": True,
            "missing_pct": round(series.isna().mean() * 100, 2),
            "unique_values": series.nunique(dropna=True),
        })

coverage = pd.DataFrame(coverage_rows)

if not coverage.empty:
    print(
        coverage.pivot(
            index="feature",
            columns="split",
            values="missing_pct"
        ).to_string()
    )


# ============================================================
# TRAIN-ONLY ASSOCIATION WITH REMAINING GAMES
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FEATURE ASSOCIATION WITH REMAINING GAMES")
print("=" * 70)

if TARGET not in train.columns:
    raise ValueError(f"Target '{TARGET}' not found in training dataset.")

association_rows = []

for feature in candidates:

    if feature not in train.columns:
        continue

    if not pd.api.types.is_numeric_dtype(train[feature]):
        continue

    # Use the feature and target only for this descriptive analysis.
    # The target is never included as a model feature.
    data = train[[feature, TARGET]].dropna()

    if len(data) < 10:
        continue

    if data[feature].nunique() <= 1:
        continue

    pearson = data[feature].corr(data[TARGET], method="pearson")

    spearman_result = spearmanr(
        data[feature],
        data[TARGET]
    )

    spearman = spearman_result.statistic

    association_rows.append({
        "feature": feature,
        "observations": len(data),
        "missing_pct_train": round(
            train[feature].isna().mean() * 100, 2
        ),
        "unique_values": train[feature].nunique(),
        "mean": round(train[feature].mean(), 3),
        "std": round(train[feature].std(), 3),
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "abs_spearman": abs(spearman),
    })

associations = pd.DataFrame(association_rows)

if associations.empty:
    print("\nNo numeric candidate features available for analysis.")

else:
    associations = associations.sort_values(
        "abs_spearman",
        ascending=False
    )

    print("\nTop features associated with remaining_games:")
    print(
        associations[
            [
                "feature",
                "observations",
                "missing_pct_train",
                "pearson",
                "spearman",
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

coverage_path = OUTPUT_DIR / "availability_feature_coverage.csv"
associations_path = OUTPUT_DIR / "availability_feature_associations.csv"

coverage.to_csv(coverage_path, index=False)

if not associations.empty:
    associations.to_csv(associations_path, index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(f"Coverage report:     {coverage_path}")
print(f"Association report:  {associations_path}")

print("\nNo model was trained.")
print("No test-set performance was used for feature selection.")