
import json
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "modeling" / "ros"
SPLIT_DIR = DATA_DIR / "splits"

MODEL_DIR = ROOT_DIR / "models" / "ros_random_forest"

METADATA_PATH = SPLIT_DIR / "metadata.json"
OUTPUT_DIR = DATA_DIR / "feature_importance"

TOP_N = 20


# ============================================================
# LOAD METADATA
# ============================================================

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

targets = metadata["targets"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ANALYZE EACH MODEL
# ============================================================

all_results = []

for target in targets:

    print("\n" + "=" * 65)
    print(f"Analyzing: {target}")
    print("=" * 65)

    model_path = MODEL_DIR / f"{target}_model.joblib"
    model = joblib.load(model_path)

    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df["importance_pct"] = (
        importance_df["importance"] * 100
    )

    importance_df["target"] = target

    all_results.append(importance_df)

    print(f"\nTop {TOP_N} features:\n")

    print(
        importance_df[
            ["feature", "importance_pct"]
        ]
        .head(TOP_N)
        .to_string(
            index=False,
            formatters={
                "importance_pct": "{:.2f}%".format
            }
        )
    )

    # Save individual target results
    output_path = OUTPUT_DIR / f"{target}_importance.csv"

    importance_df.to_csv(output_path, index=False)


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

combined = pd.concat(all_results, ignore_index=True)

combined_path = OUTPUT_DIR / "all_models_importance.csv"

combined.to_csv(combined_path, index=False)


# ============================================================
# AGGREGATE IMPORTANCE BY FEATURE FAMILY
# ============================================================

def classify_feature(feature):

    feature = feature.replace("numeric__", "")
    feature = feature.replace("categorical__", "")

    if feature == "position":
        return "Categorical"

    if "cutoff_percentage" in feature:
        return "Cutoff"

    if "last_5" in feature or "last_10" in feature or "last_20" in feature:
        return "Rolling averages"

    if "trend" in feature:
        return "Trends"

    if "per_60" in feature:
        return "Per-60 statistics"

    if feature.startswith("season_"):
        return "Season-to-date"

    if feature.startswith("games_in_last_"):
        return "Recent game counts"

    if feature == "days_since_last_game":
        return "Time since last game"

    return "Other"


combined["feature_family"] = combined["feature"].apply(
    classify_feature
)

family_importance = (
    combined.groupby(["target", "feature_family"])["importance_pct"]
    .sum()
    .reset_index()
    .sort_values(
        ["target", "importance_pct"],
        ascending=[True, False]
    )
)

family_path = OUTPUT_DIR / "importance_by_family.csv"

family_importance.to_csv(family_path, index=False)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 65)

print("\nAverage importance by feature family across all models:")

family_summary = (
    family_importance.groupby("feature_family")["importance_pct"]
    .mean()
    .sort_values(ascending=False)
)

print(family_summary.round(2).to_string())

print(f"\nIndividual results: {OUTPUT_DIR}")
print(f"Combined results: {combined_path}")
print(f"Family summary: {family_path}")
