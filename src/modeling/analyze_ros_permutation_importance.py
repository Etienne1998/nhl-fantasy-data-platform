
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "modeling" / "ros"
SPLIT_DIR = DATA_DIR / "splits"

MODEL_DIR = ROOT_DIR / "models" / "ros_random_forest"
OUTPUT_DIR = DATA_DIR / "permutation_importance"

TARGETS = [
    "remaining_goals",
    "remaining_assists",
    "remaining_shots_on_goal",
    "remaining_power_play_points",
    "remaining_time_on_ice_seconds"
]

N_REPEATS = 5
N_JOBS = 2
RANDOM_STATE = 42
TOP_N = 15


# ============================================================
# LOAD DATA
# ============================================================

print("Loading validation dataset...")

validation = pd.read_csv(SPLIT_DIR / "validation.csv")

with open(SPLIT_DIR / "metadata.json", "r") as file:
    metadata = json.load(file)

features = metadata["features"]

X_validation = validation[features]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

all_results = []

for target in TARGETS:

    print("\n" + "=" * 65)
    print(f"Analyzing: {target}")
    print("=" * 65)

    model_path = MODEL_DIR / f"{target}_model.joblib"
    model = joblib.load(model_path)

    y_validation = validation[target]

    result = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="neg_mean_absolute_error",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    importance_df = pd.DataFrame({
        "feature": features,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    })

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False
    )

    importance_df["target"] = target

    all_results.append(importance_df)

    print(f"\nTop {TOP_N} features:\n")

    print(
        importance_df[
            ["feature", "importance_mean", "importance_std"]
        ]
        .head(TOP_N)
        .to_string(
            index=False,
            formatters={
                "importance_mean": "{:.4f}".format,
                "importance_std": "{:.4f}".format
            }
        )
    )

    output_path = OUTPUT_DIR / f"{target}_permutation.csv"
    importance_df.to_csv(output_path, index=False)


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

combined = pd.concat(all_results, ignore_index=True)

combined_path = OUTPUT_DIR / "all_models_permutation.csv"

combined.to_csv(combined_path, index=False)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PERMUTATION IMPORTANCE COMPLETED")
print("=" * 65)

print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"Combined results: {combined_path}")
