
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "modeling" / "ros"
SPLIT_DIR = DATA_DIR / "splits"

MODEL_DIR = ROOT_DIR / "models" / "ros_random_forest"

METADATA_PATH = SPLIT_DIR / "metadata.json"
OUTPUT_PATH = DATA_DIR / "ros_cutoff_performance.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading ROS datasets...")

train = pd.read_csv(SPLIT_DIR / "train.csv")
validation = pd.read_csv(SPLIT_DIR / "validation.csv")

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

features = metadata["features"]
targets = metadata["targets"]

print(f"Training records: {len(train)}")
print(f"Validation records: {len(validation)}")


# ============================================================
# EVALUATE EACH CUTOFF
# ============================================================

results = []

cutoffs = sorted(validation["cutoff_percentage"].unique())

print(f"\nCutoffs found: {cutoffs}")

for target in targets:

    print("\n" + "=" * 60)
    print(f"Analyzing: {target}")
    print("=" * 60)

    model_path = MODEL_DIR / f"{target}_model.joblib"
    model = joblib.load(model_path)

    for cutoff in cutoffs:

        validation_subset = validation[
            validation["cutoff_percentage"] == cutoff
        ].copy()

        train_subset = train[
            train["cutoff_percentage"] == cutoff
        ]

        X_validation = validation_subset[features]
        y_validation = validation_subset[target]

        # Random Forest predictions
        predictions = model.predict(X_validation)

        rf_mae = mean_absolute_error(
            y_validation,
            predictions
        )

        rf_rmse = mean_squared_error(
            y_validation,
            predictions
        ) ** 0.5

        # Baseline: mean learned from training data
        # at the same cutoff percentage
        baseline_mean = train_subset[target].mean()

        baseline_predictions = [
            baseline_mean
        ] * len(validation_subset)

        baseline_mae = mean_absolute_error(
            y_validation,
            baseline_predictions
        )

        improvement_pct = (
            (baseline_mae - rf_mae) / baseline_mae
        ) * 100 if baseline_mae != 0 else 0

        results.append({
            "target": target,
            "cutoff_percentage": cutoff,
            "validation_records": len(validation_subset),
            "rf_mae": rf_mae,
            "rf_rmse": rf_rmse,
            "baseline_mae": baseline_mae,
            "improvement_pct": improvement_pct
        })

        print(
            f"Cutoff: {cutoff:.0%} | "
            f"Records: {len(validation_subset)} | "
            f"RF MAE: {rf_mae:.4f} | "
            f"Baseline MAE: {baseline_mae:.4f} | "
            f"Improvement: {improvement_pct:+.2f}%"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 60)
print("CUTOFF ANALYSIS COMPLETED")
print("=" * 60)

summary = results_df.pivot(
    index="target",
    columns="cutoff_percentage",
    values="improvement_pct"
)

summary.columns = [
    f"{column:.0%}" for column in summary.columns
]

print("\nImprovement vs baseline (%)")
print(summary.round(2).to_string())

print(f"\nResults saved to: {OUTPUT_PATH}")
