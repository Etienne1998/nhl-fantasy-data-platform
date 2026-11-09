
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT_DIR / "data" / "modeling" / "ros" / "splits"

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "ros_availability"
    / "remaining_games_model.joblib"
)

OUTPUT_DIR = ROOT_DIR / "models" / "ros_availability" / "permutation_importance"

N_REPEATS = 10
N_JOBS = 2
RANDOM_STATE = 42
TOP_N = 20


# ============================================================
# LOAD DATA
# ============================================================

print("Loading validation dataset...")

validation = pd.read_csv(SPLIT_DIR / "validation.csv")

with open(SPLIT_DIR / "metadata.json", "r") as file:
    metadata = json.load(file)

features = metadata["features"]

model = joblib.load(MODEL_PATH)

# Respect the exact feature order used during training.
model_features = list(model.feature_names_in_)

X_validation = validation[model_features]
y_validation = validation["remaining_games"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Validation records: {len(validation):,}")
print(f"Model features: {len(model_features)}")


# ============================================================
# BASELINE PERFORMANCE
# ============================================================

baseline_predictions = model.predict(X_validation)

baseline_mae = (
    (y_validation - baseline_predictions).abs().mean()
)

print("\n" + "=" * 65)
print("BASELINE MODEL PERFORMANCE")
print("=" * 65)

print(f"Validation MAE: {baseline_mae:.4f} games")


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating permutation importance...")

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
    "feature": model_features,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
})

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False
).reset_index(drop=True)

importance_df["rank"] = range(1, len(importance_df) + 1)

importance_df = importance_df[
    ["rank", "feature", "importance_mean", "importance_std"]
]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print(f"TOP {TOP_N} AVAILABILITY FEATURES")
print("=" * 65)

print(
    importance_df.head(TOP_N).to_string(
        index=False,
        formatters={
            "importance_mean": "{:.4f}".format,
            "importance_std": "{:.4f}".format
        }
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = OUTPUT_DIR / "remaining_games_permutation.csv"

importance_df.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("PERMUTATION IMPORTANCE COMPLETED")
print("=" * 65)

print(f"Results saved to: {output_path}")
