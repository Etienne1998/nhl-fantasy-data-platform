
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "modeling" / "ros"
SPLIT_DIR = DATA_DIR / "splits"
MODEL_DIR = ROOT / "models" / "ros_random_forest"

OUTPUT_DIR = DATA_DIR / "prediction_errors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS_TO_ANALYZE = [
    "remaining_goals",
    "remaining_assists",
    "remaining_shots_on_goal",
    "remaining_power_play_points",
    "remaining_time_on_ice_seconds",
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading validation dataset...")

validation = pd.read_csv(SPLIT_DIR / "validation.csv")

with open(SPLIT_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

print(f"Validation records: {len(validation)}")
print(f"Features: {len(FEATURES)}")


# ============================================================
# CREATE PLAYER ROLE GROUPS
# ============================================================

# Quartiles based on average time on ice per game.
# Q1 = least-used players, Q4 = most-used players.

validation["role_group"] = pd.qcut(
    validation["season_avg_toi_minutes"].rank(method="first"),
    q=4,
    labels=["Q1 - Low TOI", "Q2 - Mid-low TOI",
            "Q3 - Mid-high TOI", "Q4 - High TOI"]
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def calculate_metrics(group, target, predictions):

    actual = group[target].to_numpy()

    mae = mean_absolute_error(actual, predictions)

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    # Positive bias = model overestimates.
    # Negative bias = model underestimates.
    bias = np.mean(predictions - actual)

    return {
        "records": len(group),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }


# ============================================================
# ANALYZE EACH MODEL
# ============================================================

all_results = []
all_predictions = []

for target in TARGETS_TO_ANALYZE:

    print("\n" + "=" * 65)
    print(f"Analyzing: {target}")
    print("=" * 65)

    model_path = MODEL_DIR / f"{target}_model.joblib"

    model = joblib.load(model_path)

    X = validation[FEATURES]
    predictions = model.predict(X)

    actual = validation[target].to_numpy()

    errors = validation[
        [
            "player_id",
            "player_name",
            "team",
            "position",
            "season_id",
            "cutoff_date",
            "cutoff_percentage",
            "season_avg_toi_minutes",
            "role_group",
        ]
    ].copy()

    errors["target"] = target
    errors["actual"] = actual
    errors["predicted"] = predictions
    errors["absolute_error"] = np.abs(predictions - actual)
    errors["signed_error"] = predictions - actual

    all_predictions.append(errors)

    # --------------------------------------------------------
    # 1. PERFORMANCE BY CUTOFF
    # --------------------------------------------------------

    for cutoff, group in errors.groupby("cutoff_percentage"):

        metrics = calculate_metrics(
            group,
            "actual",
            group["predicted"].to_numpy()
        )

        all_results.append({
            "target": target,
            "analysis": "cutoff",
            "group": cutoff,
            **metrics,
        })

    # --------------------------------------------------------
    # 2. PERFORMANCE BY PLAYER ROLE
    # --------------------------------------------------------

    for role, group in errors.groupby("role_group", observed=False):

        if len(group) == 0:
            continue

        metrics = calculate_metrics(
            group,
            "actual",
            group["predicted"].to_numpy()
        )

        all_results.append({
            "target": target,
            "analysis": "player_role",
            "group": role,
            **metrics,
        })

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nPerformance by cutoff:")

    cutoff_results = [
        row for row in all_results
        if row["target"] == target
        and row["analysis"] == "cutoff"
    ]

    print(pd.DataFrame(cutoff_results).to_string(index=False))

    print("\nPerformance by player role:")

    role_results = [
        row for row in all_results
        if row["target"] == target
        and row["analysis"] == "player_role"
    ]

    print(pd.DataFrame(role_results).to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)
predictions_df = pd.concat(all_predictions, ignore_index=True)

results_df.to_csv(
    OUTPUT_DIR / "ros_error_metrics.csv",
    index=False
)

predictions_df.to_csv(
    OUTPUT_DIR / "ros_player_predictions.csv",
    index=False
)

print("\n" + "=" * 65)
print("ROS PREDICTION ERROR ANALYSIS COMPLETED")
print("=" * 65)

print(f"Metrics saved to: {OUTPUT_DIR / 'ros_error_metrics.csv'}")
print(f"Player predictions saved to: {OUTPUT_DIR / 'ros_player_predictions.csv'}")
