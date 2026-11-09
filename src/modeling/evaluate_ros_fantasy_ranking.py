
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "modeling" / "ros" / "splits"
MODEL_DIR = ROOT / "models" / "ros_random_forest"

OUTPUT_DIR = ROOT / "data" / "modeling" / "ros" / "fantasy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "remaining_goals",
    "remaining_assists",
    "remaining_plus_minus",
    "remaining_penalty_minutes",
    "remaining_power_play_goals",
    "remaining_power_play_points",
    "remaining_short_handed_goals",
    "remaining_short_handed_points",
    "remaining_game_winning_goals",
    "remaining_shots_on_goal",
    "remaining_time_on_ice_seconds",
]

TOP_N_VALUES = [10, 25, 50]


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation dataset...")

validation = pd.read_csv(SPLIT_DIR / "validation.csv")

# Load feature definitions
import json

with open(SPLIT_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

print(f"Validation records: {len(validation)}")


# ============================================================
# LOAD MODELS AND GENERATE PREDICTIONS
# ============================================================

predictions = validation[
    [
        "player_id",
        "player_name",
        "team",
        "position",
        "season_id",
        "cutoff_date",
        "cutoff_percentage",
    ]
].copy()

print("\nGenerating predictions...")

for target in TARGETS:

    print(f"Predicting {target}...")

    model_path = MODEL_DIR / f"{target}_model.joblib"
    model = joblib.load(model_path)

    predicted = model.predict(validation[FEATURES])

    # All these statistics are non-negative.
    if target != "remaining_plus_minus":
        predicted = np.maximum(predicted, 0)

    predictions[f"pred_{target}"] = predicted
    predictions[f"actual_{target}"] = validation[target].to_numpy()


# ============================================================
# CALCULATE FANTASY POINTS
# ============================================================

def calculate_fantasy_points(df, prefix):

    goals = df[f"{prefix}remaining_goals"]
    assists = df[f"{prefix}remaining_assists"]
    plus_minus = df[f"{prefix}remaining_plus_minus"]
    pim = df[f"{prefix}remaining_penalty_minutes"]

    ppg = df[f"{prefix}remaining_power_play_goals"]
    ppp = df[f"{prefix}remaining_power_play_points"]

    shg = df[f"{prefix}remaining_short_handed_goals"]
    shp = df[f"{prefix}remaining_short_handed_points"]

    gwg = df[f"{prefix}remaining_game_winning_goals"]
    shots = df[f"{prefix}remaining_shots_on_goal"]

    # Derived assists:
    ppa = np.maximum(ppp - ppg, 0)
    sha = np.maximum(shp - shg, 0)

    return (
        goals * 2.0
        + assists
        + plus_minus
        + pim * 0.3
        + ppg
        + ppa * 0.5
        + shg * 2.0
        + sha
        + gwg
        + shots * 0.2
    )


predictions["pred_fantasy_points"] = calculate_fantasy_points(
    predictions, "pred_"
)

predictions["actual_fantasy_points"] = calculate_fantasy_points(
    predictions, "actual_"
)


# ============================================================
# EVALUATE FANTASY RANKINGS
# ============================================================

results = []

for cutoff, group in predictions.groupby("cutoff_percentage"):

    spearman = group["actual_fantasy_points"].corr(
        group["pred_fantasy_points"],
        method="spearman"
    )

    row = {
        "cutoff": cutoff,
        "players": len(group),
        "spearman_correlation": spearman,
        "fantasy_points_mae": np.mean(
            np.abs(
                group["pred_fantasy_points"]
                - group["actual_fantasy_points"]
            )
        ),
    }

    for n in TOP_N_VALUES:

        actual_top = set(
            group.nlargest(n, "actual_fantasy_points")["player_id"]
        )

        predicted_top = set(
            group.nlargest(n, "pred_fantasy_points")["player_id"]
        )

        overlap = len(actual_top.intersection(predicted_top))

        row[f"top_{n}_overlap"] = overlap
        row[f"top_{n}_precision"] = overlap / n

    results.append(row)


# ============================================================
# SAVE OUTPUTS
# ============================================================

results_df = pd.DataFrame(results)

predictions.to_csv(
    OUTPUT_DIR / "ros_fantasy_player_predictions.csv",
    index=False
)

results_df.to_csv(
    OUTPUT_DIR / "ros_fantasy_ranking_results.csv",
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FANTASY ROS RANKING RESULTS")
print("=" * 65)

print(results_df.round(3).to_string(index=False))

print("\nFiles saved:")
print(OUTPUT_DIR / "ros_fantasy_player_predictions.csv")
print(OUTPUT_DIR / "ros_fantasy_ranking_results.csv")
