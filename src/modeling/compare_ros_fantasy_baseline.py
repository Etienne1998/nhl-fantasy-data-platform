
from pathlib import Path

import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "modeling" / "ros" / "splits"
PREDICTIONS_FILE = (
    ROOT / "data" / "modeling" / "ros" /
    "fantasy" / "ros_fantasy_player_predictions.csv"
)

OUTPUT_DIR = ROOT / "data" / "modeling" / "ros" / "fantasy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_VALUES = [10, 25, 50]

# Rolling features used for the naive baseline.
ROLLING_MAPPING = {
    "remaining_goals": "goals_last_20_avg",
    "remaining_assists": "assists_last_20_avg",
    "remaining_plus_minus": "plus_minus_last_20_avg",
    "remaining_penalty_minutes": "penalty_minutes_last_20_avg",
    "remaining_power_play_goals": "power_play_goals_last_20_avg",
    "remaining_power_play_points": "power_play_points_last_20_avg",
    "remaining_short_handed_goals": "short_handed_goals_last_20_avg",
    "remaining_short_handed_points": "short_handed_points_last_20_avg",
    "remaining_shots_on_goal": "shots_on_goal_last_20_avg",
    "remaining_time_on_ice_seconds": "time_on_ice_seconds_last_20_avg",
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

validation = pd.read_csv(SPLIT_DIR / "validation.csv")
train = pd.read_csv(SPLIT_DIR / "train.csv")
predictions = pd.read_csv(PREDICTIONS_FILE)

# Historical game-winning-goal rate, learned from training only.
gwg_rate = (
    train["remaining_game_winning_goals"].sum()
    / max(train["remaining_goals"].sum(), 1)
)

print(f"Training GWG / goals rate: {gwg_rate:.4f}")


# ============================================================
# BUILD NAIVE ROS PROJECTIONS
# ============================================================

baseline = validation[
    [
        "player_id",
        "player_name",
        "team",
        "position",
        "cutoff_percentage",
    ]
].copy()

# Approximate number of team games remaining.
# This is a simple benchmark, not an exact schedule calculation.
baseline["estimated_remaining_games"] = (
    82 * (1 - baseline["cutoff_percentage"])
)

for target, feature in ROLLING_MAPPING.items():

    baseline[target] = (
        validation[feature].fillna(0)
        * baseline["estimated_remaining_games"]
    )

# Estimate game-winning goals from projected goals.
baseline["remaining_game_winning_goals"] = (
    baseline["remaining_goals"] * gwg_rate
)


# ============================================================
# CALCULATE FANTASY POINTS
# ============================================================

def calculate_fantasy_points(df, prefix=""):

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

    ppa = np.maximum(ppp - ppg, 0)
    sha = np.maximum(shp - shg, 0)

    return (
        goals * 2
        + assists
        + plus_minus
        + pim * 0.3
        + ppg
        + ppa * 0.5
        + shg * 2
        + sha
        + gwg
        + shots * 0.2
    )


baseline["baseline_fantasy_points"] = calculate_fantasy_points(
    baseline
)

# Join actual and Random Forest fantasy results.
comparison = predictions[
    [
        "player_id",
        "cutoff_percentage",
        "actual_fantasy_points",
        "pred_fantasy_points",
    ]
].merge(
    baseline[
        ["player_id", "cutoff_percentage", "baseline_fantasy_points"]
    ],
    on=["player_id", "cutoff_percentage"],
    how="inner",
)


# ============================================================
# EVALUATE RANKINGS
# ============================================================

results = []

for cutoff, group in comparison.groupby("cutoff_percentage"):

    actual = group["actual_fantasy_points"]

    row = {
        "cutoff": cutoff,
        "players": len(group),
    }

    for model_name, prediction_column in [
        ("random_forest", "pred_fantasy_points"),
        ("last_20_baseline", "baseline_fantasy_points"),
    ]:

        predicted = group[prediction_column]

        row[f"{model_name}_spearman"] = actual.corr(
            predicted, method="spearman"
        )

        row[f"{model_name}_mae"] = np.mean(
            np.abs(actual - predicted)
        )

        for n in TOP_N_VALUES:

            actual_top = set(
                group.nlargest(n, "actual_fantasy_points")["player_id"]
            )

            predicted_top = set(
                group.nlargest(n, prediction_column)["player_id"]
            )

            overlap = len(actual_top.intersection(predicted_top))

            row[f"{model_name}_top_{n}"] = overlap

    results.append(row)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "ros_fantasy_baseline_comparison.csv",
    index=False
)

print("\n" + "=" * 75)
print("RANDOM FOREST VS LAST-20 BASELINE")
print("=" * 75)

print(results_df.round(3).to_string(index=False))

print("\nSaved to:")
print(OUTPUT_DIR / "ros_fantasy_baseline_comparison.csv")
