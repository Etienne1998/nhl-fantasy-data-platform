
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    ROOT / "data" / "modeling" / "ros" /
    "fantasy" / "ros_fantasy_test_predictions.csv"
)

SILVER_DIR = ROOT / "data" / "silver"

OUTPUT_DIR = ROOT / "data" / "modeling" / "ros" / "fantasy"


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("Loading test predictions...")

df = pd.read_csv(PREDICTIONS_FILE)

df["season_id"] = df["season_id"].astype(str)
df["player_id"] = df["player_id"].astype(str)
df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])

silver_frames = []

for season in df["season_id"].unique():

    matching_files = list(
        SILVER_DIR.glob(f"**/*{season}*.json")
    )

    if len(matching_files) != 1:
        raise FileNotFoundError(
            f"Expected one Silver file for {season}: {matching_files}"
        )

    print(f"Loading {matching_files[0].name}...")

    with open(matching_files[0], "r", encoding="utf-8") as f:
        content = json.load(f)

    season_df = pd.DataFrame(content["data"])

    season_df["season_id"] = str(season)
    season_df["player_id"] = season_df["player_id"].astype(str)
    season_df["game_date"] = pd.to_datetime(
        season_df["game_date"]
    )

    silver_frames.append(season_df)

silver = pd.concat(silver_frames, ignore_index=True)


# ============================================================
# CALCULATE HISTORICAL PLAYER AVAILABILITY
# ============================================================

print("\nCalculating historical player availability...")

historical_availability = []

for _, row in df.iterrows():

    team_games = silver[
        (silver["season_id"] == row["season_id"])
        & (silver["team"] == row["team"])
        & (silver["game_date"] <= row["cutoff_date"])
    ]["game_id"].nunique()

    player_games = silver[
        (silver["season_id"] == row["season_id"])
        & (silver["player_id"] == row["player_id"])
        & (silver["game_date"] <= row["cutoff_date"])
    ]["game_id"].nunique()

    availability = (
        player_games / team_games
        if team_games > 0
        else np.nan
    )

    historical_availability.append(availability)

df["historical_availability"] = historical_availability

df["baseline_remaining_games"] = (
    df["historical_availability"]
    * df["scheduled_remaining_games"]
)

# ============================================================
# CALCULATE ACTUAL REMAINING PLAYER GAMES
# ============================================================

print("\nCalculating actual remaining player games...")

actual_games = []

for _, row in df.iterrows():

    future_games = silver[
        (silver["season_id"] == row["season_id"])
        & (silver["player_id"] == row["player_id"])
        & (silver["game_date"] > row["cutoff_date"])
    ]

    actual_games.append(
        future_games["game_id"].nunique()
    )

df["actual_remaining_games"] = actual_games

df["missed_games"] = (
    df["scheduled_remaining_games"]
    - df["actual_remaining_games"]
).clip(lower=0)

df["availability_rate"] = np.where(
    df["scheduled_remaining_games"] > 0,
    df["actual_remaining_games"]
    / df["scheduled_remaining_games"],
    np.nan
)

df["fantasy_error"] = (
    df["pred_fantasy_points"]
    - df["actual_fantasy_points"]
)


# ============================================================
# SUMMARY BY CUTOFF
# ============================================================

print("\n" + "=" * 75)
print("PLAYER AVAILABILITY ANALYSIS")
print("=" * 75)

summary = (
    df.groupby("cutoff_percentage")
    .agg(
        players=("player_id", "count"),
        avg_scheduled_games=("scheduled_remaining_games", "mean"),
        avg_actual_games=("actual_remaining_games", "mean"),
        avg_missed_games=("missed_games", "mean"),
        avg_availability=("availability_rate", "mean"),
        players_below_75pct=(
            "availability_rate",
            lambda x: (x < 0.75).sum()
        ),
        avg_absolute_fantasy_error=(
            "fantasy_error",
            lambda x: np.abs(x).mean()
        ),
    )
    .reset_index()
)

print(summary.round(3).to_string(index=False))


# ============================================================
# AVAILABILITY VS PREDICTION ERROR
# ============================================================

print("\n" + "=" * 75)
print("PREDICTION ERROR BY AVAILABILITY")
print("=" * 75)

df["availability_group"] = pd.cut(
    df["availability_rate"],
    bins=[-0.001, 0.50, 0.75, 0.90, 1.001],
    labels=[
        "50% or less",
        "50-75%",
        "75-90%",
        "90-100%",
    ],
)

availability_summary = (
    df.groupby(
        ["cutoff_percentage", "availability_group"],
        observed=True
    )
    .agg(
        players=("player_id", "count"),
        mean_actual_games=("actual_remaining_games", "mean"),
        mean_absolute_error=(
            "fantasy_error",
            lambda x: np.abs(x).mean()
        ),
        mean_signed_error=("fantasy_error", "mean"),
    )
    .reset_index()
)

print(availability_summary.round(2).to_string(index=False))


# ============================================================
# TOP 10 FALSE POSITIVES: AVAILABILITY
# ============================================================

print("\n" + "=" * 75)
print("TOP 10 FALSE POSITIVES: AVAILABILITY")
print("=" * 75)

for cutoff, group in df.groupby("cutoff_percentage"):

    group = group.copy()

    group["actual_rank"] = group[
        "actual_fantasy_points"
    ].rank(method="min", ascending=False)

    group["predicted_rank"] = group[
        "pred_fantasy_points"
    ].rank(method="min", ascending=False)

    false_positives = group[
        (group["predicted_rank"] <= 10)
        & (group["actual_rank"] > 10)
    ].sort_values("predicted_rank")

    print(f"\nCUTOFF {cutoff:.0%}")

    print(
        false_positives[
            [
                "player_name",
                "predicted_rank",
                "actual_rank",
                "scheduled_remaining_games",
                "actual_remaining_games",
                "missed_games",
                "availability_rate",
                "pred_fantasy_points",
                "actual_fantasy_points",
            ]
        ].round(2).to_string(index=False)
    )


# ============================================================
# AVAILABILITY MODEL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("AVAILABILITY BASELINE COMPARISON")
print("=" * 75)

availability_predictions_file = (
    ROOT / "models" / "ros_availability" /
    "availability_test_predictions.csv"
)

availability_predictions = pd.read_csv(
    availability_predictions_file
)

availability_predictions["season_id"] = (
    availability_predictions["season_id"].astype(str)
)

availability_predictions["player_id"] = (
    availability_predictions["player_id"].astype(str)
)

availability_predictions["cutoff_date"] = pd.to_datetime(
    availability_predictions["cutoff_date"]
)

df["season_id"] = df["season_id"].astype(str)
df["player_id"] = df["player_id"].astype(str)
df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])

comparison = df.merge(
    availability_predictions[
        [
            "player_id",
            "season_id",
            "cutoff_date",
            "predicted_remaining_games",
        ]
    ],
    on=["player_id", "season_id", "cutoff_date"],
    how="inner",
    validate="one_to_one",
)

comparison["perfect_availability_prediction"] = (
    comparison["scheduled_remaining_games"]
)

comparison["historical_availability_prediction"] = (
    comparison["baseline_remaining_games"]
)

comparison["random_forest_prediction"] = (
    comparison["predicted_remaining_games"]
)

results = []

for cutoff, group in comparison.groupby("cutoff_percentage"):

    actual = group["actual_remaining_games"]

    for model_name, prediction_column in [
        ("Perfect availability", "perfect_availability_prediction"),
        ("Historical availability", "historical_availability_prediction"),
        ("Random Forest", "random_forest_prediction"),
    ]:

        predicted = group[prediction_column].clip(lower=0)

        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        results.append({
            "cutoff": cutoff,
            "model": model_name,
            "players": len(group),
            "MAE": mae,
            "RMSE": rmse,
        })

results_df = pd.DataFrame(results)

print(results_df.round(3).to_string(index=False))

# ============================================================
# SAVE
# ============================================================

output_file = OUTPUT_DIR / "ros_player_availability_analysis.csv"

df.to_csv(output_file, index=False)

print(f"\nDetailed results saved to: {output_file}")
