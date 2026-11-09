
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "modeling" / "ros" / "splits"

PREDICTIONS_FILE = (
    ROOT / "data" / "modeling" / "ros" /
    "fantasy" / "ros_fantasy_player_predictions.csv"
)

SILVER_DIR = ROOT / "data" / "silver"

OUTPUT_DIR = ROOT / "data" / "modeling" / "ros" / "fantasy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

TOP_N_VALUES = [10, 25, 50]


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation dataset...")

validation = pd.read_csv(SPLIT_DIR / "validation.csv")
predictions = pd.read_csv(PREDICTIONS_FILE)

validation["cutoff_date"] = pd.to_datetime(validation["cutoff_date"])
predictions["cutoff_date"] = pd.to_datetime(predictions["cutoff_date"])

print(f"Validation records: {len(validation)}")


# ============================================================
# LOAD SILVER SEASONS
# ============================================================

def load_silver_seasons():

    silver_frames = []

    seasons = validation["season_id"].astype(str).unique()

    for season in seasons:

        print(f"Loading Silver season {season}...")

        matching_files = list(
            SILVER_DIR.glob(f"**/*{season}*.json")
        )

        if len(matching_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one Silver JSON for {season}, "
                f"found {len(matching_files)}: {matching_files}"
            )

        with open(matching_files[0], "r", encoding="utf-8") as f:
            content = json.load(f)

        frame = pd.DataFrame(content["data"])

        frame["season_id"] = str(season)
        frame["game_date"] = pd.to_datetime(frame["game_date"])

        silver_frames.append(frame)

    return pd.concat(silver_frames, ignore_index=True)


silver = load_silver_seasons()


# ============================================================
# BUILD TEAM SCHEDULE
# ============================================================

print("\nBuilding team schedules...")

team_schedule = (
    silver[
        ["season_id", "team", "game_id", "game_date"]
    ]
    .drop_duplicates()
    .dropna(subset=["team", "game_id", "game_date"])
)

print(f"Unique team-game records: {len(team_schedule)}")


# ============================================================
# COUNT REMAINING TEAM GAMES
# ============================================================

print("\nCalculating remaining games for each player...")

schedule_counts = []

for _, row in validation.iterrows():

    season = str(row["season_id"])
    team = row["team"]
    cutoff = row["cutoff_date"]

    remaining_games = team_schedule[
        (team_schedule["season_id"] == season)
        & (team_schedule["team"] == team)
        & (team_schedule["game_date"] > cutoff)
    ]["game_id"].nunique()

    schedule_counts.append(remaining_games)

validation["scheduled_remaining_games"] = schedule_counts

print(
    validation["scheduled_remaining_games"]
    .describe()
    .round(2)
    .to_string()
)


# ============================================================
# BUILD NAIVE SCHEDULE-BASED PROJECTIONS
# ============================================================

baseline = validation[
    [
        "player_id",
        "cutoff_percentage",
        "scheduled_remaining_games",
    ]
].copy()

for target, feature in ROLLING_MAPPING.items():

    baseline[target] = (
        validation[feature].fillna(0)
        * baseline["scheduled_remaining_games"]
    )

# Estimate game-winning goals using a rate learned from training.
train = pd.read_csv(SPLIT_DIR / "train.csv")

gwg_rate = (
    train["remaining_game_winning_goals"].sum()
    / max(train["remaining_goals"].sum(), 1)
)

baseline["remaining_game_winning_goals"] = (
    baseline["remaining_goals"] * gwg_rate
)


# ============================================================
# CALCULATE FANTASY POINTS
# ============================================================

def calculate_fantasy_points(df):

    goals = df["remaining_goals"]
    assists = df["remaining_assists"]
    plus_minus = df["remaining_plus_minus"]
    pim = df["remaining_penalty_minutes"]

    ppg = df["remaining_power_play_goals"]
    ppp = df["remaining_power_play_points"]

    shg = df["remaining_short_handed_goals"]
    shp = df["remaining_short_handed_points"]

    gwg = df["remaining_game_winning_goals"]
    shots = df["remaining_shots_on_goal"]

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


baseline["schedule_baseline_fantasy_points"] = (
    calculate_fantasy_points(baseline)
)


# ============================================================
# JOIN RANDOM FOREST PREDICTIONS
# ============================================================

comparison = predictions[
    [
        "player_id",
        "cutoff_percentage",
        "actual_fantasy_points",
        "pred_fantasy_points",
    ]
].merge(
    baseline[
        [
            "player_id",
            "cutoff_percentage",
            "scheduled_remaining_games",
            "schedule_baseline_fantasy_points",
        ]
    ],
    on=["player_id", "cutoff_percentage"],
    how="inner",
    validate="one_to_one",
)

print(f"\nCompared records: {len(comparison)}")


# ============================================================
# EVALUATE
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
        ("schedule_baseline", "schedule_baseline_fantasy_points"),
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

            row[f"{model_name}_top_{n}"] = len(
                actual_top.intersection(predicted_top)
            )

    results.append(row)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "ros_schedule_baseline_comparison.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "ros_schedule_baseline_predictions.csv",
    index=False
)

print("\n" + "=" * 75)
print("RANDOM FOREST VS SCHEDULE-BASED BASELINE")
print("=" * 75)

print(results_df.round(3).to_string(index=False))

print("\nResults saved to:")
print(OUTPUT_DIR / "ros_schedule_baseline_comparison.csv")
