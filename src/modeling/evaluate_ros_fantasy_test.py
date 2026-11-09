
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "modeling" / "ros" / "splits"
MODEL_DIR = ROOT / "models" / "ros_random_forest"
SILVER_DIR = ROOT / "data" / "silver"

OUTPUT_DIR = ROOT / "data" / "modeling" / "ros" / "fantasy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_VALUES = [10, 25, 50]

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
# LOAD TEST DATA AND FEATURE DEFINITIONS
# ============================================================

print("Loading test dataset...")

test = pd.read_csv(SPLIT_DIR / "test.csv")
train = pd.read_csv(SPLIT_DIR / "train.csv")

test["cutoff_date"] = pd.to_datetime(test["cutoff_date"])
test["season_id"] = test["season_id"].astype(str)

with open(SPLIT_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

print(f"Test records: {len(test)}")
print(f"Test seasons: {test['season_id'].unique()}")


# ============================================================
# LOAD RANDOM FOREST MODELS
# ============================================================

print("\nGenerating Random Forest predictions...")

results = test[
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

for target in TARGETS:

    print(f"Predicting {target}...")

    model = joblib.load(MODEL_DIR / f"{target}_model.joblib")

    predicted = model.predict(test[FEATURES])

    if target != "remaining_plus_minus":
        predicted = np.maximum(predicted, 0)

    results[f"pred_{target}"] = predicted
    results[f"actual_{target}"] = test[target].to_numpy()


# ============================================================
# CALCULATE RANDOM FOREST FANTASY POINTS
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


results["pred_fantasy_points"] = calculate_fantasy_points(
    results, "pred_"
)

results["actual_fantasy_points"] = calculate_fantasy_points(
    results, "actual_"
)


# ============================================================
# LOAD SILVER AND BUILD TEAM SCHEDULE
# ============================================================

print("\nLoading Silver season data...")

seasons = test["season_id"].unique()
silver_frames = []

for season in seasons:

    matching_files = list(
        SILVER_DIR.glob(f"**/*{season}*.json")
    )

    if len(matching_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Silver JSON for {season}. "
            f"Found: {matching_files}"
        )

    print(f"Loading {matching_files[0].name}...")

    with open(matching_files[0], "r", encoding="utf-8") as f:
        content = json.load(f)

    frame = pd.DataFrame(content["data"])

    frame["season_id"] = str(season)
    frame["game_date"] = pd.to_datetime(frame["game_date"])

    silver_frames.append(frame)

silver = pd.concat(silver_frames, ignore_index=True)


print("\nBuilding team schedules...")

team_schedule = (
    silver[
        ["season_id", "team", "game_id", "game_date"]
    ]
    .drop_duplicates()
    .dropna(subset=["team", "game_id", "game_date"])
)


# ============================================================
# CALCULATE REMAINING TEAM GAMES AT EACH CUTOFF
# ============================================================

print("\nCalculating remaining team games...")

schedule_counts = []

for _, row in results.iterrows():

    remaining_games = team_schedule[
        (team_schedule["season_id"] == row["season_id"])
        & (team_schedule["team"] == row["team"])
        & (team_schedule["game_date"] > row["cutoff_date"])
    ]["game_id"].nunique()

    schedule_counts.append(remaining_games)

results["scheduled_remaining_games"] = schedule_counts

print(
    results["scheduled_remaining_games"]
    .describe()
    .round(2)
    .to_string()
)


# ============================================================
# BUILD SCHEDULE-BASED BASELINE
# ============================================================

print("\nBuilding schedule-based baseline...")

for target, feature in ROLLING_MAPPING.items():

    results[f"baseline_{target}"] = (
        test[feature].fillna(0).to_numpy()
        * results["scheduled_remaining_games"]
    )

# GWG rate learned exclusively from training data.
gwg_rate = (
    train["remaining_game_winning_goals"].sum()
    / max(train["remaining_goals"].sum(), 1)
)

results["baseline_remaining_game_winning_goals"] = (
    results["baseline_remaining_goals"] * gwg_rate
)

print(f"Training GWG / goals rate: {gwg_rate:.4f}")

# Calculate baseline fantasy points.
baseline_columns = {
    target: f"baseline_{target}"
    for target in TARGETS
}

baseline_df = results.rename(
    columns=baseline_columns
)

results["baseline_fantasy_points"] = (
    calculate_fantasy_points(baseline_df, "baseline_")
)


# ============================================================
# EVALUATE RANDOM FOREST VS BASELINE
# ============================================================

print("\nEvaluating rankings...")

evaluation = []

for cutoff, group in results.groupby("cutoff_percentage"):

    actual = group["actual_fantasy_points"]

    row = {
        "cutoff": cutoff,
        "players": len(group),
    }

    for model_name, prediction_column in [
        ("random_forest", "pred_fantasy_points"),
        ("schedule_baseline", "baseline_fantasy_points"),
    ]:

        predicted = group[prediction_column]

        row[f"{model_name}_spearman"] = actual.corr(
            predicted,
            method="spearman"
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

    evaluation.append(row)


evaluation_df = pd.DataFrame(evaluation)


# ============================================================
# SAVE RESULTS
# ============================================================

evaluation_df.to_csv(
    OUTPUT_DIR / "ros_fantasy_test_results.csv",
    index=False
)

results.to_csv(
    OUTPUT_DIR / "ros_fantasy_test_predictions.csv",
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST: RANDOM FOREST VS SCHEDULE-BASED BASELINE")
print("=" * 75)

print(evaluation_df.round(3).to_string(index=False))

print("\nFiles saved:")
print(OUTPUT_DIR / "ros_fantasy_test_results.csv")
print(OUTPUT_DIR / "ros_fantasy_test_predictions.csv")
