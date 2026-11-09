import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT / "data/modeling/ros/splits/test.csv"
METADATA_PATH = ROOT / "data/modeling/ros/splits/metadata.json"

DIRECT_PREDICTIONS_PATH = (
    ROOT / "data/modeling/ros/fantasy/ros_fantasy_test_predictions.csv"
)

AVAILABILITY_MODEL_PATH = (
    ROOT / "models/ros_availability/remaining_games_model.joblib"
)

PER_GAME_MODELS_PATH = (
    ROOT / "models/ros_per_game/per_game_models.joblib"
)

OUTPUT_DIR = ROOT / "models/ros_per_game"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

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
    "remaining_games",
]

KEYS = [
    "player_id",
    "season_id",
    "cutoff_date",
    "cutoff_percentage",
]


# ============================================================
# FANTASY SCORING
# ============================================================

def calculate_fantasy_points(df, prefix):
    return (
        df[f"{prefix}remaining_goals"] * 2
        + df[f"{prefix}remaining_assists"]
        + df[f"{prefix}remaining_plus_minus"]
        + df[f"{prefix}remaining_penalty_minutes"] * 0.3
        + df[f"{prefix}remaining_power_play_goals"]
        + (
            np.maximum(
                df[f"{prefix}remaining_power_play_points"]
                - df[f"{prefix}remaining_power_play_goals"],
                0
            ) * 0.5
        )
        + df[f"{prefix}remaining_short_handed_goals"] * 2
        + np.maximum(
            df[f"{prefix}remaining_short_handed_points"]
            - df[f"{prefix}remaining_short_handed_goals"],
            0
        )
        + df[f"{prefix}remaining_game_winning_goals"]
        + df[f"{prefix}remaining_shots_on_goal"] * 0.2
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(df, prediction_column, model_name):
    results = []

    for cutoff in sorted(df["cutoff_percentage"].unique()):
        subset = df[df["cutoff_percentage"] == cutoff].copy()

        actual = subset["actual_fantasy_points"]
        predicted = subset[prediction_column]

        actual_rank = actual.rank(ascending=False, method="min")
        predicted_rank = predicted.rank(ascending=False, method="min")

        spearman = spearmanr(actual, predicted).statistic

        results.append({
            "cutoff": cutoff,
            "model": model_name,
            "players": len(subset),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "spearman": spearman,
            "top10_overlap": (
                (actual_rank <= 10) & (predicted_rank <= 10)
            ).sum(),
            "top25_overlap": (
                (actual_rank <= 25) & (predicted_rank <= 25)
            ).sum(),
            "top50_overlap": (
                (actual_rank <= 50) & (predicted_rank <= 50)
            ).sum(),
        })

    return results


# ============================================================
# LOAD DATA
# ============================================================

print("Loading test dataset...")

test = pd.read_csv(TEST_PATH)
direct = pd.read_csv(DIRECT_PREDICTIONS_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

features = metadata["features"]

print(f"Test records: {len(test)}")
print(f"Features: {len(features)}")


# ============================================================
# NORMALIZE MERGE KEYS
# ============================================================

for df in [test, direct]:
    df["player_id"] = df["player_id"].astype(str)
    df["season_id"] = df["season_id"].astype(str)
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])
    df["cutoff_percentage"] = df["cutoff_percentage"].astype(float)


# ============================================================
# MERGE SCHEDULE AND DIRECT MODEL RESULTS
# ============================================================

print("Merging schedule and direct model predictions...")

required_direct_columns = KEYS + [
    "scheduled_remaining_games",
    "pred_fantasy_points",
    "actual_fantasy_points",
]

missing = [
    col for col in required_direct_columns
    if col not in direct.columns
]

if missing:
    raise ValueError(
        f"Missing columns in direct predictions file: {missing}"
    )

df = test.merge(
    direct[required_direct_columns],
    on=KEYS,
    how="inner",
    validate="one_to_one"
)

if len(df) != len(test):
    raise ValueError(
        f"Merge mismatch: {len(test)} test rows vs {len(df)} merged rows."
    )

print(f"Merged records: {len(df)}")


# ============================================================
# LOAD SAVED MODELS
# ============================================================

print("Loading saved models...")

availability_model = joblib.load(AVAILABILITY_MODEL_PATH)
per_game_models = joblib.load(PER_GAME_MODELS_PATH)


# ============================================================
# STAGE 1: PREDICT REMAINING GAMES
# ============================================================

print("Predicting remaining games...")

predicted_games = availability_model.predict(df[features])

# Do not allow negative games or more games than the schedule allows.
df["pred_remaining_games"] = np.clip(
    predicted_games,
    0,
    df["scheduled_remaining_games"]
)


# ============================================================
# STAGE 2: PREDICT PER-GAME STATISTICS
# ============================================================

print("Predicting per-game statistics...")

for target in TARGETS:
    if target == "remaining_games":
        continue

    model = per_game_models[target]

    predicted_rate = model.predict(df[features])

    # All statistics except plus/minus must be non-negative.
    if target != "remaining_plus_minus":
        predicted_rate = np.maximum(predicted_rate, 0)

    df[f"pred_rate_{target}"] = predicted_rate

    df[f"pred_{target}"] = (
        df["pred_remaining_games"] * predicted_rate
    )


# ============================================================
# CALCULATE TWO-STAGE FANTASY POINTS
# ============================================================

print("Calculating fantasy points...")

df["two_stage_fantasy_points"] = calculate_fantasy_points(
    df,
    prefix="pred_"
)


# ============================================================
# EVALUATE
# ============================================================

print("Evaluating models...")

evaluation_results = []

evaluation_results.extend(
    evaluate_predictions(
        df,
        "pred_fantasy_points",
        "Existing Random Forest"
    )
)

evaluation_results.extend(
    evaluate_predictions(
        df,
        "two_stage_fantasy_points",
        "Two-stage model"
    )
)

evaluation = pd.DataFrame(evaluation_results)


# ============================================================
# SAVE RESULTS
# ============================================================

predictions_path = OUTPUT_DIR / "two_stage_test_predictions.csv"
evaluation_path = OUTPUT_DIR / "two_stage_test_evaluation.csv"

df.to_csv(predictions_path, index=False)
evaluation.to_csv(evaluation_path, index=False)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 85)
print("TWO-STAGE MODEL — FINAL TEST RESULTS (2025-26)")
print("=" * 85)

print(
    evaluation.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nSchedule cap verification:")
print(
    f"Predicted games above schedule: "
    f"{(df['pred_remaining_games'] > df['scheduled_remaining_games']).sum()}"
)

print(
    f"Negative predicted games: "
    f"{(df['pred_remaining_games'] < 0).sum()}"
)

print("\nFiles saved:")
print(predictions_path)
print(evaluation_path)

print("\nEvaluation completed.")