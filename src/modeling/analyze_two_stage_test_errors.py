import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "models/ros_per_game/two_stage_test_predictions.csv"
)

OUTPUT_DIR = ROOT / "models/ros_per_game"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading test predictions...")

df = pd.read_csv(INPUT_PATH)

print(f"Records: {len(df)}")


# ============================================================
# IDENTIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "player_id",
    "season_id",
    "cutoff_date",
    "cutoff_percentage",
    "pred_fantasy_points",
    "two_stage_fantasy_points",
    "actual_fantasy_points",
    "pred_remaining_games",
    "remaining_games",
    "scheduled_remaining_games",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ============================================================
# CALCULATE ERRORS
# ============================================================

df["rf_error"] = (
    df["pred_fantasy_points"] - df["actual_fantasy_points"]
)

df["two_stage_error"] = (
    df["two_stage_fantasy_points"] - df["actual_fantasy_points"]
)

df["rf_absolute_error"] = df["rf_error"].abs()
df["two_stage_absolute_error"] = df["two_stage_error"].abs()

df["error_improvement"] = (
    df["rf_absolute_error"] - df["two_stage_absolute_error"]
)

df["prediction_difference"] = (
    df["two_stage_fantasy_points"] - df["pred_fantasy_points"]
)

df["games_prediction_error"] = (
    df["pred_remaining_games"] - df["remaining_games"]
).abs()


# ============================================================
# SUMMARY BY CUTOFF
# ============================================================

summary = []

for cutoff, group in df.groupby("cutoff_percentage"):

    summary.append({
        "cutoff": cutoff,
        "players": len(group),
        "RF_MAE": mean_absolute_error(
            group["actual_fantasy_points"],
            group["pred_fantasy_points"]
        ),
        "Two_stage_MAE": mean_absolute_error(
            group["actual_fantasy_points"],
            group["two_stage_fantasy_points"]
        ),
        "RF_better_players": (
            group["error_improvement"] < 0
        ).sum(),
        "Two_stage_better_players": (
            group["error_improvement"] > 0
        ).sum(),
        "equal_error_players": (
            group["error_improvement"] == 0
        ).sum(),
        "avg_games_prediction_error": (
            group["games_prediction_error"].mean()
        ),
    })

summary_df = pd.DataFrame(summary)


# ============================================================
# BIGGEST TWO-STAGE IMPROVEMENTS
# ============================================================

improvements = df.sort_values(
    "error_improvement",
    ascending=False
).copy()

biggest_improvements = improvements[
    [
        "player_id",
        "season_id",
        "cutoff_percentage",
        "actual_fantasy_points",
        "pred_fantasy_points",
        "two_stage_fantasy_points",
        "rf_absolute_error",
        "two_stage_absolute_error",
        "error_improvement",
        "pred_remaining_games",
        "remaining_games",
    ]
].head(25)


# ============================================================
# BIGGEST TWO-STAGE REGRESSIONS
# ============================================================

regressions = df.sort_values(
    "error_improvement",
    ascending=True
).copy()

biggest_regressions = regressions[
    [
        "player_id",
        "season_id",
        "cutoff_percentage",
        "actual_fantasy_points",
        "pred_fantasy_points",
        "two_stage_fantasy_points",
        "rf_absolute_error",
        "two_stage_absolute_error",
        "error_improvement",
        "pred_remaining_games",
        "remaining_games",
    ]
].head(25)


# ============================================================
# LARGEST MODEL DISAGREEMENTS
# ============================================================

disagreements = df.copy()

disagreements["absolute_prediction_difference"] = (
    disagreements["prediction_difference"].abs()
)

largest_disagreements = disagreements.sort_values(
    "absolute_prediction_difference",
    ascending=False
)[
    [
        "player_id",
        "season_id",
        "cutoff_percentage",
        "pred_fantasy_points",
        "two_stage_fantasy_points",
        "actual_fantasy_points",
        "prediction_difference",
        "pred_remaining_games",
        "remaining_games",
    ]
].head(25)


# ============================================================
# SAVE OUTPUTS
# ============================================================

summary_df.to_csv(
    OUTPUT_DIR / "two_stage_error_summary.csv",
    index=False
)

biggest_improvements.to_csv(
    OUTPUT_DIR / "two_stage_biggest_improvements.csv",
    index=False
)

biggest_regressions.to_csv(
    OUTPUT_DIR / "two_stage_biggest_regressions.csv",
    index=False
)

largest_disagreements.to_csv(
    OUTPUT_DIR / "two_stage_largest_disagreements.csv",
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 90)
print("TWO-STAGE ERROR ANALYSIS — TEST 2025-26")
print("=" * 90)

print("\nSUMMARY BY CUTOFF")
print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nTOP 10 BIGGEST TWO-STAGE IMPROVEMENTS")
print(biggest_improvements.head(10).to_string(index=False))

print("\nTOP 10 BIGGEST TWO-STAGE REGRESSIONS")
print(biggest_regressions.head(10).to_string(index=False))

print("\nTOP 10 LARGEST MODEL DISAGREEMENTS")
print(largest_disagreements.head(10).to_string(index=False))

print("\nFiles saved in:")
print(OUTPUT_DIR)

print("\nAnalysis completed.")