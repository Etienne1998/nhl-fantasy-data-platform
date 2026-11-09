import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


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

print(f"Records loaded: {len(df)}")


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "player_id",
    "cutoff_percentage",
    "season_games_played",
    "remaining_games",
    "pred_remaining_games",
    "scheduled_remaining_games",
    "actual_fantasy_points",
    "two_stage_fantasy_points",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ============================================================
# CREATE EXPERIENCE GROUPS
# ============================================================

bins = [-1, 10, 25, 40, np.inf]

labels = [
    "0-10 games",
    "11-25 games",
    "26-40 games",
    "41+ games",
]

df["experience_group"] = pd.cut(
    df["season_games_played"],
    bins=bins,
    labels=labels
)


# ============================================================
# CALCULATE ERRORS
# ============================================================

df["games_error"] = (
    df["pred_remaining_games"] - df["remaining_games"]
)

df["games_absolute_error"] = df["games_error"].abs()

df["fantasy_absolute_error"] = (
    df["two_stage_fantasy_points"]
    - df["actual_fantasy_points"]
).abs()

df["actual_availability"] = np.where(
    df["scheduled_remaining_games"] > 0,
    df["remaining_games"] / df["scheduled_remaining_games"],
    np.nan
)

df["predicted_availability"] = np.where(
    df["scheduled_remaining_games"] > 0,
    df["pred_remaining_games"] / df["scheduled_remaining_games"],
    np.nan
)


# ============================================================
# ANALYZE BY CUTOFF AND EXPERIENCE
# ============================================================

results = []

for cutoff in sorted(df["cutoff_percentage"].unique()):

    cutoff_df = df[df["cutoff_percentage"] == cutoff]

    for group_name in labels:

        group = cutoff_df[
            cutoff_df["experience_group"] == group_name
        ].copy()

        if len(group) == 0:
            continue

        results.append({
            "cutoff": cutoff,
            "experience_group": group_name,
            "players": len(group),
            "avg_games_played": group["season_games_played"].mean(),
            "avg_scheduled_remaining": (
                group["scheduled_remaining_games"].mean()
            ),
            "actual_remaining_games": (
                group["remaining_games"].mean()
            ),
            "pred_remaining_games": (
                group["pred_remaining_games"].mean()
            ),
            "games_MAE": mean_absolute_error(
                group["remaining_games"],
                group["pred_remaining_games"]
            ),
            "games_RMSE": np.sqrt(
                mean_squared_error(
                    group["remaining_games"],
                    group["pred_remaining_games"]
                )
            ),
            "actual_availability": (
                group["actual_availability"].mean()
            ),
            "predicted_availability": (
                group["predicted_availability"].mean()
            ),
            "fantasy_MAE": mean_absolute_error(
                group["actual_fantasy_points"],
                group["two_stage_fantasy_points"]
            ),
        })


results_df = pd.DataFrame(results)


# ============================================================
# ANALYZE LOW-EXPERIENCE PLAYERS
# ============================================================

low_experience = df[
    df["season_games_played"] <= 10
].copy()

low_experience_summary = (
    low_experience.groupby("cutoff_percentage", observed=True)
    .agg(
        players=("player_id", "count"),
        avg_games_played=("season_games_played", "mean"),
        avg_actual_remaining=("remaining_games", "mean"),
        avg_pred_remaining=("pred_remaining_games", "mean"),
        avg_games_error=("games_error", "mean"),
        avg_absolute_games_error=("games_absolute_error", "mean"),
        actual_availability=("actual_availability", "mean"),
        predicted_availability=("predicted_availability", "mean"),
    )
    .reset_index()
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_DIR / "availability_by_experience.csv"
)

low_experience_path = (
    OUTPUT_DIR / "low_experience_availability.csv"
)

results_df.to_csv(results_path, index=False)
low_experience_summary.to_csv(
    low_experience_path,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 100)
print("AVAILABILITY MODEL — PERFORMANCE BY PLAYER EXPERIENCE")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\n" + "=" * 100)
print("LOW-EXPERIENCE PLAYERS — 10 GAMES OR LESS")
print("=" * 100)

print(
    low_experience_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nFiles saved:")
print(results_path)
print(low_experience_path)

print("\nAnalysis completed.")