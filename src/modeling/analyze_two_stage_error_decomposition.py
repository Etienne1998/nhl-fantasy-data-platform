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
# CONFIGURATION
# ============================================================

STAT_TARGETS = [
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
]


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "cutoff_percentage",
    "remaining_games",
    "pred_remaining_games",
    "scheduled_remaining_games",
    "actual_fantasy_points",
    "two_stage_fantasy_points",
]

for target in STAT_TARGETS:
    required_columns.extend([
        target,
        f"pred_{target}",
        f"pred_rate_{target}",
    ])

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ============================================================
# AVAILABILITY ERROR
# ============================================================

df["games_prediction_error"] = (
    df["pred_remaining_games"] - df["remaining_games"]
)

df["games_absolute_error"] = (
    df["games_prediction_error"].abs()
)


# ============================================================
# ACTUAL AND PREDICTED PER-GAME PRODUCTION
# ============================================================

for target in STAT_TARGETS:

    # Actual production per game.
    df[f"actual_rate_{target}"] = np.where(
        df["remaining_games"] > 0,
        df[target] / df["remaining_games"],
        0
    )

    # Difference between predicted and actual production rate.
    df[f"rate_error_{target}"] = (
        df[f"pred_rate_{target}"]
        - df[f"actual_rate_{target}"]
    )


# ============================================================
# FANTASY POINTS WITH PERFECT AVAILABILITY
# ============================================================

# Keep the model's predicted per-game production,
# but replace predicted games with actual remaining games.

for target in STAT_TARGETS:
    df[f"perfect_availability_{target}"] = (
        df["remaining_games"] * df[f"pred_rate_{target}"]
    )


# ============================================================
# CALCULATE PERFECT-AVAILABILITY FANTASY POINTS
# ============================================================

df["perfect_availability_fantasy_points"] = (
    df["perfect_availability_remaining_goals"] * 2
    + df["perfect_availability_remaining_assists"]
    + df["perfect_availability_remaining_plus_minus"]
    + df["perfect_availability_remaining_penalty_minutes"] * 0.3
    + df["perfect_availability_remaining_power_play_goals"]
    + np.maximum(
        df["perfect_availability_remaining_power_play_points"]
        - df["perfect_availability_remaining_power_play_goals"],
        0
    ) * 0.5
    + df["perfect_availability_remaining_short_handed_goals"] * 2
    + np.maximum(
        df["perfect_availability_remaining_short_handed_points"]
        - df["perfect_availability_remaining_short_handed_goals"],
        0
    )
    + df["perfect_availability_remaining_game_winning_goals"]
    + df["perfect_availability_remaining_shots_on_goal"] * 0.2
)


# ============================================================
# ERROR DECOMPOSITION
# ============================================================

df["two_stage_absolute_error"] = (
    df["two_stage_fantasy_points"]
    - df["actual_fantasy_points"]
).abs()

df["perfect_availability_absolute_error"] = (
    df["perfect_availability_fantasy_points"]
    - df["actual_fantasy_points"]
).abs()

df["availability_error_impact"] = (
    df["two_stage_absolute_error"]
    - df["perfect_availability_absolute_error"]
)

df["production_rate_error_impact"] = (
    df["perfect_availability_absolute_error"]
)


# ============================================================
# SUMMARY BY CUTOFF
# ============================================================

summary = []

for cutoff, group in df.groupby("cutoff_percentage"):

    summary.append({
        "cutoff": cutoff,
        "players": len(group),
        "games_MAE": mean_absolute_error(
            group["remaining_games"],
            group["pred_remaining_games"]
        ),
        "two_stage_fantasy_MAE": (
            group["two_stage_absolute_error"].mean()
        ),
        "perfect_availability_fantasy_MAE": (
            group["perfect_availability_absolute_error"].mean()
        ),
        "availability_error_impact": (
            group["availability_error_impact"].mean()
        ),
        "production_rate_error": (
            group["production_rate_error_impact"].mean()
        ),
    })


summary_df = pd.DataFrame(summary)


# ============================================================
# STAT-BY-STAT PRODUCTION RATE ERRORS
# ============================================================

rate_results = []

for cutoff, group in df.groupby("cutoff_percentage"):

    for target in STAT_TARGETS:

        rate_results.append({
            "cutoff": cutoff,
            "stat": target,
            "actual_avg_rate": (
                group[f"actual_rate_{target}"].mean()
            ),
            "predicted_avg_rate": (
                group[f"pred_rate_{target}"].mean()
            ),
            "rate_MAE": mean_absolute_error(
                group[f"actual_rate_{target}"],
                group[f"pred_rate_{target}"]
            ),
        })


rate_df = pd.DataFrame(rate_results)


# ============================================================
# BIGGEST AVAILABILITY ERRORS
# ============================================================

availability_errors = df.copy()

availability_errors["player_name"] = (
    availability_errors["player_id"].astype(str)
)

biggest_availability_errors = availability_errors.assign(
    games_absolute_error=availability_errors["games_absolute_error"]
).sort_values(
    "games_absolute_error",
    ascending=False
)[
    [
        "player_id",
        "cutoff_percentage",
        "remaining_games",
        "pred_remaining_games",
        "games_prediction_error",
        "games_absolute_error",
        "two_stage_absolute_error",
        "perfect_availability_absolute_error",
        "availability_error_impact",
    ]
].head(30)


# ============================================================
# SAVE RESULTS
# ============================================================

summary_df.to_csv(
    OUTPUT_DIR / "two_stage_error_decomposition_summary.csv",
    index=False
)

rate_df.to_csv(
    OUTPUT_DIR / "two_stage_per_game_rate_errors.csv",
    index=False
)

biggest_availability_errors.to_csv(
    OUTPUT_DIR / "two_stage_biggest_availability_errors.csv",
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 95)
print("TWO-STAGE ERROR DECOMPOSITION — TEST 2025-26")
print("=" * 95)

print("\nERROR DECOMPOSITION BY CUTOFF")

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nPER-GAME PRODUCTION RATE ERRORS")

print(
    rate_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nTOP 15 AVAILABILITY PREDICTION ERRORS")

print(
    biggest_availability_errors.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nFiles saved in:")
print(OUTPUT_DIR)

print("\nAnalysis completed.")