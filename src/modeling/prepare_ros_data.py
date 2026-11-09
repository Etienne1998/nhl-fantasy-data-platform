
import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/modeling/ros/ros_dataset.csv")
OUTPUT_DIR = Path("data/modeling/ros/splits")

TRAIN_SEASONS = [
    "20212022",
    "20222023",
    "20232024"
]

VALIDATION_SEASONS = [
    "20242025"
]

TEST_SEASONS = [
    "20252026"
]


# ============================================================
# FEATURES
# ============================================================

FEATURE_STATS = [
    "goals",
    "assists",
    "points",
    "plus_minus",
    "penalty_minutes",
    "shots_on_goal",
    "power_play_goals",
    "power_play_points",
    "short_handed_goals",
    "short_handed_points",
    "time_on_ice_seconds"
]

WINDOWS = [5, 10, 20]

ROLLING_FEATURES = [
    f"{stat}_last_{window}_avg"
    for stat in FEATURE_STATS
    for window in WINDOWS
]

GAME_COUNT_FEATURES = [
    "games_in_last_20"
]

SEASON_FEATURES = [
    "season_games_played",
    "season_goals",
    "season_assists",
    "season_points",
    "season_shots",
    "season_total_toi_seconds",
    "season_avg_toi_minutes",
    "season_points_per_game"
]

TREND_FEATURES = [
    "points_trend_5_vs_20",
    "goals_trend_5_vs_20",
    "assists_trend_5_vs_20",
    "shots_on_goal_trend_5_vs_20",
    "time_on_ice_seconds_trend_5_vs_20"
]

PER_60_FEATURES = [
    "goals_per_60_last_5",
    "assists_per_60_last_5",
    "points_per_60_last_5",
    "shots_on_goal_per_60_last_5"
]

TEMPORAL_FEATURES = [
    "days_since_last_game",
    "games_in_last_10_days"
]

CATEGORICAL_FEATURES = [
    "position"
]

NUMERIC_FEATURES = (
    ROLLING_FEATURES
    + GAME_COUNT_FEATURES
    + SEASON_FEATURES
    + TREND_FEATURES
    + PER_60_FEATURES
    + TEMPORAL_FEATURES
    + ["cutoff_percentage"]
)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "remaining_goals",
    "remaining_assists",
    "remaining_shots_on_goal",
    "remaining_plus_minus",
    "remaining_penalty_minutes",
    "remaining_power_play_goals",
    "remaining_power_play_points",
    "remaining_short_handed_goals",
    "remaining_short_handed_points",
    "remaining_game_winning_goals",
    "remaining_time_on_ice_seconds"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading ROS dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"season_id": str}
    )

    print(f"Total snapshots: {len(df)}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = (
        FEATURES
        + TARGETS
        + ["season_id", "player_id", "player_name",
           "team", "game_date", "cutoff_date"]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print(f"Features: {len(FEATURES)}")
    print(f"Targets: {len(TARGETS)}")

    # --------------------------------------------------------
    # Validate feature separation
    # --------------------------------------------------------

    forbidden_features = [
        col for col in FEATURES
        if col.startswith("remaining_")
        or col in [
            "player_id",
            "player_name",
            "team",
            "game_id",
            "game_date",
            "season_id",
            "cutoff_date",
            "remaining_games"
        ]
    ]

    if forbidden_features:
        raise ValueError(
            f"Forbidden features detected: {forbidden_features}"
        )

    # --------------------------------------------------------
    # Split by season
    # --------------------------------------------------------

    train = df[
        df["season_id"].isin(TRAIN_SEASONS)
    ].copy()

    validation = df[
        df["season_id"].isin(VALIDATION_SEASONS)
    ].copy()

    test = df[
        df["season_id"].isin(TEST_SEASONS)
    ].copy()

    # --------------------------------------------------------
    # Validate temporal split
    # --------------------------------------------------------

    assert set(train["season_id"]) == set(TRAIN_SEASONS)
    assert set(validation["season_id"]) == set(VALIDATION_SEASONS)
    assert set(test["season_id"]) == set(TEST_SEASONS)

    assert set(train["season_id"]).isdisjoint(validation["season_id"])
    assert set(train["season_id"]).isdisjoint(test["season_id"])
    assert set(validation["season_id"]).isdisjoint(test["season_id"])

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    validation.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "total_snapshots": len(df),
        "features_count": len(FEATURES),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURES,
        "targets": TARGETS,
        "train": {
            "seasons": TRAIN_SEASONS,
            "records": len(train)
        },
        "validation": {
            "seasons": VALIDATION_SEASONS,
            "records": len(validation)
        },
        "test": {
            "seasons": TEST_SEASONS,
            "records": len(test)
        }
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("ROS DATA PREPARATION COMPLETED")
    print("=" * 55)

    print(f"\nFeatures: {len(FEATURES)}")
    print(f"Targets: {len(TARGETS)}")

    print("\nTRAIN")
    print(f"Records: {len(train)}")
    print(f"Seasons: {sorted(train['season_id'].unique())}")

    print("\nVALIDATION")
    print(f"Records: {len(validation)}")
    print(f"Seasons: {sorted(validation['season_id'].unique())}")

    print("\nTEST")
    print(f"Records: {len(test)}")
    print(f"Seasons: {sorted(test['season_id'].unique())}")

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
