
import json
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROS_PATH = Path("data/modeling/ros/ros_dataset.csv")
SILVER_DIR = Path("data/silver")

SEASONS = [
    "20212022",
    "20222023",
    "20232024",
    "20242025",
    "20252026"
]

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

TARGETS = [
    "goals",
    "assists",
    "shots_on_goal",
    "plus_minus",
    "penalty_minutes",
    "power_play_goals",
    "power_play_points",
    "short_handed_goals",
    "short_handed_points",
    "game_winning_goals",
    "time_on_ice_seconds"
]

WINDOWS = [5, 10, 20]

SAMPLE_SIZE = 5

TOLERANCE = 1e-6


# ============================================================
# LOAD SILVER
# ============================================================

def load_silver(season_id):

    path = SILVER_DIR / f"skater_game_logs_{season_id}.json"

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    df = pd.DataFrame(payload["data"])

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["player_id"] = df["player_id"].astype(str)

    return df.sort_values(
        ["player_id", "game_date", "game_id"]
    ).reset_index(drop=True)


# ============================================================
# COMPARE VALUES
# ============================================================

def values_match(actual, expected):

    if pd.isna(actual) and pd.isna(expected):
        return True

    if pd.isna(actual) or pd.isna(expected):
        return False

    return bool(
        np.isclose(
            float(actual),
            float(expected),
            atol=TOLERANCE,
            rtol=TOLERANCE
        )
    )


# ============================================================
# REBUILD HISTORICAL FEATURES
# ============================================================


def calculate_expected_features(history, cutoff_date):

    history = history.sort_values(
        ["game_date", "game_id"]
    )

    features = {}

    games_played = len(history)

    # Rolling averages
    for stat in FEATURE_STATS:

        for window in WINDOWS:

            features[f"{stat}_last_{window}_avg"] = (
                history[stat].tail(window).mean()
            )

    # Previous game counts
    for window in WINDOWS:

        features[f"games_in_last_{window}"] = min(
            games_played,
            window
        )

    # Season-to-date statistics
    features["season_games_played"] = games_played

    features["season_goals"] = history["goals"].sum()
    features["season_assists"] = history["assists"].sum()
    features["season_points"] = history["points"].sum()
    features["season_shots"] = history["shots_on_goal"].sum()

    features["season_total_toi_seconds"] = (
        history["time_on_ice_seconds"].sum()
    )

    features["season_avg_toi_minutes"] = (
        features["season_total_toi_seconds"]
        / games_played
        / 60
    )

    features["season_points_per_game"] = (
        features["season_points"] / games_played
    )

    # Trends
    for stat in [
        "points",
        "goals",
        "assists",
        "shots_on_goal",
        "time_on_ice_seconds"
    ]:

        features[f"{stat}_trend_5_vs_20"] = (
            features[f"{stat}_last_5_avg"]
            - features[f"{stat}_last_20_avg"]
        )

    # Per 60 minutes
    toi_last_5 = features["time_on_ice_seconds_last_5_avg"]

    for stat in [
        "goals",
        "assists",
        "points",
        "shots_on_goal"
    ]:

        if toi_last_5 > 0:

            features[f"{stat}_per_60_last_5"] = (
                features[f"{stat}_last_5_avg"]
                / toi_last_5
                * 3600
            )

        else:

            features[f"{stat}_per_60_last_5"] = np.nan

    # Time-based features
    last_game_date = history["game_date"].max()

    features["days_since_last_game"] = (
        cutoff_date - last_game_date
    ).days

    ten_days_before_cutoff = (
        cutoff_date - pd.Timedelta(days=10)
    )

    features["games_in_last_10_days"] = (
        (
            history["game_date"] > ten_days_before_cutoff
        )
        & (
            history["game_date"] <= cutoff_date
        )
    ).sum()

    return features


# ============================================================
# VALIDATE REMAINING TARGETS
# ============================================================

def validate_remaining_targets(ros, silver):

    errors = []

    for cutoff_date, snapshots in ros.groupby("cutoff_date"):

        cutoff_date = pd.Timestamp(cutoff_date)

        remaining = silver[
            silver["game_date"] > cutoff_date
        ]

        expected = (
            remaining
            .groupby("player_id")[TARGETS]
            .sum()
        )

        expected_games = remaining.groupby("player_id").size()

        for _, row in snapshots.iterrows():

            player_id = row["player_id"]

            if player_id in expected.index:

                expected_stats = expected.loc[player_id]
                expected_game_count = expected_games.loc[player_id]

            else:

                expected_stats = pd.Series(
                    0,
                    index=TARGETS
                )

                expected_game_count = 0

            for target in TARGETS:

                column = f"remaining_{target}"

                if not values_match(
                    row[column],
                    expected_stats[target]
                ):

                    errors.append(
                        f"{player_id} | {cutoff_date.date()} | "
                        f"{column}: expected {expected_stats[target]}, "
                        f"got {row[column]}"
                    )

            if int(row["remaining_games"]) != int(expected_game_count):

                errors.append(
                    f"{player_id} | {cutoff_date.date()} | "
                    f"remaining_games: expected {expected_game_count}, "
                    f"got {row['remaining_games']}"
                )

    return errors


# ============================================================
# VALIDATE HISTORICAL FEATURES
# ============================================================

def validate_historical_features(ros, silver):

    errors = []

    sampled = (
        ros.groupby(
            ["season_id", "cutoff_percentage"],
            group_keys=False
        )
        .sample(
            n=SAMPLE_SIZE,
            random_state=42
        )
    )

    for _, row in sampled.iterrows():

        player_id = row["player_id"]
        cutoff_date = pd.Timestamp(row["cutoff_date"])

        history = silver[
            (silver["player_id"] == player_id)
            & (silver["game_date"] <= cutoff_date)
        ].copy()

        if history.empty:

            errors.append(
                f"{player_id} | No historical data before cutoff"
            )

            continue

        expected = calculate_expected_features(
            history,
            cutoff_date
        )

        # Last historical game metadata
        expected_last_game_date = history["game_date"].max()

        if pd.Timestamp(row["game_date"]) != expected_last_game_date:

            errors.append(
                f"{player_id} | {cutoff_date.date()} | "
                "game_date does not match last historical game"
            )

        for feature, expected_value in expected.items():

            actual_value = row[feature]

            if not values_match(actual_value, expected_value):

                errors.append(
                    f"{player_id} | {cutoff_date.date()} | "
                    f"{feature}: expected {expected_value}, "
                    f"got {actual_value}"
                )

    return errors, len(sampled)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading ROS dataset...")

    ros = pd.read_csv(ROS_PATH)

    ros["player_id"] = ros["player_id"].astype(str)
    ros["season_id"] = ros["season_id"].astype(str)

    ros["game_date"] = pd.to_datetime(ros["game_date"])
    ros["cutoff_date"] = pd.to_datetime(ros["cutoff_date"])

    all_silver = []

    for season_id in SEASONS:

        silver = load_silver(season_id)
        all_silver.append(silver)

    silver_data = pd.concat(
        all_silver,
        ignore_index=True
    )

    print("\n" + "=" * 60)
    print("ROS DATASET DEEP VALIDATION")
    print("=" * 60)

    print(f"ROS snapshots: {len(ros)}")
    print(f"Silver records: {len(silver_data)}")

    # --------------------------------------------------------
    # 1. STRUCTURAL CHECKS
    # --------------------------------------------------------

    print("\n1. STRUCTURAL CHECKS")

    duplicate_count = ros.duplicated(
        subset=["player_id", "season_id", "cutoff_date"]
    ).sum()

    print(f"Duplicate snapshots: {duplicate_count}")

    # --------------------------------------------------------
    # 2. REMAINING TARGET VALIDATION
    # --------------------------------------------------------

    print("\n2. REMAINING TARGET VALIDATION")

    target_errors = []

    for season_id in SEASONS:

        season_ros = ros[
            ros["season_id"] == season_id
        ].copy()

        season_silver = silver_data[
            silver_data["season_id"].astype(str) == season_id
        ].copy()

        target_errors.extend(
            validate_remaining_targets(
                season_ros,
                season_silver
            )
        )

    print(
        f"Target mismatches: {len(target_errors)}"
    )

    # --------------------------------------------------------
    # 3. HISTORICAL FEATURE VALIDATION
    # --------------------------------------------------------

    print("\n3. HISTORICAL FEATURE VALIDATION")

    feature_errors = []

    for season_id in SEASONS:

        season_ros = ros[
            ros["season_id"] == season_id
        ].copy()

        season_silver = silver_data[
            silver_data["season_id"].astype(str) == season_id
        ].copy()

        errors, sample_count = validate_historical_features(
            season_ros,
            season_silver
        )

        feature_errors.extend(errors)

        print(
            f"{season_id}: "
            f"{sample_count} snapshots checked"
        )

    print(
        f"Feature mismatches: {len(feature_errors)}"
    )

    # --------------------------------------------------------
    # 4. ZERO REMAINING GAMES
    # --------------------------------------------------------

    print("\n4. ZERO REMAINING GAMES")

    zero_games = ros[
        ros["remaining_games"] == 0
    ]

    print(f"Snapshots with zero games remaining: {len(zero_games)}")

    zero_target_errors = 0

    for target in TARGETS:

        column = f"remaining_{target}"

        zero_target_errors += (
            zero_games[column] != 0
        ).sum()

    print(
        f"Non-zero targets among zero-game snapshots: "
        f"{zero_target_errors}"
    )

    # --------------------------------------------------------
    # 5. FINAL REPORT
    # --------------------------------------------------------

    all_errors = (
        target_errors
        + feature_errors
    )

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if duplicate_count > 0:
        print("ERROR: Duplicate snapshots detected.")

    if zero_target_errors > 0:
        print("ERROR: Zero-game snapshots have non-zero targets.")

    if all_errors:

        print(f"Total mismatches: {len(all_errors)}")

        print("\nFirst 20 mismatches:")

        for error in all_errors[:20]:
            print("-", error)

    elif duplicate_count == 0 and zero_target_errors == 0:

        print("All validation checks passed.")

    print("\nNOTE:")
    print(
        "Remaining targets were checked across all snapshots. "
        "Historical features were checked on a deterministic sample."
    )


if __name__ == "__main__":
    main()
