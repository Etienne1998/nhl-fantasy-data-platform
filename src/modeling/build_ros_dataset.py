
import json
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SILVER_DIR = Path("data/silver")
OUTPUT_DIR = Path("data/modeling/ros")

SEASONS = [
    "20212022",
    "20222023",
    "20232024",
    "20242025",
    "20252026"
]

CUTOFF_PERCENTAGES = [0.25, 0.50, 0.75]

MIN_GAMES_PLAYED = 10


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

ROLLING_STATS = FEATURE_STATS

CUMULATIVE_STATS = {
    "goals": "season_goals",
    "assists": "season_assists",
    "points": "season_points",
    "shots_on_goal": "season_shots",
    "time_on_ice_seconds": "season_total_toi_seconds"
}


# ============================================================
# LOAD SILVER DATA
# ============================================================

def load_silver_season(season_id):

    path = SILVER_DIR / f"skater_game_logs_{season_id}.json"

    print(f"Loading Silver season {season_id}...")

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    df = pd.DataFrame(payload["data"])

    df["game_date"] = pd.to_datetime(df["game_date"])

    df = df.sort_values(
        ["player_id", "game_date", "game_id"]
    ).reset_index(drop=True)

    return df


# ============================================================
# BUILD FEATURES AT CUTOFF
# ============================================================

def build_player_features(player_history, cutoff_date):

    player_history = player_history.sort_values(
        ["game_date", "game_id"]
    ).copy()

    last_game = player_history.iloc[-1]

    features = {}

    # --------------------------------------------------------
    # 1. ROLLING FEATURES
    # Use all games played up to and including cutoff
    # --------------------------------------------------------

    for stat in ROLLING_STATS:

        for window in WINDOWS:

            feature_name = f"{stat}_last_{window}_avg"

            features[feature_name] = (
                player_history[stat]
                .tail(window)
                .mean()
            )

    # Number of previous games
    games_played = len(player_history)

    for window in WINDOWS:

        features[f"games_in_last_{window}"] = min(
            games_played,
            window
        )

    # --------------------------------------------------------
    # 2. SEASON-TO-DATE FEATURES
    # --------------------------------------------------------

    features["season_games_played"] = games_played

    for source_column, feature_name in CUMULATIVE_STATS.items():

        features[feature_name] = player_history[
            source_column
        ].sum()

    features["season_avg_toi_minutes"] = (
        features["season_total_toi_seconds"]
        / games_played
        / 60
    )

    features["season_points_per_game"] = (
        features["season_points"]
        / games_played
    )

    # --------------------------------------------------------
    # 3. TREND FEATURES
    # --------------------------------------------------------

    trend_stats = [
        "points",
        "goals",
        "assists",
        "shots_on_goal",
        "time_on_ice_seconds"
    ]

    for stat in trend_stats:

        features[f"{stat}_trend_5_vs_20"] = (
            features[f"{stat}_last_5_avg"]
            - features[f"{stat}_last_20_avg"]
        )

    # --------------------------------------------------------
    # 4. PRODUCTION PER 60 MINUTES
    # --------------------------------------------------------

    toi_last_5 = features["time_on_ice_seconds_last_5_avg"]

    per_60_stats = [
        "goals",
        "assists",
        "points",
        "shots_on_goal"
    ]

    for stat in per_60_stats:

        if toi_last_5 > 0:

            features[f"{stat}_per_60_last_5"] = (
                features[f"{stat}_last_5_avg"]
                / toi_last_5
                * 3600
            )

        else:

            features[f"{stat}_per_60_last_5"] = np.nan

    # --------------------------------------------------------
    # 5. TIME-BASED FEATURES
    # Calculate relative to cutoff, not next game
    # --------------------------------------------------------

    features["days_since_last_game"] = (
        cutoff_date - last_game["game_date"]
    ).days

    ten_days_before_cutoff = (
        cutoff_date - pd.Timedelta(days=10)
    )

    features["games_in_last_10_days"] = (
        (
            player_history["game_date"] > ten_days_before_cutoff
        )
        & (
            player_history["game_date"] <= cutoff_date
        )
    ).sum()

    return features


# ============================================================
# BUILD ROS SNAPSHOTS
# ============================================================

def build_season_snapshots(silver):

    season_id = str(silver["season_id"].iloc[0])

    game_dates = sorted(silver["game_date"].unique())

    total_game_dates = len(game_dates)

    results = []

    print(f"\nSeason: {season_id}")
    print(f"Unique game dates: {total_game_dates}")

    for percentage in CUTOFF_PERCENTAGES:

        cutoff_index = int(total_game_dates * percentage) - 1
        cutoff_index = max(0, cutoff_index)

        cutoff_date = pd.Timestamp(game_dates[cutoff_index])

        print(
            f"\nCutoff: {percentage:.0%} "
            f"({cutoff_date.date()})"
        )

        # ----------------------------------------------------
        # HISTORICAL AND REMAINING DATA
        # ----------------------------------------------------

        historical = silver[
            silver["game_date"] <= cutoff_date
        ].copy()

        remaining = silver[
            silver["game_date"] > cutoff_date
        ].copy()

        if historical.empty:
            print("Skipping: no historical data.")
            continue

        # ----------------------------------------------------
        # ELIGIBLE PLAYERS
        # ----------------------------------------------------

        historical_counts = (
            historical.groupby("player_id")
            .size()
        )

        eligible_players = historical_counts[
            historical_counts >= MIN_GAMES_PLAYED
        ].index

        historical = historical[
            historical["player_id"].isin(eligible_players)
        ].copy()

        # ----------------------------------------------------
        # REMAINING TARGETS
        # ----------------------------------------------------

        remaining_stats = (
            remaining[
                remaining["player_id"].isin(eligible_players)
            ]
            .groupby("player_id")[TARGETS]
            .sum()
            .rename(
                columns={
                    target: f"remaining_{target}"
                    for target in TARGETS
                }
            )
        )

        remaining_games = (
            remaining[
                remaining["player_id"].isin(eligible_players)
            ]
            .groupby("player_id")
            .size()
            .rename("remaining_games")
        )

        # ----------------------------------------------------
        # BUILD ONE SNAPSHOT PER ELIGIBLE PLAYER
        # ----------------------------------------------------

        snapshots = []

        for player_id, player_history in historical.groupby(
            "player_id",
            sort=False
        ):

            player_history = player_history.sort_values(
                ["game_date", "game_id"]
            )

            last_game = player_history.iloc[-1]

            features = build_player_features(
                player_history,
                cutoff_date
            )

            snapshot = {
                "player_id": player_id,
                "player_name": last_game["player_name"],
                "team": last_game["team"],
                "position": last_game["position"],
                "game_id": last_game["game_id"],
                "game_date": last_game["game_date"],
                "season_id": season_id,
                "cutoff_date": cutoff_date,
                "cutoff_percentage": percentage
            }

            snapshot.update(features)

            snapshots.append(snapshot)

        snapshot_df = pd.DataFrame(snapshots)

        snapshot_df = snapshot_df.set_index("player_id")

        # ----------------------------------------------------
        # MERGE REMAINING TARGETS
        # Include players with zero remaining games
        # ----------------------------------------------------

        snapshot_df = snapshot_df.join(
            remaining_stats,
            how="left"
        )

        snapshot_df = snapshot_df.join(
            remaining_games,
            how="left"
        )

        remaining_target_columns = [
            f"remaining_{target}"
            for target in TARGETS
        ]

        snapshot_df[remaining_target_columns] = (
            snapshot_df[remaining_target_columns].fillna(0)
        )

        snapshot_df["remaining_games"] = (
            snapshot_df["remaining_games"].fillna(0).astype(int)
        )

        snapshot_df = snapshot_df.reset_index()

        print(f"Eligible players: {len(snapshot_df)}")
        print(
            "Players with zero remaining games:",
            (snapshot_df["remaining_games"] == 0).sum()
        )
        print(
            "Average remaining games:",
            round(snapshot_df["remaining_games"].mean(), 1)
        )

        results.append(snapshot_df)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ============================================================
# MAIN
# ============================================================

def main():

    all_snapshots = []

    for season_id in SEASONS:

        silver = load_silver_season(season_id)

        season_snapshots = build_season_snapshots(silver)

        if not season_snapshots.empty:
            all_snapshots.append(season_snapshots)

    if not all_snapshots:
        print("No ROS snapshots generated.")
        return

    ros_dataset = pd.concat(
        all_snapshots,
        ignore_index=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "ros_dataset.csv"

    ros_dataset.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("ROS DATASET CREATED")
    print("=" * 60)

    print(f"Total snapshots: {len(ros_dataset)}")
    print(
        "Unique players:",
        ros_dataset["player_id"].nunique()
    )

    print("\nSnapshots by season and cutoff:")

    print(
        ros_dataset.groupby(
            ["season_id", "cutoff_percentage"]
        ).size()
    )

    print("\nRemaining games distribution:")

    print(ros_dataset["remaining_games"].describe())

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
