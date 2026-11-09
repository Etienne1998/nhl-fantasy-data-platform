
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEASONS = [
    "20212022",
    "20222023",
    "20232024",
    "20242025",
    "20252026",
]

WINDOWS = [5, 10, 20]

STATS = [
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
    "time_on_ice_seconds",
]


def load_season_data(season):
    file_path = Path(f"data/silver/skater_game_logs_{season}.json")

    with open(file_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    df = pd.DataFrame(payload["data"])

    print(f"Loaded season {season}: {len(df)} records")

    return df


def build_features():
    print("Loading historical data...")

    dataframes = [
        load_season_data(season)
        for season in SEASONS
    ]

    df = pd.concat(dataframes, ignore_index=True)

    df["game_date"] = pd.to_datetime(df["game_date"])

    df = df.sort_values(
        ["player_id", "game_date", "game_id"]
    ).reset_index(drop=True)

    print(f"\nTotal records: {len(df)}")

    # --------------------------------------------------
    # 1. ROLLING FEATURES
    # Previous 5, 10 and 20 games
    # --------------------------------------------------

    print("\nBuilding rolling features...")

    player_groups = df.groupby("player_id", sort=False)

    for stat in STATS:
        for window in WINDOWS:
            feature_name = f"{stat}_last_{window}_avg"

            df[feature_name] = player_groups[stat].transform(
                lambda series: (
                    series.shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            )

    # Number of previous games in each window
    for window in WINDOWS:
        feature_name = f"games_in_last_{window}"

        df[feature_name] = player_groups["game_id"].transform(
            lambda series: (
                series.shift(1)
                .rolling(window=window, min_periods=1)
                .count()
            )
        )

    # Days since previous game
    df["days_since_last_game"] = (
        player_groups["game_date"].diff().dt.days
    )

    # --------------------------------------------------
    # 2. SEASON-TO-DATE FEATURES
    # Only information available before the current game
    # --------------------------------------------------

    print("Building season-to-date features...")

    season_groups = df.groupby(
        ["player_id", "season_id"],
        sort=False
    )

    df["season_games_played"] = season_groups.cumcount()

    cumulative_stats = {
        "goals": "season_goals",
        "assists": "season_assists",
        "points": "season_points",
        "shots_on_goal": "season_shots",
        "time_on_ice_seconds": "season_total_toi_seconds",
    }

    for source_column, feature_name in cumulative_stats.items():
        df[feature_name] = season_groups[source_column].transform(
            lambda series: (
                series.shift(1)
                .fillna(0)
                .cumsum()
            )
        )

    df["season_avg_toi_minutes"] = (
        df["season_total_toi_seconds"]
        / df["season_games_played"]
        / 60
    )

    df["season_points_per_game"] = (
        df["season_points"]
        / df["season_games_played"]
    )

    # --------------------------------------------------
    # 3. NEW FEATURE ENGINEERING
    # --------------------------------------------------

    print("Building additional features...")

    # Recent trends: last 5 games vs last 20 games
    trend_stats = [
        "points",
        "goals",
        "assists",
        "shots_on_goal",
        "time_on_ice_seconds",
    ]

    new_features = []

    for stat in trend_stats:
        feature_name = f"{stat}_trend_5_vs_20"

        df[feature_name] = (
            df[f"{stat}_last_5_avg"]
            - df[f"{stat}_last_20_avg"]
        )

        new_features.append(feature_name)

    # Production per 60 minutes over the previous 5 games
    toi_last_5 = df["time_on_ice_seconds_last_5_avg"]

    per_60_stats = [
        "goals",
        "assists",
        "points",
        "shots_on_goal",
    ]

    for stat in per_60_stats:
        feature_name = f"{stat}_per_60_last_5"

        df[feature_name] = (
            df[f"{stat}_last_5_avg"]
            / toi_last_5.replace(0, np.nan)
            * 3600
        )

        new_features.append(feature_name)

    # Number of games played in the previous 10 calendar days
    df["games_in_last_10_days"] = 0

    for _, group in df.groupby("player_id", sort=False):
        dates = group["game_date"].to_numpy()

        left_boundaries = np.searchsorted(
            dates,
            dates - np.timedelta64(10, "D"),
            side="left"
        )

        previous_games = (
            np.arange(len(group)) - left_boundaries
        )

        df.loc[group.index, "games_in_last_10_days"] = previous_games

    new_features.append("games_in_last_10_days")

    # --------------------------------------------------
    # 4. FEATURE METADATA
    # --------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if (
            "_last_" in column
            or column.startswith("games_in_last_")
            or column == "days_since_last_game"
            or column.startswith("season_")
        )
    ]

    feature_columns.extend(new_features)

    feature_columns = list(dict.fromkeys(feature_columns))

    print(f"\nOriginal features: 45")
    print(f"New features: {len(new_features)}")
    print(f"Total engineered features: {len(feature_columns)}")

    # --------------------------------------------------
    # 5. EXPORT
    # --------------------------------------------------

    output_path = Path("data/features/skater_game_features.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")

    df = df.astype(object).where(pd.notna(df), None)

    records = df.to_dict(orient="records")

    metadata = {
        "source": "NHL historical Silver data",
        "seasons": SEASONS,
        "total_records": len(df),
        "rolling_windows": WINDOWS,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "new_features": new_features,
        "leakage_prevention": (
            "All rolling, trend, per-60 and season-to-date features "
            "use information available before the current game."
        ),
    }

    output = {
        "metadata": metadata,
        "data": records,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 50)
    print(f"Records: {len(df)}")
    print(f"Players: {df['player_id'].nunique()}")
    print(f"Seasons: {df['season_id'].nunique()}")
    print(f"Features: {len(feature_columns)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    build_features()
