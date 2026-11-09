
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SILVER_FILE = (
    ROOT / "data" / "silver" /
    "skater_game_logs_20252026.json"
)

SPLIT_DIR = ROOT / "data" / "modeling" / "ros" / "splits"

AVAILABILITY_MODEL_FILE = (
    ROOT / "models" / "ros_availability" /
    "remaining_games_model.joblib"
)

PER_GAME_MODELS_FILE = (
    ROOT / "models" / "ros_per_game" /
    "per_game_models.joblib"
)

OUTPUT_DIR = (
    ROOT / "data" / "modeling" / "ros" / "fantasy"
)

OUTPUT_FILE = OUTPUT_DIR / "fantasy_rankings_20252026.csv"

CUTOFF_PERCENTAGE = 0.25
MIN_GAMES_PLAYED = 10

FEATURE_STATS = [
    "goals", "assists", "points", "plus_minus",
    "penalty_minutes", "shots_on_goal",
    "power_play_goals", "power_play_points",
    "short_handed_goals", "short_handed_points",
    "time_on_ice_seconds"
]

WINDOWS = [5, 10, 20]

CUMULATIVE_STATS = {
    "goals": "season_goals",
    "assists": "season_assists",
    "points": "season_points",
    "shots_on_goal": "season_shots",
    "time_on_ice_seconds": "season_total_toi_seconds"
}

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
# FEATURE ENGINEERING
# ============================================================

def build_player_features(player_history, cutoff_date):

    player_history = player_history.sort_values(
        ["game_date", "game_id"]
    ).copy()

    last_game = player_history.iloc[-1]

    features = {}

    for stat in FEATURE_STATS:
        for window in WINDOWS:
            features[f"{stat}_last_{window}_avg"] = (
                player_history[stat].tail(window).mean()
            )

    games_played = len(player_history)

    for window in WINDOWS:
        features[f"games_in_last_{window}"] = min(
            games_played,
            window
        )

    features["season_games_played"] = games_played

    for source, destination in CUMULATIVE_STATS.items():
        features[destination] = player_history[source].sum()

    features["season_avg_toi_minutes"] = (
        features["season_total_toi_seconds"]
        / games_played
        / 60
    )

    features["season_points_per_game"] = (
        features["season_points"] / games_played
    )

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
# ESPN FANTASY SCORING
# ============================================================

def calculate_fantasy_points(df):

    ppa = np.maximum(
        df["pred_remaining_power_play_points"]
        - df["pred_remaining_power_play_goals"],
        0
    )

    sha = np.maximum(
        df["pred_remaining_short_handed_points"]
        - df["pred_remaining_short_handed_goals"],
        0
    )

    return (
        df["pred_remaining_goals"] * 2
        + df["pred_remaining_assists"]
        + df["pred_remaining_plus_minus"]
        + df["pred_remaining_penalty_minutes"] * 0.3
        + df["pred_remaining_power_play_goals"]
        + ppa * 0.5
        + df["pred_remaining_short_handed_goals"] * 2
        + sha
        + df["pred_remaining_game_winning_goals"]
        + df["pred_remaining_shots_on_goal"] * 0.2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NHL FANTASY RANKING GENERATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata and models
    # --------------------------------------------------------

    with open(
        SPLIT_DIR / "metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    AVAILABILITY_FEATURES = metadata["features"]

    print("\nLoading trained models...")

    availability_model = joblib.load(
        AVAILABILITY_MODEL_FILE
    )

    per_game_models = joblib.load(
        PER_GAME_MODELS_FILE
    )

    # Derive the exact feature schema from the saved models.
    first_per_game_model = next(iter(per_game_models.values()))

    if hasattr(first_per_game_model, "feature_names_in_"):
        PER_GAME_FEATURES = list(
            first_per_game_model.feature_names_in_
        )
    else:
        PER_GAME_FEATURES = list(
            first_per_game_model.named_steps[
                "preprocessor"
            ].feature_names_in_
        )

    # Confirm all per-game models use the same feature schema.
    for target, model in per_game_models.items():

        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)
        else:
            model_features = list(
                model.named_steps[
                    "preprocessor"
                ].feature_names_in_
            )

        if model_features != PER_GAME_FEATURES:
            raise ValueError(
                f"Inconsistent feature schema in model: {target}"
            )

    print(
        f"Availability features: {len(AVAILABILITY_FEATURES)}"
    )
    print(
        f"Per-game features: {len(PER_GAME_FEATURES)}"
    )

    # --------------------------------------------------------
    # Load Silver data
    # --------------------------------------------------------

    print("\nLoading Silver game logs...")

    with open(SILVER_FILE, "r", encoding="utf-8") as file:
        payload = json.load(file)

    silver = pd.DataFrame(payload["data"])

    silver["game_date"] = pd.to_datetime(
        silver["game_date"]
    )

    silver = silver.sort_values(
        ["player_id", "game_date", "game_id"]
    ).reset_index(drop=True)

    season_id = str(payload["metadata"]["season_id"])

    print(f"Season: {season_id}")
    print(f"Game logs: {len(silver):,}")

    # --------------------------------------------------------
    # Determine historical cutoff
    # --------------------------------------------------------

    game_dates = sorted(silver["game_date"].unique())

    cutoff_index = max(
        0,
        int(len(game_dates) * CUTOFF_PERCENTAGE) - 1
    )

    cutoff_date = pd.Timestamp(
        game_dates[cutoff_index]
    )

    print(f"\nCutoff: {CUTOFF_PERCENTAGE:.0%}")
    print(f"Cutoff date: {cutoff_date.date()}")

    historical = silver[
        silver["game_date"] <= cutoff_date
    ].copy()

    # --------------------------------------------------------
    # Build one row per eligible player
    # --------------------------------------------------------

    snapshots = []

    for player_id, player_history in historical.groupby(
        "player_id",
        sort=False
    ):

        if len(player_history) < MIN_GAMES_PLAYED:
            continue

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
            "season_id": season_id,
            "cutoff_date": cutoff_date,
            "cutoff_percentage": CUTOFF_PERCENTAGE
        }

        snapshot.update(features)

        snapshots.append(snapshot)

    results = pd.DataFrame(snapshots)

    if results.empty:
        raise ValueError("No eligible players found.")

    print(f"\nEligible players: {len(results)}")

    # --------------------------------------------------------
    # Validate feature schemas independently
    # --------------------------------------------------------

    missing_availability = [
        feature for feature in AVAILABILITY_FEATURES
        if feature not in results.columns
    ]

    missing_per_game = [
        feature for feature in PER_GAME_FEATURES
        if feature not in results.columns
    ]

    if missing_availability:
        raise ValueError(
            f"Missing availability features: {missing_availability}"
        )

    if missing_per_game:
        raise ValueError(
            f"Missing per-game features: {missing_per_game}"
        )

    # Separate feature matrices for the two model types.
    X_availability = results[AVAILABILITY_FEATURES]
    X_per_game = results[PER_GAME_FEATURES]

    # --------------------------------------------------------
    # Predict remaining games
    # --------------------------------------------------------

    print("\nPredicting remaining games...")

    predicted_games = availability_model.predict(
        X_availability
    )

    results["predicted_remaining_games"] = np.maximum(
        predicted_games,
        0
    )

    # --------------------------------------------------------
    # Predict per-game production
    # --------------------------------------------------------

    print("\nPredicting per-game production...")

    for target in TARGETS:

        model = per_game_models[target]

        predicted_rate = model.predict(
            X_per_game
        )

        if target != "remaining_plus_minus":
            predicted_rate = np.maximum(
                predicted_rate,
                0
            )

        results[f"pred_rate_{target}"] = predicted_rate

        results[f"pred_{target}"] = (
            results["predicted_remaining_games"]
            * predicted_rate
        )

    # --------------------------------------------------------
    # Calculate Fantasy points
    # --------------------------------------------------------

    results["projected_fantasy_points"] = (
        calculate_fantasy_points(results)
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    results = results.sort_values(
        "projected_fantasy_points",
        ascending=False
    ).reset_index(drop=True)

    results.insert(
        0,
        "rank",
        range(1, len(results) + 1)
    )

    # --------------------------------------------------------
    # Select and order output columns
    # --------------------------------------------------------

    output_columns = [
        "rank",
        "player_id",
        "player_name",
        "team",
        "position",
        "season_id",
        "cutoff_date",
        "cutoff_percentage",
        "predicted_remaining_games",
    ]

    for target in TARGETS:
        output_columns.extend([
            f"pred_rate_{target}",
            f"pred_{target}"
        ])

    output_columns.append("projected_fantasy_points")

    results = results[output_columns]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FANTASY RANKINGS GENERATED")
    print("=" * 70)

    print(f"Season: {season_id}")
    print(f"Cutoff: {cutoff_date.date()}")
    print(f"Players ranked: {len(results)}")

    print("\nTOP 20 PLAYERS")

    print(
        results[
            [
                "rank",
                "player_name",
                "team",
                "position",
                "predicted_remaining_games",
                "projected_fantasy_points"
            ]
        ]
        .head(20)
        .round(2)
        .to_string(index=False)
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
