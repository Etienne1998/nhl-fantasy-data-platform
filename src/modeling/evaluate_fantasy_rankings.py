
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SILVER_FILE = (
    ROOT / "data" / "silver" /
    "skater_game_logs_20252026.json"
)

FANTASY_DIR = (
    ROOT / "data" / "modeling" / "ros" / "fantasy"
)

RANKINGS_FILE = (
    FANTASY_DIR / "fantasy_rankings_20252026.csv"
)

OUTPUT_FILE = (
    FANTASY_DIR / "fantasy_rankings_evaluation_20252026.csv"
)

TOP_N_VALUES = [10, 25, 50]


# ============================================================
# ESPN FANTASY SCORING
# ============================================================

def calculate_fantasy_points(df):

    ppa = np.maximum(
        df["power_play_points"]
        - df["power_play_goals"],
        0
    )

    sha = np.maximum(
        df["short_handed_points"]
        - df["short_handed_goals"],
        0
    )

    return (
        df["goals"] * 2
        + df["assists"]
        + df["plus_minus"]
        + df["penalty_minutes"] * 0.3
        + df["power_play_goals"]
        + ppa * 0.5
        + df["short_handed_goals"] * 2
        + sha
        + df["game_winning_goals"]
        + df["shots_on_goal"] * 0.2
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_silver_data():

    import json

    print("\nLoading Silver game logs...")

    with open(SILVER_FILE, "r", encoding="utf-8") as file:
        payload = json.load(file)

    silver = pd.DataFrame(payload["data"])

    silver["game_date"] = pd.to_datetime(
        silver["game_date"]
    )

    silver["player_id"] = (
        silver["player_id"].astype(str)
    )

    print(f"Game logs: {len(silver):,}")

    return silver


def load_rankings():

    print("\nLoading generated rankings...")

    rankings = pd.read_csv(RANKINGS_FILE)

    rankings["player_id"] = (
        rankings["player_id"].astype(str)
    )

    rankings["cutoff_date"] = pd.to_datetime(
        rankings["cutoff_date"]
    )

    print(f"Ranked players: {len(rankings)}")

    return rankings


# ============================================================
# CALCULATE ACTUAL REMAINING PRODUCTION
# ============================================================

def calculate_actual_remaining_stats(silver, cutoff_date):

    print("\nCalculating actual remaining production...")

    remaining = silver[
        silver["game_date"] > cutoff_date
    ].copy()

    if remaining.empty:
        raise ValueError(
            "No game logs found after the cutoff date."
        )

    required_columns = [
        "player_id",
        "goals",
        "assists",
        "plus_minus",
        "penalty_minutes",
        "shots_on_goal",
        "power_play_goals",
        "power_play_points",
        "short_handed_goals",
        "short_handed_points",
        "game_winning_goals"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in remaining.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing Silver columns: {missing_columns}"
        )

    aggregation = {
        "goals": "sum",
        "assists": "sum",
        "plus_minus": "sum",
        "penalty_minutes": "sum",
        "shots_on_goal": "sum",
        "power_play_goals": "sum",
        "power_play_points": "sum",
        "short_handed_goals": "sum",
        "short_handed_points": "sum",
        "game_winning_goals": "sum",
        "game_id": "nunique"
    }

    actual = (
        remaining
        .groupby("player_id")
        .agg(aggregation)
        .reset_index()
    )

    actual = actual.rename(
        columns={
            "game_id": "actual_remaining_games"
        }
    )

    actual["actual_remaining_fantasy_points"] = (
        calculate_fantasy_points(actual)
    )

    return actual


# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate_rankings(results):

    y_true = results["actual_remaining_fantasy_points"]
    y_pred = results["projected_fantasy_points"]

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    spearman = spearmanr(
        y_true,
        y_pred
    ).statistic

    print("\n" + "=" * 75)
    print("FANTASY RANKING EVALUATION")
    print("=" * 75)

    print(f"Players evaluated: {len(results)}")
    print(f"MAE:               {mae:.3f}")
    print(f"RMSE:              {rmse:.3f}")
    print(f"Spearman:          {spearman:.3f}")

    # --------------------------------------------------------
    # Top N overlap
    # --------------------------------------------------------

    print("\nTOP N RANKING OVERLAP")
    print("-" * 75)

    results = results.sort_values(
        "projected_fantasy_points",
        ascending=False
    ).reset_index(drop=True)

    results["projected_rank"] = (
        np.arange(1, len(results) + 1)
    )

    results["actual_rank"] = (
        results["actual_remaining_fantasy_points"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    for n in TOP_N_VALUES:

        projected_top = set(
            results.nlargest(
                n,
                "projected_fantasy_points"
            )["player_id"]
        )

        actual_top = set(
            results.nlargest(
                n,
                "actual_remaining_fantasy_points"
            )["player_id"]
        )

        overlap = len(projected_top & actual_top)

        print(
            f"Top {n:2d} overlap: "
            f"{overlap:2d}/{n} "
            f"({overlap / n:.1%})"
        )

    return results, mae, rmse, spearman


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("NHL FANTASY RANKING EVALUATOR")
    print("=" * 75)

    rankings = load_rankings()

    silver = load_silver_data()

    cutoff_dates = rankings["cutoff_date"].dropna().unique()

    if len(cutoff_dates) != 1:
        raise ValueError(
            "Expected exactly one cutoff date in rankings."
        )

    cutoff_date = pd.Timestamp(cutoff_dates[0])

    print(f"\nCutoff date: {cutoff_date.date()}")

    actual = calculate_actual_remaining_stats(
        silver,
        cutoff_date
    )

    # --------------------------------------------------------
    # Merge projections and actual results
    # --------------------------------------------------------

    results = rankings.merge(
        actual,
        on="player_id",
        how="left"
    )

    results["actual_remaining_games"] = (
        results["actual_remaining_games"].fillna(0)
    )

    results["actual_remaining_fantasy_points"] = (
        results["actual_remaining_fantasy_points"].fillna(0)
    )

    # --------------------------------------------------------
    # Calculate errors and actual ranking
    # --------------------------------------------------------

    results["fantasy_points_error"] = (
        results["projected_fantasy_points"]
        - results["actual_remaining_fantasy_points"]
    )

    results["absolute_fantasy_points_error"] = (
        results["fantasy_points_error"].abs()
    )

    results["games_error"] = (
        results["predicted_remaining_games"]
        - results["actual_remaining_games"]
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results, mae, rmse, spearman = evaluate_rankings(
        results
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    output_columns = [
        "projected_rank",
        "actual_rank",
        "player_id",
        "player_name",
        "team",
        "position",
        "predicted_remaining_games",
        "actual_remaining_games",
        "projected_fantasy_points",
        "actual_remaining_fantasy_points",
        "fantasy_points_error",
        "absolute_fantasy_points_error",
        "games_error"
    ]

    results = results[output_columns]

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display biggest projection errors
    # --------------------------------------------------------

    print("\nBIGGEST PROJECTION ERRORS")
    print("-" * 75)

    print(
        results.sort_values(
            "absolute_fantasy_points_error",
            ascending=False
        )[
            [
                "player_name",
                "projected_fantasy_points",
                "actual_remaining_fantasy_points",
                "fantasy_points_error"
            ]
        ]
        .head(15)
        .round(2)
        .to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)

    print(f"MAE:      {mae:.3f}")
    print(f"RMSE:     {rmse:.3f}")
    print(f"Spearman: {spearman:.3f}")

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
