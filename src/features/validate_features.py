
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURES_FILE = (
    BASE_DIR / "data" / "features" / "skater_game_features.json"
)

with open(FEATURES_FILE, "r", encoding="utf-8") as file:
    payload = json.load(file)

df = pd.DataFrame(payload["data"])

df["game_date"] = pd.to_datetime(df["game_date"])

print("\n========== FEATURE VALIDATION ==========")

print("\nDataset:")
print(f"Records: {len(df)}")
print(f"Players: {df['player_id'].nunique()}")
print(f"Seasons: {df['season_id'].nunique()}")
print(f"Columns: {len(df.columns)}")

print("\nMissing values in rolling features:")

rolling_columns = [
    col for col in df.columns if "_last_" in col
]

print(df[rolling_columns].isna().sum().sort_values(ascending=False).head(15))

print("\nFirst game per player:")

first_games = (
    df.sort_values(["player_id", "game_date", "game_id"])
    .groupby("player_id")
    .first()
)

print(
    "Players whose first-game rolling points are null:",
    first_games["points_last_5_avg"].isna().sum()
)

print("\nRolling window sample:")

player_id = df["player_id"].value_counts().index[0]

sample = (
    df[df["player_id"] == player_id]
    .sort_values(["game_date", "game_id"])
    .head(8)
)

print(
    sample[
        [
            "player_name",
            "game_date",
            "goals",
            "points",
            "points_last_5_avg",
            "games_in_last_5",
        ]
    ].to_string(index=False)
)

print("\n========================================")
print("VALIDATION COMPLETED")
print("========================================")

# --------------------------------------------------
# SEASON-TO-DATE FEATURE VALIDATION
# --------------------------------------------------

print("\n========== SEASON-TO-DATE VALIDATION ==========")

season_features = [
    "season_games_played",
    "season_goals",
    "season_assists",
    "season_points",
    "season_shots",
    "season_avg_toi_minutes",
    "season_points_per_game",
]

print("\nMissing values:")
print(df[season_features].isna().sum())

print("\nFirst game of each player-season:")

first_games = (
    df.sort_values(["player_id", "season_id", "game_date", "game_id"])
      .groupby(["player_id", "season_id"], sort=False)
      .head(1)
)

print(
    first_games[
        ["player_name", "season_id", "season_games_played",
         "season_goals", "season_assists", "season_points",
         "season_shots"]
    ].head(10).to_string(index=False)
)

print("\nSeason-to-date sample: Connor McDavid")

mcdavid = (
    df[df["player_name"] == "Connor McDavid"]
      .sort_values(["season_id", "game_date", "game_id"])
)

print(
    mcdavid[
        ["game_date", "season_id", "goals", "season_games_played",
         "season_goals", "season_assists", "season_points",
         "season_points_per_game"]
    ].tail(10).to_string(index=False)
)

# Check that first game of each player-season starts at zero
assert (first_games["season_games_played"] == 0).all()
assert (first_games["season_goals"] == 0).all()
assert (first_games["season_assists"] == 0).all()
assert (first_games["season_points"] == 0).all()
assert (first_games["season_shots"] == 0).all()

print("\nSeason-to-date checks passed!")
