
import argparse
import json
import pandas as pd
from pathlib import Path

# 1. Read season argument
parser = argparse.ArgumentParser(
    description="Validate Silver NHL game logs"
)
parser.add_argument("--season", type=int, required=True)
args = parser.parse_args()

SEASON_ID = args.season
INPUT_FILE = Path(
    f"data/silver/skater_game_logs_{SEASON_ID}.json"
)

# 2. Load Silver data
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    silver_data = json.load(file)

df = pd.DataFrame(silver_data["data"])

print("========== DATA QUALITY REPORT ==========")
print(f"File loaded: {INPUT_FILE}")
print(f"Season: {silver_data['metadata']['season_id']}")
print(f"Records loaded: {len(df)}")

# 3. Missing values
print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 4. Duplicate player-game combinations
duplicates = df.duplicated(
    subset=["game_id", "player_id"]
).sum()

print(f"\nDuplicate records: {duplicates}")

# 5. Invalid negative statistics
non_negative_columns = [
    "goals", "assists", "points", "penalty_minutes",
    "shots_on_goal", "power_play_goals", "power_play_points",
    "short_handed_goals", "short_handed_points",
    "game_winning_goals", "time_on_ice_seconds"
]

print("\nNegative values:")
for column in non_negative_columns:
    count = (df[column] < 0).sum()
    if count > 0:
        print(f"{column}: {count}")

# 6. Impossible percentages
print("\nPercentage validation:")

for column in ["shooting_pct", "faceoff_win_pct"]:
    invalid = df[
        (df[column] < 0) | (df[column] > 1)
    ]

    print(f"\n{column}")
    print(f"Invalid values: {len(invalid)}")
    print(f"Min: {df[column].min()}")
    print(f"Max: {df[column].max()}")

    if not invalid.empty:
        display_columns = [
            "player_name", "game_date",
            "goals", "shots_on_goal", column
        ]
        print(invalid[display_columns].to_string(index=False))

# 7. Points consistency
inconsistent_points = (
    df["goals"] + df["assists"] != df["points"]
).sum()

print(f"\nPoints inconsistencies: {inconsistent_points}")

# 8. Games played
invalid_games = (df["games_played"] != 1).sum()

print(f"Records where games_played != 1: {invalid_games}")

# 9. Time on ice
print("\nTime on ice (minutes):")
print((df["time_on_ice_seconds"] / 60).describe())

print("\n========== VALIDATION COMPLETED ==========")
