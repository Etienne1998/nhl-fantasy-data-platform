
import json
import argparse
import pandas as pd
from pathlib import Path

# 1. Season argument
parser = argparse.ArgumentParser(
    description="Audit NHL season game coverage"
)
parser.add_argument("--season", type=int, required=True)
args = parser.parse_args()

SEASON_ID = args.season

INPUT_FILE = Path(
    f"data/silver/skater_game_logs_{SEASON_ID}.json"
)

# 2. Load data
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    silver_data = json.load(file)

df = pd.DataFrame(silver_data["data"])

# 3. Parse dates
df["game_date"] = pd.to_datetime(df["game_date"])

# 4. Unique games by date
games_by_date = (
    df.groupby("game_date")["game_id"]
    .nunique()
    .sort_index()
)

# 5. Overall report
print("\n========== SEASON COVERAGE AUDIT ==========")
print(f"Season: {SEASON_ID}")
print(f"Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
print(f"Unique games: {df['game_id'].nunique()}")
print(f"Unique game dates: {len(games_by_date)}")

print("\nGames per date:")
print(games_by_date.to_string())

# 6. Find dates with unusually low game counts
print("\nDates with only 1 or 2 games:")
print(games_by_date[games_by_date <= 2].to_string())

# 7. Check records per game
records_per_game = df.groupby("game_id").size()

print("\nRecords per game:")
print(records_per_game.describe())

print("\nGames with fewer than 10 player records:")
print(records_per_game[records_per_game < 10].to_string())

print("\n========== AUDIT COMPLETED ==========")
