
import json
import argparse
import requests
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--season", type=int, required=True)
args = parser.parse_args()

season = args.season

# Load our Silver dataset
file_path = Path(
    f"data/silver/skater_game_logs_{season}.json"
)

with open(file_path, "r", encoding="utf-8") as file:
    silver_data = json.load(file)

df = pd.DataFrame(silver_data["data"])

silver_game_ids = set(df["game_id"].astype(str).unique())

print(f"Silver games: {len(silver_game_ids)}")

# Retrieve official NHL game list
url = "https://api.nhle.com/stats/rest/en/game"

params = {
    "cayenneExp": f"season={season} and gameType=2",
    "limit": -1
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

payload = response.json()
games = payload.get("data", [])

if not games:
    raise ValueError(
        f"No games returned by NHL API. Response: {payload}"
    )

api_df = pd.DataFrame(games)

print(f"API records: {len(api_df)}")
print(f"API columns: {list(api_df.columns)}")

# Identify game ID column
id_column = next(
    (col for col in ["gameId", "id"] if col in api_df.columns),
    None
)

if id_column is None:
    raise ValueError("Could not identify game ID column.")

api_game_ids = set(api_df[id_column].astype(str).unique())

# Compare
missing_games = api_game_ids - silver_game_ids
extra_games = silver_game_ids - api_game_ids

print("\n========== GAME COVERAGE COMPARISON ==========")
print(f"Official API games: {len(api_game_ids)}")
print(f"Silver games: {len(silver_game_ids)}")
print(f"Missing games in Silver: {len(missing_games)}")
print(f"Unexpected games in Silver: {len(extra_games)}")

if missing_games:
    print("\nMissing game IDs:")
    print(sorted(missing_games))

if extra_games:
    print("\nUnexpected game IDs:")
    print(sorted(extra_games))

print("\n========== COMPARISON COMPLETED ==========")
