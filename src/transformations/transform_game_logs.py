
import argparse
import json
from pathlib import Path

import pandas as pd
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ACCOUNT_NAME = "stnhlfantasydev2026"
BRONZE_CONTAINER = "bronze"
SILVER_CONTAINER = "silver"


# --------------------------------------------------
# 1. PARAMETERS
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Transform NHL Bronze game logs into Silver"
)

parser.add_argument(
    "--season",
    type=int,
    required=True,
    help="NHL season ID, e.g. 20242025"
)

args = parser.parse_args()
SEASON_ID = args.season

INPUT_FILE = Path(
    f"data/bronze/skater_game_logs_{SEASON_ID}.json"
)

OUTPUT_FILE = Path(
    f"data/silver/skater_game_logs_{SEASON_ID}.json"
)

print(f"Transforming season {SEASON_ID}")


# --------------------------------------------------
# 2. CONNECT TO AZURE
# --------------------------------------------------

account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"

credential = AzureCliCredential()

blob_service_client = BlobServiceClient(
    account_url=account_url,
    credential=credential
)

bronze_client = blob_service_client.get_container_client(
    BRONZE_CONTAINER
)

silver_client = blob_service_client.get_container_client(
    SILVER_CONTAINER
)

print("Azure connection successful!")


# --------------------------------------------------
# 3. LOAD BRONZE DATA
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Bronze file not found: {INPUT_FILE}"
    )

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    bronze_data = json.load(file)

records = bronze_data["data"]
df = pd.DataFrame(records)

print(f"\nBronze records loaded: {len(df)}")


# --------------------------------------------------
# 4. SELECT AND STANDARDIZE COLUMNS
# --------------------------------------------------

column_mapping = {
    "gameId": "game_id",
    "gameDate": "game_date",
    "playerId": "player_id",
    "skaterFullName": "player_name",
    "teamAbbrev": "team",
    "opponentTeamAbbrev": "opponent",
    "homeRoad": "home_road",
    "positionCode": "position",
    "goals": "goals",
    "assists": "assists",
    "points": "points",
    "plusMinus": "plus_minus",
    "penaltyMinutes": "penalty_minutes",
    "shots": "shots_on_goal",
    "shootingPct": "shooting_pct",
    "faceoffWinPct": "faceoff_win_pct",
    "ppGoals": "power_play_goals",
    "ppPoints": "power_play_points",
    "shGoals": "short_handed_goals",
    "shPoints": "short_handed_points",
    "gameWinningGoals": "game_winning_goals",
    "timeOnIcePerGame": "time_on_ice_seconds",
    "evGoals": "even_strength_goals",
    "evPoints": "even_strength_points",
    "otGoals": "overtime_goals",
    "gamesPlayed": "games_played"
}

available_columns = [
    column for column in column_mapping
    if column in df.columns
]

df = df[available_columns].rename(
    columns=column_mapping
)

df["season_id"] = SEASON_ID


# --------------------------------------------------
# 5. DATA TYPES
# --------------------------------------------------

df["game_date"] = pd.to_datetime(
    df["game_date"],
    errors="coerce"
).dt.strftime("%Y-%m-%d")

integer_columns = [
    "game_id",
    "player_id",
    "season_id",
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
    "game_winning_goals",
    "time_on_ice_seconds",
    "even_strength_goals",
    "even_strength_points",
    "overtime_goals",
    "games_played"
]

for column in integer_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).astype("Int64")

float_columns = [
    "shooting_pct",
    "faceoff_win_pct"
]

for column in float_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# --------------------------------------------------
# 6. DATA QUALITY CHECKS
# --------------------------------------------------

initial_count = len(df)

df = df.drop_duplicates(
    subset=["game_id", "player_id"]
)

duplicates_removed = initial_count - len(df)

required_columns = [
    "game_id",
    "game_date",
    "player_id",
    "player_name"
]

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values found in required identification columns."
    )

if df.duplicated(
    subset=["game_id", "player_id"]
).any():
    raise ValueError("Duplicate player-game records remain.")

df = df.sort_values(
    ["game_date", "game_id", "player_id"]
).reset_index(drop=True)


# --------------------------------------------------
# 7. BUILD SILVER DATASET
# --------------------------------------------------

silver_data = {
    "metadata": {
        "season_id": SEASON_ID,
        "source": "NHL Stats API",
        "transformation": "Standardized player game logs",
        "total_records": len(df),
        "duplicates_removed": duplicates_removed,
        "first_game_date": df["game_date"].min(),
        "last_game_date": df["game_date"].max()
    },
    "data": df.astype(object).where(
        pd.notna(df), None
    ).to_dict(orient="records")
}


# --------------------------------------------------
# 8. SAVE LOCAL SILVER DATA
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        silver_data,
        file,
        indent=2,
        ensure_ascii=False
    )

print(f"\nSilver file saved: {OUTPUT_FILE}")


# --------------------------------------------------
# 9. UPLOAD TO AZURE SILVER
# --------------------------------------------------

blob_name = f"nhl/skater_game_logs_{SEASON_ID}.json"

blob_client = silver_client.get_blob_client(blob_name)

with open(OUTPUT_FILE, "rb") as file:
    blob_client.upload_blob(
        file,
        overwrite=True
    )

print(f"Azure upload successful: {blob_name}")


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n--------------------------------")
print("SILVER TRANSFORMATION COMPLETED!")
print("--------------------------------")

print(f"Season: {SEASON_ID}")
print(f"Records processed: {initial_count}")
print(f"Duplicates removed: {duplicates_removed}")
print(f"Final records: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Azure destination: silver/{blob_name}")
