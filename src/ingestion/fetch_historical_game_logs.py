
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ACCOUNT_NAME = "stnhlfantasydev2026"
CONTAINER_NAME = "bronze"

SEASON_ID = 20252026
GAME_TYPE_ID = 2

START_DATE = date(2025, 10, 1)
END_DATE = date(2026, 4, 30)

CHUNK_DAYS = 7

BASE_URL = "https://api.nhle.com/stats/rest/en/skater/summary"


# --------------------------------------------------
# 1. CONNECT TO AZURE
# --------------------------------------------------

account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"

credential = AzureCliCredential()

blob_service_client = BlobServiceClient(
    account_url=account_url,
    credential=credential
)

container_client = blob_service_client.get_container_client(
    CONTAINER_NAME
)

print("Azure connection successful!")


# --------------------------------------------------
# 2. FETCH HISTORICAL GAME LOGS
# --------------------------------------------------

def fetch_period(start_date, end_date):

    params = {
        "isGame": "true",
        "cayenneExp": (
            f"seasonId={SEASON_ID} "
            f"and gameTypeId={GAME_TYPE_ID} "
            f"and gameDate>='{start_date}' "
            f"and gameDate<='{end_date}'"
        ),
        "limit": -1
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    if result["total"] != len(result["data"]):
        raise RuntimeError(
            f"Incomplete extraction: {start_date} to {end_date}"
        )

    return result["data"]


all_records = []
seen_records = set()

period_start = START_DATE

while period_start <= END_DATE:

    period_end = min(
        period_start + timedelta(days=CHUNK_DAYS - 1),
        END_DATE
    )

    print(f"Fetching {period_start} to {period_end}...")

    records = fetch_period(period_start, period_end)

    for record in records:

        unique_key = (
            record["gameId"],
            record["playerId"]
        )

        if unique_key not in seen_records:
            seen_records.add(unique_key)
            all_records.append(record)

    print(
        f"Retrieved: {len(records)} | "
        f"Unique total: {len(all_records)}"
    )

    period_start = period_end + timedelta(days=1)

    time.sleep(0.2)


# --------------------------------------------------
# 3. BUILD BRONZE DATASET
# --------------------------------------------------

bronze_data = {
    "metadata": {
        "source": "NHL Stats API",
        "endpoint": BASE_URL,
        "season_id": SEASON_ID,
        "game_type_id": GAME_TYPE_ID,
        "start_date": str(START_DATE),
        "end_date": str(END_DATE),
        "extraction_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_records": len(all_records)
    },
    "data": all_records
}


# --------------------------------------------------
# 4. SAVE LOCAL RAW DATA
# --------------------------------------------------

output_dir = Path("data/bronze")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = (
    output_dir / f"skater_game_logs_{SEASON_ID}.json"
)

with open(output_file, "w", encoding="utf-8") as file:

    json.dump(
        bronze_data,
        file,
        indent=2,
        ensure_ascii=False
    )

print(f"\nLocal file saved: {output_file}")


# --------------------------------------------------
# 5. UPLOAD TO AZURE BRONZE
# --------------------------------------------------

blob_name = (
    f"nhl/skater_game_logs_{SEASON_ID}.json"
)

blob_client = container_client.get_blob_client(blob_name)

json_data = json.dumps(
    bronze_data,
    indent=2,
    ensure_ascii=False
)

blob_client.upload_blob(
    json_data.encode("utf-8"),
    overwrite=True
)

print(f"\nAzure upload successful: {blob_name}")


# --------------------------------------------------
# 6. FINAL SUMMARY
# --------------------------------------------------

unique_games = len({
    record["gameId"] for record in all_records
})

unique_players = len({
    record["playerId"] for record in all_records
})

print("\n--------------------------------")
print("HISTORICAL INGESTION COMPLETED!")
print("--------------------------------")

print(f"Season: {SEASON_ID}")
print(f"Records: {len(all_records)}")
print(f"Unique games: {unique_games}")
print(f"Unique players: {unique_players}")
print(f"Azure destination: bronze/{blob_name}")
