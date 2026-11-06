
import json
from datetime import datetime, timezone

import requests
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ACCOUNT_NAME = "stnhlfantasydev2026"
CONTAINER_NAME = "bronze"

SEASON_ID = 20262027
GAME_TYPE_ID = 2
PAGE_SIZE = 100

BASE_URL = "https://api.nhle.com/stats/rest/en/skater"


# --------------------------------------------------
# 1. CONNECT TO AZURE
# --------------------------------------------------

account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"

credential = AzureCliCredential()

blob_service_client = BlobServiceClient(
    account_url=account_url,
    credential=credential
)

print("Azure connection successful!")


# --------------------------------------------------
# 2. FETCH COMPLETE NHL REPORT WITH PAGINATION
# --------------------------------------------------

def fetch_all_players(report_name):

    url = f"{BASE_URL}/{report_name}"

    params = {
        "cayenneExp": (
            f"seasonId={SEASON_ID} and gameTypeId={GAME_TYPE_ID}"
        ),
        "limit": PAGE_SIZE,
        "start": 0
    }

    all_players = []

    print(f"\nDownloading {report_name}...")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    first_page = response.json()
    total_players = first_page["total"]

    all_players.extend(first_page["data"])

    print(f"Total players available: {total_players}")

    # Download remaining pages
    while len(all_players) < total_players:

        params["start"] = len(all_players)

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        page = response.json()
        new_players = page["data"]

        if not new_players:
            raise RuntimeError(
                f"Pagination stopped unexpectedly for {report_name}."
            )

        all_players.extend(new_players)

        print(
            f"Downloaded {len(all_players)} / {total_players} players"
        )

    print(f"{report_name} download successful!")

    return {
        "total": total_players,
        "data": all_players
    }


# --------------------------------------------------
# 3. DOWNLOAD BOTH REPORTS
# --------------------------------------------------

summary_data = fetch_all_players("summary")

realtime_data = fetch_all_players("realtime")


# --------------------------------------------------
# 4. UPLOAD RAW DATA TO AZURE BRONZE
# --------------------------------------------------

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

reports = {
    "summary": summary_data,
    "realtime": realtime_data
}

container_client = blob_service_client.get_container_client(
    CONTAINER_NAME
)

for report_name, report_data in reports.items():

    blob_name = f"nhl/skater_{report_name}_{timestamp}.json"

    blob_client = container_client.get_blob_client(blob_name)

    json_data = json.dumps(
        report_data,
        indent=2,
        ensure_ascii=False
    )

    blob_client.upload_blob(
        json_data.encode("utf-8"),
        overwrite=True
    )

    print(f"\nAzure upload successful: {blob_name}")
    print(f"Records uploaded: {len(report_data['data'])}")


# --------------------------------------------------
# 5. FINAL SUMMARY
# --------------------------------------------------

print("\n--------------------------------")
print("NHL INGESTION COMPLETED!")
print("--------------------------------")

print(f"Summary players: {len(summary_data['data'])}")
print(f"Realtime players: {len(realtime_data['data'])}")
print(f"Season: {SEASON_ID}")
print(f"Game type: {GAME_TYPE_ID}")
