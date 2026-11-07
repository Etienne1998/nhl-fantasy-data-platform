import json
from datetime import datetime, timezone

import requests
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient


# Configuration
STORAGE_ACCOUNT_NAME = "stnhlfantasydev2026"
CONTAINER_NAME = "bronze"

BASE_URL = "https://api.nhle.com/stats/rest/en/goalie"

SEASON_ID = 20262027
GAME_TYPE_ID = 2

REPORTS = {
    "summary": "summary"
}


def fetch_goalie_report(report_name: str) -> list:
    """Fetch all goalie records from the NHL Stats API."""

    url = f"{BASE_URL}/{report_name}"

    params = {
        "cayenneExp": (
            f"seasonId={SEASON_ID} and gameTypeId={GAME_TYPE_ID}"
        ),
        "limit": 100,
        "start": 0
    }

    all_records = []

    print(f"\nFetching goalie {report_name} report...")

    while True:
        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        records = data.get("data", [])

        if not records:
            break

        all_records.extend(records)

        print(f"Records fetched: {len(all_records)}")

        if len(records) < params["limit"]:
            break

        params["start"] += params["limit"]

    print(f"Total records: {len(all_records)}")

    return all_records


def upload_to_bronze(data: list, report_name: str, timestamp: str):
    """Upload raw NHL data to Azure Bronze."""

    credential = AzureCliCredential()

    account_url = (
        f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
    )

    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=credential
    )

    blob_name = (
        f"nhl/goalie_{report_name}_{timestamp}.json"
    )

    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME,
        blob=blob_name
    )

    json_data = json.dumps(
        data,
        indent=4,
        ensure_ascii=False
    )

    blob_client.upload_blob(
        json_data.encode("utf-8"),
        overwrite=True
    )

    print(f"Azure Bronze upload successful: {blob_name}")


def main():
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    for report_name in REPORTS.values():

        records = fetch_goalie_report(report_name)

        upload_to_bronze(
            records,
            report_name,
            timestamp
        )

    print("\nGOALIE INGESTION COMPLETED!")


if __name__ == "__main__":
    main()