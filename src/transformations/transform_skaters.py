
import json
from datetime import datetime, timezone

from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient


# --------------------------------------------------
# 1. CONNECT TO AZURE
# --------------------------------------------------

ACCOUNT_NAME = "stnhlfantasydev2026"

account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"

credential = AzureCliCredential()

blob_service_client = BlobServiceClient(
    account_url=account_url,
    credential=credential
)

print("Azure connection successful!")


# --------------------------------------------------
# 2. READ LATEST BRONZE FILE FROM AZURE
# --------------------------------------------------

bronze_container = blob_service_client.get_container_client("bronze")

bronze_blobs = list(
    bronze_container.list_blobs(
        name_starts_with="nhl/skater_stats_"
    )
)

if not bronze_blobs:
    raise FileNotFoundError("No Bronze file found in Azure.")

latest_blob = max(bronze_blobs, key=lambda blob: blob.name)

print(f"Latest Bronze file: {latest_blob.name}")

blob_client = bronze_container.get_blob_client(latest_blob.name)

downloaded_data = blob_client.download_blob().readall()

data = json.loads(downloaded_data.decode("utf-8"))

print("Bronze data downloaded successfully!")


# --------------------------------------------------
# 3. TRANSFORM DATA INTO SILVER
# --------------------------------------------------

players = {}

for category, records in data.items():
    for record in records:

        player_id = record["id"]

        if player_id not in players:
            players[player_id] = {
                "player_id": player_id,
                "first_name": record["firstName"]["default"],
                "last_name": record["lastName"]["default"],
                "team": record["teamAbbrev"],
                "position": record["position"],
            }

        players[player_id][category] = record["value"]


silver_data = list(players.values())

silver_json = json.dumps(
    silver_data,
    indent=2,
    ensure_ascii=False
)

print("Silver transformation successful!")
print(f"Unique players: {len(silver_data)}")


# --------------------------------------------------
# 4. UPLOAD SILVER DATA TO AZURE
# --------------------------------------------------

silver_container = blob_service_client.get_container_client("silver")

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

silver_blob_name = f"nhl/skaters_{timestamp}.json"

silver_blob_client = silver_container.get_blob_client(
    silver_blob_name
)

silver_blob_client.upload_blob(
    silver_json.encode("utf-8"),
    overwrite=True
)

print("Azure Silver upload successful!")
print(f"Container: silver")
print(f"Blob: {silver_blob_name}")
