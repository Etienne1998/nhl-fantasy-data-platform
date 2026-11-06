
import json
from datetime import datetime, timezone

import requests
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient

# Configuration
ACCOUNT_NAME = "stnhlfantasydev2026"
CONTAINER_NAME = "bronze"

# 1. Extract data from NHL API
url = "https://api-web.nhle.com/v1/skater-stats-leaders/current"
params = {"limit": 5}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

data = response.json()

print("NHL data extracted successfully!")

# 2. Connect to Azure Storage
account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"

credential = AzureCliCredential()

blob_service_client = BlobServiceClient(
    account_url=account_url,
    credential=credential
)

# 3. Upload raw JSON to Bronze
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

blob_name = f"nhl/skater_stats_{timestamp}.json"

blob_client = blob_service_client.get_blob_client(
    container=CONTAINER_NAME,
    blob=blob_name
)

blob_client.upload_blob(
    json.dumps(data, ensure_ascii=False, indent=2),
    overwrite=True
)

print("Azure upload successful!")
print(f"Container: {CONTAINER_NAME}")
print(f"Blob: {blob_name}")
