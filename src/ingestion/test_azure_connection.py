
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient

# Azure Storage Account
account_name = "stnhlfantasydev2026"

account_url = f"https://{account_name}.blob.core.windows.net"

# Authenticate using your Azure CLI session
credential = AzureCliCredential(
    process_timeout=30
)

# Connect to Azure Storage
blob_service_client = BlobServiceClient(
    account_url=account_url,
    credential=credential
)

# List available containers
containers = blob_service_client.list_containers()

print("Azure connection successful!")
print("Available containers:")

for container in containers:
    print(f"- {container['name']}")
