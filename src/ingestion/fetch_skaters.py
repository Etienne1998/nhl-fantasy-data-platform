
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

# NHL API
url = "https://api-web.nhle.com/v1/skater-stats-leaders/current"

params = {"limit": 5}

# Extract
response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

data = response.json()

# Save raw response in Bronze
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

output_dir = Path("data/bronze")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"skater_stats_{timestamp}.json"

with output_file.open("w", encoding="utf-8") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)

print(f"Extraction successful!")
print(f"File saved: {output_file}")
print(f"Number of statistic categories: {len(data)}")
