import requests
import json

filename = "urls.txt"
output_jsonl = "Jan25-Data.jsonl"

with open(filename, "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

with open(output_jsonl, "w", encoding="utf-8") as out:
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()

            parts = url.rsplit("/", 1)
            if parts[-1].endswith(".md"):
                base = parts[0]
                name = parts[1][:-3]
                new_url = f"{base}/#/{name}"
            else:
                new_url = url 

            out.write(json.dumps({
                "url": new_url,
                "content": response.text
            }, ensure_ascii=False) + "\n")

            print(f"Fetched and stored: {new_url}")

        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

