import json
import os
import requests
import typesense
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("AIPROXY_TOKEN")
typesense_api_key = os.getenv("TYPESENSE_ADMIN_KEY")
typesense_host = "9crqf8ga1kxbhvtdp-1.a1.typesense.net"

typesense_client = typesense.Client({
    "nodes": [{
        "host": typesense_host,
        "port": "443",
        "protocol": "https"
    }],
    "api_key": typesense_api_key,
    "connection_timeout_seconds": 2
})

typesense_client.collections['discourse-book'].delete()

try:
    typesense_client.collections.create({
      "name": "discourse-book",
      "fields": [
          {"name": "id", "type": "string"},
          {"name": "url", "type": "string", "filter": True},
          {"name": "parent_url", "type": "string", "filter": True},
          {"name": "content", "type": "string"},
          {"name": "embedding", "type": "float[]", "num_dim": 1536}
      ],
  })

except Exception as e:
    print("Collection exists or error:", e)


BATCH_SIZE = 10
buffer = []
count = 0
    
with open("discourse-data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        url = data.get("url")
        parent_url = data.get("parent_url")
        content = data.get("content")
        discourse_id = data.get("id")

        if not content or not url:
            continue

        buffer.append({
            "id": str(discourse_id),
            "url": url,
            "parent_url": parent_url,
            "content": content,
        })

        if len(buffer) == BATCH_SIZE:
            texts = [item["content"] for item in buffer]

            response = requests.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "text-embedding-3-small",
                    "input": texts
                }
            )
            response.raise_for_status()
            embeddings = [d["embedding"] for d in response.json()["data"]]

            for i, doc in enumerate(buffer):
                try:
                    typesense_client.collections["discourse-book"].documents.upsert({
                        "id": doc["id"],
                        "url": doc["url"],
                        "parent_url": doc["parent_url"],
                        "content": doc["content"],
                        "embedding": embeddings[i],
                    })
                except Exception as e:
                    print(f"Error inserting document: {e}")

            count+=BATCH_SIZE
            print(count)

            buffer = []

if buffer:
    texts = [item["content"] for item in buffer]
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "text-embedding-3-small",
            "input": texts
        }
    )
    response.raise_for_status()
    embeddings = [d["embedding"] for d in response.json()["data"]]

    for i, doc in enumerate(buffer):
        try:
            typesense_client.collections["discourse-book"].documents.upsert({
                "id": doc["id"],
                "url": doc["url"],
                "parent_url": doc["parent_url"],
                "content": doc["content"],
                "embedding": embeddings[i],
            })
        except Exception as e:
            print(f"Error inserting document: {e}")