from fastapi import FastAPI, Query
from pydantic import BaseModel
import typesense
from typing import List, Optional
import requests
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import re


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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["OPTIONS", "POST", "GET"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None
    link: Optional[str] = None

def get_embedding(text: str) -> List:
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": text
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()['data'][0]['embedding']

def cosine_sim(vec1: List[float], vec2: List[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def search_typesense_with_vector(embedding: list, k=2) -> list:
    query_vector = ",".join(map(str, embedding))

    collections = ["discourse-book", "course-content-book"]
    all_hits = []

    for collection in collections:
        results = typesense_client.multi_search.perform({
            "searches": [
                {
                    "collection": collection,
                    "q": "placeholder",
                    "query_by": "content",
                    "vector_query": f"embedding:([{query_vector}], k:{k})"
                }
            ]
        })

        hits = results["results"][0]["hits"]
        for hit in hits:
            doc = hit["document"]
            doc_embedding = doc.get("embedding")
            if doc_embedding:
                sim = cosine_sim(embedding, doc_embedding)
                all_hits.append({
                    "url": doc.get("url"),
                    "content": doc["content"],
                    "similarity": sim
                })

    sorted_hits = sorted(all_hits, key=lambda x: x["similarity"], reverse=True)
    return sorted_hits[:k]

def search_typesense_with_link(link: str, query_embedding: list) -> list:
    all_hits = []

    collections = ["discourse-book", "course-content-book"]

    for collection in collections:
        # query_by_fields = "url,parent_url" if collection == "discourse-book" else "url"
        query_by_fields = "url"
        try:
            result = typesense_client.collections[collection].documents.search({
                "q": link,
                "query_by": query_by_fields
            })

            for hit in result.get("hits", []):
                doc = hit["document"]
                doc_embedding = doc.get("embedding")
                if doc_embedding:
                    sim = cosine_sim(query_embedding, doc_embedding)
                    all_hits.append({
                        "url": doc["url"],
                        "content": doc["content"],
                        "similarity": sim
                    })

        except Exception as e:
            print(f"Error searching link in {collection}: {e}")

    sorted_hits = sorted(all_hits, key=lambda x: x["similarity"], reverse=True)
    return sorted_hits

def fetch_surrounding_context(matches: list, window: int = 2) -> list:
    updated_matches = []

    for match in matches:
        url = match.get("url")
        content = match.get("content")
        updated_matches.append({"url": url, "content": content})

        discourse_prefix = "https://discourse.onlinedegree.iitm.ac.in/t/"
        if url and url.startswith(discourse_prefix):
            m = re.match(rf"{re.escape(discourse_prefix)}([^/]+)/(\d+)/(\d+)$", url)
            if m:
                slug = m.group(1)
                topic_id = m.group(2)
                post_id = int(m.group(3))

                for offset in range(-window, window + 1):
                    if offset == 0:
                        continue
                    new_post_id = post_id + offset
                    new_url = f"{discourse_prefix}{slug}/{topic_id}/{new_post_id}"

                    try:
                        result = typesense_client.collections["discourse-book"].documents.search({
                            "q": new_url,
                            "query_by": "url"
                        })
                        for hit in result.get("hits", []):
                            doc = hit["document"]
                            if doc.get("url") == new_url:
                                updated_matches.append({
                                    "url": doc["url"],
                                    "content": doc["content"]
                                })
                    except Exception as e:
                        print(f"Could not fetch {new_url}: {e}")

    return matches if updated_matches==[] else updated_matches

def ask_gpt(query: str, matches: list, image_base64: str = None) -> str:
    context_str = "\n".join([m["content"] for m in matches])

    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if image_base64:
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions using the given context. If the context does not have any answer, tell that you do not have the answer"},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": f"The image is attached above. Use this image along with the context to answer the question."}
                ]
            }
        ]

    else:
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions using the given context. If the context does not have any answer, tell that you do not have the answer"},   
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]

    data={
        "model": "gpt-4o-mini",
        "messages": messages
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]


@app.get('/')
async def default():
    return {"message": "server is running"}

@app.post("/api")
async def handle_query(payload: QueryRequest):
    embedding = get_embedding(payload.question)
    # if(payload.link):
    #     matches = search_typesense_with_link(payload.link,embedding)
    # else:
    #     matches = search_typesense_with_vector(embedding)
    
    updated_matches = search_typesense_with_link(payload.link,embedding)

    # updated_matches = fetch_surrounding_context(matches)
    
    gpt_answer = ask_gpt(payload.question, updated_matches, payload.image)

    return {"answer": gpt_answer["content"], 
            "links": updated_matches}