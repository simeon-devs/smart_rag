"""Embedding helpers used for product search and memory retrieval."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN    = os.getenv("HF_TOKEN")
API_URL     = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5"
VECTOR_SIZE = 1024


def embed(text: str) -> list[float]:
    text     = f"Represent this sentence for searching relevant passages: {text}"
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=30)
    vector   = response.json()
    if isinstance(vector[0], list):
        vector = vector[0]
    return vector


def embed_batch(texts: list[str]) -> list[list[float]]:
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": texts}, timeout=60)
    return response.json()
