"""Embedding helpers used for product search and memory retrieval."""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

HF_TOKEN  = os.getenv("HF_TOKEN")
API_URL   = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5"
VECTOR_SIZE = 1024


def embed(text: str) -> list[float]:
    """Encode a single text into an embedding vector via HF Inference API."""
    text = f"Represent this sentence for searching relevant passages: {text}"
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    vector   = response.json()
    # HF returns nested list for single input — flatten
    if isinstance(vector[0], list):
        vector = vector[0]
    return vector


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Encode multiple texts into embedding vectors via HF Inference API."""
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": texts})
    return response.json()
