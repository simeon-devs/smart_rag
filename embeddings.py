"""Embedding helpers used for product search and memory retrieval."""

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME  = "BAAI/bge-large-en-v1.5"
VECTOR_SIZE = 1024
HF_TOKEN    = os.getenv("HF_TOKEN")


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, token=HF_TOKEN)


def embed(text: str) -> list[float]:
    text  = f"Represent this sentence for searching relevant passages: {text}"
    model = _load_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    model   = _load_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]
