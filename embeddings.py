"""Embedding helpers backed by a Hugging Face dedicated TEI endpoint."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

MODEL_NAME = "BAAI/bge-large-en-v1.5"
VECTOR_SIZE = 1024
_BGE_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingConfigurationError(RuntimeError):
    """Raised when the Hugging Face embedding endpoint is not configured."""


@dataclass(frozen=True)
class EmbeddingSettings:
    endpoint_url: str
    token: str
    timeout_sec: float
    batch_size: int
    max_retries: int


def _read_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise EmbeddingConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise EmbeddingConfigurationError(f"{name} must be > 0, got {value}.")
    return value


def _read_non_negative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise EmbeddingConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < 0:
        raise EmbeddingConfigurationError(f"{name} must be >= 0, got {value}.")
    return value


def _read_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise EmbeddingConfigurationError(f"{name} must be a number, got {raw!r}.") from exc
    if value <= 0:
        raise EmbeddingConfigurationError(f"{name} must be > 0, got {value}.")
    return value


@lru_cache(maxsize=1)
def get_embedding_settings() -> EmbeddingSettings:
    """Read and validate runtime settings for the remote TEI endpoint."""
    endpoint_url = os.getenv("HF_EMBEDDING_ENDPOINT_URL", "").strip()
    token = os.getenv("HF_TOKEN", "").strip()

    if not endpoint_url:
        raise EmbeddingConfigurationError(
            "Missing HF_EMBEDDING_ENDPOINT_URL. Point it at your dedicated Hugging Face TEI endpoint URL."
        )
    if not endpoint_url.startswith(("http://", "https://")):
        raise EmbeddingConfigurationError(
            "HF_EMBEDDING_ENDPOINT_URL must be a full dedicated endpoint URL, not a model id."
        )
    if not token:
        raise EmbeddingConfigurationError(
            "Missing HF_TOKEN. Set a Hugging Face token that can access the dedicated endpoint."
        )

    return EmbeddingSettings(
        endpoint_url=endpoint_url,
        token=token,
        timeout_sec=_read_positive_float("HF_EMBED_TIMEOUT_SEC", 120.0),
        batch_size=_read_positive_int("HF_EMBED_BATCH_SIZE", 64),
        max_retries=_read_non_negative_int("HF_EMBED_MAX_RETRIES", 2),
    )


def validate_embedding_config() -> EmbeddingSettings:
    """Fail fast when the embedding runtime is not configured correctly."""
    settings = get_embedding_settings()
    _get_client()
    return settings


def describe_embedding_backend() -> str:
    """Return a concise, non-secret summary of the embedding backend."""
    settings = validate_embedding_config()
    parsed = urlparse(settings.endpoint_url)
    host = parsed.netloc or settings.endpoint_url
    return (
        f"HF TEI endpoint={host} model={MODEL_NAME} dims={VECTOR_SIZE} "
        f"batch={settings.batch_size} timeout={settings.timeout_sec:.0f}s retries={settings.max_retries}"
    )


@lru_cache(maxsize=1)
def _get_client():
    """Create and cache the Hugging Face inference client."""
    from huggingface_hub import InferenceClient

    settings = get_embedding_settings()
    return InferenceClient(
        model=settings.endpoint_url,
        token=settings.token,
        timeout=settings.timeout_sec,
    )


def _coerce_vectors(raw_output, expected_count: int) -> list[list[float]]:
    """Normalize TEI output into a list of dense float vectors."""
    payload = raw_output.tolist() if hasattr(raw_output, "tolist") else raw_output

    if expected_count == 1 and isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
        vectors = [payload]
    elif isinstance(payload, list) and payload and isinstance(payload[0], list):
        vectors = payload
    else:
        raise RuntimeError(
            f"Unexpected embedding response shape from HF endpoint: expected {expected_count} vector(s), got {type(payload).__name__}."
        )

    if len(vectors) != expected_count:
        raise RuntimeError(
            f"HF endpoint returned {len(vectors)} vectors for {expected_count} input texts."
        )

    normalized: list[list[float]] = []
    for index, vector in enumerate(vectors, start=1):
        if len(vector) != VECTOR_SIZE:
            raise RuntimeError(
                f"HF endpoint returned vector {index} with dimension {len(vector)}; expected {VECTOR_SIZE}."
            )
        normalized.append([float(value) for value in vector])
    return normalized


def _request_embeddings(inputs: str | list[str], request_label: str) -> list[list[float]]:
    """Call the remote TEI endpoint with retry logging and shape validation."""
    client = _get_client()
    settings = get_embedding_settings()
    expected_count = 1 if isinstance(inputs, str) else len(inputs)

    for attempt in range(1, settings.max_retries + 2):
        started = time.perf_counter()
        try:
            response = client.feature_extraction(inputs, normalize=True)
            vectors = _coerce_vectors(response, expected_count)
            elapsed = time.perf_counter() - started
            print(
                f"[embeddings] {request_label} size={expected_count} attempt={attempt} took={elapsed:.2f}s"
            )
            return vectors
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(
                f"[embeddings] {request_label} size={expected_count} attempt={attempt} failed after {elapsed:.2f}s: {exc}"
            )
            if attempt > settings.max_retries:
                raise RuntimeError(
                    f"HF embedding request failed after {attempt} attempt(s): {exc}"
                ) from exc

            sleep_seconds = min(2 ** (attempt - 1), 8)
            print(f"[embeddings] retrying {request_label} in {sleep_seconds}s")
            time.sleep(sleep_seconds)

    raise RuntimeError("Embedding request retry loop exited unexpectedly.")


def embed(text: str) -> list[float]:
    """Encode a single query string into a normalized embedding vector."""
    return _request_embeddings(_BGE_PREFIX + text, request_label="query")[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Encode multiple documents into normalized embedding vectors."""
    if not texts:
        return []

    settings = get_embedding_settings()
    all_vectors: list[list[float]] = []
    total_batches = (len(texts) + settings.batch_size - 1) // settings.batch_size

    for start in range(0, len(texts), settings.batch_size):
        batch_index = start // settings.batch_size + 1
        batch = texts[start:start + settings.batch_size]
        batch_vectors = _request_embeddings(
            batch,
            request_label=f"batch {batch_index}/{total_batches}",
        )
        all_vectors.extend(batch_vectors)

    return all_vectors
