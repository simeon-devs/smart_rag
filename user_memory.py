"""Persistence and retrieval for user memory stored in Qdrant."""

import os
import time
import math
import uuid
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
)

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "user_memory"
VECTOR_SIZE     = 1024

LAMBDA = {
    "structural": 0.01,
    "semantic":   0.10,
    "episodic":   0.30,
}

TOP_STRUCTURAL = 5
TOP_SEMANTIC   = 5
TOP_EPISODIC   = 3

@dataclass
class MemoryEntry:
    user_id:     str
    memory_type: str
    text:        str
    source:      str = "chat"


from embeddings import embed


def _get_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


def setup_collection():
    """Ensure the memory collection and filter indexes exist."""
    client = _get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="user_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="memory_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def _decay(memory_type: str, timestamp: float) -> float:
    """Return the decay multiplier for a stored memory."""
    days_elapsed = (time.time() - timestamp) / 86400
    lam = LAMBDA.get(memory_type, 0.10)
    return math.exp(-lam * days_elapsed)


def _delete_field_memories(client: QdrantClient, user_id: str, text_prefix: str) -> int:
    """Delete all structural memories for a user whose text starts with text_prefix.

    Used before saving an updated constraint so only the latest value is kept.
    Qdrant doesn't support prefix-match filters, so we scroll + delete by ID.
    Returns the number of entries deleted.
    """
    user_filter = Filter(
        must=[
            FieldCondition(key="user_id",     match=MatchValue(value=user_id)),
            FieldCondition(key="memory_type", match=MatchValue(value="structural")),
        ]
    )
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=user_filter,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    ids_to_delete = [
        point.id for point in results
        if (point.payload.get("text") or "").startswith(text_prefix)
    ]
    if ids_to_delete:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=ids_to_delete),
        )
        print(f"  Deleted {len(ids_to_delete)} old '{text_prefix}*' memories for {user_id}")
    return len(ids_to_delete)


def _text_already_exists(client: QdrantClient, user_id: str, memory_type: str, text: str) -> bool:
    """Return True if an identical text entry already exists for this user."""
    user_filter = Filter(
        must=[
            FieldCondition(key="user_id",     match=MatchValue(value=user_id)),
            FieldCondition(key="memory_type", match=MatchValue(value=memory_type)),
        ]
    )
    # Scroll through all entries of this type for the user and check text equality.
    # In practice structural memories are few (< 20) so this is fast.
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=user_filter,
        limit=100,
        with_payload=True,
    )
    for point in results:
        if point.payload.get("text") == text:
            return True
    return False


def save_memory(entry: MemoryEntry) -> str:
    """Persist a single memory entry and return its generated id.

    Skips saving if an identical text entry already exists for this user
    (prevents duplicate constraint memories from repeated chip confirms).
    """
    setup_collection()
    client  = _get_client()

    if _text_already_exists(client, entry.user_id, entry.memory_type, entry.text):
        print(f"  Skipping duplicate memory for {entry.user_id}: {entry.text!r}")
        return "duplicate"

    vector  = embed(entry.text)
    mem_id  = str(uuid.uuid4())
    now     = time.time()

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id      = abs(hash(mem_id)) % (2**63),
                vector  = vector,
                payload = {
                    "mem_id":      mem_id,
                    "user_id":     entry.user_id,
                    "memory_type": entry.memory_type,
                    "text":        entry.text,
                    "source":      entry.source,
                    "timestamp":   now,
                    "lambda":      LAMBDA[entry.memory_type],
                },
            )
        ],
    )
    return mem_id


def save_many(entries: list[MemoryEntry]) -> list[str]:
    """Persist multiple memory entries."""
    return [save_memory(e) for e in entries]


def get_user_context(user_id: str, query: str) -> dict:
    """Return the current structured memory context for a user."""
    setup_collection()
    client       = _get_client()
    query_vector = embed(query)

    def fetch(memory_type: str, limit: int) -> list[dict]:
        type_filter = Filter(
            must=[
                FieldCondition(key="user_id",     match=MatchValue(value=user_id)),
                FieldCondition(key="memory_type", match=MatchValue(value=memory_type)),
            ]
        )
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=type_filter,
            limit=limit,
            with_payload=True,
        ).points

        memories = []
        for r in results:
            p           = r.payload
            raw_score   = r.score
            decay_w     = _decay(memory_type, p.get("timestamp", time.time()))
            final_score = raw_score * decay_w

            memories.append({
                "text":         p.get("text", ""),
                "memory_type":  memory_type,
                "source":       p.get("source", "chat"),
                "raw_score":    round(raw_score, 4),
                "decay_weight": round(decay_w, 4),
                "final_score":  round(final_score, 4),
                "timestamp":    p.get("timestamp", 0),
            })

        return sorted(memories, key=lambda x: x["final_score"], reverse=True)

    structural = fetch("structural", TOP_STRUCTURAL)
    semantic   = fetch("semantic",   TOP_SEMANTIC)
    episodic   = fetch("episodic",   TOP_EPISODIC)

    lines = []

    if structural:
        lines.append("HARD CONSTRAINTS (must never be violated):")
        for m in structural:
            lines.append(f"  - {m['text']}")

    if semantic:
        lines.append("USER PREFERENCES (learned over time):")
        for m in semantic:
            lines.append(f"  - {m['text']} (confidence: {m['decay_weight']:.0%})")

    if episodic:
        lines.append("RECENT ACTIVITY (fades quickly):")
        for m in episodic:
            lines.append(f"  - {m['text']}")

    summary = "\n".join(lines) if lines else "No prior context for this user."

    return {
        "structural": structural,
        "semantic":   semantic,
        "episodic":   episodic,
        "summary":    summary,
    }


def save_constraints_as_memory(user_id: str, constraints: dict):
    """Persist explicit hard constraints as structural memory.

    Each constraint field is stored as a single, always-current entry.
    Before saving a new value for a field, any existing entry for that
    field is deleted (delete-then-replace), so only the latest value
    appears in the memory panel — never stale old values.

    Only fields that are explicitly set (non-None / non-empty) in the
    incoming constraints dict are touched; unset fields are left alone.
    """
    setup_collection()
    client = _get_client()
    saved = []

    if constraints.get("max_wattage"):
        _delete_field_memories(client, user_id, "maximum wattage ")
        saved.append(save_memory(MemoryEntry(
            user_id=user_id,
            memory_type="structural",
            text=f"maximum wattage {constraints['max_wattage']}W",
            source="constraint",
        )))

    if constraints.get("max_price_chf"):
        _delete_field_memories(client, user_id, "maximum budget ")
        saved.append(save_memory(MemoryEntry(
            user_id=user_id,
            memory_type="structural",
            text=f"maximum budget {constraints['max_price_chf']} CHF",
            source="constraint",
        )))

    for mat in constraints.get("forbidden_materials", []):
        # Forbidden materials are additive — only deduplicate exact text, no delete.
        saved.append(save_memory(MemoryEntry(
            user_id=user_id,
            memory_type="structural",
            text=f"forbidden material: {mat}",
            source="constraint",
        )))

    if constraints.get("kelvin_min") or constraints.get("kelvin_max"):
        # Delete all previous kelvin memories regardless of which variant was stored.
        _delete_field_memories(client, user_id, "warm white light ")
        _delete_field_memories(client, user_id, "cool white light ")
        _delete_field_memories(client, user_id, "color temperature ")
        kmin = constraints.get("kelvin_min")
        kmax = constraints.get("kelvin_max")
        if kmin is not None and kmax is not None:
            kelvin_text = f"color temperature {kmin}K to {kmax}K"
        elif kmax is not None:
            kelvin_text = f"warm white light max {kmax}K"
        else:
            kelvin_text = f"cool white light min {kmin}K"
        saved.append(save_memory(MemoryEntry(
            user_id=user_id,
            memory_type="structural",
            text=kelvin_text,
            source="constraint",
        )))

    if constraints.get("room_type"):
        _delete_field_memories(client, user_id, "room type: ")
        saved.append(save_memory(MemoryEntry(
            user_id=user_id,
            memory_type="structural",
            text=f"room type: {constraints['room_type']}",
            source="constraint",
        )))

    if constraints.get("location"):
        _delete_field_memories(client, user_id, "location: ")
        saved.append(save_memory(MemoryEntry(
            user_id=user_id,
            memory_type="structural",
            text=f"location: {constraints['location']}",
            source="constraint",
        )))

    print(f"  Saved {len(saved)} structural memories for {user_id}")
    return saved


def save_browse_as_memory(user_id: str, product_name: str, product_description: str):
    """Persist a browse event as episodic memory."""
    entry = MemoryEntry(
        user_id     = user_id,
        memory_type = "episodic",
        text        = f"browsed: {product_name} — {product_description}",
        source      = "browse",
    )
    return save_memory(entry)


def save_chat_preference(user_id: str, preference_text: str):
    """Persist a chat-derived preference as semantic memory."""
    entry = MemoryEntry(
        user_id     = user_id,
        memory_type = "semantic",
        text        = preference_text,
        source      = "chat",
    )
    return save_memory(entry)
