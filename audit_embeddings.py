"""
MARA Embedding Quality Audit
=============================
Verifies Qdrant health, payload completeness, and semantic relevance.
Run with:
  python3.11 audit_embeddings.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, PayloadSchemaType

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

SEP = "─" * 60


def connect() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


# ── Step 1: Collection Counts ──────────────────────────────────────────────────

def step1_counts(client: QdrantClient) -> None:
    print(f"\n{SEP}")
    print("STEP 1 — Collection Counts")
    print(SEP)
    for name in ["hard_constraints", "soft_preferences"]:
        info = client.get_collection(name)
        print(f"  {name}: {info.points_count} points")


# ── Step 2: Random Product Sample ─────────────────────────────────────────────

def step2_samples(client: QdrantClient) -> None:
    print(f"\n{SEP}")
    print("STEP 2 — Random Product Payload Samples (hard_constraints)")
    print(SEP)

    # scroll to pull all point IDs, then randomly select 3
    all_points, _ = client.scroll(
        "hard_constraints", limit=100, with_payload=True, with_vectors=False
    )
    samples = random.sample(all_points, min(3, len(all_points)))
    for p in samples:
        print(f"\n  Point ID: {p.id}")
        print(json.dumps(p.payload, indent=4, ensure_ascii=False))

    print(f"\n  (showing 3 of {len(all_points)} fetched in first-page scroll)")


# ── Step 3: Semantic Search ────────────────────────────────────────────────────

QUERIES = [
    "warm light for a reading corner, scandinavian style",
    "outdoor waterproof spotlight for a facade",
    "energy efficient office ceiling light with DALI control",
]


def step3_semantic(client: QdrantClient) -> list[list[float]]:
    from embeddings import embed

    print(f"\n{SEP}")
    print("STEP 3 — Semantic Search  (soft_preferences, top-3 per query)")
    print(SEP)

    vectors = []
    for q in QUERIES:
        vec = embed(q)
        vectors.append(vec)
        print(f"\n  Query: \"{q}\"")
        results = client.query_points(
            "soft_preferences", query=vec, limit=3, with_payload=True
        ).points
        for r in results:
            p = r.payload
            name = p.get("name", "(no name)")
            cat  = p.get("category", "")
            mfr  = p.get("manufacturer", "")
            desc = (p.get("description") or "")[:80]
            print(f"    {r.score:.4f}  {name}  [{cat}]  {mfr}")
            if desc:
                print(f"           {desc}…")

    return vectors


# ── Step 4: Constraint Filter Test ───────────────────────────────────────────

def ensure_numeric_indices(client: QdrantClient) -> None:
    """Create float payload indices required for Qdrant range filters."""
    numeric_fields = ["wattage", "price_chf", "kelvin"]
    for collection in ["hard_constraints", "soft_preferences"]:
        for field in numeric_fields:
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.FLOAT,
                )
            except Exception:
                pass  # already exists or not needed for this collection


def step4_filter(client: QdrantClient, query_vectors: list[list[float]]) -> None:
    from embeddings import embed

    print(f"\n{SEP}")
    print("STEP 4 — Constraint Filter  (wattage ≤ 40 W  +  price ≤ 200 CHF)")
    print(SEP)

    print("  Creating numeric payload indices for range filtering ...")
    ensure_numeric_indices(client)
    print("  Indices ready.\n")

    f = Filter(must=[
        FieldCondition(key="wattage",    range=Range(lte=40)),
        FieldCondition(key="price_chf",  range=Range(lte=200)),
    ])

    q_vec = query_vectors[0]  # "warm light for a reading corner, scandinavian style"
    filtered = client.query_points(
        "hard_constraints", query=q_vec, query_filter=f,
        limit=5000, with_payload=True
    )
    print(f"\n  Products passing wattage≤40 + price≤200 CHF: {len(filtered.points)}")

    print(f"\n  MARA vs Baseline — Query: \"{QUERIES[0]}\"")
    print(f"  {'Rank':<5} {'Score':<8} {'Name':<45} {'Wattage':>8} {'Price CHF':>10}")
    print(f"  {'-'*4} {'-'*7} {'-'*44} {'-'*8} {'-'*9}")

    for i, r in enumerate(filtered.points[:5], start=1):
        p    = r.payload
        name = (p.get("name") or "")[:44]
        w    = p.get("wattage")
        pr   = p.get("price_chf")
        w_s  = f"{w}W"  if w is not None else "–"
        pr_s = f"{pr}"  if pr is not None else "–"
        print(f"  {i:<5} {r.score:<8.4f} {name:<45} {w_s:>8} {pr_s:>10}")


# ── Step 5: Field Population Stats ───────────────────────────────────────────

def step5_field_stats(client: QdrantClient) -> None:
    print(f"\n{SEP}")
    print("STEP 5 — Soft Field Population Stats  (soft_preferences, first 500 pts)")
    print(SEP)

    all_points, _ = client.scroll(
        "soft_preferences", limit=500, with_payload=True, with_vectors=False
    )

    fields = ["style", "finish", "mood", "description", "tags", "image_url"]
    counts = {f: 0 for f in fields}
    total  = len(all_points)

    for p in all_points:
        pl = p.payload
        for f in fields:
            val = pl.get(f)
            if val not in (None, "", []):
                counts[f] += 1

    print(f"\n  Sample size: {total} points")
    print(f"  {'Field':<16} {'Populated':>10} {'% filled':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10}")
    for f in fields:
        pct = 100 * counts[f] / total if total else 0
        print(f"  {f:<16} {counts[f]:>10} {pct:>9.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from embeddings import describe_embedding_backend, validate_embedding_config

    print("=" * 60)
    print("  MARA Embedding Quality Audit")
    print("=" * 60)

    validate_embedding_config()
    print(f"  Embedding backend: {describe_embedding_backend()}")

    client = connect()

    step1_counts(client)
    step2_samples(client)
    vecs = step3_semantic(client)
    step4_filter(client, vecs)
    step5_field_stats(client)

    print(f"\n{SEP}")
    print("Audit complete.")
    print(SEP)


if __name__ == "__main__":
    random.seed(42)
    main()
