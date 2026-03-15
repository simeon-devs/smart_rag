"""
MARA — Product Enrichment
==========================
Infers and backfills missing style / mood / finish payload fields in both
Qdrant collections WITHOUT touching vectors (payload-only update via
client.set_payload).

Usage:
  python3.11 enrich_products.py
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from tqdm import tqdm

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

COLLECTIONS    = ["hard_constraints", "soft_preferences"]
SCROLL_BATCH   = 250   # points per scroll page
SET_PAYLOAD_BATCH = 512  # max point IDs per set_payload call

SEP = "─" * 60


# ── Inference rules ───────────────────────────────────────────────────────────

def infer_mood(payload: dict[str, Any]) -> str:
    kelvin = payload.get("kelvin")
    if kelvin is None:
        return "ambient"
    if kelvin <= 2700:
        return "cozy"
    if kelvin >= 4000:
        return "focused"
    return "ambient"   # 3000 K


def infer_style(payload: dict[str, Any]) -> str:
    blob = _searchable(payload)
    if any(kw in blob for kw in ("pendant", "suspended", "pendel")):
        return "minimalist"
    if any(kw in blob for kw in ("outdoor", "aussen", "façade", "facade")):
        return "industrial"
    if any(kw in blob for kw in ("spot", "strahler", "downlight")):
        return "minimalist"
    if any(kw in blob for kw in ("profile", "profil", "lichtband")):
        return "minimalist"
    if any(kw in blob for kw in ("acoustic", "akustik")):
        return "scandinavian"
    if any(kw in blob for kw in ("warm", "warmwhite", "2700")):
        return "scandinavian"
    return "minimalist"


def infer_finish(payload: dict[str, Any]) -> str:
    blob = _searchable(payload)
    if any(kw in blob for kw in ("white", "weiß", "weiss", "blanc")):
        return "white"
    if any(kw in blob for kw in ("black", "schwarz", "noir")):
        return "matte black"
    if any(kw in blob for kw in ("chrome", "chrom")):
        return "chrome"
    if any(kw in blob for kw in ("brass", "messing")):
        return "brushed brass"
    return "white"


def _searchable(payload: dict[str, Any]) -> str:
    """Build a lowercase string from description + tags + name for matching."""
    parts = [
        payload.get("description") or "",
        " ".join(payload.get("tags") or []),
        payload.get("name") or "",
    ]
    return " ".join(parts).lower()


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def connect() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


def scroll_all(client: QdrantClient, collection: str):
    """Yield every point in the collection (payload only, no vectors)."""
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        yield from batch
        if next_offset is None:
            break
        offset = next_offset


def set_payload_batched(
    client: QdrantClient,
    collection: str,
    payload: dict[str, Any],
    point_ids: list[int],
) -> None:
    """Call set_payload in chunks to avoid oversized requests."""
    for i in range(0, len(point_ids), SET_PAYLOAD_BATCH):
        client.set_payload(
            collection_name=collection,
            payload=payload,
            points=point_ids[i : i + SET_PAYLOAD_BATCH],
        )


# ── Field population stats ────────────────────────────────────────────────────

def field_stats(points: list, fields: list[str]) -> dict[str, int]:
    counts = {f: 0 for f in fields}
    for p in points:
        pl = p.payload
        for f in fields:
            val = pl.get(f)
            if val not in (None, "", []):
                counts[f] += 1
    return counts


def print_stats(label: str, total: int, counts: dict[str, int]) -> None:
    print(f"\n  {label} ({total} points)")
    print(f"  {'Field':<10} {'Count':>8}  {'%':>7}")
    print(f"  {'-'*9} {'-'*8}  {'-'*7}")
    for f, n in counts.items():
        pct = 100 * n / total if total else 0
        print(f"  {f:<10} {n:>8}  {pct:>6.1f}%")


# ── Core enrichment ───────────────────────────────────────────────────────────

FIELDS = ["style", "mood", "finish"]


def enrich_collection(client: QdrantClient, collection: str) -> None:
    print(f"\n{SEP}")
    print(f"Collection: {collection}")
    print(SEP)

    # ── 1. Load all points ────────────────────────────────────────────────────
    print("  Loading all points …")
    total_count = client.get_collection(collection).points_count
    points = list(tqdm(
        scroll_all(client, collection),
        total=total_count,
        desc=f"  scroll {collection[:16]}",
        unit="pt",
        ncols=72,
    ))
    print(f"  Loaded {len(points)} points.")

    # ── 2. Before stats ───────────────────────────────────────────────────────
    before = field_stats(points, FIELDS)
    print_stats("BEFORE", len(points), before)

    # ── 3. Build update groups ────────────────────────────────────────────────
    # Group points by field-to-set → inferred value → [point_ids]
    # Only touch a field if it is currently absent.
    style_groups:  dict[str, list[int]] = defaultdict(list)
    mood_groups:   dict[str, list[int]] = defaultdict(list)
    finish_groups: dict[str, list[int]] = defaultdict(list)

    for pt in points:
        pl = pt.payload
        if not pl.get("style"):
            style_groups[infer_style(pl)].append(pt.id)
        if not pl.get("mood"):
            mood_groups[infer_mood(pl)].append(pt.id)
        if not pl.get("finish"):
            finish_groups[infer_finish(pl)].append(pt.id)

    style_total  = sum(len(v) for v in style_groups.values())
    mood_total   = sum(len(v) for v in mood_groups.values())
    finish_total = sum(len(v) for v in finish_groups.values())
    print(f"\n  Updates needed — style: {style_total}  mood: {mood_total}  finish: {finish_total}")

    # ── 4. Apply updates ──────────────────────────────────────────────────────
    total_calls  = (
        len(style_groups) + len(mood_groups) + len(finish_groups)
    )

    with tqdm(total=total_calls, desc="  set_payload  ", unit="call", ncols=72) as pbar:
        for val, ids in style_groups.items():
            set_payload_batched(client, collection, {"style": val}, ids)
            pbar.update(1)
        for val, ids in mood_groups.items():
            set_payload_batched(client, collection, {"mood": val}, ids)
            pbar.update(1)
        for val, ids in finish_groups.items():
            set_payload_batched(client, collection, {"finish": val}, ids)
            pbar.update(1)

    # ── 5. Reload and show after stats ────────────────────────────────────────
    print("  Reloading to verify …")
    points_after = list(tqdm(
        scroll_all(client, collection),
        total=total_count,
        desc=f"  verify {collection[:16]}",
        unit="pt",
        ncols=72,
    ))
    after = field_stats(points_after, FIELDS)
    print_stats("AFTER", len(points_after), after)

    # ── 6. Value distribution ─────────────────────────────────────────────────
    print(f"\n  Value distribution:")
    for field in FIELDS:
        dist: dict[str, int] = defaultdict(int)
        for pt in points_after:
            val = pt.payload.get(field) or "(none)"
            dist[val] += 1
        row = "  ".join(f"{v}={n}" for v, n in sorted(dist.items()))
        print(f"    {field:<8}  {row}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MARA — Product Enrichment")
    print("=" * 60)

    client = connect()

    for collection in COLLECTIONS:
        enrich_collection(client, collection)

    print(f"\n{SEP}")
    print("Enrichment complete.")
    print("preference_boost() will now fire for mood and style matches.")
    print(SEP)


if __name__ == "__main__":
    main()
