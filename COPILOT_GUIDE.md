# MARA — Copilot Guide

Quick reference for AI assistants working on this codebase.


## What This Project Is

MARA (Memory-Augmented Retail Agent) recommends lighting products by combining
semantic vector search with hard constraint filtering, user memory, and
time-based decay. It was built for the GenAI Zürich Hackathon.

The backend is a FastAPI app backed by Qdrant Cloud. The frontend lives in
`old_files/braight/` and is not yet wired to the backend.


## Key Files

| File | What it does |
|------|--------------|
| `main.py` | FastAPI app — `/constraints`, `/browse`, `/chat`, debug routes |
| `mara_engine.py` | Retrieval + reranking: constraint filter → preference boost → decay |
| `embeddings.py` | Remote Hugging Face TEI client for `BAAI/bge-large-en-v1.5` (1024-dim). `embed()` adds the BGE prefix for queries; `embed_batch()` does not |
| `setup_qdrant.py` | Creates Qdrant collections, indexes catalog, creates numeric payload indices |
| `enrich_products.py` | Backfills `style`/`mood`/`finish` on existing payloads — no re-embedding |
| `user_memory.py` | Reads/writes user memory (structural / semantic / episodic) to Qdrant |
| `extract_supabase_catalog.py` | Pulls products from Supabase → `catalog_export.json` |
| `audit_embeddings.py` | Health check: counts, samples, semantic queries, constraint filter |


## Qdrant Collections

| Collection | Purpose | Key payload fields |
|---|---|---|
| `hard_constraints` | Spec-oriented text + numeric filtering | `wattage`, `price_chf`, `kelvin`, `material`, `inside`/`outside` |
| `soft_preferences` | Semantic description text + soft signals | `description`, `tags`, `style`, `finish`, `mood` |
| `user_memory` | Per-user memory entries | `user_id`, `memory_type`, `text`, `timestamp`, `lambda` |

All product collections have float payload indices on `wattage`, `price_chf`,
and `kelvin` (required for Qdrant range filters — without them queries return
HTTP 400).


## Scoring Formula

```
final_score = (similarity * decay(hard, 0)) * constraint_weight
            + preference_boost(style, finish, mood)
```

- `constraint_weight` is binary: 1.0 if no violations, 0.0 if any violation
- max boost: style +0.15, finish +0.10, mood +0.05 = 0.30 total
- decay is applied to user memory, not to product retrieval scores


## Running the Backend

```bash
source .venv/bin/activate
uvicorn main:app --reload        # FastAPI on :8000
python3.11 setup_qdrant.py       # re-index via the configured HF TEI endpoint
python3.11 enrich_products.py    # backfill style/mood/finish after re-index
```


## Known Issues

### Numeric payload indices required

Qdrant range filters (`wattage ≤ 40`, `price ≤ 200`) require explicit float
payload indices. These are created by `setup_qdrant.py` via
`create_payload_indices()`. If you recreate collections without running the
full setup script, range filters will fail with HTTP 400.

### `forbidden_materials` is post-filtered, not pre-filtered

`build_qdrant_filter()` in `mara_engine.py` handles `max_wattage`,
`max_price_chf`, and `kelvin` as Qdrant pre-filters. `forbidden_materials` is
checked post-retrieval in `constraint_weight()`. This means plastic products
can enter the candidate set and be zeroed out after retrieval, potentially
shrinking results below `top_k`.

### `style_age_days` resets on server restart

Style/finish preference timestamps are tracked in `style_timestamp_store` in
`main.py` (in-process dict). The age correctly accumulates within a session
but resets when the server restarts. A future fix would persist the first
expression timestamp to the `user_memory` Qdrant collection.


### Style and Mood Enrichment

**Why it exists**

Lu's Supabase catalog has no `style`, `mood`, or `finish` fields. Without
them, `preference_boost()` in `mara_engine.py` never fires, silencing the
+0.15 style boost and +0.10 finish boost entirely.

**How `enrich_products.py` infers the fields**

`mood` — from the numeric `kelvin` value:

- `kelvin ≤ 2700` → `"cozy"`
- `kelvin == 3000` → `"ambient"`
- `kelvin ≥ 4000` → `"focused"`
- `kelvin` absent → `"ambient"`

`style` — from keywords in `description` + `tags` + `name`:

- `warm`, `warmwhite`, `2700` → `"scandinavian"`
- `outdoor`, `aussen`, `façade` → `"industrial"`
- `pendant`, `suspended`, `spot`, `downlight`, `profile`, `lichtband`, etc. → `"minimalist"`
- default → `"minimalist"`

`finish` — from keywords in `description`:

- `white`, `weiß`, `blanc` → `"white"`
- `black`, `schwarz`, `noir` → `"matte black"`
- `chrome`, `Chrom` → `"chrome"`
- `brass`, `Messing` → `"brushed brass"`
- default → `"white"`

**Important caveat**

`mood` reads `kelvin`, which only exists in `hard_constraints`. The
`soft_preferences` collection has no `kelvin`, so every soft product gets
`mood = "ambient"`. Since `preference_boost()` reads from the soft payload,
the mood boost (+0.05) only fires when a user sets `preferred_mood = "ambient"`.
Style (+0.15) and finish (+0.10) boosts are unaffected and work correctly.

**When to re-run**

Run `enrich_products.py` after every `setup_qdrant.py` run (i.e., any time the
catalog is replaced with fresh Supabase data). The script is idempotent — it
only writes fields that are currently absent.
