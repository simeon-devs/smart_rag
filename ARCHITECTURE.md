# MARA Architecture

## High-Level Goal

MARA is a memory-augmented retail retrieval system for lighting products.

Its purpose is to recommend products more intelligently than standard RAG by
combining:

- semantic retrieval
- hard constraint preservation
- user memory
- time-based decay of different memory types

The key principle is that not all user information should have the same weight
or lifetime.


## System Boundaries

The target architecture separates the system into four roles:

- Supabase: full product source of truth
- Qdrant: retrieval-optimized catalog and memory store
- FastAPI: orchestration and MARA logic
- Lovable frontend: user interaction and rendering

The LLM is used only to explain and present the chosen results naturally.
It is not the system of record for product retrieval.


## Target Data Flow

```text
User message / action (Lovable frontend)
        |
        v
FastAPI
        |
        +--> embed(query) via Hugging Face model
        |
        +--> retrieve candidate products from Qdrant
        |
        +--> retrieve relevant user memory from Qdrant
        |
        +--> apply MARA logic
        |      - hard constraint filtering
        |      - soft preference boosting
        |      - decay-aware memory weighting
        |
        +--> send final shortlisted products + context to Llama 3.3 via Groq
        |
        v
Response to Lovable
        |
        +--> ranked product IDs
        +--> MARA metadata
        +--> natural-language reply
        |
        v
Lovable fetches or hydrates full product details from Supabase
```


## Request Lifecycle

### 1. User sends a request

The frontend sends one of three main event types:

- constraints
- browse event
- chat request

Example request content:

- `user_id`
- `message`
- explicit preferences
- clicked product metadata
- hard rules like budget or wattage limits

### 2. FastAPI receives the request

FastAPI is the main orchestrator.

It does not own the product catalog.
It owns the retrieval flow and the MARA business logic.

### 3. Query embedding

The user message is converted into an embedding vector using the Hugging Face
model in `embeddings.py`.

This vector is used to search:

- the product catalog representation in Qdrant
- the user memory representation in Qdrant

### 4. Product retrieval from Qdrant

Qdrant returns the initial product candidates.

This is not the full Supabase product row.
It is a retrieval-optimized representation of the product.

That representation should contain:

- stable product ID
- product name
- embedding text
- hard constraint fields
- selected soft or semantic fields
- minimal display fallback fields

### 5. Memory retrieval from Qdrant

Qdrant also returns relevant user memory for the current query.

Memory types:

- structural
- semantic
- episodic

Each memory type has different decay behavior.

### 6. MARA reranking in FastAPI

FastAPI applies MARA logic over the retrieved candidates.

This includes:

- hard constraint filtering
- violation detection
- soft preference scoring
- decay-aware weighting

This is where MARA differs from baseline retrieval.

### 7. LLM generation

After the shortlist is built, FastAPI sends the selected products and the user
context to Llama 3.3 via Groq.

The LLM is responsible for:

- writing the final reply
- explaining why products fit
- sounding natural and conversational

The LLM is not responsible for deciding the ranking from raw catalog data.

### 8. Response to frontend

FastAPI returns:

- `llm_reply`
- ranked product IDs or lightweight product records
- `baseline_results`
- `mara_results`
- `violation_count`
- `user_context`

The frontend can then fetch the full product details from Supabase using the
returned IDs.


## Role Of Each Core Technology

## Supabase

Supabase is the source of truth for the real lighting catalog.

It stores:

- product identity
- technical attributes
- classifications
- manufacturer/category/family relations
- images and related product assets

Supabase should remain the master database.

## Qdrant

Qdrant is the retrieval layer.

It stores:

- searchable product vectors
- retrieval-oriented product payloads
- user memory vectors

Qdrant is important because the project depends on:

- semantic search
- memory retrieval
- filtered retrieval
- reranking using structured signals

Qdrant is a central part of the innovation, not just storage.

## FastAPI

FastAPI is the orchestration layer.

It:

- receives frontend requests
- embeds text
- calls Qdrant
- applies MARA logic
- calls the LLM
- returns the final response

## Hugging Face embeddings

The embedding model turns text into vectors for semantic retrieval.

This is used for:

- product search
- memory search

Without embeddings, MARA would fall back to keyword matching or rigid filtering.

## Groq and Llama 3.3

Groq provides the inference service.
Llama 3.3 generates the final user-facing answer.

Their role is presentation and explanation, not source-of-truth retrieval.


## Memory Model

MARA uses three memory types:

### Structural memory

Examples:

- maximum budget
- maximum wattage
- forbidden material
- room constraints

These should persist strongly and should not be violated.

### Semantic memory

Examples:

- likes warm light
- prefers brushed finish
- tends toward minimalist products

These should affect ranking but fade slowly.

### Episodic memory

Examples:

- recently clicked a product
- recently browsed a category

These should matter briefly and decay quickly.


## Why This Is Different From Standard RAG

Standard RAG usually retrieves by semantic similarity only.

MARA adds:

- hard-rule preservation
- memory retrieval
- decay-aware weighting

The real difference is:

```text
Baseline: Similarity(product, query)

MARA: Similarity(product, query)
    + structural compliance
    + user memory
    + memory-type weighting
    + time-based decay
```


## Product Data Strategy

The long-term architecture is:

```text
Supabase (full catalog)
    |
    v
Extraction / transformation layer
    |
    v
Canonical MARA product representation
    |
    v
Qdrant indexing
```

This means MARA should not depend directly on raw Supabase tables everywhere.
Instead, it should depend on a stable canonical product schema built from
Supabase data.


## Canonical MARA Product Representation

Each retrieval-ready product should contain:

- stable product ID
- product name
- canonical semantic description
- key technical constraint fields
- selected semantic/category fields
- lightweight display metadata

Example categories of fields:

- identity
- hard constraints
- soft semantic attributes
- display fields

Supabase remains the full record.
Qdrant stores the best searchable version of that record.


## Qdrant Payload Strategy

Qdrant should contain enough detail for high-quality retrieval, but not every
raw field from Supabase.

Too little detail:

- weak retrieval quality
- bad matching

Too much raw detail:

- noisy embeddings
- harder maintenance
- irrelevant metadata polluting search

So the rule is:

- store retrieval-relevant fields in Qdrant
- keep full product detail in Supabase


## Recommended Retrieval Split

The current MARA pattern of separating the product representation into hard and
soft retrieval views should be preserved.

### Hard/spec side

Used for:

- constraints
- filtering
- strict compatibility checks

Typical fields:

- price
- wattage
- kelvin
- material
- product type
- indoor/outdoor
- mounting class

### Soft/semantic side

Used for:

- semantic similarity
- stylistic preferences
- descriptive matching

Typical fields:

- composed product description
- category/family
- finish
- derived style tags
- use-case language


## Expected Frontend Integration

The frontend should not rely on the LLM to return the full product payload.

Preferred flow:

1. FastAPI returns ranked product IDs and MARA metadata.
2. Lovable uses those IDs to fetch or hydrate the full product details from
   Supabase.
3. Lovable renders the real product cards and detail views.

This keeps:

- retrieval in MARA/Qdrant
- source-of-truth data in Supabase
- rendering in Lovable


## File Responsibilities

| File | Role |
|------|------|
| `embeddings.py` | Loads `BAAI/bge-large-en-v1.5` (1024-dim). `embed()` adds BGE query prefix; `embed_batch()` does not (correct for asymmetric search). |
| `setup_qdrant.py` | Creates `hard_constraints` and `soft_preferences` collections, embeds catalog, uploads points, creates numeric payload indices. Re-run this when replacing the catalog. |
| `enrich_products.py` | Backfills `style`, `mood`, `finish` fields on existing Qdrant payloads using keyword rules. Payload-only update — no re-embedding. Re-run after each catalog reload. |
| `mara_engine.py` | Core retrieval and reranking: Qdrant pre-filter → `constraint_weight()` → `preference_boost()` → decay → sort. |
| `main.py` | FastAPI app. Endpoints: `POST /constraints`, `POST /browse`, `POST /chat`, plus debug routes. |
| `user_memory.py` | Reads and writes user memory (structural / semantic / episodic) to the `user_memory` Qdrant collection. |
| `extract_supabase_catalog.py` | Pulls products from Supabase and writes `catalog_export.json`. |
| `validate_catalog.py` | Checks `catalog_export.json` for schema compliance before indexing. |
| `audit_embeddings.py` | One-off health check: collection counts, random samples, semantic queries, constraint filter test, field population stats. |


### Style and Mood Enrichment

**Why it exists**

Lu's Supabase catalog has no `style`, `mood`, or `finish` fields. Without them,
`preference_boost()` in `mara_engine.py` never fires, silencing the +0.15 style
boost and the +0.10 finish boost entirely.

**How `enrich_products.py` infers the fields**

`mood` — from the numeric `kelvin` value in the payload:

- `kelvin ≤ 2700` → `"cozy"`
- `kelvin == 3000` → `"ambient"`
- `kelvin ≥ 4000` → `"focused"`
- `kelvin` absent → `"ambient"`

`style` — from keywords in `description` + `tags` + `name` (case-insensitive):

- `warm`, `warmwhite`, `2700` → `"scandinavian"`
- `outdoor`, `aussen`, `façade` → `"industrial"`
- `pendant`, `suspended`, `Pendel`, `spot`, `Strahler`, `downlight`, `profile`, `Profil`, `lichtband` → `"minimalist"`
- default → `"minimalist"`

`finish` — from keywords in `description`:

- `white`, `weiß`, `blanc` → `"white"`
- `black`, `schwarz`, `noir` → `"matte black"`
- `chrome`, `Chrom` → `"chrome"`
- `brass`, `Messing` → `"brushed brass"`
- default → `"white"`

**Important caveat**

`mood` inference reads `kelvin`, which only lives in `hard_constraints`.
The `soft_preferences` collection does not carry `kelvin`, so every product in
that collection gets `mood = "ambient"`. Because `preference_boost()` reads
from the soft payload, the mood boost (+0.05) will only fire when a user
explicitly sets `preferred_mood = "ambient"`. Style (+0.15) and finish (+0.10)
boosts are unaffected and work correctly.

**When to re-run**

Run `enrich_products.py` any time `catalog_export.json` is replaced with fresh
data from Supabase and `setup_qdrant.py` has re-indexed the collections. The
script is idempotent — it only touches fields that are currently absent.


## Implementation Direction

The migration path is:

1. inspect real Supabase product tables
2. define canonical MARA schema
3. map direct and derived fields
4. extract and normalize Supabase data
5. reindex Qdrant from real catalog data
6. adapt root retrieval logic
7. validate with real queries

The architecture goal is not just "connect Supabase".
The real goal is to preserve MARA's memory-aware retrieval logic while using
the real production catalog as the only product source.
