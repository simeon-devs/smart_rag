# MARA Context

## Project Purpose

MARA is a memory-augmented retail retrieval system for lighting products.

The goal is not to build a generic chatbot. The goal is to build a product
recommendation system that:

- understands the user's current request
- remembers the user's constraints and preferences over time
- treats different kinds of memory differently
- retrieves better products than standard similarity-only RAG

The project is centered on the idea that not all memory should behave the same.

- Hard constraints should persist and should not be violated.
- Soft preferences should influence ranking but fade slowly.
- Recent browsing should matter briefly and then fade fast.


## Main Stack

The root MARA project uses:

- FastAPI for the API layer
- Python for orchestration and business logic
- Qdrant for vector search and memory storage
- Hugging Face sentence-transformers for embeddings
- Groq for model inference
- Llama 3.3 as the response-generation LLM
- Supabase as the real product source for the future catalog integration

The `braight/` folder is not the current source of truth for MARA logic. It is
only a reference clone of the Lovable frontend project that will be connected
later.


## Core Roles Of The Systems

### FastAPI

FastAPI is the orchestrator. It receives frontend requests, embeds the query,
calls Qdrant, applies MARA logic, calls the LLM, and returns the final response.

### Hugging Face Embeddings

The embedding model turns text into vectors so that products and memories can be
searched by meaning instead of exact keyword match.

This is used for:

- product retrieval
- user memory retrieval

### Groq And Llama 3.3

Groq provides the inference layer.
Llama 3.3 is used to generate the final natural-language answer.

The LLM should not be the search engine.
The LLM should explain and present the products chosen by MARA.

### Qdrant

Qdrant is the core retrieval engine of the project.

It stores:

- product vectors
- user memory vectors

It enables:

- semantic search over products
- memory retrieval by relevance
- the baseline vs MARA comparison
- constraint-aware retrieval workflows

Without Qdrant, the project loses its main technical point.


## Current Root Architecture

Current root files:

- `main.py` wires the API together
- `mara_engine.py` contains baseline retrieval and MARA reranking logic
- `user_memory.py` stores and retrieves structural, semantic, and episodic memory
- `embeddings.py` provides the embedding model
- `extract_supabase_catalog.py` exports the real catalog into the canonical MARA schema
- `setup_qdrant.py` indexes the canonical catalog into Qdrant
- `CATALOG_SCHEMA.md` defines the canonical MARA product representation

The canonical MARA schema is defined separately in `CATALOG_SCHEMA.md`.
It includes:

- product identity
- pricing
- technical fields
- classification fields
- semantic fields
- embedding-ready semantic description
- media references


## Current Request Flow

When a user sends a request:

1. The frontend sends the request to FastAPI.
2. FastAPI embeds the user message.
3. FastAPI queries Qdrant for product candidates.
4. FastAPI also retrieves user memory from Qdrant.
5. MARA logic applies:
   - hard constraint filtering
   - soft preference boosting
   - decay-aware memory handling
6. FastAPI builds the final product shortlist.
7. FastAPI sends the shortlisted products and user context to the LLM.
8. The LLM writes the final natural-language answer.
9. FastAPI returns:
   - `llm_reply`
   - `baseline_results`
   - `mara_results`
   - `violation_count`
   - `user_context`


## Why Qdrant Matters

Qdrant is important because MARA is fundamentally a retrieval system, not just a
chat layer.

Standard RAG retrieves by similarity only.
MARA retrieves by similarity plus structured memory logic.

The real differentiator is:

- product retrieval
- memory retrieval
- different treatment of different memory types

Qdrant is what makes those capabilities practical.


## Real Innovation

The real innovation is not "LLM plus products".

The real innovation is a memory architecture where:

- hard constraints persist and must not be violated
- soft preferences matter but decay slowly
- episodic interactions fade quickly

This changes retrieval from:

`Similarity(product, query)`

to a richer ranking process that includes:

- semantic relevance
- structural constraint compliance
- memory type
- time-based decay


## Product Data Direction

The active product path is now based on Supabase data, extracted into the
canonical MARA schema before indexing.

The real production catalog is stored in Supabase.
The minimum important source tables are:

- `articles`
- `article_technical_profiles`
- `article_character_profiles`
- `article_classifications`

Supporting tables:

- `manufacturers`
- `light_categories`
- `light_families`
- image/media references

The real Supabase schema is more technical and relational than the original
flat MARA prototype schema.

This means the migration is not just "connect to Supabase".
It is a schema adaptation problem.


## Future Architecture With Supabase

The intended future flow is:

1. Supabase remains the source of truth for the full product catalog.
2. A curated product representation is built from Supabase data.
3. That curated representation is indexed into Qdrant.
4. FastAPI uses Qdrant to retrieve and rank product candidates.
5. FastAPI returns product IDs and MARA metadata.
6. Lovable can fetch or hydrate the full product details from Supabase.

This separates responsibilities clearly:

- Supabase = full product database
- Qdrant = retrieval-optimized catalog and memory layer
- FastAPI = orchestration and MARA logic
- Lovable = UI rendering


## What Qdrant Should Store

Qdrant should not store the full raw Supabase schema blindly.

Qdrant should store a retrieval-optimized representation of each product:

- stable product ID
- product name
- semantic text for embeddings
- hard constraint fields
- selected soft or categorical fields
- lightweight fallback display fields

Qdrant should contain enough detail for strong retrieval, filtering, and
reranking, but Supabase should remain the master source for full product detail.

Practical rule:

- Supabase = source of truth
- Qdrant = searchable representation


## Expected Migration Strategy

The Supabase-to-MARA flow should follow this order:

1. identify the exact Supabase source tables and fields
2. define a canonical MARA product schema based on real data
3. map direct fields and derived fields
4. build an extraction layer from Supabase to MARA-ready product records
5. reindex Qdrant from the real catalog
6. adapt MARA retrieval logic where weak or unavailable fields no longer apply
7. validate with real queries and real products


## Important Product-Level Truth

This project is not trying to let the LLM decide what products to recommend on
its own.

The intended logic is:

- embeddings create semantic vectors
- Qdrant finds product and memory candidates
- MARA logic filters and reranks them
- the LLM explains the result

So the LLM is the presentation layer for the final recommendation, not the
source of truth for retrieval.
