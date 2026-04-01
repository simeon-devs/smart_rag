# MARA — Architecture

## Data Flow

```
User message (Lovable frontend)
        │
        ▼
POST /chat  (FastAPI — main.py)
        │
        ├──► embed(message)          → 1024-dimensional vector
        │         (embeddings.py)
        │
        ├──► run_baseline(vector)    → top 5 products, pure similarity
        │         (mara_engine.py)      NO constraints, NO decay
        │
        ├──► run_mara(vector,        → top 5 products, constraint-filtered
        │      constraints,             decay-weighted, preference-boosted
        │      preferences)
        │         (mara_engine.py)
        │
        ├──► get_user_context(       → what MARA learned about this user
        │      user_id, message)        structural + semantic + episodic
        │         (user_memory.py)
        │
        ├──► call_groq(              → natural language reply
        │      system_prompt,           "Based on your style, here are..."
        │      user_message)
        │         (main.py)
        │
        └──► ChatResponse {
               llm_reply,           → chat bubble text
               baseline_results,    → left panel (standard RAG)
               mara_results,        → right panel (MARA)
               violation_count,     → live counter badge
               user_context,        → memory summary
             }
```

## Three Memory Types

```
┌─────────────────────────────────────────────────────────────┐
│                    QDRANT COLLECTIONS                        │
├──────────────────┬──────────────────┬───────────────────────┤
│ hard_constraints │ soft_preferences │ user_memory           │
│                  │                  │                       │
│ Product catalog  │ Product catalog  │ User interactions     │
│ (specs side)     │ (style side)     │ (all 3 types)         │
│                  │                  │                       │
│ wattage          │ style            │ STRUCTURAL (λ=0.01)   │
│ price_chf        │ finish           │ "max 40W"             │
│ kelvin           │ mood             │ "no plastic"          │
│ material         │ description      │ "budget 200 CHF"      │
│                  │                  │                       │
│                  │                  │ SEMANTIC (λ=0.10)     │
│                  │                  │ "loves scandinavian"  │
│                  │                  │ "prefers warm light"  │
│                  │                  │                       │
│                  │                  │ EPISODIC (λ=0.30)     │
│                  │                  │ "clicked brass sconce"│
│                  │                  │ "browsed floor lamps" │
└──────────────────┴──────────────────┴───────────────────────┘
```

## Decay Formula

```python
FinalScore = Similarity(product, query)
           × StructuralWeight(constraints)   # 1.0 = pass, 0.0 = fail
           × DecayFunction(memory_type, t)   # e^(-λ × days)
```

**Decay rates by memory type:**

```
Day     hard (λ=0.01)   soft (λ=0.10)   episodic (λ=0.30)
─────   ─────────────   ─────────────   ─────────────────
0       1.000           1.000           1.000
7       0.932           0.497           0.122
30      0.741           0.050           0.000
90      0.407           0.000           0.000
180     0.165           0.000           0.000
```

Hard constraints are still at 74% after 30 days.
Style preferences fade to 5% after 30 days.
Episodic browsing is essentially gone after 7 days.

## File Responsibilities

```
products.json
  └─ 30 lighting products with all attributes
  └─ Owner: Simeon (Lu replaces with real data)

setup_qdrant.py
  └─ Runs ONCE — creates collections, indexes all 30 products
  └─ Creates: hard_constraints + soft_preferences collections
  └─ Owner: Simeon

mara_engine.py
  └─ Core MARA logic — never touches the API layer
  └─ decay()              → exponential decay calculation
  └─ constraint_weight()  → 1.0 (pass) or 0.0 (fail) + violations list
  └─ preference_boost()   → small score boost for matching soft preferences
  └─ run_baseline()       → pure cosine similarity, no logic
  └─ run_mara()           → full pipeline: filter → decay → boost → rerank
  └─ Owner: Simeon

user_memory.py
  └─ Stores + retrieves user interactions from Qdrant
  └─ save_constraints_as_memory() → called after POST /constraints
  └─ save_browse_as_memory()      → called after POST /browse
  └─ save_chat_preference()       → called after POST /chat
  └─ get_user_context()           → called inside POST /chat
  └─ Owner: Simeon

embeddings.py
  └─ One job: text → 1024-dimensional vector
  └─ Model: BAAI/bge-large-en-v1.5 (downloaded locally, ~1.3GB)
  └─ embed(text)        → single text to vector
  └─ embed_batch(texts) → multiple texts, faster
  └─ Owner: Nursena

main.py
  └─ FastAPI — wires all modules together
  └─ POST /constraints  → save rules + structural memory
  └─ POST /browse       → log click + episodic memory
  └─ POST /chat         → search + memory + LLM + save preferences
  └─ GET /debug/memory  → inspect user memory (demo moment)
  └─ Owner: Nursena (API structure) + Simeon (memory integration)
```

## How Nursena Plugs In Her Embedding Model

In `mara_engine.py`, `user_memory.py`, and `main.py` there is a mock
embedding function. Nursena replaces it with one import:

```python
# ADD at the top of the file
from embeddings import embed

# DELETE the mock functions below (about 10 lines each)
def _mock_embed(...): ...
def embed(...): return _mock_embed(...)
```

That's the entire integration. The function signature `embed(text) -> list[float]`
is identical — nothing else changes.

## How Lu Connects the Frontend

Three API calls from Lovable:

```javascript
// 1. When judge sets constraints (once at start)
POST http://localhost:8001/constraints
Body: { user_id, max_wattage, max_price_chf, forbidden_materials, kelvin_min, kelvin_max }

// 2. When user clicks a product
POST http://localhost:8001/browse
Body: { user_id, product_id, name, description }

// 3. When user sends a chat message
POST http://localhost:8001/chat
Body: { user_id, message, preferred_style, preferred_mood }

// Response fields Lu uses:
response.llm_reply          → chat bubble text
response.mara_results       → right panel product cards
response.baseline_results   → left panel product cards
response.violation_count    → live counter badge number
response.user_context.summary → memory panel text
```

## Qdrant Collections Summary

```
Collection          Contents              Vector text           λ
──────────────────  ────────────────────  ────────────────────  ────
hard_constraints    product catalog       name + specs          0.01
soft_preferences    product catalog       full description      0.10
user_memory         user interactions     constraint/pref text  varies
```

All collections use:
- Distance: COSINE
- Vector size: 1024
- Index: HNSW (default)
