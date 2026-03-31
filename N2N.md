# MARA — N2N Integration Guide

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           USER BROWSER                               │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  braight (React/Vite)  :8080                                  │   │
│  │                                                               │   │
│  │  Session identity:                                            │   │
│  │    • logged-in user  → maraUserId = supabase user.id          │   │
│  │    • guest           → maraUserId = sessionStorage "mara_sid" │   │
│  │                         (UUID generated once per browser tab) │   │
│  │                                                               │   │
│  │  ChatWindow → POST /extract     (constraint detection)        │   │
│  │            → POST /chat         (MARA recommendation)         │   │
│  │            → POST /browse       (episodic memory)             │   │
│  │            → POST /constraints  (hard constraint save)        │   │
│  │                                                               │   │
│  │  Product cards hydrated from Supabase using                   │   │
│  │  source_article_id returned by MARA                           │   │
│  └───────────────────────────────────────────────────────────────┘   │
└────────────────────────┬───────────────────────┬─────────────────────┘
                         │ HTTP                  │ HTTP
                         ▼                       ▼
      ┌───────────────────────────┐   ┌─────────────────────┐
      │  mara_backend             │   │  Supabase            │
      │  FastAPI  :8001           │   │  (product catalog)   │
      │                           │   │                      │
      │  /chat  /extract          │   │  articles            │
      │  /constraints  /browse    │   │  profiles            │
      │  /debug/*                 │   │  wishlists/projects  │
      │                           │   │  product_interactions│
      │  In-process stores:       │   └─────────────────────┘
      │  • constraints_store      │
      │  • browsing_store         │
      │  • conversation_store     │  ← rolling 6-turn history
      │  • style_timestamp_store  │     auto-summarized by Groq
      └────────────┬──────────────┘
                   │
           ┌───────┴────────┐
           │                │
           ▼                ▼
     ┌──────────┐    ┌──────────────┐
     │  Qdrant  │    │  Groq API    │
     │  :6333   │    │  Llama 3.3   │
     │          │    │  70B         │
     │ hard_    │    │              │
     │ constr.  │    │  ① extract   │
     │ soft_    │    │  ② generate  │
     │ prefs    │    │  ③ summarize │
     │ user_    │    └──────────────┘
     │ memory   │
     └──────────┘
```

---

## How the N2N Flow Works

1. **User types a message** in the chat (e.g. "warm ceiling light under 300 CHF")
2. **Frontend resolves user identity** — uses Supabase `user.id` if logged in, otherwise
   reads/generates a `guest_<uuid>` from `sessionStorage` (key: `mara_sid`).
   Each browser tab gets its own isolated memory — no more shared constraints.
3. **Frontend calls `/extract`** — Groq detects constraints (max_price_chf: 300, kelvin_max: 2700)
4. **Frontend shows constraint chips** — user confirms or skips
5. **Frontend calls `/chat`** — MARA:
   - Embeds the query (BAAI/bge-large-en-v1.5, 1024-dim)
   - Retrieves candidates from Qdrant (hard_constraints + soft_preferences)
   - Applies constraint filtering (wattage, price, kelvin, location)
   - Applies preference boosts (style, finish, mood with exponential decay)
   - Loads user memory from Qdrant (structural / semantic / episodic)
   - Loads conversation history from `conversation_store` (up to 6 prior turns)
   - Generates LLM reply via Groq with full context
   - Saves the new exchange to conversation history
   - Returns ranked products + hydration metadata
6. **Frontend hydrates products** from Supabase using `source_article_id`
7. **Frontend renders** MARA-ranked product cards with full Supabase data
8. **User clicks a product** → `/browse` saves it as episodic memory in Qdrant
9. **Next query** uses accumulated memory + conversation history for better personalization

---

## Memory System

MARA maintains three distinct memory types per user, all stored as 1024-dim vectors in Qdrant:

| Type | λ (decay) | What is stored | Lifetime |
|------|-----------|---------------|----------|
| `structural` | 0.01 | Hard constraints (budget, wattage, location, kelvin) | ~permanent |
| `semantic` | 0.10 | Learned style/mood/finish preferences | ~weeks |
| `episodic` | 0.30 | Recently viewed products | ~3 days |

Decay formula: `score × exp(−λ × days_elapsed)`

Memories are retrieved by **semantic similarity** (cosine distance between the query vector and memory vectors), then re-ranked by the decayed score. This means contextually relevant old memories surface even if they use different words.

### Conversation Memory

In addition to the Qdrant memory layer, `conversation_store` keeps a rolling chat history in the FastAPI process:

- Last **6 exchanges** (12 messages) are passed to Groq as `messages[]` context
- When the limit is exceeded, older turns are **compressed into a summary** via a dedicated Groq call and stored as a `system` message
- This gives MARA short-term conversational coherence (e.g. "what was the second option?")
- `conversation_store` is in-process only — it resets on server restart. Persistent preferences live in Qdrant.

---

## User Identity & Session Isolation

### How `maraUserId` is resolved (frontend)

```typescript
// braight/src/pages/Index.tsx

function getGuestSessionId(): string {
  const KEY = 'mara_sid';
  let sid = sessionStorage.getItem(KEY);
  if (!sid) {
    sid = `guest_${crypto.randomUUID()}`;
    sessionStorage.setItem(KEY, sid);
  }
  return sid;
}

// Inside the component:
const maraUserId = useMemo(() => user?.id ?? getGuestSessionId(), [user?.id]);
```

| Scenario | maraUserId value | Scope |
|----------|-----------------|-------|
| Logged-in user | Supabase `user.id` (UUID) | Persists across devices/sessions |
| Guest (same tab) | `guest_<uuid>` from sessionStorage | Lives until tab is closed |
| Guest (new tab) | New `guest_<uuid>` | Fully isolated from other tabs |

This eliminates the previously shared `braight_user_01` user ID that caused all visitors to pollute each other's memory.

---

## API Reference

### POST /extract
Detect constraints in a user message. Returns UI chips for confirmation. **Does not save anything.**

```json
{ "user_id": "guest_abc123", "message": "something under 200 CHF outdoors" }
```

```json
{
  "suggestions": [
    { "field": "max_price_chf", "label": "Budget max 200 CHF?", "value": 200 },
    { "field": "location",      "label": "Outdoor use?",        "value": "outdoor" }
  ]
}
```

### POST /constraints
Save confirmed hard constraints. Merges with existing values; each field is stored as a single structural memory in Qdrant (old value deleted before new one is written).

```json
{ "user_id": "guest_abc123", "max_price_chf": 200, "location": "outdoor" }
```

### POST /chat
Full MARA pipeline: embed → retrieve → rank → load memory → load history → generate → save history.

```json
{ "user_id": "guest_abc123", "message": "show me something for my garden" }
```

Response includes `llm_reply`, `mara_results`, `baseline_results`, and `hydration` metadata.

### POST /browse
Log a product view as episodic memory (fire-and-forget from frontend).

```json
{ "user_id": "guest_abc123", "product_id": "article_42", "name": "Arc Lamp", "description": "..." }
```

### GET /debug/memory/{user_id}
Inspect all current Qdrant memories for a user.

### DELETE /debug/memory/{user_id}
Wipe all memories for a user: Qdrant entries + `constraints_store` + `browsing_store` + `style_timestamp_store` + `conversation_store`.

```bash
# Reset before a demo session
curl -X DELETE http://localhost:8001/debug/memory/guest_abc123
```

---

## Quick Start (Local, no Docker)

### Prerequisites
- Python 3.11+
- Node.js 20+
- A Groq API key → https://console.groq.com
- Qdrant Cloud cluster OR local Qdrant (see below)

### 1. Configure backend secrets

Edit `mara_backend/.env`:

```env
# Qdrant Cloud
QDRANT_URL=https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# OR local Qdrant (start with: docker run -p 6333:6333 qdrant/qdrant)
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=

# Groq
GROQ_API_KEY=gsk_your_key_here

# HuggingFace (optional — only needed if model download fails without auth)
HF_TOKEN=

# Supabase — already configured
SUPABASE_URL=https://xgjiulkqwqxprgvlzpld.supabase.co
SUPABASE_ANON_KEY=...

# Additional CORS origins (optional, comma-separated)
# CORS_ORIGINS=https://your-production-domain.com
```

### 2. One-time catalog setup (run once)

```bash
./setup.sh
```

This:
1. Extracts the product catalog from Supabase → `catalog_export.json`
2. Embeds all products and indexes them in Qdrant (downloads ~1.3GB model on first run)
3. Enriches missing style/mood/finish fields
4. Runs a quality audit on the indexed data

### 3. Start everything

```bash
./start.sh
```

| Service  | URL                             |
|----------|---------------------------------|
| Frontend | http://localhost:8080           |
| Backend  | http://localhost:8001           |
| API Docs | http://localhost:8001/docs      |
| Qdrant   | http://localhost:6333/dashboard |

---

## Quick Start (Docker Compose)

```bash
# Start all services
make up

# One-time catalog setup
make index
make enrich

# View logs
make logs

# Stop
make down
```

---

## Demo Reset

To start a fresh demo session without stale memories from a previous run:

```bash
# If you know the user ID (shown in browser console as [MARA] → /chat logs)
curl -X DELETE http://localhost:8001/debug/memory/<user_id>

# For the old shared ID (if you were using a previous version)
curl -X DELETE http://localhost:8001/debug/memory/braight_user_01
```

With the current version this is rarely needed — each browser session gets its own guest UUID automatically.

---

## Environment Variables Reference

### `mara_backend/.env`

| Variable | Required | Description |
|----------|----------|-------------|
| `QDRANT_URL` | Yes | Qdrant cluster URL |
| `QDRANT_API_KEY` | Cloud only | Qdrant API key |
| `GROQ_API_KEY` | Yes | Groq LLM API key |
| `HF_TOKEN` | No | HuggingFace token (model download) |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_ANON_KEY` | Yes | Supabase anon key |
| `CORS_ORIGINS` | No | Additional CORS origins (comma-separated) |

### `braight/.env`

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_SUPABASE_URL` | Yes | Supabase URL |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Yes | Supabase anon key |
| `VITE_MARA_BASE_URL` | Yes | MARA backend URL (default: `http://localhost:8001`) |

---

## Folder Structure

```
START/
├── braight/                        ← Frontend (React 18 / Vite / TypeScript)
│   ├── src/
│   │   ├── lib/maraApi.ts          ← MARA API client (extract/chat/browse/constraints)
│   │   ├── pages/Index.tsx         ← Main page — maraUserId resolution + MARA integration
│   │   └── components/             ← UI components (ChatWindow, ProductCard, ...)
│   └── .env                        ← Frontend secrets
│
├── mara_backend/                   ← Backend (FastAPI / Python 3.11)
│   ├── main.py                     ← API endpoints + agent orchestration
│   │                                  conversation_store (rolling history + auto-summarize)
│   ├── mara_engine.py              ← Retrieval (Qdrant) + reranking + decay scoring
│   ├── user_memory.py              ← Qdrant memory CRUD (structural/semantic/episodic)
│   ├── embeddings.py               ← BAAI/bge-large-en-v1.5 (1024-dim vectors)
│   ├── extract_supabase_catalog.py ← One-time: Supabase → catalog_export.json
│   ├── setup_qdrant.py             ← One-time: catalog_export.json → Qdrant index
│   ├── enrich_products.py          ← One-time: backfill style/mood/finish payloads
│   ├── audit_embeddings.py         ← QA: verify semantic search + constraint filtering
│   ├── validate_catalog.py         ← QA: field coverage stats before indexing
│   └── .env                        ← Backend secrets
│
├── agents/
│   └── mara.yml                    ← MARA agent catalog definition
│
├── flows/
│   └── mara.yml                    ← MARA recommendation flow definition
│
├── Mara/                           ← Old Mistral-based prototype (not used)
│
├── docker-compose.yml              ← Full N2N Docker setup (qdrant + backend + frontend)
├── Makefile                        ← Common commands (up / down / index / enrich / logs)
├── start.sh                        ← Local startup without Docker
├── setup.sh                        ← One-time catalog indexing pipeline
└── N2N.md                          ← This file
```

---

## Changelog

### 2026-04-01
- **Session isolation** — replaced hardcoded `braight_user_01` user ID with per-session identity:
  logged-in users use their Supabase `user.id`; guests get a `guest_<uuid>` stored in
  `sessionStorage` (one per browser tab, auto-generated on first interaction).
- **Conversation memory** — backend now maintains a rolling LLM conversation history per user
  (`MAX_HISTORY_TURNS = 6`). History is passed to Groq as `messages[]` context on every `/chat`
  call, enabling multi-turn coherence. When the buffer exceeds 12 messages, older turns are
  compressed into a summary via a dedicated Groq call.
- **Full reset endpoint** — `DELETE /debug/memory/{user_id}` now wipes Qdrant entries,
  `constraints_store`, `browsing_store`, `style_timestamp_store`, and `conversation_store`
  in one call.
- **`delete_all_user_memory()`** added to `user_memory.py`.

### 2026-03-17
- Initial N2N integration: braight frontend + mara_backend wired together
- Docker Compose, Makefile, start.sh, setup.sh
- CORS configured for localhost + Lovable preview URLs
- Episode memory (`/browse`) hooked into product click handler
