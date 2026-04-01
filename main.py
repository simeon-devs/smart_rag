"""FastAPI application for MARA product retrieval and memory orchestration."""

import os
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from mara_engine import (
    run_baseline,
    run_mara,
    UserConstraints,
    UserPreferences,
    ScoredProduct,
)
from user_memory import (
    save_constraints_as_memory,
    save_browse_as_memory,
    save_chat_preference,
    get_user_context,
    delete_all_user_memory,
)
from embeddings import describe_embedding_backend, embed, validate_embedding_config


MAX_HISTORY_TURNS = 6  # keep last 6 exchanges (12 messages) in context
_SUMMARY_PROMPT = (
    "Summarize the following conversation in 3-5 concise sentences, "
    "preserving only the user's key constraints, preferences, and decisions. "
    "Be factual and brief — no greetings or filler."
)


def call_groq(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
) -> str:
    """Return a natural-language answer from Groq or a fallback message."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "LLM not configured — add GROQ_API_KEY to .env. "
            "Product results are still available above."
        )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            temperature = 0.6,
            max_tokens  = 400,
            messages    = messages,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Groq error: {str(e)}"


def _summarize_history(history: list[dict]) -> str:
    """Use Groq to produce a short summary of an older conversation chunk."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fallback: just concatenate the last few exchanges as plain text
        lines = [f"{m['role'].upper()}: {m['content']}" for m in history[-6:]]
        return " | ".join(lines)
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        convo_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            temperature = 0.2,
            max_tokens  = 200,
            messages    = [
                {"role": "system", "content": _SUMMARY_PROMPT},
                {"role": "user",   "content": convo_text},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        lines = [f"{m['role'].upper()}: {m['content']}" for m in history[-4:]]
        return " | ".join(lines)


def build_llm_prompt(
    user_context: dict,
    mara_products: list,
    baseline_products: list,
) -> str:
    """Build the prompt passed to the LLM."""
    mara_lines = []
    for i, p in enumerate(mara_products[:3], 1):
        name = p.get("name", "?")
        details = []

        price = p.get("price_chf")
        if price is not None:
            details.append(f"{price} CHF")

        watt = p.get("wattage")
        if watt is not None:
            details.append(f"{watt}W")

        kelvin = p.get("kelvin")
        if kelvin is not None:
            details.append(f"{int(kelvin) if float(kelvin).is_integer() else kelvin}K")

        finish = p.get("finish")
        if finish:
            details.append(str(finish))

        manufacturer = p.get("manufacturer")
        if manufacturer:
            details.append(str(manufacturer))

        mara_lines.append(f"  {i}. {name}" + (f" — {', '.join(details)}" if details else ""))

    baseline_top = baseline_products[0]["name"] if baseline_products else "unknown"

    return f"""You are MARA, a memory-augmented lighting assistant.
You remember this user's preferences and constraints across sessions.

WHAT YOU KNOW ABOUT THIS USER:
{user_context.get("summary", "No prior context.")}

YOUR TOP RECOMMENDATIONS (constraint-aware):
{chr(10).join(mara_lines) if mara_lines else "No matching products found."}

WITHOUT YOUR MEMORY (standard search would suggest):
  {baseline_top}

YOUR RULES:
1. Be warm, concise, and confident — max 3 sentences.
2. Reference the user's specific constraints or style preferences naturally.
3. Recommend from YOUR list above, not from the standard search result.
4. Never mention "MARA", "baseline", "vectors", or technical terms.
5. If no products match, say so honestly and suggest relaxing a constraint."""

app = FastAPI(
    title       = "MARA API",
    description = "Memory-Augmented Retail Agent — Qdrant-powered lighting recommendations",
    version     = "2.0.0",
)

_CORS_ORIGINS = [
    "http://localhost:8080",   # Vite dev server
    "http://localhost:5173",   # Vite alt port
    "http://localhost:3000",   # CRA fallback
    "http://127.0.0.1:8080",
]
# Allow additional origins from env (e.g. production Vercel/Lovable URL)
_extra = os.getenv("CORS_ORIGINS", "")
if _extra:
    _CORS_ORIGINS += [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _CORS_ORIGINS,
    allow_origin_regex = r"https://.*\.lovable\.app",   # Lovable preview URLs
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.on_event("startup")
async def startup_checks() -> None:
    """Validate critical runtime integrations before serving traffic."""
    validate_embedding_config()
    print(f"[startup] Embeddings ready via {describe_embedding_backend()}")

# In-memory state for the current process. Persistent memory lives in Qdrant.
constraints_store:  dict[str, UserConstraints] = {}
browsing_store:     dict[str, list[dict]]       = {}
# Tracks when each user first expressed a given style/finish/mood preference.
# Keys: user_id → {"preferred_style": datetime, "preferred_finish": datetime, ...}
style_timestamp_store: dict[str, dict[str, datetime]] = {}
# Rolling LLM conversation history per user.
# Each entry: {"role": "user"|"assistant", "content": str}
# Older turns are compressed into a summary message when MAX_HISTORY_TURNS is exceeded.
conversation_store: dict[str, list[dict]] = {}


class ConstraintsRequest(BaseModel):
    user_id:             str
    max_wattage:         Optional[float] = None
    max_price_chf:       Optional[float] = None
    forbidden_materials: list[str]       = Field(default_factory=list)
    kelvin_min:          Optional[float] = None
    kelvin_max:          Optional[float] = None
    room_type:           Optional[str]   = None
    location:            Optional[str]   = None   # "outdoor" | "indoor"

class BrowseRequest(BaseModel):
    user_id:     str
    product_id:  str
    name:        str
    description: str

class ChatRequest(BaseModel):
    user_id:          str
    message:          str
    preferred_style:  Optional[str] = None
    preferred_finish: Optional[str] = None
    preferred_mood:   Optional[str] = None

class ProductResult(BaseModel):
    product_id:       str
    source_article_id: Optional[int] = None
    source_article_number: Optional[str] = None
    source_l_number:  Optional[int] = None
    name:             str
    manufacturer:     Optional[str] = None
    category:         Optional[str] = None
    family:           Optional[str] = None
    price_chf:        Optional[float] = None
    wattage:          Optional[float] = None
    kelvin:           Optional[float] = None
    material:         Optional[str] = None
    style:            Optional[str] = None
    finish:           Optional[str] = None
    mood:             Optional[str] = None
    room_type:        Optional[str] = None
    image_url:        Optional[str] = None
    tags:             list[str] = Field(default_factory=list)
    similarity_score: float
    final_score:      float
    violations:       list[str] = Field(default_factory=list)


class HydrationTarget(BaseModel):
    rank:                  int
    product_id:            str
    source_article_id:     Optional[int] = None
    source_article_number: Optional[str] = None
    source_l_number:       Optional[int] = None


class FrontendHydration(BaseModel):
    provider:              str
    preferred_key:         str
    ordered_article_ids:   list[int] = Field(default_factory=list)
    ranked_targets:        list[HydrationTarget] = Field(default_factory=list)


class FrontendContractInfo(BaseModel):
    version:               str
    primary_result_field:  str
    hydration_provider:    str
    hydration_key:         str
    baseline_available:    bool
    merge_strategy:        str


class ChatResponse(BaseModel):
    user_id:          str
    query:            str
    llm_reply:        str
    baseline_results: list[dict]
    mara_results:     list[ProductResult]
    violation_count:  int
    constraints_used: dict
    user_context:     dict
    hydration:        FrontendHydration
    frontend:         FrontendContractInfo


class ConstraintSuggestion(BaseModel):
    field:   str
    label:   str
    value:   object
    options: list[str] = Field(default_factory=lambda: ["Yes", "Skip"])

class ExtractRequest(BaseModel):
    user_id: str
    message: str

class ExtractResponse(BaseModel):
    user_id:     str
    message:     str
    suggestions: list[ConstraintSuggestion]


_EXTRACT_SYSTEM_PROMPT = (
    "You are a constraint extractor for a lighting recommendation system.\n"
    "Read the user message and extract ONLY constraints the user explicitly\n"
    "mentioned or clearly implied. Do not invent constraints.\n\n"
    "Return a JSON array only. Each item has exactly these fields:\n"
    "  field: one of [max_price_chf, max_wattage, kelvin_max, kelvin_min,\n"
    "                 room_type, forbidden_materials, location]\n"
    "  label: short confirmation question in the same language as the user\n"
    "  value: extracted value (number, string, or array of strings)\n\n"
    "CRITICAL — price vs wattage disambiguation:\n"
    "  - Wattage is ALWAYS between 1W and 200W maximum for lighting products.\n"
    "  - If the number is above 200 and has no explicit 'W' or 'watt' unit →\n"
    "    it is ALWAYS max_price_chf, NEVER max_wattage.\n"
    "  - Never save a value above 200 as max_wattage under any circumstances.\n"
    "  - If the user mentions CHF, Fr., francs, budget, cost, price, afford, spend,\n"
    "    or any monetary context → field: max_price_chf\n"
    "  - If the user mentions watts, W, watt, power consumption, energy → field: max_wattage\n"
    "  - A bare number with no unit (e.g. '1500', 'under 150') in a product-search\n"
    "    context means budget → field: max_price_chf\n\n"
    "Conversion rules:\n"
    "  'nothing over 200 CHF' → field: max_price_chf, value: 200\n"
    "  '1500 for a studio'    → field: max_price_chf, value: 1500\n"
    "  'under 150'            → field: max_price_chf, value: 150\n"
    "  'warm light'           → field: kelvin_max, value: 2700\n"
    "  'cool white'           → field: kelvin_min, value: 4000\n"
    "  'for the bedroom'      → field: room_type, value: 'bedroom'\n"
    "  'no plastic'           → field: forbidden_materials, value: ['plastic']\n"
    "  'outdoor'              → field: location, value: 'outdoor'\n"
    "  'max 40 watts'         → field: max_wattage, value: 40\n\n"
    "If nothing specific was mentioned → return []\n"
    "Return only the JSON array. No explanation. No markdown. No code blocks."
)


def get_constraints(user_id: str) -> UserConstraints:
    return constraints_store.get(user_id, UserConstraints())


def get_history(user_id: str) -> list[dict]:
    return conversation_store.get(user_id, [])


def append_to_history(user_id: str, user_msg: str, assistant_msg: str) -> None:
    """Add the latest exchange and compress older turns if limit is exceeded."""
    history = conversation_store.setdefault(user_id, [])
    history.append({"role": "user",      "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})

    max_msgs = MAX_HISTORY_TURNS * 2
    if len(history) > max_msgs:
        older  = history[:-max_msgs]
        recent = history[-max_msgs:]
        summary = _summarize_history(older)
        conversation_store[user_id] = [
            {"role": "system", "content": f"[Earlier conversation summary: {summary}]"},
            *recent,
        ]


def _record_style_timestamps(user_id: str, overrides: dict) -> None:
    """Record the first time this user expressed each style preference."""
    now = datetime.now(timezone.utc)
    if user_id not in style_timestamp_store:
        style_timestamp_store[user_id] = {}
    store = style_timestamp_store[user_id]
    for key in ("preferred_style", "preferred_finish", "preferred_mood"):
        if overrides.get(key) and key not in store:
            store[key] = now


def _style_age_days(user_id: str) -> float:
    """Return how many days ago the oldest active style preference was expressed."""
    store = style_timestamp_store.get(user_id, {})
    if not store:
        return 0.0
    now   = datetime.now(timezone.utc)
    oldest = min(store.values())
    return (now - oldest).total_seconds() / 86400


def get_preferences(user_id: str, overrides: dict) -> UserPreferences:
    history      = browsing_store.get(user_id, [])
    browsing_age = 0.0

    if history:
        last  = history[-1]["timestamp"]
        now   = datetime.now(timezone.utc)
        delta = now - datetime.fromisoformat(last)
        browsing_age = delta.total_seconds() / 86400

    _record_style_timestamps(user_id, overrides)

    return UserPreferences(
        preferred_style   = overrides.get("preferred_style"),
        preferred_finish  = overrides.get("preferred_finish"),
        preferred_mood    = overrides.get("preferred_mood"),
        style_age_days    = _style_age_days(user_id),
        browsing_age_days = browsing_age,
    )


def scored_to_model(p: ScoredProduct) -> ProductResult:
    return ProductResult(
        product_id       = p.product_id,
        source_article_id = p.source_article_id,
        source_article_number = p.source_article_number,
        source_l_number = p.source_l_number,
        name             = p.name,
        manufacturer     = p.manufacturer,
        category         = p.category,
        family           = p.family,
        price_chf        = p.price_chf,
        wattage          = p.wattage,
        kelvin           = p.kelvin,
        material         = p.material,
        style            = p.style,
        finish           = p.finish,
        mood             = p.mood,
        room_type        = p.room_type,
        image_url        = p.image_url,
        tags             = p.tags,
        similarity_score = p.similarity_score,
        final_score      = p.final_score,
        violations       = p.violations,
    )


def build_hydration_payload(products: list[ProductResult]) -> FrontendHydration:
    ranked_targets = [
        HydrationTarget(
            rank=index,
            product_id=product.product_id,
            source_article_id=product.source_article_id,
            source_article_number=product.source_article_number,
            source_l_number=product.source_l_number,
        )
        for index, product in enumerate(products, start=1)
    ]
    ordered_article_ids = [
        product.source_article_id
        for product in products
        if product.source_article_id is not None
    ]
    return FrontendHydration(
        provider="supabase",
        preferred_key="source_article_id",
        ordered_article_ids=ordered_article_ids,
        ranked_targets=ranked_targets,
    )

@app.get("/")
def root():
    return {
        "project":   "MARA",
        "version":   "2.0.0",
        "status":    "running",
        "endpoints": ["/constraints", "/browse", "/chat"],
    }


@app.post("/constraints")
def save_constraints(req: ConstraintsRequest):
    """Store explicit hard constraints for the user, merging with existing values."""
    existing = constraints_store.get(req.user_id, UserConstraints())
    constraints = UserConstraints(
        max_wattage         = req.max_wattage         if req.max_wattage         is not None else existing.max_wattage,
        max_price_chf       = req.max_price_chf       if req.max_price_chf       is not None else existing.max_price_chf,
        forbidden_materials = req.forbidden_materials if req.forbidden_materials else existing.forbidden_materials,
        kelvin_min          = req.kelvin_min          if req.kelvin_min          is not None else existing.kelvin_min,
        kelvin_max          = req.kelvin_max          if req.kelvin_max          is not None else existing.kelvin_max,
        room_type           = req.room_type           if req.room_type           is not None else existing.room_type,
        location            = req.location            if req.location            is not None else existing.location,
    )
    constraints_store[req.user_id] = constraints

    save_constraints_as_memory(req.user_id, {
        "max_wattage":         req.max_wattage,
        "max_price_chf":       req.max_price_chf,
        "forbidden_materials": req.forbidden_materials,
        "kelvin_min":          req.kelvin_min,
        "kelvin_max":          req.kelvin_max,
        "room_type":           req.room_type,
        "location":            req.location,
    })

    return {
        "status":      "saved",
        "user_id":     req.user_id,
        "memory":      "saved to Qdrant as structural (λ=0.01)",
        "constraints": {
            "max_wattage":         req.max_wattage,
            "max_price_chf":       req.max_price_chf,
            "forbidden_materials": req.forbidden_materials,
            "kelvin_min":          req.kelvin_min,
            "kelvin_max":          req.kelvin_max,
            "room_type":           req.room_type,
            "location":            req.location,
        },
    }


@app.post("/browse")
def log_browse(req: BrowseRequest):
    """Record a browse event for session state and long-term memory."""
    if req.user_id not in browsing_store:
        browsing_store[req.user_id] = []

    browsing_store[req.user_id].append({
        "product_id": req.product_id,
        "name":       req.name,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })

    save_browse_as_memory(req.user_id, req.name, req.description)

    return {
        "status":        "logged",
        "user_id":       req.user_id,
        "product_id":    req.product_id,
        "history_count": len(browsing_store[req.user_id]),
        "memory":        "saved to Qdrant as episodic (λ=0.30)",
    }


@app.post("/extract", response_model=ExtractResponse)
def extract_constraints(req: ExtractRequest):
    """Detect constraints in a user message and return confirmation suggestions.

    Never saves anything — detection only. The frontend shows one chip per
    suggestion; clicking Yes triggers a separate POST /constraints call.
    """
    import json
    import re

    raw = call_groq(_EXTRACT_SYSTEM_PROMPT, req.message)

    suggestions: list[ConstraintSuggestion] = []
    try:
        # Strip markdown code fences Groq occasionally adds
        cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`").strip()
        items = json.loads(cleaned)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "field" in item and "label" in item and "value" in item:
                    suggestions.append(ConstraintSuggestion(
                        field=str(item["field"]),
                        label=str(item["label"]),
                        value=item["value"],
                    ))
    except Exception:
        pass  # JSON parse failed — return empty suggestions, never crash

    return ExtractResponse(
        user_id=req.user_id,
        message=req.message,
        suggestions=suggestions,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Run retrieval, load memory, generate the reply, and return results."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    query_vector = embed(req.message)
    print(f"[/chat] user={req.user_id!r} msg={req.message!r} vec_len={len(query_vector)} first3={query_vector[:3]}")

    constraints = get_constraints(req.user_id)
    preferences = get_preferences(req.user_id, {
        "preferred_style":  req.preferred_style,
        "preferred_finish": req.preferred_finish,
        "preferred_mood":   req.preferred_mood,
    })
    baseline = run_baseline(query_vector)
    mara     = run_mara(query_vector, constraints, preferences)

    user_context = get_user_context(req.user_id, req.message)

    mara_dicts = [
        {
            "name": p.name,
            "price_chf": p.price_chf,
            "wattage": p.wattage,
            "kelvin": p.kelvin,
            "finish": p.finish,
            "manufacturer": p.manufacturer,
        }
        for p in mara
    ]
    system_prompt = build_llm_prompt(user_context, mara_dicts, baseline)
    history       = get_history(req.user_id)
    llm_reply     = call_groq(system_prompt, req.message, history)
    append_to_history(req.user_id, req.message, llm_reply)

    if req.preferred_style:
        save_chat_preference(req.user_id, f"prefers {req.preferred_style} style lighting")
    if req.preferred_mood:
        save_chat_preference(req.user_id, f"wants {req.preferred_mood} mood atmosphere")
    if req.preferred_finish:
        save_chat_preference(req.user_id, f"likes {req.preferred_finish} finish")

    mara_models = [scored_to_model(p) for p in mara]
    return ChatResponse(
        user_id          = req.user_id,
        query            = req.message,
        llm_reply        = llm_reply,
        baseline_results = baseline,
        mara_results     = mara_models,
        violation_count  = sum(len(p.violations) for p in mara),
        constraints_used = {
            "max_wattage":         constraints.max_wattage,
            "max_price_chf":       constraints.max_price_chf,
            "forbidden_materials": constraints.forbidden_materials,
            "kelvin_min":          constraints.kelvin_min,
            "kelvin_max":          constraints.kelvin_max,
            "room_type":           constraints.room_type,
            "location":            constraints.location,
        },
        user_context = {
            "structural_count": len(user_context["structural"]),
            "semantic_count":   len(user_context["semantic"]),
            "episodic_count":   len(user_context["episodic"]),
            "summary":          user_context["summary"],
        },
        hydration = build_hydration_payload(mara_models),
        frontend = FrontendContractInfo(
            version="2026-03-12",
            primary_result_field="mara_results",
            hydration_provider="supabase",
            hydration_key="source_article_id",
            baseline_available=True,
            merge_strategy="hydrate_supabase_then_enrich_with_mara",
        ),
    )
@app.get("/debug/constraints/{user_id}")
def debug_constraints(user_id: str):
    c = constraints_store.get(user_id)
    if not c:
        return {"user_id": user_id, "constraints": None}
    return {
        "user_id": user_id,
        "constraints": {
            "max_wattage":         c.max_wattage,
            "max_price_chf":       c.max_price_chf,
            "forbidden_materials": c.forbidden_materials,
            "kelvin_min":          c.kelvin_min,
            "kelvin_max":          c.kelvin_max,
            "room_type":           c.room_type,
            "location":            c.location,
        }
    }

@app.get("/debug/history/{user_id}")
def debug_history(user_id: str):
    return {
        "user_id": user_id,
        "count":   len(browsing_store.get(user_id, [])),
        "history": browsing_store.get(user_id, []),
    }

@app.get("/debug/memory/{user_id}")
async def debug_memory(user_id: str):
    """Return the current memory context for a user."""
    context = get_user_context(user_id, "show me everything")
    return {
        "user_id":    user_id,
        "structural": context["structural"],
        "semantic":   context["semantic"],
        "episodic":   context["episodic"],
        "summary":    context["summary"],
    }

@app.delete("/debug/memory/{user_id}")
async def clear_memory(user_id: str):
    """Wipe all Qdrant memory + in-process state for a user (useful for demo resets)."""
    deleted = delete_all_user_memory(user_id)
    constraints_store.pop(user_id, None)
    browsing_store.pop(user_id, None)
    style_timestamp_store.pop(user_id, None)
    conversation_store.pop(user_id, None)
    return {
        "status":   "cleared",
        "user_id":  user_id,
        "deleted_memory_entries": deleted,
    }
