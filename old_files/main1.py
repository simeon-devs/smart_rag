"""
MARA — Step 4: FastAPI Endpoints
==================================
Three endpoints Nursena connects to from the frontend:

  POST /constraints   → save user's hard constraints
  POST /browse        → log browsing history (builds episodic memory)
  POST /chat          → run baseline RAG + MARA in parallel, return both

HOW TO RUN:
  uvicorn main:app --reload

DEPENDENCIES:
  pip install fastapi uvicorn

This file imports from mara_engine.py (Step 3).
Nursena replaces mock_embed() with real HuggingFace embeddings.
"""

import hashlib
import random
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mara_engine import (
    run_baseline,
    run_mara,
    UserConstraints,
    UserPreferences,
    ScoredProduct,
)

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="MARA API",
    description="Memory-Augmented Retail Agent — Qdrant-powered lighting recommendations",
    version="1.0.0",
)

# Allow Lovable frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to Lovable domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# MOCK EMBEDDING
# Nursena replaces this with HuggingFace model
# ─────────────────────────────────────────────

def mock_embed(text: str, size: int = 384) -> list[float]:
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(size)]
    mag = sum(x**2 for x in raw) ** 0.5
    return [x / mag for x in raw]

def embed(text: str) -> list[float]:
    """
    Single entry point for embeddings.
    Nursena replaces the body of this function with:

        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        def embed(text: str) -> list[float]:
            return _model.encode(text).tolist()
    """
    return mock_embed(text)


# ─────────────────────────────────────────────
# SESSION STORE
# In-memory for demo. Replace with Redis or
# Supabase for production persistence.
# ─────────────────────────────────────────────

# Stores constraints per user_id
# { "user_id": UserConstraints }
constraints_store: dict[str, UserConstraints] = {}

# Stores browsing history per user_id
# { "user_id": [{"text": ..., "timestamp": ...}] }
browsing_store: dict[str, list[dict]] = {}


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class ConstraintsRequest(BaseModel):
    user_id:              str
    max_wattage:          Optional[float] = None
    max_price_chf:        Optional[float] = None
    forbidden_materials:  list[str]       = []
    kelvin_min:           Optional[float] = None
    kelvin_max:           Optional[float] = None
    room_type:            Optional[str]   = None

class BrowseRequest(BaseModel):
    user_id:    str
    product_id: str
    text:       str      # description of what was browsed

class ChatRequest(BaseModel):
    user_id:           str
    message:           str
    preferred_style:   Optional[str] = None
    preferred_finish:  Optional[str] = None
    preferred_mood:    Optional[str] = None

class ProductResult(BaseModel):
    product_id:        str
    name:              str
    price_chf:         float
    wattage:           float
    kelvin:            float
    material:          str
    style:             str
    finish:            str
    mood:              str
    room_type:         str
    image_url:         str
    similarity_score:  float
    final_score:       float
    violations:        list[str]

class ChatResponse(BaseModel):
    user_id:          str
    query:            str
    baseline_results: list[dict]          # left side of split-screen
    mara_results:     list[ProductResult] # right side of split-screen
    violation_count:  int                 # for the live counter in frontend
    constraints_used: dict                # shows judge what constraints are active


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_constraints(user_id: str) -> UserConstraints:
    """Returns saved constraints for user, or empty defaults."""
    return constraints_store.get(user_id, UserConstraints())


def get_preferences(user_id: str, overrides: dict) -> UserPreferences:
    """
    Builds UserPreferences from:
    1. Browsing history age (episodic memory)
    2. Any style/finish/mood passed in the chat request
    """
    history = browsing_store.get(user_id, [])
    browsing_age = 0.0

    if history:
        last = history[-1]["timestamp"]
        now  = datetime.now(timezone.utc)
        delta = now - datetime.fromisoformat(last)
        browsing_age = delta.total_seconds() / 86400  # convert to days

    return UserPreferences(
        preferred_style   = overrides.get("preferred_style"),
        preferred_finish  = overrides.get("preferred_finish"),
        preferred_mood    = overrides.get("preferred_mood"),
        style_age_days    = 0.0,         # stated in this session → fresh
        browsing_age_days = browsing_age,
    )


def scored_to_model(p: ScoredProduct) -> ProductResult:
    return ProductResult(
        product_id       = p.product_id,
        name             = p.name,
        price_chf        = p.price_chf,
        wattage          = p.wattage,
        kelvin           = p.kelvin,
        material         = p.material,
        style            = p.style,
        finish           = p.finish,
        mood             = p.mood,
        room_type        = p.room_type,
        image_url        = p.image_url,
        similarity_score = p.similarity_score,
        final_score      = p.final_score,
        violations       = p.violations,
    )


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "project": "MARA",
        "status":  "running",
        "version": "1.0.0",
        "endpoints": ["/constraints", "/browse", "/chat"],
    }


# ── ENDPOINT 1 ──────────────────────────────
@app.post("/constraints")
def save_constraints(req: ConstraintsRequest):
    """
    Saves the user's hard constraints.
    Call this when the judge sets their preferences at the start.

    These are stored with lambda = 0.01 — they almost never fade.
    MARA will use these for every /chat call for this user_id.

    Example body:
    {
        "user_id": "judge_01",
        "max_wattage": 40,
        "max_price_chf": 200,
        "forbidden_materials": ["plastic"],
        "kelvin_min": 2700,
        "kelvin_max": 3200
    }
    """
    constraints = UserConstraints(
        max_wattage          = req.max_wattage,
        max_price_chf        = req.max_price_chf,
        forbidden_materials  = req.forbidden_materials,
        kelvin_min           = req.kelvin_min,
        kelvin_max           = req.kelvin_max,
        room_type            = req.room_type,
    )
    constraints_store[req.user_id] = constraints

    return {
        "status":      "saved",
        "user_id":     req.user_id,
        "constraints": {
            "max_wattage":         req.max_wattage,
            "max_price_chf":       req.max_price_chf,
            "forbidden_materials": req.forbidden_materials,
            "kelvin_min":          req.kelvin_min,
            "kelvin_max":          req.kelvin_max,
            "room_type":           req.room_type,
        }
    }


# ── ENDPOINT 2 ──────────────────────────────
@app.post("/browse")
def log_browse(req: BrowseRequest):
    """
    Logs a product the user viewed. Builds episodic memory.
    Call this when the user clicks on a product in the frontend.

    These memories have lambda = 0.30 — they fade fast.
    After 7 days, browsing history has almost no influence.

    Example body:
    {
        "user_id":    "judge_01",
        "product_id": "prod_001",
        "text":       "brass wall sconce warm light scandinavian"
    }
    """
    if req.user_id not in browsing_store:
        browsing_store[req.user_id] = []

    browsing_store[req.user_id].append({
        "product_id": req.product_id,
        "text":       req.text,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })

    return {
        "status":        "logged",
        "user_id":       req.user_id,
        "product_id":    req.product_id,
        "history_count": len(browsing_store[req.user_id]),
    }


# ── ENDPOINT 3 ──────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    The main endpoint. Runs baseline RAG and MARA in parallel.
    Returns both result sets for the split-screen demo.

    The frontend uses:
      - baseline_results  → left panel (standard RAG)
      - mara_results      → right panel (MARA)
      - violation_count   → live counter badge

    Example body:
    {
        "user_id":         "judge_01",
        "message":         "I need a warm light for my reading corner",
        "preferred_style": "scandinavian",
        "preferred_mood":  "cozy"
    }
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 1. Embed the user query
    query_vector = embed(req.message)

    # 2. Load saved constraints for this user
    constraints = get_constraints(req.user_id)

    # 3. Build preferences from session + request overrides
    preferences = get_preferences(req.user_id, {
        "preferred_style":  req.preferred_style,
        "preferred_finish": req.preferred_finish,
        "preferred_mood":   req.preferred_mood,
    })

    # 4. Run both searches
    baseline = run_baseline(query_vector)
    mara     = run_mara(query_vector, constraints, preferences)

    # 5. Count total violations across all MARA results
    violation_count = sum(len(p.violations) for p in mara)

    # 6. Build response
    return ChatResponse(
        user_id          = req.user_id,
        query            = req.message,
        baseline_results = baseline,
        mara_results     = [scored_to_model(p) for p in mara],
        violation_count  = violation_count,
        constraints_used = {
            "max_wattage":         constraints.max_wattage,
            "max_price_chf":       constraints.max_price_chf,
            "forbidden_materials": constraints.forbidden_materials,
            "kelvin_min":          constraints.kelvin_min,
            "kelvin_max":          constraints.kelvin_max,
            "room_type":           constraints.room_type,
        },
    )


# ─────────────────────────────────────────────
# DEBUG ENDPOINTS (remove before final demo)
# ─────────────────────────────────────────────

@app.get("/debug/constraints/{user_id}")
def debug_constraints(user_id: str):
    """Check what constraints are saved for a user."""
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
        }
    }

@app.get("/debug/history/{user_id}")
def debug_history(user_id: str):
    """Check browsing history for a user."""
    history = browsing_store.get(user_id, [])
    return {
        "user_id": user_id,
        "count":   len(history),
        "history": history,
    }
