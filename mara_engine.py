"""Retrieval and reranking logic for MARA."""

import math
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

COLLECTION_HARD = "hard_constraints"
COLLECTION_SOFT = "soft_preferences"
TOP_K           = 5   # how many results to return


@dataclass
class UserConstraints:
    """Strict filters applied during retrieval and reranking."""
    max_wattage:          Optional[float] = None
    max_price_chf:        Optional[float] = None
    forbidden_materials:  list[str]       = field(default_factory=list)
    kelvin_min:           Optional[float] = None
    kelvin_max:           Optional[float] = None
    room_type:            Optional[str]   = None
    location:             Optional[str]   = None   # "outdoor" | "indoor"


@dataclass
class UserPreferences:
    """Soft signals that can boost ranking without filtering results."""
    preferred_style:    Optional[str]  = None
    preferred_finish:   Optional[str]  = None
    preferred_mood:     Optional[str]  = None
    style_age_days:     float          = 0.0
    browsing_age_days:  float          = 0.0


@dataclass
class ScoredProduct:
    """A product with its final MARA score and violation report."""
    product_id:        str
    source_article_id: Optional[int]
    source_article_number: Optional[str]
    source_l_number:   Optional[int]
    name:              str
    manufacturer:      Optional[str]
    category:          Optional[str]
    family:            Optional[str]
    price_chf:         Optional[float]
    wattage:           Optional[float]
    kelvin:            Optional[float]
    material:          Optional[str]
    style:             Optional[str]
    finish:            Optional[str]
    mood:              Optional[str]
    room_type:         Optional[str]
    image_url:         Optional[str]
    similarity_score:  float
    decay_score:       float
    final_score:       float
    tags:              list[str] = field(default_factory=list)
    violations:        list[str] = field(default_factory=list)

LAMBDA = {
    "hard":     0.01,   # hard constraints — almost never fade
    "soft":     0.10,   # style/finish preferences — slow drift
    "episodic": 0.30,   # recent browsing — fades fast
}

# Same keyword list as setup_qdrant.py — filters accessories post-retrieval
# so the current live Qdrant data is clean even before a full re-index.
_ACCESSORY_KEYWORDS = (
    "kit ",
    "bracket",
    "cover",
    "accessory",
    "accessories",
    "abdeckung",
    "einbaurahmen",
    "gegengewicht",
    "staffa",
    "rotazione",
    "seil",
    "schiene",
    "rail",
    "halter",
    "end cap",
    "adapter",
)


def _is_accessory(name: str, wattage: Optional[float], price_chf: Optional[float]) -> bool:
    """Return True if the product looks like an accessory/spare part."""
    low = name.lower()
    for kw in _ACCESSORY_KEYWORDS:
        if kw in low:
            return True
    if wattage is None and price_chf is not None and price_chf < 20:
        return True
    return False


def decay(initial_score: float, memory_type: str, time_elapsed_days: float) -> float:
    """Apply exponential decay to a score."""
    lam = LAMBDA.get(memory_type, 0.10)
    return initial_score * math.exp(-lam * time_elapsed_days)


def constraint_weight(product: dict, constraints: UserConstraints) -> tuple[float, list[str]]:
    """Return the constraint weight and any violations for a product."""
    violations = []

    wattage = product.get("wattage")
    price = product.get("price_chf")
    kelvin = product.get("kelvin")
    material = product.get("material")
    room_type = product.get("room_type")

    if constraints.max_wattage is not None:
        if wattage is None:
            violations.append("× wattage unknown")
        elif wattage > constraints.max_wattage:
            violations.append(
                f"× {wattage}W exceeds limit of {constraints.max_wattage}W"
            )

    if constraints.max_price_chf is not None:
        if price is None:
            violations.append("× price unknown")
        elif price > constraints.max_price_chf:
            violations.append(
                f"× {price} CHF exceeds budget of {constraints.max_price_chf} CHF"
            )

    if constraints.forbidden_materials:
        if material is not None:
            # Only flag if we know the material and it IS forbidden.
            # A product with unknown material is treated as neutral.
            mat = str(material).lower()
            for forbidden in constraints.forbidden_materials:
                if forbidden.lower() in mat:
                    violations.append(f"× material '{mat}' is forbidden")

    if constraints.kelvin_min is not None:
        if kelvin is None:
            violations.append("× color temperature unknown")
        elif kelvin < constraints.kelvin_min:
            violations.append(
                f"× {kelvin}K below minimum {constraints.kelvin_min}K"
            )

    if constraints.kelvin_max is not None:
        if kelvin is None:
            violations.append("× color temperature unknown")
        elif kelvin > constraints.kelvin_max:
            violations.append(
                f"× {kelvin}K above maximum {constraints.kelvin_max}K"
            )

    if constraints.room_type is not None:
        # Only flag if the catalog specifies a DIFFERENT room type.
        # Products with no room_type metadata are treated as general-purpose.
        if room_type is not None and str(room_type).lower() != constraints.room_type.lower():
            violations.append(
                f"× room type '{room_type}' doesn't match '{constraints.room_type}'"
            )

    weight = 0.0 if violations else 1.0
    return weight, violations


def preference_boost(product: dict, preferences: UserPreferences) -> float:
    """Return a ranking boost based on soft preference matches."""
    boost = 0.0

    if preferences.preferred_style:
        if product.get("style", "").lower() == preferences.preferred_style.lower():
            boost += 0.15 * decay(1.0, "soft", preferences.style_age_days)

    if preferences.preferred_finish:
        if product.get("finish", "").lower() == preferences.preferred_finish.lower():
            boost += 0.10 * decay(1.0, "soft", preferences.style_age_days)

    if preferences.preferred_mood:
        if product.get("mood", "").lower() == preferences.preferred_mood.lower():
            boost += 0.05 * decay(1.0, "episodic", preferences.browsing_age_days)

    return boost


def get_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


def build_qdrant_filter(constraints: UserConstraints) -> Optional[Filter]:
    """Build a Qdrant pre-filter from strict numeric and boolean constraints."""
    conditions = []

    if constraints.max_wattage is not None:
        conditions.append(FieldCondition(
            key="wattage",
            range=Range(lte=constraints.max_wattage)
        ))

    if constraints.max_price_chf is not None:
        conditions.append(FieldCondition(
            key="price_chf",
            range=Range(lte=constraints.max_price_chf)
        ))

    if constraints.kelvin_min is not None:
        conditions.append(FieldCondition(
            key="kelvin",
            range=Range(gte=constraints.kelvin_min)
        ))

    if constraints.kelvin_max is not None:
        conditions.append(FieldCondition(
            key="kelvin",
            range=Range(lte=constraints.kelvin_max)
        ))

    if constraints.location is not None:
        loc = constraints.location.lower()
        if loc == "outdoor":
            # Match products where outside=True (payload bool field).
            conditions.append(FieldCondition(
                key="outside",
                match=MatchValue(value=True)
            ))
        elif loc == "indoor":
            conditions.append(FieldCondition(
                key="inside",
                match=MatchValue(value=True)
            ))

    if not conditions:
        return None

    return Filter(must=conditions)


def run_baseline(query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
    """Return similarity-only search results from the soft collection."""
    client = get_client()

    results = client.query_points(
        collection_name=COLLECTION_SOFT,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    output = []
    for r in results:
        p = r.payload
        if _is_accessory(p.get("name", ""), None, None):
            continue
        output.append({
            "product_id":       p.get("product_id"),
            "source_article_id": p.get("source_article_id"),
            "source_article_number": p.get("source_article_number"),
            "source_l_number": p.get("source_l_number"),
            "name":             p.get("name"),
            "manufacturer":     p.get("manufacturer"),
            "category":         p.get("category"),
            "family":           p.get("family"),
            "style":            p.get("style"),
            "finish":           p.get("finish"),
            "mood":             p.get("mood"),
            "image_url":        p.get("image_url"),
            "tags":             p.get("tags", []),
            "similarity_score": round(r.score, 4),
            "method":           "baseline",
        })

    return output


def _fetch_and_score(
    client: QdrantClient,
    query_vector: list[float],
    constraints: UserConstraints,
    preferences: UserPreferences,
    top_k: int,
) -> list[ScoredProduct]:
    """Single retrieval + scoring pass with the given constraints.

    Fetches top_k*3 candidates from both collections, filters accessories,
    applies constraint checking + preference boosts, and returns the top_k
    scored products sorted by final_score descending.
    """
    qdrant_filter = build_qdrant_filter(constraints)

    hard_results = client.query_points(
        collection_name=COLLECTION_HARD,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k * 3,
        with_payload=True,
    ).points

    soft_results = client.query_points(
        collection_name=COLLECTION_SOFT,
        query=query_vector,
        limit=top_k * 3,
        with_payload=True,
    ).points

    soft_lookup = {
        r.payload.get("product_id"): r.payload
        for r in soft_results
    }

    scored = []

    for r in hard_results:
        hp = r.payload
        if _is_accessory(hp.get("name", ""), hp.get("wattage"), hp.get("price_chf")):
            continue
        sp = soft_lookup.get(hp.get("product_id"), {})

        similarity         = r.score
        c_weight, violations = constraint_weight(hp, constraints)
        decayed_similarity = decay(similarity, "hard", 0)
        boost              = preference_boost(sp, preferences)
        final              = (decayed_similarity + boost) * c_weight

        scored.append(ScoredProduct(
            product_id            = hp.get("product_id", ""),
            source_article_id     = hp.get("source_article_id"),
            source_article_number = hp.get("source_article_number"),
            source_l_number       = hp.get("source_l_number"),
            name                  = hp.get("name", ""),
            manufacturer          = hp.get("manufacturer") or sp.get("manufacturer"),
            category              = hp.get("category") or sp.get("category"),
            family                = hp.get("family") or sp.get("family"),
            price_chf             = hp.get("price_chf"),
            wattage               = hp.get("wattage"),
            kelvin                = hp.get("kelvin"),
            material              = hp.get("material"),
            style                 = sp.get("style"),
            finish                = sp.get("finish"),
            mood                  = sp.get("mood"),
            room_type             = hp.get("room_type"),
            image_url             = hp.get("image_url") or sp.get("image_url"),
            tags                  = sp.get("tags", []),
            similarity_score      = round(similarity, 4),
            decay_score           = round(decayed_similarity, 4),
            final_score           = round(final, 4),
            violations            = violations,
        ))

    scored.sort(key=lambda x: x.final_score, reverse=True)
    return scored[:top_k]


def run_mara(
    query_vector: list[float],
    constraints: UserConstraints,
    preferences: UserPreferences,
    top_k: int = TOP_K,
) -> list[ScoredProduct]:
    """Return MARA-ranked results using filters, decay, and preference boosts.

    Price-relaxation fallback
    ─────────────────────────
    Strict price constraints can leave the user with very few clean results.
    To keep the UI useful, we progressively relax ONLY the price constraint:

      Attempt 1  — full constraints as given by the user.
      Attempt 2  — price budget relaxed by +20% (gives a little headroom).
      Attempt 3  — price constraint removed entirely (no upper limit).

    Wattage and material constraints are NEVER relaxed — they represent hard
    safety/preference requirements the user explicitly set.

    The fallback only kicks in when fewer than 3 products are clean (no
    violations). If the initial query already yields ≥ 3 clean products, or
    if no price constraint is active, we return immediately.
    """
    client = get_client()

    # Attempt 1: full user constraints.
    results = _fetch_and_score(client, query_vector, constraints, preferences, top_k)
    clean_count = sum(1 for p in results if not p.violations)

    if clean_count >= 3 or constraints.max_price_chf is None:
        return results

    # Attempt 2: relax price by +20%.
    relaxed_price = constraints.max_price_chf * 1.20
    relaxed_constraints = UserConstraints(
        max_wattage         = constraints.max_wattage,
        max_price_chf       = relaxed_price,
        forbidden_materials = constraints.forbidden_materials,
        kelvin_min          = constraints.kelvin_min,
        kelvin_max          = constraints.kelvin_max,
        room_type           = constraints.room_type,
    )
    results = _fetch_and_score(client, query_vector, relaxed_constraints, preferences, top_k)
    clean_count = sum(1 for p in results if not p.violations)
    print(f"[run_mara] price relaxed +20% ({relaxed_price:.0f} CHF) → {clean_count} clean results")

    if clean_count >= 3:
        return results

    # Attempt 3: remove price constraint entirely.
    no_price_constraints = UserConstraints(
        max_wattage         = constraints.max_wattage,
        max_price_chf       = None,
        forbidden_materials = constraints.forbidden_materials,
        kelvin_min          = constraints.kelvin_min,
        kelvin_max          = constraints.kelvin_max,
        room_type           = constraints.room_type,
    )
    results = _fetch_and_score(client, query_vector, no_price_constraints, preferences, top_k)
    print(f"[run_mara] price removed entirely → {sum(1 for p in results if not p.violations)} clean results")
    return results
