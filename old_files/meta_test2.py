from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, MatchValue
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# Connect to local Qdrant
# ─────────────────────────────────────────
client = QdrantClient("localhost", port=6333)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────
# STEP 1 — Create collection (fresh every run)
# ─────────────────────────────────────────
client.delete_collection("lighting_products")
print("🗑️  Old collection deleted")

client.create_collection(
    collection_name="lighting_products",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
print("✅ Collection created successfully")

# ─────────────────────────────────────────
# STEP 2 — Add lighting products
# ─────────────────────────────────────────
products = [
    {
        "id": 1,
        "text": "warm white wall sconce brushed brass minimal scandinavian",
        "payload": {
            "name": "Brass Wall Sconce",
            "price": 175,
            "wattage": 35,
            "kelvin": 2800,
            "material": "metal",
            "style": "scandinavian"
        }
    },
    {
        "id": 2,
        "text": "cool white ceiling spotlight plastic modern bright",
        "payload": {
            "name": "Chrome Ceiling Spot",
            "price": 240,
            "wattage": 60,
            "kelvin": 5000,
            "material": "plastic",
            "style": "modern"
        }
    },
    {
        "id": 3,
        "text": "warm white pendant light wooden natural cozy dining",
        "payload": {
            "name": "Wooden Pendant",
            "price": 190,
            "wattage": 40,
            "kelvin": 2700,
            "material": "wood",
            "style": "natural"
        }
    },
    {
        "id": 4,
        "text": "warm white floor lamp linen fabric soft reading cozy",
        "payload": {
            "name": "Linen Floor Lamp",
            "price": 145,
            "wattage": 25,
            "kelvin": 2700,
            "material": "metal",
            "style": "minimal"
        }
    },
    {
        "id": 5,
        "text": "cool white industrial metal pendant bar kitchen bright",
        "payload": {
            "name": "Industrial Pendant",
            "price": 310,
            "wattage": 75,
            "kelvin": 4000,
            "material": "metal",
            "style": "industrial"
        }
    }
]

# Convert text to vectors and store in Qdrant
points = []
for product in products:
    vector = model.encode(product["text"]).tolist()
    points.append(
        PointStruct(
            id=product["id"],
            vector=vector,
            payload=product["payload"]
        )
    )

client.upsert(
    collection_name="lighting_products",
    points=points
)
print(f"✅ {len(products)} products stored in Qdrant\n")

# ─────────────────────────────────────────
# STEP 3 — Search WITHOUT constraints
# Standard RAG — baseline
# ─────────────────────────────────────────
query = "something cozy for a reading corner"
query_vector = model.encode(query).tolist()

baseline_results = client.query_points(
    collection_name="lighting_products",
    query=query_vector,
    limit=3
).points

print("❌ BASELINE RAG results (no constraints):")
print("─" * 50)
for r in baseline_results:
    print(f"  → {r.payload['name']:<25} | "
          f"{r.payload['price']} CHF | "
          f"{r.payload['wattage']}W | "
          f"{r.payload['material']}")

# ─────────────────────────────────────────
# STEP 4 — Search WITH hard constraints
# MARA — constraint-aware retrieval
# Hard constraints:
#   - max 40W
#   - max 200 CHF
#   - no plastic (metal only)
# ─────────────────────────────────────────
mara_results = client.query_points(
    collection_name="lighting_products",
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="wattage",
                range=Range(lte=40)         # max 40W — hard constraint
            ),
            FieldCondition(
                key="price",
                range=Range(lte=200)        # max 200 CHF — hard constraint
            ),
            FieldCondition(
                key="material",
                match=MatchValue(value="metal")  # no plastic — hard constraint
            )
        ]
    ),
    limit=3
).points

print("\n✅ MARA results (hard constraints respected):")
print("─" * 50)
for r in mara_results:
    print(f"  → {r.payload['name']:<25} | "
          f"{r.payload['price']} CHF | "
          f"{r.payload['wattage']}W | "
          f"{r.payload['material']}")

# ─────────────────────────────────────────
# STEP 5 — Show the difference
# ─────────────────────────────────────────
print("\n📊 CONSTRAINT VIOLATION CHECK:")
print("─" * 50)
print("Baseline RAG violations:")
for r in baseline_results:
    violations = []
    if r.payload['wattage'] > 40:
        violations.append(f"wattage {r.payload['wattage']}W > 40W")
    if r.payload['price'] > 200:
        violations.append(f"price {r.payload['price']} CHF > 200 CHF")
    if r.payload['material'] == 'plastic':
        violations.append("material is plastic")
    if violations:
        print(f"  ❌ {r.payload['name']}: {', '.join(violations)}")
    else:
        print(f"  ✅ {r.payload['name']}: no violations")

print("\nMARA violations:")
for r in mara_results:
    print(f"  ✅ {r.payload['name']}: all constraints respected")