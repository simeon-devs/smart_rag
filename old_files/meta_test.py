from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# connect to my local Qdrant 
client = QdrantClient("localhost", port=6333)

# Load embeding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#----
# create my first collection

# Delete if already exists
client.delete_collection("lighting_products")
print("🗑️ Old collection deleted")


client.create_collection(
    collection_name="lighting_products",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE

    )
)

print("Collection created successfully.👍")

# 2 > Addling some lighting products to the collection


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
    }
]

#  Convert text to vectors and store in QdrandtClient

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

# Search without constraint ( standard RAG - baseline)

query = "something cosy for reading corner"
query_vector = model.encode(query).tolist()

results = client.search(
    collection_name="lighting_products",
    query_vector=query_vector,
    limit=3
)

print("\n ❌BASELINE RAG reults (no constraints):")
for r in results:
    print(f" -> {r.payload['name']} | " 
          f"{r.payload['price']} CHF | "
          f"{r.payload['wattage']}W | "
          f"{r.payload['material']} | ")
    

# step 4 - search WITH hard constraints (e.g. max price and specific material)

from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

mara_results = client.search(
    collection_name="lighting_products",
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="wattage",
                range=Range(lte=40) # max 40 W - hard constraint
            ),
            FieldCondition(
                key="price",
                range=Range(lte=200) # max 200 CHF - hard constraint
            ),
            FieldCondition(
                key="material",
                match=MatchValue(value="metal") # no plastic - hard constraint
            )
        ]
    ),
    limit=3
)


print("\n ✅MARA RAG results (with constraints):")
for r in mara_results:
    print(f" -> {r.payload['name']} | " 
          f"{r.payload['price']} CHF | "
          f"{r.payload['wattage']}W | "
          f"{r.payload['material']} | ")