"""
MARA Data Explorer
==================
Pulls all products from Qdrant and all raw products from Supabase,
saves them as CSV files, and prints a summary you can read in the terminal.

Setup (run once):
  pip install qdrant-client python-dotenv pandas

Usage:
  python3 explore_data.py

Output files:
  qdrant_products.csv   — 5000+ indexed products with scores/fields
  supabase_products.csv — raw catalog rows from Supabase

Fill in your credentials below OR put them in a .env file in the same folder.
"""

import os
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

# ─── CREDENTIALS ──────────────────────────────────────────────────────────────
# Either fill these in directly, or put them in a .env file
QDRANT_URL     = os.getenv("QDRANT_URL",     "YOUR_QDRANT_URL_HERE")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "YOUR_QDRANT_API_KEY_HERE")
SUPABASE_URL   = os.getenv("SUPABASE_URL",   "YOUR_SUPABASE_URL_HERE")
SUPABASE_KEY   = os.getenv("SUPABASE_ANON_KEY", "YOUR_SUPABASE_ANON_KEY_HERE")
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# PART 1 — QDRANT
# =============================================================================

def fetch_all_qdrant_products():
    """Scroll through all points in the hard_constraints collection."""
    from qdrant_client import QdrantClient

    print("\n── Connecting to Qdrant ──────────────────────────────")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    info = client.get_collection("hard_constraints")
    total = info.points_count
    print(f"  Collection: hard_constraints — {total} points")

    records = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name="hard_constraints",
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            records.append(point.payload)

        print(f"  Fetched {len(records)}/{total} products...", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\n  Done — {len(records)} products fetched.")
    return records


def save_qdrant_csv(records):
    import pandas as pd

    df = pd.DataFrame(records)

    # Clean up list columns so they display nicely in Excel
    for col in ["mounting", "luminaire_types", "kelvin_values"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ", ".join(str(i) for i in x) if isinstance(x, list) else x
            )

    # Reorder most useful columns first
    priority = [
        "name", "manufacturer", "category", "family",
        "price_chf", "wattage", "kelvin", "material",
        "inside", "outside", "mounting", "luminaire_types",
        "room_type", "image_url", "product_id",
        "source_article_id", "source_article_number",
    ]
    cols = [c for c in priority if c in df.columns] + \
           [c for c in df.columns if c not in priority]
    df = df[cols]

    out = "qdrant_products.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}  ({len(df)} rows, {len(df.columns)} columns)")
    return df


def print_qdrant_summary(df):
    print("\n── Qdrant Summary ────────────────────────────────────")
    print(f"  Total products:    {len(df)}")

    if "price_chf" in df.columns:
        prices = df["price_chf"].dropna()
        print(f"  Price range:       {prices.min():.0f} – {prices.max():.0f} CHF")
        print(f"  Median price:      {prices.median():.0f} CHF")

    if "wattage" in df.columns:
        watts = df["wattage"].dropna()
        print(f"  Wattage range:     {watts.min():.0f} – {watts.max():.0f} W")

    if "manufacturer" in df.columns:
        top_brands = df["manufacturer"].value_counts().head(5)
        print(f"\n  Top 5 manufacturers:")
        for brand, count in top_brands.items():
            print(f"    {brand:<30} {count} products")

    if "category" in df.columns:
        top_cats = df["category"].value_counts().head(5)
        print(f"\n  Top 5 categories:")
        for cat, count in top_cats.items():
            print(f"    {str(cat):<30} {count} products")

    if "inside" in df.columns and "outside" in df.columns:
        n_inside  = df["inside"].sum()
        n_outside = df["outside"].sum()
        print(f"\n  Indoor products:   {int(n_inside)}")
        print(f"  Outdoor products:  {int(n_outside)}")


# =============================================================================
# PART 2 — SUPABASE
# =============================================================================

def fetch_supabase_products(page_size=1000):
    """Pull all articles from Supabase using the REST API."""
    print("\n── Connecting to Supabase ────────────────────────────")

    headers = {
        "apikey":         SUPABASE_KEY,
        "Authorization":  f"Bearer {SUPABASE_KEY}",
        "Accept":         "application/json",
        "Accept-Profile": "public",
    }

    records = []
    offset = 0

    while True:
        params = urllib.parse.urlencode({
            "select": "id,l_number,article_number,"
                      "price_pp_chf,price_sp_chf,"
                      "hero_image_url,"
                      "very_short_description_de,short_description_de",
            "limit":  page_size,
            "offset": offset,
        })
        url = f"{SUPABASE_URL}/rest/v1/articles?{params}"
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                batch = json.loads(resp.read())
        except Exception as e:
            print(f"  Supabase error at offset {offset}: {e}")
            break

        if not batch:
            break

        records.extend(batch)
        print(f"  Fetched {len(records)} Supabase rows...", end="\r")
        offset += page_size

        if len(batch) < page_size:
            break

    print(f"\n  Done — {len(records)} rows fetched.")
    return records


def save_supabase_csv(records):
    import pandas as pd

    if not records:
        print("  No Supabase records to save.")
        return None

    df = pd.DataFrame(records)
    out = "supabase_products.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}  ({len(df)} rows, {len(df.columns)} columns)")
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed. Run: pip install pandas")
        raise SystemExit(1)

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("qdrant-client not installed. Run: pip install qdrant-client")
        raise SystemExit(1)

    print("=" * 54)
    print("  MARA Data Explorer")
    print("=" * 54)

    # Qdrant
    qdrant_records = fetch_all_qdrant_products()
    qdrant_df      = save_qdrant_csv(qdrant_records)
    print_qdrant_summary(qdrant_df)

    # Supabase
    supabase_records = fetch_supabase_products()
    save_supabase_csv(supabase_records)

    print("\n── Done ──────────────────────────────────────────────")
    print("  Open qdrant_products.csv and supabase_products.csv")
    print("  in Excel, Google Sheets, or any CSV viewer.\n")
