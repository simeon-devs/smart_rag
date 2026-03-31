# MARA — Memory-Augmented Retail Agent
### Constraint-Preserving Agent Architecture for Long-Term Retail Reasoning

[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%20Cloud-red)](https://qdrant.tech/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Mistral](https://img.shields.io/badge/LLM-Mistral%207B-blue)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
[![Python](https://img.shields.io/badge/Language-Python%203.9+-yellow)](https://www.python.org/)

---

## 1. Executive Summary

Retail AI agents frequently fail at long-term reasoning because they treat all memory equally. Over time, critical budget constraints and size preferences fade into conversational noise. **MARA (Memory-Augmented Retail Agent)** solves this by treating memory not just as data to be retrieved, but as a **reparameterized retrieval space**.

Built on **Qdrant Cloud**, MARA implements a three-strata memory architecture that distinguishes between **numeric invariants** (e.g., budget, size) and **adaptive variables** (e.g., style, mood). By applying type-aware exponential decay, MARA ensures that "Hard Constraints" remain conserved quantities over 6-month customer journeys, while seasonal trends evolve naturally.

> [!IMPORTANT]
> MARA reduces constraint violation rates from ~37% (baseline RAG) to **~4%** in long-term simulated retail environments.

---

## 2. Core Innovation: Retrieval Space Reparameterization

Traditional RAG systems compute simple similarity: `Score = Similarity(x, q)`. MARA modifies the retrieval geometry itself:

$$\text{FinalScore} = \text{Similarity}(x, q) \times \text{StructuralWeight}(x) \times \text{DecayFunction}(\text{type}, t)$$

### 2.1 The Three Strata of Memory

| Memory Type | Examples | Decay Rate (λ) | Physics |
| :--- | :--- | :--- | :--- |
| **🏗 Structural** | Budget, Size, Material | λ ≈ 0.01 | **Conserved** (Invariant) |
| **🎨 Semantic** | Style, Brand Affinity | λ ≈ 0.10 | **Slow Decay** (Adaptive) |
| **⚡ Episodic** | Recent Browsing, Vibe | λ ≈ 0.30 | **Fast Decay** (Volatile) |

---

## 3. Technical Architecture

The backend is built with **Python/FastAPI** and utilizes a **dual-collection Qdrant architecture** to separate invariant constraints from adaptive preferences.

### 3.1 Tech Stack
- **Vector Database:** Qdrant Cloud (Dual-collection: `structural_memory` & `semantic_episodic_memory`)
- **LLM:** Mistral-7B-Instruct-v0.2 (via HuggingFace Router)
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- **API Framework:** FastAPI

### 3.2 Dual-Collection Logic
- **`structural_memory`**: Stores hard rules (Budget: 200 CHF). Uses near-zero decay to ensure these facts are always "near" in the retrieval space.
- **`semantic_episodic_memory`**: Stores preferences and history. Uses exponential decay $e^{-\lambda t}$ to let older "noise" fade out while keeping recent context relevant.

---

## 4. Setup & Installation

### Prerequisites
- Python 3.9+
- Qdrant Cloud Cluster
- HuggingFace API Token (for Mistral 7B)

### 1. Clone & Environment Setup
```bash
git clone <repository-url>
cd Mara-cluster/mara_env
pip install -r requirements.txt # Or install manually: fastapi qdrant-client sentence-transformers python-dotenv requests uvicorn
```

### 2. Configure Environment Variables
Create a `.env` file in `mara_env/`:
```env
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
HF_TOKEN=your_huggingface_token
```

### 3. Initialize Database
Run the initialization script to create the necessary Qdrant collections:
```bash
python init_db.py
```

---

## 5. Usage

### Starting the Backend
```bash
python main.py
# Server runs on http://0.0.0.0:8000
```

### API Endpoints
- **POST `/chat`**: Main interaction endpoint.
  - **Payload:** `{"user_id": "user123", "message": "Suggest a minimalist lamp"}`
  - **Response:** Includes the LLM reply, retrieved context, and a `constraint_violation` flag.

### Populating Memory (Example)
You can use `memory_manager.py` to simulate a customer profile:
```python
from memory_manager import MARAMemoryManager
manager = MARAMemoryManager()
manager.add_structural_constraint("user123", "budget", 200, "Maximum price 200 CHF")
manager.add_interaction("user123", "I love earth tones", "semantic")
```

---

## 6. Evaluation Framework: LLM-as-a-Judge

MARA includes a dedicated evaluation pipeline that measures:
1. **Numerical Stability:** Accuracy of budget/size retention.
2. **Context Coherence:** Reasoning consistency over 6 months.
3. **Retrieval Precision:** Re-ranking accuracy of reparameterized scores.

---

*Confidential — Qdrant Challenge: GenAI in Retail Hackathon Submission*
