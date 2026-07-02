# SmartRec - Hybrid Recommendation System

A production-ready hybrid recommender combining **collaborative filtering** and **content-based signals** via a neural matrix factorization model, served through a FastAPI REST API backed by PostgreSQL.

## Architecture

```mermaid
graph TD
    User["User Request"] --> API["FastAPI (Async Endpoints)"]
    
    subgraph Storage_Layer ["Storage & Cache"]
        API --> DB[("PostgreSQL DB<br>(SQLAlchemy Async)")]
        API --> Cache[("Embedding Cache<br>(Pre-computed Items)")]
    end
    
    subgraph Model_Layer ["Neural MF Model (PyTorch)"]
        API --> NMF["Neural MF Engine"]
        NMF --> User_Emb["User Embedding Layer<br>(Collaborative Signal)"]
        NMF --> Item_Emb["Item Embedding Layer<br>(Collaborative Signal)"]
        NMF --> Content_MLP["Content Feature MLP<br>(Item Metadata)"]
        
        User_Emb --> Fusion["Fusion Head (Concatenation)"]
        Item_Emb --> Fusion
        Content_MLP --> Fusion
        
        Fusion --> Scoring["Scoring Head<br>σ(MLP)"]
        Scoring --> Recs["Top-K Recommendations"]
    end

    %% Styling
    style User fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000
    style API fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    style DB fill:#ede7f6,stroke:#5e35b1,stroke-width:2px,color:#000
    style Cache fill:#ede7f6,stroke:#5e35b1,stroke-width:2px,color:#000
    style Recs fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    
    classDef default font-family:sans-serif,font-size:13px;

```

---

## Tech Stack

* **Model**: PyTorch — Neural Matrix Factorization with content fusion
* **API**: FastAPI + Uvicorn (async)
* **Database**: PostgreSQL + SQLAlchemy (async)
* **Caching**: Embedding cache in PostgreSQL, Redis-ready
* **Containerization**: Docker + Docker Compose

---

## Project Structure

```text
smartrec/
├── app/
│   ├── api/routes/          # FastAPI route handlers
│   ├── core/                # Config, settings
│   ├── db/                  # Database engine, session
│   ├── models/              # SQLAlchemy ORM models
│   ├── schemas/             # Pydantic request/response schemas
│   └── services/            # Business logic (recommender, training)
├── scripts/
│   ├── train.py             # Model training entrypoint
│   └── seed_data.py         # Seed synthetic interaction data
├── tests/
│   └── test_recommend.py    # API + service tests
├── docker-compose.yml
├── Dockerfile
└── requirements.txt

```

---

## Quickstart

```bash
# 1. Start PostgreSQL
docker-compose up -d db

# 2. Install dependencies
pip install -r requirements.txt

# 3. Seed database with synthetic data
python scripts/seed_data.py

# 4. Train the model
python scripts/train.py

# 5. Run the API
uvicorn app.main:app --reload

```

API docs at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/recommend/{user_id}` | Get top-K recommendations |
| POST | `/users/` | Create a user |
| POST | `/items/` | Create an item |
| POST | `/interactions/` | Log a user-item interaction |
| GET | `/health` | Health check |

---

## Model Pipeline & Architecture

Neural Matrix Factorization fusing two signals:

* **Collaborative signal**: learned user & item embeddings from interaction history (implicit feedback).
* **Content signal**: item metadata (genre, tags) encoded via a small MLP, concatenated with CF embeddings before the scoring head.

$$\hat{r}_{ui} = \sigma\left(MLP\left([e_u \oplus e_i \oplus c_i]\right)\right)$$

Where $e_u$, $e_i$ are learned embeddings and $c_i$ is the content feature vector.

```mermaid
graph LR
    subgraph Inputs [Input Features]
        User_ID["User ID"]
        Item_ID["Item ID"]
        Metadata["Item Metadata<br>(Genres/Tags)"]
    end

    subgraph Latent_Layers [Representation Layers]
        User_ID --> UE["User Embedding (e_u)"]
        Item_ID --> IE["Item Embedding (e_i)"]
        Metadata --> MLP["Dense MLP (c_i)"]
    end

    subgraph Scoring_Network [Fusion & Output]
        UE --> Concat["Concatenation [e_u ⊕ e_i ⊕ c_i]"]
        IE --> Concat
        MLP --> Concat
        Concat --> Layer_Dense["Scoring MLP"]
        Layer_Dense --> Sigmoid["Sigmoid (σ)"]
        Sigmoid --> Output["Predicted Score (r̂_ui)"]
    end

    %% Styling
    style Inputs fill:#f9f9f9,stroke:#ccc,stroke-dasharray: 5 5
    style Output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    style Concat fill:#fff3e0,stroke:#f57c00,stroke-width:1px,color:#000

```

```

```
