# Technical Documentation  
## Phase 6: REST API & Serving  
**Project:** PharmaSupply-ML  
**Framework:** Standard MLOps Framework  
**File:** `docs/06_REST_API_SERVING.md`  
**Core Scripts & Files:** `src/api.py`, `tests/test_api.py`, `Dockerfile.api`, `docker-compose.yml`, `models/xgboost_pharma_demand.joblib`  
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)  

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

### *Ce document détaille l'industrialisation du modèle XGBoost sous forme d'une API REST performante (FastAPI), la validation rigoureuse des contrats de données, la suite de tests automatisés, ainsi que la conteneurisation et l'orchestration multi-services.*

---

## 1. Vue d'ensemble & Objectif MLOps
L'objectif de cette étape est de faire passer le modèle de prédiction de demande et de réapprovisionnement de l'état d'expérimentation (Notebook) à un **service de production robuste, testé et conteneurisé**.

Pour un Data Scientist, ce passage au MLOps implique 4 piliers majeurs :
1. **L'exposition (API REST)** : Transformer la logique d'inférence et de réapprovisionnement en un service Web synchrone et rapide.
2. **La validation des données (Pydantic v2)** : Empêcher les erreurs de production en filtrant strictement les types, les identifiants et les plages de valeurs avant toute exécution.
3. **L'intégration logistique & FEFO** : Exposer des points d'accès pour interroger les stocks, les catalogues produits, et calculer des quantités de réapprovisionnement intégrant le MOQ, la taille des lots et les contraintes de péremption.
4. **L'isolation (Docker & Tests)** : Garantir la reproductibilité du service via `Dockerfile.api`, son orchestration via `docker-compose.yml`, et la non-régression via `pytest`.

---

## 2. Architecture des Services & Orchestration

L'application repose sur une architecture multi-services orchestrée par **`docker-compose.yml`** :

* **`pharmasupply-db` (PostgreSQL 16)** : Base de données relationnelle persistante initialisée avec le schéma DDL et les données d'inventaire.
* **`pharmasupply-api` (FastAPI)** : Service Web Python conteneurisé via **`Dockerfile.api`**, chargeant l'artefact `.joblib` en mémoire et connecté à la base de données.
* **`pharmasupply-dashboard` (Streamlit)** : Interface utilisateur consommant l'API pour piloter la chaîne logistique.

---

## 3. Le Service API : `src/api.py`

Le script `src/api.py` structure toute la logique de service, depuis la configuration initiale jusqu'aux règles métiers d'approvisionnement :

### A. Configuration & Variables Globales
```python
MODEL_PATH = os.getenv(
    "MODEL_PATH", "models/xgboost_pharma_demand.joblib"
)
STATUS_FILE_PATH = os.getenv(
    "STATUS_FILE_PATH", "model_status.txt"
)

FEATURE_COLS = [
    "month",
    "dayofweek",
    "is_weekend",
    "stock_on_hand",
    "supplier_lead_time",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
]
```
* **`MODEL_PATH`** : Définit l'emplacement du modèle. L'utilisation de `os.getenv` permet de modifier ce chemin dynamiquement via les variables d'environnement.
* **`FEATURE_COLS`** : Liste stricte des colonnes dans l'ordre exact attendu par le modèle XGBoost pour l'inférence.

### B. Validation des Données (Pydantic v2)
```python
class PredictionInput(BaseModel):
    store_id: Union[int, str] = Field(
        ..., json_schema_extra={"example": "1"}
    )
    product_id: Union[int, str] = Field(
        ..., json_schema_extra={"example": "MED_001"}
    )
    stock_on_hand: float = Field(
        ..., ge=0, json_schema_extra={"example": 100.0}
    )
    supplier_lead_time: float = Field(
        ..., ge=0, json_schema_extra={"example": 3.0}
    )
    safety_threshold: float = Field(
        ..., ge=0, json_schema_extra={"example": 150.0}
    )
    batch_size: int = Field(
        default=1, ge=1, json_schema_extra={"example": 50}
    )
    moq: int = Field(
        default=1, ge=1, json_schema_extra={"example": 100}
    )
    shelf_life_days: int = Field(
        default=365,
        ge=1,
        json_schema_extra={"example": 365},
    )
    lag_1: Optional[float] = Field(default=0.0, ge=0)
    lag_7: float = Field(
        ..., ge=0, json_schema_extra={"example": 630.0}
    )
    lag_14: float = Field(
        ..., ge=0, json_schema_extra={"example": 1260.0}
    )
    lag_30: float = Field(
        ..., ge=0, json_schema_extra={"example": 2700.0}
    )


class PredictionOutput(BaseModel):
    store_id: Union[int, str]
    product_id: Union[int, str]
    predicted_demand: float
    projected_stock: float
    reorder_needed: bool
    order_quantity: int
    model_status: str
    model_version: str = "1.0.0"
```
* **Contrats de données** : Filtrent automatiquement les entrées. Des contraintes comme `ge=0` protègent l'inférence contre les stocks ou historiques négatifs.

### C. Gestion du Cycle de Vie (*Lifespan*)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model
    if os.path.exists(MODEL_PATH):
        try:
            loaded_model = joblib.load(MODEL_PATH)
            logger.info(
                f"✅ Model successfully loaded from : {MODEL_PATH}"
            )
        except Exception as e:
            logger.error(
                f"❌ Error loading model artifact : {e}"
            )
            loaded_model = None
    yield
    loaded_model = None
```
* **Chargement unique en mémoire** : L'artefact `.joblib` est chargé une seule fois au démarrage du serveur FastAPI pour garantir un haut débit de prédiction.

### D. Endpoints HTTP
1. **`/stores` (GET)** : Récupère la liste des identifiants de magasins disponibles.
2. **`/stores/{store_id}/products` (GET)** : Retourne le catalogue des produits associés à un magasin donné.
3. **`/stores/{store_id}/alerts` (GET)** : Analyse les stocks et génère des alertes multicritères (Niveau 1 : Paramétrage, Niveau 2 : Rupture & Péremption FEFO, Niveau 3 : Dérive / Drift).
4. **`/features/{store_id}/{product_id}` (GET)** : Récupère automatiquement l'historique récent et calcule les features dynamiques.
5. **`/predict` (POST)** : Exécute l'inférence XGBoost, applique un garde-fou métier (`np.maximum(0.0, raw_pred)`), et calcule la quantité de réapprovisionnement en respectant les contraintes de lots (MOQ, `batch_size`) et de péremption.

---

## 4. La Suite de Tests Automatisés : `tests/test_api.py`

Le fichier `tests/test_api.py` assure la non-régression de la couche de service via `pytest` et le `TestClient` de FastAPI.

```python
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

VALID_PAYLOAD = {
    "store_id": "1",
    "product_id": "MED_001",
    "stock_on_hand": 500.0,
    "supplier_lead_time": 3.0,
    "safety_threshold": 150.0,
    "batch_size": 50,
    "moq": 100,
    "shelf_life_days": 365,
    "lag_1": 90.0,
    "lag_7": 630.0,
    "lag_14": 1260.0,
    "lag_30": 2700.0,
}


def test_root_endpoint():
    response = client.get("/stores")
    assert response.status_code == 200
    assert "stores" in response.json()


def test_predict_success():
    response = client.post("/predict", json=VALID_PAYLOAD)
    if response.status_code == 200:
        data = response.json()
        assert data["store_id"] == "1"
        assert data["product_id"] == "MED_001"
        assert "predicted_demand" in data
        assert data["predicted_demand"] >= 0.0
    else:
        assert response.status_code == 503


def test_predict_model_not_loaded():
    with patch("src.api.loaded_model", None):
        response = client.post(
            "/predict", json=VALID_PAYLOAD
        )
        assert response.status_code == 503
        assert (
            response.json()["detail"]
            == "ML model artifact is not loaded in memory."
        )


def test_predict_invalid_schema():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["stock_on_hand"] = -50.0
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
```

### Lancement des tests :
```bash
pytest tests/test_api.py
```

---

## 5. Configuration de la Conteneurisation & Orchestration

### A. Le `Dockerfile.api`
Situé à la racine du projet, il définit l'environnement de conteneurisation du backend :
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    pandas \
    numpy \
    xgboost \
    joblib \
    scikit-learn \
    python-dotenv \
    psycopg2-binary \
    streamlit \
    requests \
    evidently \
    shap

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### B. L'Orchestration avec `docker-compose.yml`
```yaml
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: pharmasupply-api
    restart: always
    env_file:
      - .env
    environment:
      POSTGRES_HOST: pharmasupply-db
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@pharmasupply-db:5432/${POSTGRES_DB}
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
      - ./model_status.txt:/app/model_status.txt
      - ./reports:/app/reports
    depends_on:
      postgres:
        condition: service_healthy
```

---

## 6. Démarrage et Validation en Ligne de Commande

### A. Lancement de la pile Docker
```bash
docker compose up --build -d
```

### B. Validation avec `curl`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "1",
    "product_id": "MED_001",
    "stock_on_hand": 500.0,
    "supplier_lead_time": 3.0,
    "safety_threshold": 150.0,
    "batch_size": 50,
    "moq": 100,
    "shelf_life_days": 365,
    "lag_1": 90.0,
    "lag_7": 630.0,
    "lag_14": 1260.0,
    "lag_30": 2700.0
  }'
```

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *This document details the industrialization of the XGBoost model into a high-performance REST API (FastAPI), rigorous data contract validation, automated test suites, and multi-service containerization and orchestration.*

---

## 1. Overview & MLOps Objective
The objective of this phase is to transition the demand forecasting and inventory replenishment model from an experimental state (Notebook) to a **robust, tested, and containerized production service**.

For a Data Scientist, moving to MLOps relies on four core pillars:
1. **Exposition (REST API)**: Converting inference and replenishment logic into a fast, synchronous Web service.
2. **Data Validation (Pydantic v2)**: Preventing production failures by strictly filtering data types, identifiers, and value ranges prior to execution.
3. **Logistics & FEFO Integration**: Exposing endpoints to query inventory levels, product catalogs, and compute replenishment quantities integrating MOQ, batch sizing, and expiry constraints.
4. **Isolation (Docker & Testing)**: Ensuring service reproducibility via `Dockerfile.api`, orchestration via `docker-compose.yml`, and non-regression through `pytest`.

---

## 2. Service Architecture & Orchestration

The application relies on a multi-service architecture orchestrated by **`docker-compose.yml`**:

* **`pharmasupply-db` (PostgreSQL 16)**: Persistent relational database initialized with DDL schemas and inventory data.
* **`pharmasupply-api` (FastAPI)**: Python Web service containerized via **`Dockerfile.api`**, loading the `.joblib` artifact into memory and connected to the database.
* **`pharmasupply-dashboard` (Streamlit)**: User interface consuming the API to manage the supply chain.

---

## 3. The API Service: `src/api.py`

The `src/api.py` script structures the complete service logic, from initial configuration to automated replenishment business rules:

### A. Configuration & Global Variables
```python
MODEL_PATH = os.getenv(
    "MODEL_PATH", "models/xgboost_pharma_demand.joblib"
)
STATUS_FILE_PATH = os.getenv(
    "STATUS_FILE_PATH", "model_status.txt"
)

FEATURE_COLS = [
    "month",
    "dayofweek",
    "is_weekend",
    "stock_on_hand",
    "supplier_lead_time",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
]
```

### B. Data Validation (Pydantic v2)
```python
class PredictionInput(BaseModel):
    store_id: Union[int, str] = Field(
        ..., json_schema_extra={"example": "1"}
    )
    product_id: Union[int, str] = Field(
        ..., json_schema_extra={"example": "MED_001"}
    )
    stock_on_hand: float = Field(
        ..., ge=0, json_schema_extra={"example": 100.0}
    )
    supplier_lead_time: float = Field(
        ..., ge=0, json_schema_extra={"example": 3.0}
    )
    safety_threshold: float = Field(
        ..., ge=0, json_schema_extra={"example": 150.0}
    )
    batch_size: int = Field(
        default=1, ge=1, json_schema_extra={"example": 50}
    )
    moq: int = Field(
        default=1, ge=1, json_schema_extra={"example": 100}
    )
    shelf_life_days: int = Field(
        default=365,
        ge=1,
        json_schema_extra={"example": 365},
    )
    lag_1: Optional[float] = Field(default=0.0, ge=0)
    lag_7: float = Field(
        ..., ge=0, json_schema_extra={"example": 630.0}
    )
    lag_14: float = Field(
        ..., ge=0, json_schema_extra={"example": 1260.0}
    )
    lag_30: float = Field(
        ..., ge=0, json_schema_extra={"example": 2700.0}
    )


class PredictionOutput(BaseModel):
    store_id: Union[int, str]
    product_id: Union[int, str]
    predicted_demand: float
    projected_stock: float
    reorder_needed: bool
    order_quantity: int
    model_status: str
    model_version: str = "1.0.0"
```

### C. Lifecycle Management (*Lifespan*)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model
    if os.path.exists(MODEL_PATH):
        try:
            loaded_model = joblib.load(MODEL_PATH)
            logger.info(
                f"✅ Model successfully loaded from : {MODEL_PATH}"
            )
        except Exception as e:
            logger.error(
                f"❌ Error loading model artifact : {e}"
            )
            loaded_model = None
    yield
    loaded_model = None
```

### D. HTTP Endpoints
1. **`/stores` (GET)**: Retrieves available store identifiers.
2. **`/stores/{store_id}/products` (GET)**: Returns the product catalog associated with a specific store.
3. **`/stores/{store_id}/alerts` (GET)**: Analyzes stock levels and generates multi-criteria alerts (Level 1: Parameterization, Level 2: Stockout & FEFO Expiry, Level 3: Model Drift).
4. **`/features/{store_id}/{product_id}` (GET)**: Automatically retrieves recent history and computes dynamic feature lags.
5. **`/predict` (POST)**: Executes XGBoost inference, applies clipping rules, and calculates replenishment quantities respecting batch multiples and expiration limits.

---

## 4. Automated Test Suite: `tests/test_api.py`

API non-regression is validated using FastAPI's test client (`TestClient`) and `pytest`.

```python
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

VALID_PAYLOAD = {
    "store_id": "1",
    "product_id": "MED_001",
    "stock_on_hand": 500.0,
    "supplier_lead_time": 3.0,
    "safety_threshold": 150.0,
    "batch_size": 50,
    "moq": 100,
    "shelf_life_days": 365,
    "lag_1": 90.0,
    "lag_7": 630.0,
    "lag_14": 1260.0,
    "lag_30": 2700.0,
}


def test_root_endpoint():
    response = client.get("/stores")
    assert response.status_code == 200
    assert "stores" in response.json()


def test_predict_success():
    response = client.post("/predict", json=VALID_PAYLOAD)
    if response.status_code == 200:
        data = response.json()
        assert data["store_id"] == "1"
        assert data["product_id"] == "MED_001"
        assert "predicted_demand" in data
        assert data["predicted_demand"] >= 0.0
    else:
        assert response.status_code == 503


def test_predict_model_not_loaded():
    with patch("src.api.loaded_model", None):
        response = client.post(
            "/predict", json=VALID_PAYLOAD
        )
        assert response.status_code == 503
        assert (
            response.json()["detail"]
            == "ML model artifact is not loaded in memory."
        )


def test_predict_invalid_schema():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["stock_on_hand"] = -50.0
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
```

### Running Tests:
```bash
pytest tests/test_api.py
```

---

## 5. Containerization and Orchestration

### A. The `Dockerfile.api`
Located at the project root, it sets up the execution environment for the API:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    pandas \
    numpy \
    xgboost \
    joblib \
    scikit-learn \
    python-dotenv \
    psycopg2-binary \
    streamlit \
    requests \
    evidently \
    shap

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### B. Orchestration with `docker-compose.yml`
Links the PostgreSQL database, the API, and the Dashboard, utilizing healthchecks to ensure proper service startup order and sharing persistence volumes (`model_status.txt` and report outputs).

---

## 6. Startup and Validation

Launching the containerized stack:
```bash
docker compose up --build -d
```

Verifying inference via `curl` on the `/predict` endpoint using a payload matching the Pydantic schema.

</details>