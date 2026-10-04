# Technical Documentation
## Phase 6: REST API & SERVING
**Project:** PharmaSupply-ML  
**File:** `docs/REST_API_SERVING.md`  
**Status:** Pending approval

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

## 1. Vue d'ensemble & Objectif MLOps
L'objectif de cette étape est de faire passer le modèle de prédiction de demande de médicaments (**XGBoost**) de l'état d'expérimentation (Notebook) à un **service de production robuste, testé et conteneurisé**.

Pour un Data Scientist, ce passage au MLOps implique 3 grands piliers :
1. **L'exposition (API REST)** : Transformer la fonction Python `predict()` en un service Web accessible par n'importe quel autre système.
2. **La validation des données (Schémas)** : Empêcher le modèle de planter en vérifiant les types et plages de valeurs avant l'inférence.
3. **L'isolation (Docker & Tests)** : Garantir que le code tourne de la même façon sur votre machine, en CI/CD ou sur un serveur cloud, tout en automatisant les tests non-régressifs.

---

## 2. Architecture des Services
L'application repose sur une architecture multi-services orchestrée par **Docker Compose** :

* **`pharmasupply-db` (PostgreSQL)** : Base de données relationnelle stockant les données historiques d'inventaire.
* **`pharmasupply-api` (FastAPI)** : Service Web Python chargeant l'artefact `.joblib` pour exposer les points d'accès HTTP (*endpoints*).

---

## 3. Le Fichier d'API : `src/api.py`

Le script `src/api.py` est le cœur de l'application MLOps. Il est découpé en 5 sections principales :

### A. Configuration & Variables Globales
```python
MODEL_PATH = os.getenv(
    "MODEL_PATH", "models/xgboost_pharma_demand.joblib"
)
loaded_model = None

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
* **`MODEL_PATH`** : Définit l'emplacement du modèle. L'utilisation de `os.getenv` permet de modifier ce chemin dynamiquement via les variables d'environnement (pratique pour le CI/CD).
* **`FEATURE_COLS`** : Liste stricte des colonnes dans l'ordre exact attendu par le modèle XGBoost pour l'inférence.

### B. Validation des Données (Pydantic v2)
```python
class PredictionInput(BaseModel):
    store_id: int = Field(
        ..., json_schema_extra={"example": 101}
    )
    product_id: int = Field(
        ..., json_schema_extra={"example": 2045}
    )
    month: int = Field(
        ..., ge=1, le=12, json_schema_extra={"example": 10}
    )
    dayofweek: int = Field(
        ..., ge=0, le=6, json_schema_extra={"example": 2}
    )
    is_weekend: int = Field(
        ..., ge=0, le=1, json_schema_extra={"example": 0}
    )
    stock_on_hand: float = Field(
        ..., ge=0, json_schema_extra={"example": 500.0}
    )
    supplier_lead_time: float = Field(
        ..., ge=0, json_schema_extra={"example": 3.0}
    )
    lag_7: float = Field(
        ..., ge=0, json_schema_extra={"example": 120.5}
    )
    lag_14: float = Field(
        ..., ge=0, json_schema_extra={"example": 115.0}
    )
    lag_30: float = Field(
        ..., ge=0, json_schema_extra={"example": 110.0}
    )
    rolling_mean_7: float = Field(
        ..., ge=0, json_schema_extra={"example": 118.2}
    )
    rolling_mean_30: float = Field(
        ..., ge=0, json_schema_extra={"example": 112.4}
    )
```
* **Contrats de données (`PredictionInput`)** : Définit le format attendu. Les contraintes comme `ge=1, le=12` (greater/equal, lower/equal) agissent comme un filtre automatique : si une requête envoie un mois `13` ou un stock négatif, FastAPI rejette immédiatement la requête avec une erreur `HTTP 422 Unprocessable Entity` sans même solliciter le modèle.

```python
class PredictionOutput(BaseModel):
    store_id: int
    product_id: int
    predicted_demand: float
    model_version: str = "1.0.0"
```
* **Format de sortie (`PredictionOutput`)** : Standardise la réponse renvoyée au client, incluant la prédiction et la version du modèle pour assurer la traçabilité.

### C. Gestion du Cycle de Vie (*Lifespan*)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model
    if os.path.exists(MODEL_PATH):
        try:
            loaded_model = joblib.load(MODEL_PATH)
            logger.info(
                f"✅ Model successfully loaded from: {MODEL_PATH}"
            )
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            loaded_model = None
    yield
    loaded_model = None
```
* **Chargement unique en mémoire** : Au lieu de charger le fichier `.joblib` à chaque requête (ce qui détruirait les performances), le gestionnaire `lifespan` le charge une seule fois au démarrage du serveur FastAPI.

### D. Endpoints HTTP
1. **`/health` (GET)** : Endpoint de contrôle pour les orchestrateurs (Kubernetes, Docker, Load Balancers). Il confirme si l'API est en ligne et si le modèle est chargé en mémoire.
2. **`/predict` (POST)** : Recevant une liste d'objets `PredictionInput`, il :
   * Convertit la charge utile JSON en `pandas.DataFrame`.
   * Filtre et ordonne les colonnes selon `FEATURE_COLS`.
   * Exécute l'inférence : `loaded_model.predict(X)`.
   * Applique un garde-fou métier : `np.maximum(0.0, raw_preds)` pour garantir qu'aucune prédiction de demande ne soit négative.
   * Renvoie le résultat sous forme de liste `PredictionOutput`.

---

## 4. La Suite de Tests : `tests/test_api.py`

En MLOps, on ne déploie jamais sans tests automatisés. Le fichier `tests/test_api.py` permet de valider le comportement de l'API via `pytest` et le `TestClient` de FastAPI.

### Structure des tests automatisés :
```python
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_valid_data():
    payload = [{
        "store_id": 101,
        "product_id": 2045,
        "month": 10,
        "dayofweek": 2,
        "is_weekend": 0,
        "stock_on_hand": 500.0,
        "supplier_lead_time": 3.0,
        "lag_7": 120.5,
        "lag_14": 115.0,
        "lag_30": 110.0,
        "rolling_mean_7": 118.2,
        "rolling_mean_30": 112.4
    }]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert "predicted_demand" in data[0]
    assert data[0]["predicted_demand"] >= 0.0

def test_predict_endpoint_invalid_data():
    # Test d'envoi d'un mois invalide (13)
    payload = [{"month": 13, ...}]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Erreur de validation Pydantic
```

### Lancement des tests :
```bash
pytest tests/test_api.py
```

---

## 5. Configuration de la Conteneurisation (Docker)

### A. Le `Dockerfile`
Fichier de configuration situé à la racine permettant de créer l'image Docker autonome de l'API.

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
    "psycopg[binary]"

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Rôle des instructions :
* **`FROM python:3.11-slim`** : Base Linux Debian minimale pré-équipée avec Python 3.11.
* **`WORKDIR /app`** : Dossier racine à l'intérieur du conteneur.
* **`RUN apt-get update ...`** : Installation des bibliothèques C/C++ requises par PostgreSQL et XGBoost.
* **`RUN pip install ...`** : Installation des dépendances Python requises pour le Runtime.
* **`COPY`** : Importation du code source (`src/`) et des modèles sérialisés (`models/`).
* **`CMD`** : Commande exécutée au démarrage pour lancer le serveur ASGI Uvicorn.

---

### B. L'Orchestration avec `docker-compose.yml`
Permet de lier l'API à la base de données PostgreSQL dans un réseau conteneurisé unique.

```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: pharmasupply-db
    restart: always
    env_file:
      - .env
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: [ "CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}" ]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build: .
    container_name: pharmasupply-api
    restart: always
    env_file:
      - .env
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
    driver: local
```

* **`depends_on` + `condition: service_healthy`** : Clé MLOps essentielle. L'API ne tente de démarrer que lorsque la base de données est complètement prête à accepter des connexions.

---

## 6. Démarrage et Validation en Ligne de Commande

### A. Lancement de la pile Docker
```bash
docker compose up --build -d
```

### B. Validation avec `curl`

1. **Vérification du statut (`/health`) :**
   ```bash
   curl -X GET http://localhost:8000/health
   ```
   *Réponse attendue :* `{"status":"healthy","model_loaded":true}`

2. **Exécution d'une prédiction d'inférence (`/predict`) :**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '[
       {
         "store_id": 101,
         "product_id": 2045,
         "month": 10,
         "dayofweek": 2,
         "is_weekend": 0,
         "stock_on_hand": 500.0,
         "supplier_lead_time": 3.0,
         "lag_7": 120.5,
         "lag_14": 115.0,
         "lag_30": 110.0,
         "rolling_mean_7": 118.2,
         "rolling_mean_30": 112.4
       }
     ]'
   ```
   *Réponse métier obtenue :*
   ```json
   [
     {
       "store_id": 101,
       "product_id": 2045,
       "predicted_demand": 133.93557739257812,
       "model_version": "1.0.0"
     }
   ]
   ```

### Interprétation de l'Inférence :
Pour la pharmacie **101** et le produit **2045**, un mardi d'octobre avec un stock de 500 unités et une moyenne récente de 118 ventes/jour, le modèle XGBoost prédit un besoin de **~134 boîtes**. Sachant que le fournisseur met 3 jours à livrer (`supplier_lead_time: 3.0`), le système de gestion d'inventaire peut réagir en toute sécurité.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

## 1. Overview & MLOps Objective
The objective of this phase is to transition the pharmaceutical demand forecasting model (**XGBoost**) from an experimental state (Notebook) to a **robust, tested, and containerized production service**.

For a Data Scientist, moving to MLOps relies on three key pillars:
1. **Exposition (REST API)**: Converting the Python `predict()` function into an accessible Web service for any external system.
2. **Data Validation (Schemas)**: Preventing model failures by verifying data types and value ranges prior to inference.
3. **Isolation (Docker & Testing)**: Ensuring code runs identically on local environments, CI/CD pipelines, or cloud servers, while automating non-regression testing.

---

## 2. Service Architecture
The application relies on a multi-service architecture orchestrated by **Docker Compose**:

* **`pharmasupply-db` (PostgreSQL)**: Relational database storing historical inventory data.
* **`pharmasupply-api` (FastAPI)**: Python Web service loading the `.joblib` artifact to expose HTTP endpoints.

---

## 3. The API Script: `src/api.py`

The `src/api.py` script serves as the core of the MLOps application. It is divided into 5 main sections:

### A. Configuration & Global Variables
```python
MODEL_PATH = os.getenv(
    "MODEL_PATH", "models/xgboost_pharma_demand.joblib"
)
loaded_model = None

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
* **`MODEL_PATH`**: Specifies the location of the model artifact. Utilizing `os.getenv` allows dynamic path adjustments through environment variables (ideal for CI/CD).
* **`FEATURE_COLS`**: Strict list of feature columns in the exact order required by the XGBoost model for inference.

### B. Data Validation (Pydantic v2)
```python
class PredictionInput(BaseModel):
    store_id: int = Field(
        ..., json_schema_extra={"example": 101}
    )
    product_id: int = Field(
        ..., json_schema_extra={"example": 2045}
    )
    month: int = Field(
        ..., ge=1, le=12, json_schema_extra={"example": 10}
    )
    dayofweek: int = Field(
        ..., ge=0, le=6, json_schema_extra={"example": 2}
    )
    is_weekend: int = Field(
        ..., ge=0, le=1, json_schema_extra={"example": 0}
    )
    stock_on_hand: float = Field(
        ..., ge=0, json_schema_extra={"example": 500.0}
    )
    supplier_lead_time: float = Field(
        ..., ge=0, json_schema_extra={"example": 3.0}
    )
    lag_7: float = Field(
        ..., ge=0, json_schema_extra={"example": 120.5}
    )
    lag_14: float = Field(
        ..., ge=0, json_schema_extra={"example": 115.0}
    )
    lag_30: float = Field(
        ..., ge=0, json_schema_extra={"example": 110.0}
    )
    rolling_mean_7: float = Field(
        ..., ge=0, json_schema_extra={"example": 118.2}
    )
    rolling_mean_30: float = Field(
        ..., ge=0, json_schema_extra={"example": 112.4}
    )
```
* **Data Contracts (`PredictionInput`)**: Defines the required input schema. Constraints such as `ge=1, le=12` act as an automatic validation filter: if a request sends month `13` or a negative stock value, FastAPI immediately rejects the request with an `HTTP 422 Unprocessable Entity` error without invoking the model.

```python
class PredictionOutput(BaseModel):
    store_id: int
    product_id: int
    predicted_demand: float
    model_version: str = "1.0.0"
```
* **Output Schema (`PredictionOutput`)**: Standardizes the client response format, including prediction results and model versioning for full traceability.

### C. Lifecycle Management (*Lifespan*)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model
    if os.path.exists(MODEL_PATH):
        try:
            loaded_model = joblib.load(MODEL_PATH)
            logger.info(
                f"✅ Model successfully loaded from: {MODEL_PATH}"
            )
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            loaded_model = None
    yield
    loaded_model = None
```
* **Single Memory Loading**: Instead of reloading the `.joblib` file on every request (which would degrade throughput), the `lifespan` manager loads it once during FastAPI server startup.

### D. HTTP Endpoints
1. **`/health` (GET)**: Monitoring endpoint for orchestrators (Kubernetes, Docker, Load Balancers). It confirms whether the API is online and the model is loaded in memory.
2. **`/predict` (POST)**: Receives a list of `PredictionInput` objects, then:
   * Converts the JSON payload into a `pandas.DataFrame`.
   * Filters and orders columns according to `FEATURE_COLS`.
   * Runs inference: `loaded_model.predict(X)`.
   * Applies domain rules: `np.maximum(0.0, raw_preds)` to prevent negative demand predictions.
   * Returns predictions formatted as a list of `PredictionOutput`.

---

## 4. Test Suite: `tests/test_api.py`

In MLOps practices, deployment requires automated tests. The `tests/test_api.py` module verifies API functionality using `pytest` and FastAPI's `TestClient`.

### Automated Testing Structure:
```python
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_valid_data():
    payload = [{
        "store_id": 101,
        "product_id": 2045,
        "month": 10,
        "dayofweek": 2,
        "is_weekend": 0,
        "stock_on_hand": 500.0,
        "supplier_lead_time": 3.0,
        "lag_7": 120.5,
        "lag_14": 115.0,
        "lag_30": 110.0,
        "rolling_mean_7": 118.2,
        "rolling_mean_30": 112.4
    }]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert "predicted_demand" in data[0]
    assert data[0]["predicted_demand"] >= 0.0

def test_predict_endpoint_invalid_data():
    # Test sending invalid month (13)
    payload = [{"month": 13, ...}]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error
```

### Running Tests:
```bash
pytest tests/test_api.py
```

---

## 5. Containerization Setup (Docker)

### A. The `Dockerfile`
Configuration file located at the project root to build an autonomous Docker image for the API.

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
    "psycopg[binary]"

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Instruction Overview:
* **`FROM python:3.11-slim`**: Minimal Debian Linux base image with pre-installed Python 3.11.
* **`WORKDIR /app`**: Working directory inside the container.
* **`RUN apt-get update ...`**: Installation of C/C++ libraries required for PostgreSQL and XGBoost.
* **`RUN pip install ...`**: Installation of runtime Python dependencies.
* **`COPY`**: Copies source code (`src/`) and serialized model artifacts (`models/`).
* **`CMD`**: Execution command on startup to run the Uvicorn ASGI server.

---

### B. Service Orchestration with `docker-compose.yml`
Links the API container to the PostgreSQL database within a shared virtual network.

```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: pharmasupply-db
    restart: always
    env_file:
      - .env
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: [ "CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}" ]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build: .
    container_name: pharmasupply-api
    restart: always
    env_file:
      - .env
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
    driver: local
```

* **`depends_on` + `condition: service_healthy`**: Critical MLOps dependency configuration. The API service waits for PostgreSQL to pass its health check before starting.

---

## 6. Execution and CLI Validation

### A. Launching the Docker Stack
```bash
docker compose up --build -d
```

### B. Validation using `curl`

1. **Status Verification (`/health`):**
   ```bash
   curl -X GET http://localhost:8000/health
   ```
   *Expected Response:* `{"status":"healthy","model_loaded":true}`

2. **Executing Inference (`/predict`):**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '[
       {
         "store_id": 101,
         "product_id": 2045,
         "month": 10,
         "dayofweek": 2,
         "is_weekend": 0,
         "stock_on_hand": 500.0,
         "supplier_lead_time": 3.0,
         "lag_7": 120.5,
         "lag_14": 115.0,
         "lag_30": 110.0,
         "rolling_mean_7": 118.2,
         "rolling_mean_30": 112.4
       }
     ]'
   ```
   *Returned Response:*
   ```json
   [
     {
       "store_id": 101,
       "product_id": 2045,
       "predicted_demand": 133.93557739257812,
       "model_version": "1.0.0"
     }
   ]
   ```

### Inference Interpretation:
For store **101** and product **2045**, on an October Tuesday with 500 units on hand and recent average sales of 118 units/day, the XGBoost model predicts a demand of **~134 units**. Given a 3-day supplier lead time (`supplier_lead_time: 3.0`), inventory management systems can trigger reordering safely.

</details>