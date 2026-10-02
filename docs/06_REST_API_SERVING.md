# Technical Documentation
## Phase 6: REST API Serving (`src/api.py`)  
**Project:** PharmaSupply-ML  
**Framework:** Standard MLOps Blueprint  
**File:** `docs/06_REST_API_SERVING.md` (`src/api.py`)    
**Status:** Validated

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>  

## 1. Vue d'ensemble et objectifs

Ce document détaille l'architecture et le fonctionnement du script `src/api.py`. Son rôle principal est d'exposer le modèle de Machine Learning (XGBoost) préalablement entraîné via une API RESTful robuste, hautement performante et documentée automatiquement (Swagger UI).

### Concepts Clés :
- **Swagger UI** : Interface web interactive générée automatiquement par FastAPI (basée sur la norme OpenAPI) permettant de documenter et de tester visuellement les endpoints de l'API depuis un navigateur.  
- **Pydantic** : Bibliothèque Python de validation de données et de gestion de paramètres. Elle garantit que les objets JSON manipulés par l'API respectent un schéma strict (types de données, valeurs autorisées) et convertit les requêtes HTTP en objets Python typés.

---

## 2. Architecture logicielle

Le script est placé dans le répertoire `src/` (le cœur métier de l'application) car il constitue le service principal de déploiement, par opposition aux scripts utilitaires ou d'automatisation qui résident dans `scripts/`.

---

## 3. Explication de l'implémentation

### 3.1. Imports et variables globales
- **Bibliothèques principales** : Utilisation de `FastAPI` pour le routage, `pandas` pour la manipulation des données d'inférence, et `joblib` pour la désérialisation du modèle.
- **`MODEL_PATH`** : Définit l'emplacement de l'artefact. L'utilisation de `os.getenv` permet de surcharger ce chemin dynamiquement via les variables d'environnement (pratique pour l'isolation Docker).
- **`loaded_model`** : Variable globale initialisée à `None`. Elle stocke le modèle en mémoire RAM pour éviter tout Re-loading E/S coûteux à chaque requête.

### 3.2. Schémas de validation (Pydantic)
Les classes héritant de `BaseModel` agissent comme des boucliers pour l'API. Elles garantissent que les données entrantes et sortantes sont strictement conformes aux attentes :
- **`PredictionInput`** : Définit les caractéristiques requises par le modèle XGBoost. L'utilisation de `Field(...)` permet d'imposer des règles de validation (ex. : `ge=1, le=12` limite le mois entre 1 et 12).
- **`PredictionOutput`** : Standardise la forme de la réponse renvoyée au client (identifiants et prédiction finale).
- **`HealthCheckOutput`** : Définit la structure JSON renvoyée par la route `/health` (`status` et `model_loaded`).

### 3.3. Gestion du cycle de vie (Lifespan)
La fonction `lifespan(app: FastAPI)` est un gestionnaire asynchrone critique pour les performances MLOps :
- **Au démarrage (Before `yield`)** : Le serveur vérifie l'existence du fichier `.joblib` et le charge en mémoire (`loaded_model`). Cette opération lourde n'est exécutée qu'une seule fois.
- **En cours d'exécution (`yield`)** : Cède le contrôle à FastAPI pour écouter et traiter les requêtes HTTP.
- **À l'arrêt (After `yield`)** : Vide la variable pour libérer proprement la mémoire RAM lors de la fermeture du serveur.

### 3.4. Les routes (Endpoints)

#### `GET /health`
- **Rôle** : Sonde de viabilité (*Health check*).
- **Fonctionnement** : Indique si l'API est en ligne et si le modèle est bien chargé. C'est indispensable pour les orchestrateurs (Docker Compose, Kubernetes) afin de vérifier si le conteneur est prêt à recevoir du trafic.

#### `POST /predict`
- **Rôle** : Point de terminaison principal dédié à l'inférence par lots (*batch*).
- **Fonctionnement** :
  1. **Clause de garde** : Renvoie une erreur HTTP 503 immédiate si le modèle n'a pas pu être chargé.
  2. **Transformation** : Convertit la liste d'objets JSON validés par Pydantic en un `DataFrame` Pandas, format exigé par le modèle.
  3. **Prédiction** : Exécute `loaded_model.predict(df)`.
  4. **Règle métier** : Applique un `max(0, pred)` sur chaque résultat. Cette contrainte physique garantit qu'une prédiction de ventes (demande) ne sera jamais négative.
  5. **Retour** : Renvoie les prédictions formatées sous forme de liste JSON.

---

## 4. Exécution

Pour lancer le serveur d'API en mode développement dans ton environnement Conda (`pharmasupply-ml`) :

~~~~bash
conda activate pharmasupply-ml
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
~~~~

La documentation Swagger interactive est ensuite accessible à l'adresse : `http://localhost:8000/docs`.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

## 1. Overview & Objectives

This document details the architecture and inner mechanics of the `src/api.py` script. Its primary role is to expose the previously trained Machine Learning model (XGBoost) via a robust, high-performance, and production-ready RESTful API featuring interactive auto-generated documentation (Swagger UI).

### Key Concepts:
- **Swagger UI**: Interactive web interface automatically generated by FastAPI (based on the OpenAPI specification) allowing developers to visually document and test API endpoints directly from a browser.
- **Pydantic**: Python library for data validation and settings management. It ensures that JSON payloads handled by the API strictly follow a predefined schema (data types, allowed value ranges) and parses HTTP request data into typed Python objects.

---

## 2. Software Architecture

The script is located in the `src/` directory (the core application logic) as it serves as the main serving entry point, as opposed to automation and DevOps helper scripts residing in `scripts/`.

---

## 3. Section-by-Section Implementation Breakdown

### 3.1. Imports and Global Configuration
- **Core Libraries**: Uses `FastAPI` for routing, `pandas` for handling tabular inference payloads, and `joblib` for model deserialization.
- **`MODEL_PATH`**: Defines the target model artifact path. Leveraging `os.getenv` allows dynamic runtime overrides via environment variables (essential for Docker containerization).
- **`loaded_model`**: A global variable initialized to `None`. It holds the model in RAM to prevent redundant file I/O operations across incoming HTTP requests.

### 3.2. Data Validation Schemas (Pydantic)
Classes inheriting from `BaseModel` act as structural shields for the API, ensuring incoming and outgoing payloads strictly conform to business contracts:
- **`PredictionInput`**: Defines the feature payload expected by the XGBoost model. Using `Field(...)` allows enforcing validation constraints (e.g., `ge=1, le=12` restricts the month range between 1 and 12).
- **`PredictionOutput`**: Standardizes the response payload returned to API clients (identifiers and final demand predictions).
- **`HealthCheckOutput`**: Defines the JSON schema returned by the `/health` endpoint (`status` and `model_loaded`).

### 3.3. Lifespan Management (Resource Handling)
The `lifespan(app: FastAPI)` function is an asynchronous context manager critical for MLOps efficiency:
- **Before `yield` (Startup)**: The server verifies the `.joblib` file existence and loads the model into RAM (`loaded_model`). This high-latency task runs strictly once at startup.
- **`yield`**: Yields execution back to FastAPI to start listening for and processing incoming HTTP requests.
- **After `yield` (Shutdown)**: Clears the model variable to properly release system RAM during server shutdown.

### 3.4. API Endpoints (Routes)

#### `GET /health`
- **Role**: Liveness and readiness probe (Health check).
- **Mechanics**: Checks if the API service is alive and whether the ML model is successfully loaded in RAM. Essential for container orchestrators (such as Docker Compose or Kubernetes) and load balancers to determine service readiness.

#### `POST /predict`
- **Role**: Main inference endpoint designed for batch processing.
- **Mechanics**:
  1. **Guard Clause**: Returns an immediate HTTP 503 error if the model artifact failed to load.
  2. **Transformation**: Converts the list of Pydantic-validated JSON items into a Pandas `DataFrame`, matching the tabular format expected by the model.
  3. **Inference**: Executes `loaded_model.predict(df)`.
  4. **Business Rule**: Applies a `max(0, pred)` floor constraint on each output to physically prevent negative demand predictions.
  5. **Response**: Returns the formatted prediction payload list as JSON.

---

## 4. Execution

To run the API server in development mode within your Conda environment (`pharmasupply-ml`):

~~~~bash
conda activate pharmasupply-ml
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
~~~~

Interactive Swagger documentation is then accessible at: `http://localhost:8000/docs`.


</details>
