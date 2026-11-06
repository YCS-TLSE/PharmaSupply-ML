# PharmaSupply-ML — From Exploratory Notebooks to Industrial MLOps & LLM Architecture
> **MLOps Framework (V1) & Generative AI (V2) Blueprint | P0 Stabilized, P1 Industrialized, Production-Ready Educational Guide**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Model](https://img.shields.io/badge/Model-XGBoost-orange)
![API](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Observability](https://img.shields.io/badge/Observability-Prometheus-E6522C?logo=prometheus&logoColor=white)
![Security](https://img.shields.io/badge/Security-X--API--Key%20%2F%20CORS%20%2F%20SlowAPI-blue)
![Tests](https://img.shields.io/badge/Tests-92%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-80.91%25-brightgreen)
[![Status](https://img.shields.io/badge/Status-P0%20%26%20P1%20Validated%20%7C%20P2%20Planned-brightgreen)]()
![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

## Philosophie du Projet

Ce dépôt répond à un double enjeu d'ingénierie :

- **Une architecture MLOps industrialisée (Production-Ready) :** Implémentation complète du cycle de vie ML — ingestion idempotente, validation stricte, feature engineering, entraînement XGBoost, serving FastAPI modulaire, observabilité, sécurité et monitoring — avec une trajectoire d'extension vers l'IA générative (RAG / LLM en V2).
- **Un guide pédagogique « From Lab to Prod » :** Démarche pas à pas montrant comment encapsuler une analyse exploratoire dans une enveloppe MLOps modulaire, testable, observable et conteneurisée.

Les phases **P0 (stabilisation)** et **P1 (structuration & industrialisation)** sont clôturées et validées : le socle est stabilisé, l'API est découpée en routers/services/repositories, les migrations Alembic et l'ingestion UPSERT sont opérationnelles, l'observabilité et la sécurité sont en place, et la suite de tests atteint **92 tests au vert pour 80,91 % de couverture**.

> **Lexique des priorités (pour lecteurs non initiés)**
> - **P0 — Stabilisation du socle (Urgent) :** correctifs critiques, suppression des fallbacks silencieux et fiabilisation des entrées/sorties.
> - **P1 — Industrialisation MLOps (Majeur) :** refactorisation modulaire, tests unitaires et d'intégration (couverture > 80 %), observabilité, sécurité et migrations.
> - **P2 — Industrialisation avancée (Évolutions) :** fonctionnalités futures (réentraînement automatique piloté par le drift, scaling cloud, Copilote IA V2).

---

## Vue d'ensemble de l'architecture

```text
┌───────────────────────┐      ┌───────────────────────────┐      ┌──────────────────────────┐
│ Synthetic Data Gen    │ ---> │ PostgreSQL 16             │ ---> │ Quality Gate & Features  │
│ src/data/generate...  │      │ Alembic + UPSERT          │      │ validation + lags        │
└───────────────────────┘      └───────────────────────────┘      └────────────┬─────────────┘
                                                                                │
                                                                                v
┌───────────────────────┐      ┌───────────────────────────┐      ┌──────────────────────────┐
│ FastAPI Serving       │ <--- │ SHAP + Evidently          │ <--- │ XGBoost + MLflow         │
│ api_routers/services/ │      │ weights + drift reports   │      │ training + registry      │
│ repositories/schemas  │      └───────────────────────────┘      └──────────────────────────┘
└──────────┬────────────┘
           │
           ├── Observabilité : logs JSON, Prometheus /metrics, /livez, /readyz, X-Request-ID
           ├── Sécurité       : X-API-Key, CORS, rate limiting slowapi
           v
┌───────────────────────┐
│ Streamlit Dashboard   │
│ thin client API       │
└──────────┬────────────┘
           │ (V2)
           v
┌──────────────────────────────────────────────────────────────────────────────┐
│ RAG / LLM Copilot · Redistribution automatisée · Déploiement Cloud & SLOs   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Architecture modulaire de l'API

```text
src/api.py                 # composition root : lifespan, sécurité, observabilité, routers
src/api_routers/           # health, data, alerts, features, predict
src/services/              # règles métier pures : alertes, réapprovisionnement, FEFO, features
src/repositories/          # engine SQLAlchemy poolé + requêtes SQL read-only
src/schemas.py             # contrats Pydantic v2 entrée/sortie
src/security.py            # X-API-Key, CORS, slowapi
src/observability.py       # logs JSON, métriques Prometheus, journal de prédiction
src/config.py              # source de vérité des constantes et variables d'environnement
```

---

## Stack technique

| Domaine | Technologies |
|---|---|
| **Serving API** | FastAPI 0.162, Pydantic v2, Uvicorn, architecture routers/services/repositories |
| **Données** | PostgreSQL 16, SQLAlchemy 2.x (pool + `pool_pre_ping`), Alembic, ingestion UPSERT sur clé métier `(store_id, product_id, date)` |
| **ML** | XGBoost 2.1, scikit-learn 1.5, joblib, parité train/serve via un constructeur de features unique |
| **Explicabilité & drift** | SHAP 0.46, export `reports/feature_weights.json`, Evidently 0.7 (holdout offline + drift production sur logs agrégés) |
| **Migrations** | Alembic 1.14 (`alembic upgrade head`, `alembic upgrade head --sql`) |
| **Observabilité** | Logs JSON, `X-Request-ID`, métriques Prometheus (`/metrics`), sondes `/livez` et `/readyz` |
| **Sécurité** | `X-API-Key`, CORS, rate limiting slowapi (`/predict`, `/alerts`) |
| **Dashboard** | Streamlit 1.40 (thin client, fallback démo explicitement marqué) |
| **Qualité** | Ruff (`E`, `F`, `I`), pytest 8.3, pytest-cov 6, Testcontainers PostgreSQL, Trivy (CI) |
| **Conteneurisation** | Docker multi-images, dépendances épinglées, utilisateur non-root, healthchecks |

---

## Environnement de Développement

Le projet est développé sous **WSL2 (Ubuntu)** en environnement shell **Bash / Zsh** couplé à **Miniconda**.

**Pourquoi ce choix d'architecture de dev ?**
- **Parité Dev/Prod (Linux native) :** Exécution dans un sous-système Linux garantissant une parité stricte avec les conteneurs Docker et les serveurs de production.
- **Isolation stricte avec Conda :** Gestion précise des dépendances Python et C/C++ (XGBoost, SHAP, FastAPI, etc.).
- **Productivité Shell :** Automatisation fluide des scripts CLI, des commandes Docker et des pipelines CI/CD.

---

## Données : Génération Synthétique & Prêt pour le Réel

En raison de la confidentialité et de la réglementation sur les données pharmaceutiques (GDPR / HIPAA), le projet s'appuie sur une **génération automatisée de données synthétiques** (`src/data/generate_data.py`).

L'architecture respecte le principe de **découplage** :
- **Simulation réaliste :** saisonnalité des pathologies, délais de réapprovisionnement variables, ruptures, contraintes logistiques (MOQ, taille de lot, shelf-life) et suivi FEFO.
- **Ingestion incrémentale :** la table `raw_pharmaceutical_demand` possède une clé métier unique `(store_id, product_id, date)` ; l'ingestion utilise un **UPSERT** (`ON CONFLICT DO UPDATE`) et non plus un `TRUNCATE` destructif.
- **Connecteur réel (« Plug & Play ») :** le pipeline d'ingestion est générique ; un adaptateur de source peut être branché sans modifier le cœur applicatif.

---

## Démarrage rapide (Quickstart)

> ⚠️ **Prérequis :** Docker et Conda (ou Python 3.11 avec `venv`) installés.

1. **Cloner le dépôt et se placer dans le projet**

```bash
git clone https://github.com/YCS-TLSE/PharmaSupply-ML.git
cd PharmaSupply-ML
```

2. **Créer ou mettre à jour l'environnement virtuel**

```bash
# Première installation
conda env create -f environment.yml
conda activate pharma-supply

# Mise à jour après modification des dépendances
conda env update -f environment.yml --prune
```

Variante Pip / venv :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Configurer les variables d'environnement**

```bash
cp .env.example .env
# Renseigner POSTGRES_*, puis éventuellement API_KEY, SMTP_PASSWORD, MLFLOW_*
```

4. **Démarrer l'infrastructure (PostgreSQL + API + Dashboard)**

```bash
docker compose up -d --build
```

> `--build` est important après un changement de dépendances Python : l'image API est reconstruite avec `requirements.api.txt`.

5. **Appliquer les migrations sur une base existante (optionnel)**

```bash
alembic upgrade head
# Vérification du SQL généré, sans base live :
alembic upgrade head --sql
```

> Un volume Docker neuf est initialisé automatiquement via `sql/schema.sql`. Alembic est la voie recommandée pour les bases existantes.

6. **Exécuter le pipeline complet de bout en bout**

```bash
python run_pipeline.py
```

Le script orchestre : vérification Docker, génération et **UPSERT** des données, quality gate, feature engineering, entraînement XGBoost, **rebuild de l'image API + attente de `/readyz`**, export SHAP, drift Evidently offline et smoke tests `/predict` + `/alerts`.

7. **(Optionnel) Lancer l'API REST en autonome**

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Sondes et métriques :

```bash
curl http://localhost:8000/livez
curl http://localhost:8000/readyz
curl http://localhost:8000/metrics
```

Avec authentification activée :

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/stores
```

8. **Lancer le Dashboard Streamlit**

```bash
streamlit run src/dashboard.py
```

9. **Lancer les lints et les tests**

```bash
# Lint
ruff check src tests run_pipeline.py alembic

# Tests unitaires + couverture (seuil 70 %)
pytest -q -m "not integration" --cov=src --cov-report=term-missing

# Tests d'intégration (Docker / Testcontainers PostgreSQL)
pytest -q -m integration
```

10. **(Optionnel) Activer MLflow tracking + registry**

```bash
MLFLOW_ENABLED=true python -m src.train
mlflow ui --backend-store-uri file:./mlruns
```

---

## Feuille de route du projet & Documentation

> Le document `docs/00_MLOPS_BLUEPRINT.md` est le **document-cadre méthodologique**. La roadmap comporte **7 phases MLOps validées (01 → 07)** et **2 phases V2 en perspective (08 → 09)**.

### Roadmap opérationnelle — 7 phases MLOps validées (01 → 07) + 2 phases V2 (08 → 09)

- [x] **Phase 01 — `01_INITIALIZATION_AND_SCOPING.md` :** Cadrage métier, choix technologiques et structure du dépôt.
- [x] **Phase 02 — `02_DATA_INFRASTRUCTURE_AND_INGESTION.md` :** PostgreSQL 16, clé métier unique, **Alembic** et ingestion **UPSERT** incrémentale (`src/ingestion.py`).
- [x] **Phase 03 — `03_OPS_GUARDRILS_DATA_QUALITY_AND_CLEANING.md` :** Quality gate hard fail / soft correction, bornes métier et imputation (`src/validation.py`).
- [x] **Phase 04 — `04_ML_PIPELINE_PREPROCESSING_FEATURE_ENGINEERING.md` :** Lags/rolling means, split temporel, XGBoost, **constructeur de features partagé** et test de parité train/serve (`src/features.py`, `src/train.py`).
- [x] **Phase 05 — `05_EXPLAINABILITY_AND_MONITORING.md` :** SHAP, export des poids normalisés, drift Evidently offline **et drift production** sur logs agrégés (`src/explainability.py`, `src/monitoring.py`).
- [x] **Phase 06 — `06_REST_API_SERVING.md` :** FastAPI modulaire (`api_routers/`, `services/`, `repositories/`), contrats Pydantic, `X-API-Key`, CORS, slowapi, logs JSON, Prometheus, `/livez`, `/readyz` (`src/api.py`).
- [x] **Phase 07 — `07_DECISION_DASHBOARD_AND_USER_INTERFACE.md` :** Dashboard Streamlit thin client, sélection par exception, simulation de réapprovisionnement et historique 30 jours (`src/dashboard.py`).
- [ ] **Phase 08 — Copilote RAG / LLM :** Recommandation de médicaments substitutifs en cas de rupture active.
- [ ] **Phase 09 — Industrialisation Cloud & Optimisation :** Redistribution automatisée inter-officines, autoscaling et politique de promotion SLOs.

### Statut des chantiers transverses P0 / P1 / P2

- [x] **P0 — Stabilisation du socle (clôturée & validée) :**
  e-mail SMTP hors du chemin de requête, erreurs explicites (404/503), contrat `/features` typé (`critical_expiry_days`, `sales_history`), pool PostgreSQL, constantes centralisées, poids SHAP consommés, plancher MOQ corrigé, CI minimale.
- [x] **P1 — Structuration & Industrialisation MLOps (clôturée & validée) :**
  découpage API en routers/services/repositories, parité train/serve, MLflow tracking + registry + signature, Alembic + UPSERT, observabilité JSON/Prometheus/sondes, sécurité API key/CORS/rate limiting, couverture ≥ 70 %, Dockerfiles épinglés non-root + scan Trivy, drift de production.
- [ ] **P2 — Industrialisation avancée (perspectives) :**
  réentraînement automatique piloté par le drift, endpoint batch et cache distribué, model server dédié, partitionnement/dimensions, déploiement cloud avec SLOs, extension V2 RAG/LLM.

### Documentation complémentaire

- `docs/00_MLOPS_BLUEPRINT.md` — blueprint MLOps bilingue.
- `docs/01_INITIALIZATION_AND_SCOPING.md` — initialisation et cadrage.
- `docs/02_DATA_INFRASTRUCTURE_AND_INGESTION.md` — infrastructure de données et ingestion.
- `docs/03_OPS_GUARDRILS_DATA_QUALITY_AND_CLEANING.md` — garde-fous et qualité des données.
- `docs/04_ML_PIPELINE_PREPROCESSING_FEATURE_ENGINEERING.md` — pipeline ML et feature engineering.
- `docs/05_EXPLAINABILITY_AND_MONITORING.md` — explicabilité et monitoring.
- `docs/06_REST_API_SERVING.md` — serving REST.

---

## Qualité & métriques

| Indicateur | Valeur |
|---|---:|
| Tests unitaires | **91** |
| Tests d'intégration (Testcontainers) | **1** |
| **Total au vert** | **92** |
| Couverture `src/` | **80,91 %** |
| Seuil de couverture | **70 %** |
| Lint | Ruff `E`, `F`, `I` — **All checks passed** |
| Scan images | Trivy CRITICAL/HIGH en CI |
| Parité train/serve | Test dédié `tests/test_parity.py` |

---

## Sécurité & exploitation

| Variable | Rôle |
|---|---|
| `API_KEY` | Active `X-API-Key` ; vide = désactivé en local |
| `API_KEY_HEADER` | Nom de l'en-tête (défaut `X-API-Key`) |
| `CORS_ORIGINS` | Origines autorisées (dashboard) |
| `RATE_LIMIT_PREDICT` / `RATE_LIMIT_ALERTS` | Quotas slowapi |
| `MLFLOW_ENABLED` | Active tracking + registry |
| `PREDICTION_LOG_PATH` | Journal JSONL utilisé par le drift production |
| `SMTP_PASSWORD` | Notifications de drift (best-effort) |

---

> Les jalons V2 sont désormais suivis directement dans la roadmap ci-dessus, aux phases 08 et 09.

</details>

---

### [EN] English Version

## Project Philosophy

This repository addresses a dual engineering challenge:

- **A production-ready, industrialized MLOps architecture:** complete ML lifecycle implementation — idempotent ingestion, strict validation, feature engineering, XGBoost training, modular FastAPI serving, observability, security and monitoring — with a Generative AI extension path (RAG / LLM in V2).
- **A "From Lab to Prod" educational guide:** a step-by-step approach showing how to wrap a classic exploratory analysis inside a modular, testable, observable and containerized MLOps architecture.

The **P0 (stabilization)** and **P1 (structuring & industrialization)** phases are closed and validated: the core is stabilized, the API is split into routers/services/repositories, Alembic migrations and UPSERT ingestion are operational, observability and security are in place, and the test suite reports **92 passing tests with 80.91% coverage**.

> **Priority legend (for non-specialist readers)**
> - **P0 — Core stabilization (Urgent):** critical fixes, removal of silent fallbacks and I/O reliability.
> - **P1 — MLOps industrialization (Major):** modular refactoring, unit/integration tests (80%+ coverage), observability, security and migrations.
> - **P2 — Advanced industrialization (Outlook):** future capabilities (drift-triggered retraining, cloud scaling, V2 AI Copilot).

---

## Architecture Overview

```text
┌───────────────────────┐      ┌───────────────────────────┐      ┌──────────────────────────┐
│ Synthetic Data Gen    │ ---> │ PostgreSQL 16             │ ---> │ Quality Gate & Features  │
│ src/data/generate...  │      │ Alembic + UPSERT          │      │ validation + lags        │
└───────────────────────┘      └───────────────────────────┘      └────────────┬─────────────┘
                                                                                │
                                                                                v
┌───────────────────────┐      ┌───────────────────────────┐      ┌──────────────────────────┐
│ FastAPI Serving       │ <--- │ SHAP + Evidently          │ <--- │ XGBoost + MLflow         │
│ api_routers/services/ │      │ weights + drift reports   │      │ training + registry      │
│ repositories/schemas  │      └───────────────────────────┘      └──────────────────────────┘
└──────────┬────────────┘
           │
           ├── Observability: JSON logs, Prometheus /metrics, /livez, /readyz, X-Request-ID
           ├── Security: X-API-Key, CORS, slowapi rate limiting
           v
┌───────────────────────┐
│ Streamlit Dashboard   │
│ thin API client       │
└──────────┬────────────┘
           │ (V2)
           v
┌──────────────────────────────────────────────────────────────────────────────┐
│ RAG / LLM Copilot · Automated redistribution · Cloud deployment & SLOs      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Modular API Architecture

```text
src/api.py                 # composition root: lifespan, security, observability, routers
src/api_routers/           # health, data, alerts, features, predict
src/services/              # pure business rules: alerts, replenishment, FEFO, features
src/repositories/          # pooled SQLAlchemy engine + read-only SQL queries
src/schemas.py             # Pydantic v2 input/output contracts
src/security.py            # X-API-Key, CORS, slowapi
src/observability.py       # JSON logs, Prometheus metrics, prediction log
src/config.py              # single source of truth for constants and environment settings
```

---

## Technical Stack

| Area | Technologies |
|---|---|
| **Serving API** | FastAPI 0.162, Pydantic v2, Uvicorn, routers/services/repositories |
| **Data** | PostgreSQL 16, SQLAlchemy 2.x (pool + `pool_pre_ping`), Alembic, UPSERT on `(store_id, product_id, date)` |
| **ML** | XGBoost 2.1, scikit-learn 1.5, joblib, shared feature builder for train/serve parity |
| **Explainability & drift** | SHAP 0.46, `reports/feature_weights.json`, Evidently 0.7 (offline holdout + production logs) |
| **Migrations** | Alembic 1.14 (`alembic upgrade head`, `alembic upgrade head --sql`) |
| **Observability** | JSON logs, `X-Request-ID`, Prometheus `/metrics`, `/livez`, `/readyz` |
| **Security** | `X-API-Key`, CORS, slowapi rate limiting (`/predict`, `/alerts`) |
| **Dashboard** | Streamlit 1.40 (thin client, explicitly labelled demo fallback) |
| **Quality** | Ruff (`E`, `F`, `I`), pytest 8.3, pytest-cov 6, Testcontainers PostgreSQL, Trivy (CI) |
| **Containers** | Multi-image Docker, pinned dependencies, non-root user, healthchecks |

---

## Development Environment

The project is developed under **WSL2 (Ubuntu)** using a **Bash / Zsh** shell paired with **Miniconda**.

**Why this dev architecture choice?**
- **Dev/Prod parity (native Linux):** Linux execution guarantees strict parity with Docker containers and production servers.
- **Strict Conda isolation:** precise Python and C/C++ dependency management (XGBoost, SHAP, FastAPI, etc.).
- **Shell productivity:** seamless CLI, Docker and CI/CD automation.

---

## Data: Synthetic Generation & Real-World Readiness

Due to pharmaceutical data privacy and regulations (GDPR / HIPAA), the project relies on an **automated synthetic data generator** (`src/data/generate_data.py`).

The architecture follows a **decoupling** principle:
- **Realistic simulation:** pathology seasonality, variable lead times, stockouts, logistics constraints (MOQ, batch size, shelf life) and FEFO tracking.
- **Incremental ingestion:** `raw_pharmaceutical_demand` has a natural key `(store_id, product_id, date)`; ingestion uses an **UPSERT** (`ON CONFLICT DO UPDATE`) instead of a destructive `TRUNCATE`.
- **Plug & Play connector:** the ingestion pipeline is generic; a production source adapter can be plugged in without changing the application core.

---

## Quickstart

> ⚠️ **Prerequisites:** Docker and Conda (or Python 3.11 with `venv`) installed.

1. **Clone the repository and enter the directory**

```bash
git clone https://github.com/YCS-TLSE/PharmaSupply-ML.git
cd PharmaSupply-ML
```

2. **Create or update the virtual environment**

```bash
# First installation
conda env create -f environment.yml
conda activate pharma-supply

# Update after dependency changes
conda env update -f environment.yml --prune
```

Pip / venv alternative:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Configure environment variables**

```bash
cp .env.example .env
# Fill POSTGRES_*, then optionally API_KEY, SMTP_PASSWORD, MLFLOW_*
```

4. **Start the infrastructure (PostgreSQL + API + Dashboard)**

```bash
docker compose up -d --build
```

> `--build` matters after a Python dependency change: the API image is rebuilt from `requirements.api.txt`.

5. **Apply migrations on an existing database (optional)**

```bash
alembic upgrade head
# Inspect generated SQL without a live database:
alembic upgrade head --sql
```

> A fresh Docker volume is bootstrapped from `sql/schema.sql`. Alembic is the recommended path for existing databases.

6. **Run the full end-to-end pipeline**

```bash
python run_pipeline.py
```

The orchestrator runs: Docker check, data generation and **UPSERT**, quality gate, feature engineering, XGBoost training, **API image rebuild + `/readyz` wait**, SHAP export, offline Evidently drift, and `/predict` + `/alerts` smoke tests.

7. **(Optional) Launch the REST API independently**

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Probes and metrics:

```bash
curl http://localhost:8000/livez
curl http://localhost:8000/readyz
curl http://localhost:8000/metrics
```

With authentication enabled:

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/stores
```

8. **Launch the Streamlit Dashboard**

```bash
streamlit run src/dashboard.py
```

9. **Run linters and tests**

```bash
# Lint
ruff check src tests run_pipeline.py alembic

# Unit tests + coverage (70% threshold)
pytest -q -m "not integration" --cov=src --cov-report=term-missing

# Integration tests (Docker / Testcontainers PostgreSQL)
pytest -q -m integration
```

10. **(Optional) Enable MLflow tracking + registry**

```bash
MLFLOW_ENABLED=true python -m src.train
mlflow ui --backend-store-uri file:./mlruns
```

---

## Project Roadmap & Documentation

> `docs/00_MLOPS_BLUEPRINT.md` is the **methodological framing document**. The roadmap contains **7 validated MLOps phases (01 → 07)** and **2 V2 outlook phases (08 → 09)**.

### Operational roadmap — 7 validated MLOps phases (01 → 07) + 2 V2 phases (08 → 09)

- [x] **Phase 01 — `01_INITIALIZATION_AND_SCOPING.md`:** Business scoping, technology choices and repository structure.
- [x] **Phase 02 — `02_DATA_INFRASTRUCTURE_AND_INGESTION.md`:** PostgreSQL 16, natural business key, **Alembic** and incremental **UPSERT** ingestion (`src/ingestion.py`).
- [x] **Phase 03 — `03_OPS_GUARDRILS_DATA_QUALITY_AND_CLEANING.md`:** Hard-fail/soft-correction quality gate, business bounds and imputation (`src/validation.py`).
- [x] **Phase 04 — `04_ML_PIPELINE_PREPROCESSING_FEATURE_ENGINEERING.md`:** Lags/rolling means, temporal split, XGBoost, **shared feature builder** and train/serve parity test (`src/features.py`, `src/train.py`).
- [x] **Phase 05 — `05_EXPLAINABILITY_AND_MONITORING.md`:** SHAP, normalized weight export, offline Evidently drift **and production drift** on aggregated logs (`src/explainability.py`, `src/monitoring.py`).
- [x] **Phase 06 — `06_REST_API_SERVING.md`:** Modular FastAPI (`api_routers/`, `services/`, `repositories/`), Pydantic contracts, `X-API-Key`, CORS, slowapi, JSON logs, Prometheus, `/livez`, `/readyz` (`src/api.py`).
- [x] **Phase 07 — `07_DECISION_DASHBOARD_AND_USER_INTERFACE.md`:** Streamlit thin client, exception-driven selection, replenishment simulation and 30-day history (`src/dashboard.py`).
- [ ] **Phase 08 — RAG / LLM Copilot:** Alternative drug recommendations during active stockouts.
- [ ] **Phase 09 — Cloud Industrialization & Optimization:** Automated inter-pharmacy redistribution, autoscaling and SLO promotion policy.

### Cross-cutting P0 / P1 / P2 workstream status

- [x] **P0 — Core stabilization (closed & validated):**
  SMTP out of the request path, explicit errors (404/503), typed `/features` contract (`critical_expiry_days`, `sales_history`), PostgreSQL pool, centralized constants, SHAP weights consumed, MOQ floor fixed, minimal CI.
- [x] **P1 — Structuring & MLOps industrialization (closed & validated):**
  API split into routers/services/repositories, train/serve parity, MLflow tracking + registry + signature, Alembic + UPSERT, JSON/Prometheus/probe observability, API key/CORS/rate limiting, coverage ≥ 70%, pinned non-root Dockerfiles + Trivy scan, production drift.
- [ ] **P2 — Advanced industrialization (outlook):**
  drift-triggered retraining, batch endpoint and distributed cache, dedicated model server, partitioning/dimensions, cloud deployment with SLOs, RAG/LLM V2 extension.

### Additional documentation

- `docs/00_MLOPS_BLUEPRINT.md` — bilingual MLOps blueprint.
- `docs/01_INITIALIZATION_AND_SCOPING.md` — initialization and scoping.
- `docs/02_DATA_INFRASTRUCTURE_AND_INGESTION.md` — data infrastructure and ingestion.
- `docs/03_OPS_GUARDRILS_DATA_QUALITY_AND_CLEANING.md` — guardrails and data quality.
- `docs/04_ML_PIPELINE_PREPROCESSING_FEATURE_ENGINEERING.md` — ML pipeline and feature engineering.
- `docs/05_EXPLAINABILITY_AND_MONITORING.md` — explainability and monitoring.
- `docs/06_REST_API_SERVING.md` — REST serving.

---

## Quality & Metrics

| Indicator | Value |
|---|---:|
| Unit tests | **91** |
| Integration tests (Testcontainers) | **1** |
| **Total passing** | **92** |
| `src/` coverage | **80.91%** |
| Coverage threshold | **70%** |
| Lint | Ruff `E`, `F`, `I` — **All checks passed** |
| Image scan | Trivy CRITICAL/HIGH in CI |
| Train/serve parity | Dedicated `tests/test_parity.py` |

---

## Security & Operations

| Variable | Purpose |
|---|---|
| `API_KEY` | Enables `X-API-Key`; empty disables auth locally |
| `API_KEY_HEADER` | Header name (default `X-API-Key`) |
| `CORS_ORIGINS` | Allowed dashboard origins |
| `RATE_LIMIT_PREDICT` / `RATE_LIMIT_ALERTS` | slowapi quotas |
| `MLFLOW_ENABLED` | Enables tracking + registry |
| `PREDICTION_LOG_PATH` | JSONL log used by production drift |
| `SMTP_PASSWORD` | Best-effort drift notifications |

---

> V2 milestones are now tracked directly in the roadmap above, as Phases 08 and 09.
