# PharmaSupply-ML — From Exploratory Notebooks to Industrial MLOps & LLM Architecture
> **Standard MLOps Framework (V1) & Generative AI (V2) Blueprint | Production-Ready Educational Guide**


![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat&logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-111111?style=flat)
![Status](https://img.shields.io/badge/Status-In--Development-orange?style=flat)
![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)

---
<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>  

## Philosophie du Projet

Ce dépôt répond à un double enjeu d'ingénierie :

* **Une architecture MLOps de base (Production-Ready) :** Implémentation pratique des fondamentaux du cycle de vie ML (ingestion idempotente, validation stricte des données, serving par API, monitoring) avec une trajectoire d'extension vers l'IA générative (RAG / LLM en V2).
* **Un guide pédagogique « From Lab to Prod » :** Démarche pas à pas montrant comment encapsuler une analyse exploratoire classique dans une enveloppe MLOps modulaire, testable et conteneurisée.

---

## Vue d'ensemble de l'architecture

```text
+-----------------------+     +------------------------+     +------------------------+
|  Synthetic Data Gen   | --> | PostgreSQL (Docker)    | --> | Data Quality & Ops     |
|  (Raw CSV Generation) |     | (Idempotent Ingestion) |     | (Automated Assertions) |
+-----------------------+     +------------------------+     +------------------------+
                                                                         |
                                                                         v
+-----------------------+     +------------------------+     +------------------------+
|  FastAPI / Streamlit  | <-- | SHAP & Monitoring      | <-- | Feature Engineering    |
|  (Serving & Dashboard)|     | (Drift & Explainability) |     | (Lags, XGBoost Train)  |
+-----------------------+     +------------------------+     +------------------------+
            |
            | (V2 Extension)
            v
+-------------------------------------------------------------------------------------+
| Generative AI Module: RAG / LLM Agent                                              |
| (Automated Alternative Drug Recommendations during Active Stockouts)               |
+-------------------------------------------------------------------------------------+
```

## Environnement de Développement

Le projet est développé sous **WSL2 (Ubuntu)** en environnement shell **Bash / Zsh** couplé à **Miniconda**.

**Pourquoi ce choix d'architecture de dev ?**
* **Parité Dev/Prod (Linux native) :** L'exécution du code dans un sous-système Linux (WSL2) garantit une parité stricte avec les conteneurs Docker et les serveurs de déploiement en production, évitant les problèmes de compatibilité spécifiques à Windows.
* **Isolation stricte avec Conda :** Utilisation de `miniconda` pour gérer précisément les dépendances Python et C/C++ complexes (indispensables pour XGBoost, SHAP, FastAPI, etc.).
* **Productivité Shell :** L'écosystème Bash/Zsh offre une automatisation fluide des scripts CLI, des commandes Docker et des pipelines CI/CD.

---

## Données : Génération Synthétique & Prêt pour le Réel

En raison de la confidentialité stricte et de la réglementation sur les données pharmaceutiques (GDPR / HIPAA), ce projet s'appuie sur une **génération automatisée de données synthétiques** (`src/generate_data.py`). 

Cependant, l'architecture respecte le principe de **découplage** :
* **Simulation réaliste :** Le générateur reproduit les comportements d'une vraie Supply Chain (saisonnalité des pathologies, délais de réapprovisionnement, risques de rupture).
* **Connecteur Réel ("Plug & Play") :** Le pipeline d'ingestion est conçu de manière générique. Un simple basculement de variable d'environnement (`DATA_SOURCE=production`) permet de brancher le pipeline sur une vraie base de données sans modifier le cœur de l'application.

---

## Démarrage rapide (Quickstart)

> ⚠️ **Prérequis :** Assurez-vous d'avoir Docker et Conda (ou Python 3.11 avec `venv`) installés sur votre machine.

1. **Cloner le dépôt et se placer dans le projet**

~~~~bash
git clone https://github.com/YCS-TLSE/PharmaSupply-ML.git
cd PharmaSupply-ML
~~~~

2. **Créer et activer l'environnement virtuel**

* **Option A : Avec Conda (Recommandé)**
~~~~bash
conda env create -f environment.yml
conda activate pharmasupply-ml
~~~~

* **Option B : Avec Pip / venv classique**
~~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~~

3. **Démarrer l'infrastructure (PostgreSQL via Docker)**

~~~~bash
docker compose up -d
~~~~

4. **Exécuter le pipeline complet de bout en bout (Orchestration E2E)**

~~~~bash
python run_pipeline.py
~~~~
*(Ce script orchestre automatiquement : la génération de données synthétiques, l'ingestion PostgreSQL, la validation des données, l'entraînement XGBoost et l'explicabilité/monitoring).*

5. **(Optionnel) Lancer l'API REST FastAPI en autonome**

~~~~bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
~~~~

6. **Lancer le Dashboard Streamlit (Phase 07 - à venir)**

~~~~bash
streamlit run src/streamlit.py
~~~~

---

## Feuille de route du projet & Documentation (Phases 00 à 07)

### V1 — Moteur MLOps de prévision de la demande (Cœur d'infrastructure)

* [x] **`00_MLOPS_BLUEPRINT.md` — Blueprint & Architecture :** Vision globale, cycle de vie MLOps et stratégie d'encapsulation.
* [x] **`01_INITIALIZATION_AND_SCOPING.md` — Initialisation :** Cadrage du besoin métier, choix technologiques et structure du dépôt.
* [x] **`02_DATA_INFRASTRUCTURE_AND_INGESTION.md` — Ingestion & Persistance :** Base PostgreSQL conteneurisée et pipeline d'ingestion idempotent (`src/ingestion.py`).
* [x] **`03_OPS_GUARDRUILS_DATA_QUALITY_AND_CLEANING.md` — Garde-fous Ops :** Validation automatique des schémas, assertions et nettoyage métier (`src/validation.py`).
* [x] **`04_ML_PIPELINE_PREPROCESSING_FEATURE_ENGINEERING.md` — Pipeline ML :** Ingestion des lags/moyennes glissantes, découpage temporel et XGBoost (`src/train.py`).
* [x] **`05_EXPLAINABILITY_AND_MONITORING.md` — Explicabilité & Monitoring :** Valeurs SHAP et suivi du Data Drift avec Evidently (`src/explain.py`).
* [x] **`06_REST_API_SERVING.md` — Serving REST API :** FastAPI, validation Pydantic, gestion du cycle de vie et routes d'inférence (`src/api.py`).
* [ ] **`07_DECISION_DASHBOARD_AND_USER_INTERFACE.md` — Interface Décisionnelle :** Tableau de bord Streamlit connecté à l'API (`src/streamlit.py`).

---

### V2 — Intelligence Générative & Optimisation Avancée (Perspectives)

* [ ] **Copilote RAG / LLM :** Agent d'IA générative pour recommander des médicaments substitutifs en cas de rupture de stock.
* [ ] **Redistribution Automatisée :** Algorithme d'optimisation sous contraintes pour rééquilibrer les stocks inter-officines.
* [ ] **Déploiement Cloud & CI/CD :** Pipeline GitHub Actions et hébergement conteneurisé.

</details>

---

### [EN] English Version

## Project Philosophy

This repository addresses a dual engineering challenge:

* **A Production-Ready Core MLOps Architecture:** Practical implementation of ML lifecycle fundamentals (idempotent ingestion, strict data validation, API serving, monitoring) with an extension roadmap toward Generative AI (RAG / LLM in V2).
* **A "From Lab to Prod" Educational Guide:** A step-by-step approach demonstrating how to wrap a classic exploratory analysis inside a modular, testable, and containerized MLOps architecture.

---

## Architecture Overview

```text
+-----------------------+     +------------------------+     +------------------------+
|  Synthetic Data Gen   | --> | PostgreSQL (Docker)    | --> | Data Quality & Ops     |
|  (Raw CSV Generation) |     | (Idempotent Ingestion) |     | (Automated Assertions) |
+-----------------------+     +------------------------+     +------------------------+
                                                                          |
                                                                          v
+-----------------------+     +------------------------+     +------------------------+
|  FastAPI / Streamlit  | <-- | SHAP & Monitoring      | <-- | Feature Engineering    |
|  (Serving & Dashboard)|     | (Drift & Explainability) |     | (Lags, XGBoost Train)  |
+-----------------------+     +------------------------+     +------------------------+
            |
            | (V2 Extension)
            v
+-------------------------------------------------------------------------------------+
| Generative AI Module: RAG / LLM Agent                                               |
| (Automated Alternative Drug Recommendations during Active Stockouts)                |
+-------------------------------------------------------------------------------------+
```

---

## Development Environment

The project is developed under **WSL2 (Ubuntu)** using a **Bash / Zsh** shell environment paired with **Miniconda**.

**Why this dev architecture choice?**
* **Dev/Prod Parity (Native Linux):** Executing code within a Linux subsystem (WSL2) guarantees strict parity with Docker containers and production deployment servers.
* **Strict Isolation with Conda:** Using `miniconda` to precisely manage Python and C/C++ dependencies without polluting the host environment.
* **Shell Productivity:** Seamless automation of CLI scripts, Docker commands, and CI/CD workflows.

---

## Quickstart

> ⚠️ **Prerequisites:** Ensure Docker and Conda (or Python 3.11 with `venv`) are installed in your environment.

1. **Clone the repository and enter the directory**

~~~~bash
git clone https://github.com/YCS-TLSE/PharmaSupply-ML.git
cd PharmaSupply-ML
~~~~

2. **Create and activate the virtual environment**

* **Option A: Using Conda (Recommended)**
~~~~bash
conda env create -f environment.yml
conda activate pharmasupply-ml
~~~~

* **Option B: Using Pip / Standard venv**
~~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~~

3. **Start the infrastructure (PostgreSQL via Docker)**

~~~~bash
docker compose up -d
~~~~

4. **Run the full End-to-End Orchestrated Pipeline**

~~~~bash
python run_pipeline.py
~~~~
*(This script automatically orchestrates: synthetic data generation, PostgreSQL ingestion, data quality validation, XGBoost training, and SHAP/Evidently monitoring).*

5. **(Optional) Launch the FastAPI REST API service independently**

~~~~bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
~~~~

6. **Launch the Streamlit Dashboard (Phase 07 - upcoming)**

~~~~bash
streamlit run src/streamlit.py
~~~~

---

## Project Roadmap & Documentation (Phases 00 to 07)

### V1 — Demand Forecasting MLOps Engine (Core Infrastructure)

* [x] **`00_MLOPS_BLUEPRINT.md` — Blueprint & Architecture:** Global vision, MLOps lifecycle, and lab-to-prod encapsulation strategy.
* [x] **`01_INITIALIZATION_AND_SCOPING.md` — Initialization:** Business scoping, technology stack selection, and repo setup.
* [x] **`02_DATA_INFRASTRUCTURE_AND_INGESTION.md` — Ingestion & Persistence:** Containerized PostgreSQL database and idempotent ingestion pipeline (`src/ingestion.py`).
* [x] **`03_OPS_GUARDRUILS_DATA_QUALITY_AND_CLEANING.md` — Ops Guardrails:** Automated schema validation, data quality assertions, and business cleaning (`src/validation.py`).
* [x] **`04_ML_PIPELINE_PREPROCESSING_FEATURE_ENGINEERING.md` — ML Pipeline:** Lags/rolling features, temporal train/test split, and XGBoost training (`src/train.py`).
* [x] **`05_EXPLAINABILITY_AND_MONITORING.md` — Explainability & Monitoring:** SHAP values integration and Data Drift tracking via Evidently (`src/explain.py`).
* [x] **`06_REST_API_SERVING.md` — REST API Serving:** FastAPI, Pydantic validation schemas, lifespan management, and inference endpoints (`src/api.py`).
* [ ] **`07_DECISION_DASHBOARD_AND_USER_INTERFACE.md` — Decision Dashboard:** Streamlit user interface connected to the REST API (`src/streamlit.py`).

---

### V2 — Generative AI & Advanced Supply Chain Optimization (Outlook)

* [ ] **RAG / LLM Copilot:** Generative AI agent for automated alternative drug recommendations during active stockouts.
* [ ] **Automated Inventory Redistribution:** Constrained optimization algorithm for balancing inventory across pharmacies.
* [ ] **Cloud Deployment & CI/CD:** GitHub Actions pipeline and containerized infrastructure hosting.