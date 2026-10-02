# Technical Documentation  
## Blueprint: From ML Notebook to MLOps   
**Project:** PharmaSupply-ML (Standard MLOps Framework)  
**File:** `docs/00_MLOPS_BLUEPRINT.md`  
**Status:** Reference Architecture  

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>  


#### *Ce document sert de trame de référence pour structurer, industrialiser et déployer un projet de Machine Learning selon les standards MLOps. Il est applicable aussi bien pour refactoriser un notebook exploratoire existant que pour démarrer un projet à partir d'une page blanche*

---

## Matrice d'adaptation du projet

| Mode de départ | Objectif principal de la démarche |
| :--- | :--- |
| **Option A : Refactoring (depuis un Notebook)** | Découper le code monolithique, extraire la logique métier dans des modules `.py`, et sécuriser l'environnement. |
| **Option B : Greenfield (depuis zéro)** | Initialiser directement l'architecture logicielle, le conteneur de données et le pipeline de bout en bout. |

---

# Cadre Méthodologique MLOps

## 1. Vue d'Ensemble Macro : Piliers & Maturité MLOps

### A. Les 3 Piliers Fonctionnels du MLOps
Sur le plan architectural, tout système MLOps repose sur l'isolement de 3 piliers :

```text
+--------------------+      +--------------------+      +--------------------+
|  DATA ENGINEERING  | ---> | MODEL ENGINEERING  | ---> |    OPERATIONS &    |
|  (Ingestion, Data  |      | (Training, SHAP,   |      |     GOVERNANCE     |
|  Quality, Features)|      |   Serialization)   |      |  (API & Monitoring)|
+--------------------+      +--------------------+      +--------------------+
```

### B. Niveaux de Maturité (MLOps Maturity Levels)
* **Level 0 (Manuel) :** Expérimentation pure dans des notebooks.
* **Level 1 (Pipeline Automatisé) :** [Cible de notre projet] Code modularisé (`src/`), artefacts traçables, validation automatique des données et monitoring du drift.
* **Level 2 (Continuous Training - CI/CD) :** Réentraînement automatique du modèle déclenché par alerte de drift.

---

## 2. Déclinaison Opérationnelle : Les 7 Phases du Projet

*(Rappel : La Phase 05 fait la jonction entre le Model Engineering [Explicabilité] et les Opérations [Monitoring] pour valider la transparence du modèle avant son exposition en API REST).*



### 1. Initialization & Scoping
* **Objectif :** Poser les bases de l'infrastructure logicielle, du contrôle de version et de l'environnement de développement sur un socle **Linux natif / WSL2**.
* **Livrables :**
  * Structure standard de dépôt Git (`src/`, `data/`, `models/`, `reports/`, `notebooks/`, `docs/`).
  * Environnement d'exécution conteneurisé (Docker Engine natif sous Linux / WSL2 géré via `systemd`).
  * Fichier de configuration d'environnement déclaratif (`requirements.txt`).
  * Fichier `.gitignore` adapté aux données brutes, identifiants et artefacts de modèles.

---

### 2. Data Infrastructure & Ingestion
* **Objectif :** Garantir un accès fiable, reproductible et idempotent aux données sources via des services conteneurisés standard.
* **Livrables :**
  * **Option Standard (Entreprise) :** Service de base de données PostgreSQL conteneurisé et orchestré par **`docker compose`** standard (sans wrapper interactif).
  * **Option Légère (POC / Fichiers) :** Module d'accès aux fichiers plats (`.csv`, `.parquet`) sur un stockage local.
  * Module d'ingestion automatisé (`src/ingestion.py`).

---

### 3. Ops Guardrails, Data Quality & Cleaning
* **Objectif :** Valider l'intégrité technique des données brutes et effectuer un nettoyage de surface avant tout traitement analytique.
* **Livrables :**
  * Pipeline de validation et nettoyage automatisé (`src/validation.py`).
  * Contrôle des schémas et types de données.
  * Gestion des doublons, suppression/imputation d'urgence des valeurs manquantes et filtrage des valeurs aberrantes ou incohérentes métier.

---

### 4. ML Pipeline, Preprocessing & Feature Engineering
* **Objectif :** Transformer les données propres en représentations matricielles exploitables par l'algorithme, puis entraîner et sérialiser le modèle de manière reproductible.
* **Livrables :**
  * Module de pré-traitement spécifique au modèle (`src/preprocessing.py`) : normalisation/scaling, encodage des variables catégorielles, imputation avancée.
  * Module de création de variables (*Feature Engineering*) : calcul de retards (*lags*), statistiques glissantes, ratios métier.
  * Script d'entraînement et d'évaluation (`src/train.py`).
  * Sauvegarde sérialisée des artefacts du modèle (`.joblib`, `.json`, `.onnx`).

---

### 5. Explainability & Monitoring
* **Objectif :** Rendre les décisions du modèle intelligibles pour les équipes métiers et surveiller l'évolution des données dans le temps.
* **Livrables :**
  * Module d'explicabilité (`src/explainability.py`) intégrant les valeurs **SHAP** (explications globales et locales).
  * Script de détection de dérive des données (*Data Drift*) comparant le jeu de données de référence avec les nouvelles données de production (via Evidently AI).

---

### 6. REST API Serving
* **Objectif :** Exposer le modèle sous forme de service web haute performance pour permettre les prédictions en temps réel ou en batch.
* **Livrables :**
  * Application API (`src/api.py`) développée avec **FastAPI**.
  * Contrats de validation stricte des données d'entrée/sortie via **Pydantic**.
  * Encapsulation et exécution du service dans des conteneurs **Docker natifs Linux**.

---

### 7. Decision Dashboard & User Interface
* **Objectif :** Offrir une interface métier aux utilisateurs finaux pour restituer les prédictions, piloter la prise de décision et visualiser l'état du système.
* **Livrables :**
  * Application d'interface utilisateur (`src/dashboard.py`) développée avec **Streamlit**.
  * Tableaux de bord décisionnels intégrant la restitution des prédictions, les graphiques d'explicabilité SHAP et les alertes de dérive des données.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>


---

#### *This document serves as the master reference blueprint to structure, industrialize, and deploy a Machine Learning project according to MLOps standards. It applies equally to refactoring an existing exploratory notebook or starting a new project from scratch*

---

## Project Adaptation Matrix

| Starting Point | Core Objective |
| :--- | :--- |
| **Option A: Refactoring (From a Notebook)** | Break down monolithic code, extract business logic into `.py` modules, and secure the environment. |
| **Option B: Greenfield (From Scratch)** | Initialize software architecture, data containers, and the end-to-end pipeline directly. |

---


# MLOps Methodological Framework

## 1. Macro Overview: MLOps Pillars & Maturity Levels

### A. The 3 Functional Pillars of MLOps
Architecturally, every MLOps system relies on the isolation of three core pillars:

```text
+--------------------+      +--------------------+      +--------------------+
|  DATA ENGINEERING  | ---> | MODEL ENGINEERING  | ---> |    OPERATIONS &    |
|  (Ingestion, Data  |      | (Training, SHAP,   |      |     GOVERNANCE     |
|  Quality, Features)|      |   Serialization)   |      |  (API & Monitoring)|
+--------------------+      +--------------------+      +--------------------+
```

### B. MLOps Maturity Levels
* **Level 0 (Manual):** Pure experimentation within notebooks.
* **Level 1 (Automated Pipeline):** [Project Target] Modular code (`src/`), traceable artifacts, automated data validation, and drift monitoring.
* **Level 2 (Continuous Training - CI/CD):** Automated model retraining triggered by drift alerts.

---

## 2. Operational Breakdown: The 7 Project Phases

*(Note: Phase 05 bridges Model Engineering [Explainability] and Operations [Monitoring] to validate model transparency prior to REST API deployment).*




### 1. Initialization & Scoping
* **Objective:** Establish core software infrastructure, version control, and development environment on a **native Linux / WSL2** host platform.
* **Deliverables:**
  * Standard Git repository structure (`src/`, `data/`, `models/`, `reports/`, `notebooks/`, `docs/`).
  * Containerized execution environment (Native Linux Docker Engine / WSL2 managed via `systemd`).
  * Declarative environment configuration file (`requirements.txt`).
  * Custom `.gitignore` file tracking raw data, credentials, and model artifacts safely.

---

### 2. Data Infrastructure & Ingestion
* **Objective:** Ensure reliable, reproducible, and idempotent access to source data via standard containerized services.
* **Deliverables:**
  * **Standard Option (Enterprise):** Containerized PostgreSQL database service orchestrated via standard **`docker compose`** (without interactive wrappers).
  * **Light Option (POC / Files):** Data access module for flat files (`.csv`, `.parquet`) on local storage.
  * Automated ingestion module (`src/ingestion.py`).

---

### 3. Ops Guardrails, Data Quality & Cleaning
* **Objective:** Validate raw data integrity and perform surface cleaning before downstream processing.
* **Deliverables:**
  * Automated validation and cleaning pipeline (`src/validation.py`).
  * Schema enforcement and data type validation.
  * Duplicate management, missing value strategy, and domain-specific outlier filtering.

---

### 4. ML Pipeline, Preprocessing & Feature Engineering
* **Objective:** Transform clean data into matrix representations, train the model, and serialize artifacts reproducibly.
* **Deliverables:**
  * Model-specific preprocessing module (`src/preprocessing.py`): scaling, categorical encoding, advanced imputation.
  * Feature engineering module: lag calculations, rolling statistics, business ratios.
  * Training and evaluation script (`src/train.py`).
  * Serialized model artifacts (`.joblib`, `.json`, `.onnx`).

---

### 5. Explainability & Monitoring
* **Objective:** Provide interpretable insights for business stakeholders and monitor data distribution over time.
* **Deliverables:**
  * Explainability module (`src/explainability.py`) integrating **SHAP** values (global and local interpretations).
  * Data drift detection script comparing baseline datasets against incoming production data (via Evidently AI).

---

### 6. REST API Serving
* **Objective:** Expose the trained model via a high-performance web service for real-time or batch inference.
* **Deliverables:**
  * API application (`src/api.py`) powered by **FastAPI**.
  * Strict input/output payload data validation contracts using **Pydantic**.
  * Application containerization running on **native Linux Docker containers**.

---

### 7. Decision Dashboard & User Interface
* **Objective:** Deliver a business interface for end-users to serve predictions, drive decision-making, and monitor system health.
* **Deliverables:**
  * User interface application (`src/dashboard.py`) built with **Streamlit**.
  * Executive dashboards featuring prediction displays, SHAP explainability plots, and data drift alerts.

</details>