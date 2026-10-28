# Technical Documentation  
## Phase 1: Initialization & Scoping  
**Project:** PharmaSupply-ML  
**Framework:** Standard MLOps Blueprint  
**File:** `docs/01_INITIALIZATION_AND_SCOPING.md`  
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)  

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

### *Ce document définit la procédure d'initialisation du projet PharmaSupply-ML : structure du dépôt, environnement Conda, gestion des dépendances et contrôle de qualité du code.*

---

## 1. Structure du Dépôt Git

L'arborescence du projet est organisée de manière modulaire pour isoler le code source, la documentation, les tests et la configuration :

```text
PharmaSupply-ML/
├── .vscode/               # Configuration partagée VS Code
│   └── settings.json
├── docs/                  # Documentation MLOps (00 à 07)
├── notebooks/             # Notebooks d'exploration (Phase 0)
├── src/                   # Code source modulaire Python
│   ├── api.py
│   ├── dashboard.py
│   ├── explainability.py
│   ├── ingestion.py
│   ├── monitoring.py
│   ├── train.py
│   └── validation.py
├── tests/                 # Tests unitaires et d'intégration (pytest)
├── reports/               # Artefacts de sortie (figures SHAP, rapports de drift)
├── data/                  # Données brutes et synthétiques
├── sql/                   # Schéma de base de données relationnelle
├── .env                   # Variables d'environnement (secrets, BDD)
├── .gitignore             # Exclusion des fichiers lourds et secrets
├── environment.yml        # Configuration déclarative Conda (avec versions stables épinglées)
├── pyproject.toml         # Configuration du linter/formatter (Ruff)
├── docker-compose.yml     # Orchestration de l'infrastructure PostgreSQL
├── run_pipeline.py        # Orchestrateur global End-to-End
└── README.md              # Présentation du projet et Roadmap
```

---

## 2. Environnement Virtuel & Dépendances

### 2.1. Création et Activation (Conda)
L'environnement utilise **Python 3.11** sous WSL2 (Linux) avec l'environnement `pharma-supply` :

```bash
conda env create -f environment.yml
conda activate pharma-supply
```

### 2.2. Stack des Dépendances Principales
Les dépendances du projet intègrent des verrous de versions pour garantir la parfaite stabilité de la chaîne MLOps (notamment la compatibilité SHAP et XGBoost) :

* **Ingestion & Data :** `pandas<2.0`, `numpy`
* **Base de Données & SQL :** `sqlalchemy`, `psycopg2-binary`
* **ML & Preprocessing :** `scikit-learn`, `xgboost<3.0`, `joblib`
* **Explicabilité & Monitoring :** `shap`, `evidently`
* **API & Serving :** `fastapi`, `uvicorn`, `pydantic`
* **Dashboard :** `streamlit`
* **Qualité & Config :** `python-dotenv`, `ruff`, `pytest`

### 2.3. Exportation de l'Environnement
Pour garantir la reproductibilité :

```bash
conda env export --no-builds > environment.yml
```

---

## 3. Sécurité & Contrôle de Version (Git)

### 3.1. Fichier `.gitignore`
Un fichier `.gitignore` strict est configuré à la racine pour empêcher le commit d'artefacts volumineux, de données brutes ou de clés d'accès :

```gitignore
# Données et artefacts lourds
src/data/*.csv
src/data/*.parquet
models/*.joblib
reports/figures/*.png

# Environnements Python & Caches
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.env

# Configuration IDE & Système
.DS_Store
.vscode/*
!.vscode/settings.json
```

---

## 4. Qualité de Code & Outillage de Développement

### 4.1. Linter & Formatter (`pyproject.toml`)
La qualité du code Python est assurée par **Ruff** :

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]
```

### 4.2. Configuration VS Code (`.vscode/settings.json`)
L'intégration directe dans l'éditeur force le formatage automatique à la sauvegarde :

```json
{
  "python.defaultInterpreterPath": "~/.conda/envs/pharma-supply/bin/python",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit"
  },
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *This document outlines the setup procedure for PharmaSupply-ML: repository structure, Conda environment, dependency management, and code quality controls.*

---

## 1. Git Repository Structure

The project directory layout is strictly modularized to separate source code, documentation, tests, and IDE settings:

```text
PharmaSupply-ML/
├── .vscode/               # Shared VS Code settings
│   └── settings.json
├── docs/                  # MLOps documentation framework (00 to 07)
├── notebooks/             # Exploratory notebooks (Phase 0)
├── src/                   # Modular Python source code
│   ├── api.py
│   ├── dashboard.py
│   ├── explainability.py
│   ├── ingestion.py
│   ├── monitoring.py
│   ├── train.py
│   └── validation.py
├── tests/                 # Unit & integration tests (pytest)
├── reports/               # Output artifacts (SHAP figures, drift reports)
├── data/                  # Raw and synthetic datasets
├── sql/                   # Relational database schema
├── .env                   # Local environment secrets & DB credentials
├── .gitignore             # Git exclusion rules for raw data & secrets
├── environment.yml        # Declarative Conda environment setup (with pinned stable versions)
├── pyproject.toml         # Ruff linter/formatter configuration
├── docker-compose.yml     # PostgreSQL infrastructure orchestration
├── run_pipeline.py        # End-to-end global orchestrator
└── README.md              # Project overview & MLOps roadmap
```

---

## 2. Virtual Environment & Dependencies

### 2.1. Creation & Activation (Conda)
The project runs on **Python 3.11** under Linux (WSL2) using the `pharma-supply` environment:

```bash
conda env create -f environment.yml
conda activate pharma-supply
```

### 2.2. Core Stack Overview
Dependencies include strict version constraints to ensure absolute stability of the MLOps pipeline (particularly for SHAP and XGBoost compatibility):

* **Data & Ingestion:** `pandas<2.0`, `numpy`
* **Database & SQL:** `sqlalchemy`, `psycopg2-binary`
* **ML & Preprocessing:** `scikit-learn`, `xgboost<3.0`, `joblib`
* **Explainability & Drift:** `shap`, `evidently`
* **Serving & API:** `fastapi`, `uvicorn`, `pydantic`
* **Dashboard:** `streamlit`
* **Quality & Config:** `python-dotenv`, `ruff`, `pytest`

### 2.3. Reproducibility Export
Export the environment configuration for deployment consistency:

```bash
conda env export --no-builds > environment.yml
```

---

## 3. Version Control & Security (Git)

### 3.1. `.gitignore` Policy
A strict `.gitignore` file prevents committing confidential keys, local cache files, or large data files:

```gitignore
# Data files & Heavy artifacts
src/data/*.csv
src/data/*.parquet
models/*.joblib
reports/figures/*.png

# Python Environment & Caches
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.env

# System & IDE Config
.DS_Store
.vscode/*
!.vscode/settings.json
```

---

## 4. Code Quality & Developer Tooling

### 4.1. Linter & Formatter (`pyproject.toml`)
Code standards are enforced using **Ruff**:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]
```

### 4.2. Editor Workspace Setup (`.vscode/settings.json`)
VS Code is configured to apply auto-formatting and import sorting on every save:

```json
{
  "python.defaultInterpreterPath": "~/.conda/envs/pharma-supply/bin/python",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit"
  },
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

</details>