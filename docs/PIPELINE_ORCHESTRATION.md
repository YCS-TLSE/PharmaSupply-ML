# Technical Documentation
## Master Pipeline Orchestration
**Project:** PharmaSupply-ML  
**File:** `docs/PIPELINE_ORCHESTRATION.md`  
**Status:** Pending approval

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

## 1. Vue d'ensemble & Rôle du Script
Le fichier `run_pipeline.py` est l'**orchestrateur principal** du projet PharmaSupply-ML. Son rôle est d'exécuter de manière séquentielle, automatisée et reproductible l'intégralité de la chaîne MLOps : de la vérification de l'infrastructure Docker jusqu'à la validation du service d'inférence en ligne.

---

## 2. Découpage Séquentiel du Pipeline

```text
[Step 0] Infra Check (Docker Compose / Ports 5432 & 8000)
   └── [Step 1] Ingestion Données Brutes -> PostgreSQL
          └── [Step 2] Extraction BDD & Feature Engineering
                 └── [Step 3] Entraînement Modèle XGBoost
                        └── [Step 4] Explicabilité (SHAP)
                               └── [Step 5] Monitoring Dérive (Evidently AI)
                                      └── [Step 7] Serving REST API Validation (/predict)
```

### **Step 0 : Vérification d'Infrastructure**
* **Fonction :** `ensure_docker_running()`
* **Rôle :** Teste la connectivité réseau via sockets sur les ports **5432** (PostgreSQL) et **8000** (FastAPI).
* **Résilience :** Si l'un des ports ne répond pas, le script déclenche automatiquement `docker compose up -d` pour lancer les services.

### **Step 1 : Ingestion des Données Brutes**
* **Fonction :** `ingest_csv_to_postgres()`
* **Rôle :** Vérifie la présence du fichier `data/pharmaceutical_demand_row.csv`. S'il est absent, il régénère le jeu de données synthétique via `generate_data.py`. Chargement ensuite des 10 950 lignes dans la table `raw_pharmaceutical_demand`.

### **Step 2 : Feature Engineering**
* **Fonctions :** `load_data_from_db()` & `create_features()`
* **Rôle :** Extraction des données brutes depuis PostgreSQL et génération des variables dynamiques (variables calendaires, lags à 7, 14 et 30 jours, moyennes mobiles à 7 et 30 jours).

### **Step 3 : Entraînement du Modèle XGBoost**
* **Fonction :** `train_xgboost()`
* **Rôle :** Séparation temporelle stricte entre jeu d'entraînement et jeu de test. Évaluation des métriques (MAE, RMSE, WMAPE) et sauvegarde de l'artefact sous `models/xgboost_pharma_demand.joblib`.

### **Step 4 : Explicabilité des Prédictions (SHAP)**
* **Classe :** `ModelExplainer`
* **Rôle :** Calcul des valeurs SHAP pour générer les graphiques d'importance globale des variables et l'analyse locale d'une prédiction individuelle sous `reports/figures/`.

### **Step 5 : Monitoring de Dérive des Données (Evidently AI)**
* **Fonction :** `run_drift_monitoring()`
* **Rôle :** Comparaison statistique du jeu de référence (Train) et du jeu courant (Test). Génération du rapport interactif HTML dans `reports/drift_report.html`.

### **Step 7 : Validation de l'Inférence API REST**
* **Fonction :** `verify_api_serving()`
* **Rôle :** Envoie une requête HTTP POST sur `http://localhost:8000/predict` via `urllib.request`. Elle valide en direct que le service FastAPI en conteneur est opérationnel et capable de charger le modèle pour renvoyer une prédiction.

---

## 3. Guide d'Exécution

Pour lancer l'intégralité du pipeline MLOps, exécutez simplement :

```bash
python run_pipeline.py
```

### Artefacts Générés en Fin d'Exécution :
* **Modèle sérialisé :** `models/xgboost_pharma_demand.joblib`
* **Rapports SHAP :** `reports/figures/`
* **Rapport de dérive :** `reports/drift_report.html`
* **Endpoint API validé :** `http://localhost:8000/predict`

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

## 1. Overview & Script Role
The `run_pipeline.py` script serves as the **master orchestrator** for the PharmaSupply-ML project. Its role is to execute the entire MLOps lifecycle in a sequential, automated, and reproducible manner: from Docker infrastructure checks to live REST API inference validation.

---

## 2. Pipeline Execution Steps

```text
[Step 0] Infra Check (Docker Compose / Ports 5432 & 8000)
   └── [Step 1] Raw Data Ingestion -> PostgreSQL
          └── [Step 2] DB Extraction & Feature Engineering
                 └── [Step 3] XGBoost Model Training
                        └── [Step 4] Explainability (SHAP)
                               └── [Step 5] Data Drift Monitoring (Evidently AI)
                                      └── [Step 7] REST API Serving Validation (/predict)
```

### **Step 0: Infrastructure Check**
* **Function:** `ensure_docker_running()`
* **Role:** Verifies socket connections on ports **5432** (PostgreSQL) and **8000** (FastAPI).
* **Resilience:** If either port is unreachable, the script automatically executes `docker compose up -d` to spin up services.

### **Step 1: Raw Data Ingestion**
* **Function:** `ingest_csv_to_postgres()`
* **Role:** Checks for the presence of `data/pharmaceutical_demand_row.csv`. If missing, it triggers synthetic dataset generation via `generate_data.py`. Ingests 10,950 records into the `raw_pharmaceutical_demand` PostgreSQL table.

### **Step 2: Feature Engineering**
* **Functions:** `load_data_from_db()` & `create_features()`
* **Role:** Extracts raw data from PostgreSQL and constructs dynamic features (calendar features, 7/14/30-day lags, and 7/30-day rolling averages).

### **Step 3: XGBoost Model Training**
* **Function:** `train_xgboost()`
* **Role:** Performs time-based splitting between training and testing datasets. Computes evaluation metrics (MAE, RMSE, WMAPE) and exports the model artifact to `models/xgboost_pharma_demand.joblib`.

### **Step 4: Model Explainability (SHAP)**
* **Class:** `ModelExplainer`
* **Role:** Calculates SHAP values to generate global feature importance plots and individual local breakdown visualizations saved under `reports/figures/`.

### **Step 5: Data Drift Monitoring (Evidently AI)**
* **Function:** `run_drift_monitoring()`
* **Role:** Conducts statistical drift analysis between reference (Train) and current (Test) datasets. Exports an interactive HTML monitoring dashboard to `reports/drift_report.html`.

### **Step 7: REST API Serving Validation**
* **Function:** `verify_api_serving()`
* **Role:** Dispatches an HTTP POST payload to `http://localhost:8000/predict` using `urllib.request`. Validates live containerized FastAPI deployment and model loading readiness.

---

## 3. Execution Guide

To trigger the end-to-end MLOps pipeline, execute:

```bash
python run_pipeline.py
```

### Generated Artifacts:
* **Serialized Model:** `models/xgboost_pharma_demand.joblib`
* **SHAP Plots:** `reports/figures/`
* **Drift Monitoring Report:** `reports/drift_report.html`
* **Validated API Endpoint:** `http://localhost:8000/predict`

</details>