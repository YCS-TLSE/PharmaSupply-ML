# Technical Documentation  
## Phase 5: Explainability & Monitoring  
**Project:** PharmaSupply-ML (Standard MLOps Framework)  
**File:** `docs/05_EXPLAINABILITY_AND_MONITORING.md`  
**Status:** Validated  

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>  

## 1. Objectifs de la Phase
La Phase 05 vise à garantir l'auditabilité (traçabilité), la transparence et la pérennité de tout modèle de Machine Learning destiné à la production.

Elle répond à deux enjeux majeurs :
1. **Explicabilité (Explainability) :** Comprendre le processus de décision du modèle au niveau global (variables les plus influentes) et local (explication d'une prédiction individuelle).
2. **Surveillance (Monitoring) :** Détecter la dérive des données (*Data Drift*) et la dérive de performance (*Target / Prediction Drift*) entre l'entraînement et la production pour anticiper la dégradation du modèle.

---

## 2. Justification des Choix Techniques

### A. SHAP (*SHapley Additive exPlanations*) pour l'Explicabilité

#### Pourquoi SHAP ?
* **Fondement mathématique solide :** Basé sur la théorie des jeux (valeurs de Shapley), SHAP garantit l'équité et la consistance de la répartition des contributions de chaque variable.
* **Explicabilité globale et locale :** Offre à la fois une vue d'ensemble sur le comportement du modèle et une décomposition exacte pour une observation donnée.
* **Optimisation pour les modèles à base d'arbres :** L'algorithme `TreeExplainer` permet un calcul exact et performant des valeurs SHAP pour les modèles de type Gradient Boosting ou Random Forest.

#### Solutions comparées et motifs de non-rétention :
* **Feature Importance native des algorithmes :** Écartée car elle est souvent biaisée en faveur des variables ayant un grand nombre de valeurs distinctes (cardinalité) et n'indique pas le sens de l'impact (positif ou négatif).
* **LIME :** Écarté en raison de son approche par approximation locale stochastique (instabilité des explications selon les tirages), incompatible avec les exigences de reproductibilité strictes.

---

### B. Evidently AI pour le Monitoring (Data & Target Drift)

#### Pourquoi Evidently AI ?
* **Spécialisation Drift :** Intègre des tests statistiques automatisés et adaptés à la nature des variables (tests Kolmogorov-Smirnov, Wasserstein, Chi-Square, PSI).
* **Autonomie et Légèreté :** Génère des rapports HTML interactifs pour l'analyse visuelle et des exports JSON pour l'intégration automatisée dans les pipelines.
* **Interopérabilité MLOps :** S'intègre facilement dans les scripts Python et s'interface avec les outils de métriques (Prometheus, Grafana).

#### Solutions comparées et motifs de non-rétention :
* **Great Expectations :** Très performant pour la validation de schéma et de qualité de données statiques, mais non optimisé pour mesurer la dérive statistique temporelle des distributions.
* **NannyML :** Très complet pour l'estimation de performance sans vérité terrain, mais présente une complexité d'infrastructure plus élevée au stade actuel.

---

### C. Rôle et complémentarité des livrables (Notebook vs Scripts `.py`)

L'architecture repose sur un découplage clair entre la phase d'expérimentation et l'industrialisation pour garantir la reproductibilité du projet :

1. **Le Notebook d'exploration (`notebooks/05_explainability.ipynb`) :**
   * **Rôle :** Espace de recherche, de validation visuelle et de calibrage.
   * **Apport :** Permet d'analyser le comportement du modèle, de mesurer l'importance relative des variables, de déterminer la valeur de référence (*baseline*) et de déceler les contraintes d'affichage. Il sert à fixer les règles métiers appliquées ensuite dans le code de production.

2. **Le Script d'Explicabilité (`src/explainability.py`) :**
   * **Rôle :** Module Python réutilisable pour le calcul d'explicabilité automatisé.
   * **Apport :** Encapsule le moteur SHAP, génère des visuels nettoyés au format image (`.png`), filtre l'affichage sur les variables principales et fournit des explications locales (*Individual Breakdown Bar Plots*) pour les prédictions atypiques.

3. **Le Script de Surveillance (`src/monitoring.py`) :**
   * **Rôle :** Pipeline de surveillance continue en production.
   * **Apport :** Compare les données entrantes aux distributions de référence et applique une stratégie d'alerte paramétrable en fonction du poids réel des variables.

---

## 3. Méthodologie d'Analyse SHAP et Implémentation Production

### A. Rôle des 3 graphiques SHAP
* **Summary Bar Plot (Global Feature Importance) :** Quantifie et classe l'impact moyen absolu ($\vert{}SHAP\ value\vert$) de chaque variable sur l'ensemble du jeu de données pour identifier les moteurs principaux du modèle.
* **Beeswarm Plot (Distribution et sens d'influence) :** Combine le classement des variables avec l'échelle de leurs valeurs (fortes vs faibles) pour révéler le sens de leur influence (hausse ou baisse de la prédiction).
* **Individual Breakdown Bar Plot (Explicabilité locale) :** Explique une prédiction individuelle en détaillant le passage de la valeur de référence moyenne $E[f(X)]$ à la prédiction finale $f(x)$ grâce aux contributions positives (+) et négatives (-) de chaque paramètre.

---

### B. Anatomie et Explication Détaillée du Script `src/explainability.py`

Pour accompagner l'acculturation de l'équipe Data Science aux bonnes pratiques MLOps, le module `src/explainability.py` a été conçu selon les principes de la programmation orientée objet (POO), du typage strict et de l'optimisation des ressources serveur.

Voici le rôle détaillé de chaque composant du script :

#### 1. En-tête et Stratégie d'Architecture (Docstring)
Le module débute par une documentation claire exposant les deux arbitrages MLOps majeurs :
* **Échantillonnage dynamique (Subsampling) :** Sur des données volumineuses ($> 300$ lignes), le calcul SHAP complet devient lent et surcharge visuellement le *Beeswarm plot*. Un sous-échantillonnage est donc déclenché automatiquement.
* **Mode Prod (`n_repeats=1`) vs Mode Audit (`n_repeats > 1`) :** 
  * En production (API REST / Streamlit), un seul tirage déterministe (`random_state=42`) garantit un temps de réponse $< 0.3$s.
  * En audit / rapport officiel, un calcul agrégé sur $N$ tirages (*Cross-Validation / Aggregated Subsampling*) élimine le biais de sous-échantillonnage.

#### 2. Constructeur `__init__(self, model_path, output_dir)`
* **Rôle :** Initialise l'instance du module d'explicabilité.
* **Fonctionnement :**
  * Convertit les chemins passés en paramètres en objets `Path` (gestion multiplateforme propre).
  * Crée le dossier de destination des visuels (`reports/figures/`) s'il n'existe pas encore.
  * Déclenche le chargement du modèle sérialisé via la méthode privée `_load_model()`.
  * Initialise les attributs d'état (`explainer`, `shap_values`, `X_sample`) à `None`.

#### 3. Méthode privée `_load_model(self)`
* **Rôle :** Sécurise le chargement du fichier modèle `.joblib` depuis le disque.
* **Fonctionnement :**
  * Vérifie la présence physique du fichier sur le disque et lève une exception explicite (`FileNotFoundError`) si l'artefact est introuvable.
  * Désérialise et retourne l'objet modèle XGBoost via `joblib.load()`.

#### 4. Calculateur de valeurs SHAP `fit_explainer(self, X_sample, threshold=300, sample_size=200, n_repeats=1)`
* **Rôle :** Instancie un `shap.TreeExplainer` optimisé pour XGBoost et applique la logique de calcul.
* **Fonctionnement :**
  * Si la taille de `X_sample` est inférieure ou égale au seuil (`threshold`), le calcul est effectué sur l'intégralité du jeu de données.
  * Si le jeu dépasse le seuil :
    * **Si `n_repeats == 1` (Fast Prod Mode) :** Extrait un échantillon unique de `sample_size` lignes (`random_state=42`) pour un calcul instantané.
    * **Si `n_repeats > 1` (Robust Audit Mode) :** Effectue `n_repeats` tirages indépendants, calcule les matrices SHAP associées et réalise la moyenne élément par élément (`np.mean`) pour stabiliser l'explicabilité.

#### 5. Générateur d'Explications Globales `generate_global_plots(self)`
* **Rôle :** Génère et sauvegarde les visuels d'interprétabilité globale du modèle.
* **Fonctionnement :**
  * Vérifie que `fit_explainer()` a bien été exécuté (sinon lève une `ValueError`).
  * Génère le **Beeswarm Plot** (`shap_summary_beeswarm.png`) et le **Bar Plot** (`shap_summary_bar.png`).
  * Exporte les images en haute définition (`dpi=300`) dans `reports/figures/` puis libère la mémoire RAM avec `plt.close()`.

#### 6. Générateur d'Explication Locale `generate_local_breakdown(self, sample_index=0, filename=...)`
* **Rôle :** Explique la prédiction individuelle d'une observation précise.
* **Fonctionnement :**
  * Génère un **Individual Breakdown Bar Plot** montrant pas à pas comment les caractéristiques d'une commande font dévier la prédiction à partir de la valeur moyenne de référence $E[f(X)]$.

#### 7. Bloc d'exécution autonome `if __name__ == "__main__":`
* **Rôle :** Permet de tester et de valider le script en ligne de commande (`python -m src.explainability`) sans impacter les imports de modules externes.

---

### C. Anatomie et Explication Détaillée du Script `src/monitoring.py`

Le module `src/monitoring.py` assure le contrôle continu de la santé statistique des données et des prédictions. Conçu pour s'exécuter aussi bien de manière autonome (CLI / Tâches CRON) que dans un pipeline d'orchestration global MLOps, il repose sur la bibliothèque **Evidently AI (v0.6+)**.

Voici l'analyse détaillée de ses composants et de sa logique de fonctionnement :

#### 1. Configuration des Colonnes (`FEATURE_COLS` & `TARGET_COL`)
* **Rôle :** Isole précisément le périmètre d'analyse pour Evidently AI.
* **Fonctionnement :** Définit la liste exacte des 10 variables explicatives (*features*) et de la variable cible (`target_demand`). Cela garantit que les métadonnées non pertinentes (comme les identifiants bruts ou clés primaires) ne viennent pas perturber le calcul des tests statistiques.

#### 2. Architecture Double Mode (In-Memory vs Database Fallback)
* **Rôle :** Garantit la polyvalence du module entre l'exécution autonome et l'orchestration.
* **Fonctionnement :**
  ```python
  if df is None:
      logging.info(
          "Loading dataset for monitoring from PostgreSQL..."
      )
      df_raw = load_data_from_db()
      df = create_features(df_raw)
  ```
  * **Mode Pipeline Orchestré :** La fonction `run_drift_monitoring(df=...)` reçoit directement le `DataFrame` pré-transformé en mémoire depuis l'orchestrateur. Aucune requête BDD inutile n'est exécutée.
  * **Mode Standalone (CLI) :** Si `df` vaut `None`, le script bascule automatiquement sur la base de données PostgreSQL via `load_data_from_db()` et recalcule les variables dynamiques avec `create_features()`.

#### 3. Stratégie de Split Temporel (Reference vs Current)
* **Rôle :** Reconstitue une séparation temporelle réaliste entre le passé (données d'entraînement) et le présent (données récentes en production).
* **Fonctionnement :** 
  * Extrait la liste ordonnée des dates uniques (`df["date"]`).
  * Applique un découpage temporel **80% / 20%** : les 80% les plus anciens forment le jeu de **Référence** (*Baseline*), et les 20% les plus récents forment le jeu **Actuel** (*Current*).

#### 4. Calcul du Drift avec Evidently AI (`Report` & `DataDriftPreset`)
* **Rôle :** Applique les tests statistiques appropriés selon le type de variable (numérique vs catégorielle).
* **Fonctionnement :**
  ```python
  report = Report([DataDriftPreset()])
  my_eval = report.run(current_df, reference_df)
  ```
  * Instancie le preset `DataDriftPreset()` compatible avec l'API v0.6+.
  * Compare la distribution de `current_df` par rapport à `reference_df`.
  * Utilise automatiquement les tests adaptés (ex: *Wasserstein distance* pour les variables numériques comme `rolling_mean_7`, *Jensen-Shannon distance* pour les variables catégorielles comme `month`).

#### 5. Exportation et Persistance du Rapport HTML
* **Rôle :** Génère l'artefact visuel d'auditabilité pour les équipes Data et Métier.
* **Fonctionnement :**
  * S'assure de l'existence du dossier de destination (`reports/`).
  * Utilise l'API de sauvegarde de l'évaluation (`my_eval.save_html()`) pour produire un rapport HTML interactif complet sans dépendances externes (`reports/drift_report.html`).

---

## 4. Déroulement & Livrables de la Phase 05

1. **`notebooks/05_explainability.ipynb`** : Notebook d'expérimentation pour l'analyse des graphiques SHAP, la définition de la *baseline* et la validation des tests de dérive.
2. **`src/explainability.py`** : Module Python automatisé d'extraction des valeurs SHAP et de génération des graphiques (`.png` / `.csv`).
3. **`src/monitoring.py`** : Script de surveillance automatisé basé sur Evidently AI avec gestion des alertes et export des rapports HTML/JSON.
4. **`reports/figures/`** : Dossier de stockage des artefacts visuels d'explicabilité et des rapports de dérive pour l'auditabilité.

</details>

---

### [EN] English Version

## 1. Phase Objectives
Phase 05 aims to ensure auditability (traceability), transparency, and long-term reliability for any Machine Learning model deployed in production.

It addresses two core challenges:
1. **Explainability:** Understanding the model's decision-making process at both global (most influential features) and local (individual prediction explanation) levels.
2. **Monitoring:** Detecting data drift (*Data Drift*) and performance drift (*Target / Prediction Drift*) between training and production environments to anticipate model degradation.

---

## 2. Technical Choices Justification

### A. SHAP (*SHapley Additive exPlanations*) for Explainability

#### Why SHAP?
* **Solid Mathematical Foundation:** Grounded in game theory (Shapley values), SHAP guarantees fairness and consistency in allocating contributions to each feature.
* **Global and Local Explainability:** Provides both an overview of model behavior and an exact decomposition for a specific observation.
* **Optimization for Tree-Based Models:** The `TreeExplainer` algorithm allows exact and efficient calculation of SHAP values for Gradient Boosting and Random Forest architectures.

#### Compared Solutions and Reasons for Non-Selection:
* **Native Feature Importance:** Discarded because it is often biased toward high-cardinality features (variables with many distinct values) and does not indicate directionality (positive or negative impact).
* **LIME:** Discarded due to its local stochastic approximation approach (instability of explanations across runs), which conflicts with strict reproducibility requirements.

---

### B. Evidently AI for Monitoring (Data & Target Drift)

#### Why Evidently AI?
* **Drift Specialization:** Integrates automated statistical tests tailored to variable types (Kolmogorov-Smirnov, Wasserstein, Chi-Square, PSI).
* **Autonomy & Lightweight Design:** Generates interactive HTML reports for visual analysis and JSON exports for automated pipeline integrations.
* **MLOps Interoperability:** Easily integrates into Python scripts and interfaces seamlessly with metrics infrastructure (Prometheus, Grafana).

#### Compared Solutions and Reasons for Non-Selection:
* **Great Expectations:** Excellent for schema validation and static data quality checks, but not optimized for measuring statistical temporal drift in distributions.
* **NannyML:** Comprehensive for estimating performance without ground truth, but involves higher infrastructure complexity at the current stage.

---

### C. Role and Complementarity of Deliverables (Notebook vs. `.py` Scripts)

The architecture relies on a clear decoupling between experimentation and industrialization to guarantee project reproducibility:

1. **Exploration Notebook (`notebooks/05_explainability.ipynb`):**
   * **Role:** Research, visual validation, and calibration sandbox.
   * **Value:** Used to analyze model behavior, measure relative feature importance, set the baseline value, and identify rendering constraints. It establishes the business logic applied later in production code.

2. **Explainability Script (`src/explainability.py`):**
   * **Role:** Reusable Python module for automated explainability computations.
   * **Value:** Encapsulates the SHAP engine, exports cleaned static images (`.png`), filters views to top features, and generates individual breakdown plots (*Individual Breakdown Bar Plots*) for atypical predictions.

3. **Monitoring Script (`src/monitoring.py`):**
   * **Role:** Continuous production monitoring pipeline.
   * **Value:** Compares incoming data against reference distributions and enforces customizable alert strategies based on feature importance weights.

---

## 3. SHAP Analysis Methodology & Production Implementation

### A. Role of the 3 SHAP Plots
* **Summary Bar Plot (Global Feature Importance):** Quantifies and ranks average absolute impact ($\vert{}SHAP\ value\vert$) of each feature across the dataset to identify primary model drivers.
* **Beeswarm Plot (Impact Distribution & Direction):** Combines feature ranking with feature value scales (high vs. low) to reveal directionality (positive or negative influence on predictions).
* **Individual Breakdown Bar Plot (Local Explainability):** Explains a single prediction by breaking down the shift from the average reference baseline $E[f(X)]$ to the final output $f(x)$ through positive (+) and negative (-) contributions.

---

### B. Anatomy and Detailed Explanation of `src/explainability.py`

To support the Data Science team's upskilling in MLOps best practices, the `src/explainability.py` module was designed following object-oriented programming (OOP) principles, strict typing, and server resource optimization.

Here is the detailed role of each script component:

#### 1. Header and Architectural Strategy (Docstring)
The module opens with clear documentation outlining two key MLOps trade-offs:
* **Dynamic Subsampling:** On large datasets ($> 300$ rows), full SHAP computation becomes slow and visually cluttering for the *Beeswarm plot*. Subsampling is therefore automatically triggered.
* **Prod Mode (`n_repeats=1`) vs. Audit Mode (`n_repeats > 1`):** 
  * In production (REST API / Streamlit), a single deterministic run (`random_state=42`) guarantees response times $< 0.3$s.
  * In audit / official reports, an aggregated calculation across $N$ runs (*Cross-Validation / Aggregated Subsampling*) eliminates subsampling bias.

#### 2. Constructor `__init__(self, model_path, output_dir)`
* **Role:** Initializes the explainability module instance.
* **How it works:**
  * Converts input paths into `Path` objects for clean cross-platform file handling.
  * Creates the output directory (`reports/figures/`) if it does not already exist.
  * Triggers model loading via the private `_load_model()` method.
  * Initializes state attributes (`explainer`, `shap_values`, `X_sample`) to `None`.

#### 3. Private Method `_load_model(self)`
* **Role:** Securely loads the `.joblib` model artifact from disk.
* **How it works:**
  * Checks physical file existence on disk and raises an explicit `FileNotFoundError` if the artifact is missing.
  * Deserializes and returns the XGBoost model object using `joblib.load()`.

#### 4. SHAP Values Calculator `fit_explainer(self, X_sample, threshold=300, sample_size=200, n_repeats=1)`
* **Role:** Instantiates a `shap.TreeExplainer` optimized for XGBoost and applies execution logic.
* **How it works:**
  * If `X_sample` size is below or equal to `threshold`, SHAP values are computed on the full dataset.
  * If the dataset exceeds the threshold:
    * **If `n_repeats == 1` (Fast Prod Mode):** Extracts a single sample of `sample_size` rows (`random_state=42`) for near-instant computation.
    * **If `n_repeats > 1` (Robust Audit Mode):** Performs `n_repeats` independent random draws, computes corresponding SHAP matrices, and averages them element-wise (`np.mean`) to stabilize explanations.

#### 5. Global Explanations Generator `generate_global_plots(self)`
* **Role:** Generates and saves model global interpretability visualizations.
* **How it works:**
  * Verifies that `fit_explainer()` has been executed (raises a `ValueError` otherwise).
  * Generates the **Beeswarm Plot** (`shap_summary_beeswarm.png`) and **Bar Plot** (`shap_summary_bar.png`).
  * Exports high-resolution images (`dpi=300`) to `reports/figures/` and clears RAM using `plt.close()`.

#### 6. Local Explanation Generator `generate_local_breakdown(self, sample_index=0, filename=...)`
* **Role:** Explains individual predictions for a specific observation.
* **How it works:**
  * Generates an **Individual Breakdown Bar Plot** detailing step-by-step how an order's specific features push the prediction away from the mean reference baseline $E[f(X)]$.

#### 7. Autonomous Execution Block `if __name__ == "__main__":`
* **Role:** Allows direct CLI testing and validation (`python -m src.explainability`) without affecting external module imports.

---

### C. Anatomy and Detailed Explanation of `src/monitoring.py`  

The `src/monitoring.py` module ensures continuous control over the statistical health of input data and model outputs. Designed to execute both standalone (CLI / CRON jobs) and within a master MLOps pipeline, it relies on the **Evidently AI (v0.6+)** framework.

Here is the detailed breakdown of its components and operating logic:

#### 1. Column Definition (`FEATURE_COLS` & `TARGET_COL`)
* **Role:** Strictly isolates the analysis scope for Evidently AI.
* **How it works:** Explicitly lists the 10 feature variables and the target column (`target_demand`). This prevents non-predictive metadata (such as primary keys or raw timestamps) from diluting statistical drift scoring.

#### 2. Dual-Mode Architecture (In-Memory vs Database Fallback)
* **Role:** Guarantees module versatility across standalone execution and pipeline orchestration.
* **How it works:**
  ```python
  if df is None:
      logging.info(
          "Loading dataset for monitoring from PostgreSQL..."
      )
      df_raw = load_data_from_db()
      df = create_features(df_raw)
  ```
  * **Orchestrated Pipeline Mode:** The `run_drift_monitoring(df=...)` function receives a pre-transformed in-memory `DataFrame` directly from the master orchestrator, bypassing database calls.
  * **Standalone Mode (CLI):** If `df` is `None`, the script automatically falls back to querying the PostgreSQL database via `load_data_from_db()` and generating features using `create_features()`.

#### 3. Temporal Split Strategy (Reference vs Current)
* **Role:** Recreates a realistic time-based separation between baseline training data and recent production data.
* **How it works:**
  * Extracts sorted unique timestamps (`df["date"]`).
  * Applies an **80/20 time split**: the oldest 80% constitutes the **Reference** dataset, while the most recent 20% constitutes the **Current** evaluation dataset.

#### 4. Drift Calculation via Evidently AI (`Report` & `DataDriftPreset`)
* **Role:** Runs appropriate statistical tests depending on variable data types (numerical vs categorical).
* **How it works:**
  ```python
  report = Report([DataDriftPreset()])
  my_eval = report.run(current_df, reference_df)
  ```
  * Instantiates `DataDriftPreset()` compliant with Evidently v0.6+ API standards.
  * Evaluates distribution shifts between `current_df` and `reference_df`.
  * Automatically selects optimal statistical tests (e.g., *Wasserstein distance* for numerical features like `rolling_mean_7`, *Jensen-Shannon distance* for categorical features like `month`).

#### 5. Export & Report Persistence
* **Role:** Generates an auditable visual artifact for Data and Business stakeholders.
* **How it works:**
  * Guarantees target directory existence (`reports/`).
  * Calls the evaluation object's native export method (`my_eval.save_html()`) to output a self-contained, interactive HTML dashboard (`reports/drift_report.html`).

---

## 4. Phase 05 Roadmap & Deliverables

1. **`notebooks/05_explainability.ipynb`**: Experimental notebook for SHAP visual analysis, baseline configuration, and drift testing.
2. **`src/explainability.py`**: Automated Python module for SHAP extraction and figure/data export (`.png` / `.csv`).
3. **`src/monitoring.py`**: Automated monitoring script built on Evidently AI with alert handling and HTML/JSON report generation.
4. **`reports/figures/`**: Storage directory for explainability artifacts and drift report audits.
