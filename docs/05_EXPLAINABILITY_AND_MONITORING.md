# Technical Documentation  
## Phase 5: Explainability & Monitoring  
**Project:** PharmaSupply-ML (Standard MLOps Framework)  
**File:** `docs/05_EXPLAINABILITY_AND_MONITORING.md`  
**Core Scripts & Files:** `src/explainability.py`, `src/monitoring.py`, `reports/figures/`, `reports/drift_report.html`, `model_status.txt`  
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)  

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>  

## 1. Objectifs de la Phase
La Phase 05 vise à garantir l'auditabilité (traçabilité), la transparence et la pérennité de tout modèle de Machine Learning destiné à la production.

Elle répond à deux enjeux majeurs :
1. **Explicabilité (Explainability) :** Comprendre le processus de décision du modèle au niveau global (variables les plus influentes) et local (explication d'une prédiction individuelle).
2. **Surveillance (Monitoring) :** Détecter la dérive des données (*Data Drift*) et la dérive de performance entre l'entraînement et la production pour anticiper la dégradation du modèle.

---

## 2. Justification des Choix Techniques

### A. SHAP (*SHapley Additive exPlanations*) pour l'Explicabilité

#### Pourquoi SHAP ?
* **Fondement mathématique solide :** Basé sur la théorie des jeux (valeurs de Shapley), SHAP garantit l'équité et la consistance de la répartition des contributions de chaque variable.
* **Explicabilité globale et locale :** Offre à la fois une vue d'ensemble sur le comportement du modèle et une décomposition exacte pour une observation donnée.
* **Optimisation pour les modèles à base d'arbres :** L'algorithme `TreeExplainer` permet un calcul exact et performant des valeurs SHAP pour les modèles de type Gradient Boosting.

#### Solutions comparées et motifs de non-rétention :
* **Feature Importance native des algorithmes :** Écartée car elle est souvent biaisée en faveur des variables ayant un grand nombre de valeurs distinctes et n'indique pas le sens de l'impact.
* **LIME :** Écarté en raison de son approche par approximation locale stochastique, incompatible avec les exigences de reproductibilité strictes.

---

### B. Evidently AI pour le Monitoring (Data & Target Drift)

#### Pourquoi Evidently AI ?
* **Spécialisation Drift :** Intègre des tests statistiques automatisés et adaptés à la nature des variables (tests Kolmogorov-Smirnov, Wasserstein, Chi-Square, PSI).
* **Autonomie et Légèreté :** Génère des rapports HTML interactifs pour l'analyse visuelle et des exports pour l'intégration automatisée dans les pipelines.
* **Interopérabilité MLOps :** S'intègre facilement dans les scripts Python et s'interface avec les outils de métriques.

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
   * **Apport :** Encapsule le moteur SHAP, intègre un correctif de robustesse pour XGBoost >= 3.0 (`base_score`), génère des visuels au format image (`.png`), et exporte les poids de dérive normalisés vers un artefact JSON (`feature_weights.json`) consommé par le monitoring.

3. **Le Script de Surveillance (`src/monitoring.py`) :**
   * **Rôle :** Pipeline de surveillance continue en production.
   * **Apport :** Compare les données entrantes aux distributions de référence via Evidently AI, calcule un score de dérive pondéré, pilote la persistance des statuts dans `model_status.txt` (trois états : `STABLE`, `WARNING`, `CRITICAL`), et déclenche des alertes e-mail SMTP en cas de risque critique.

---

## 3. Méthodologie d'Analyse SHAP et Implémentation Production

### A. Rôle des 3 graphiques SHAP
* **Summary Bar Plot (Global Feature Importance) :** Quantifie et classe l'impact moyen absolu ($\vert{}SHAP\ value\vert$) de chaque variable sur l'ensemble du jeu de données pour identifier les moteurs principaux du modèle.
* **Beeswarm Plot (Distribution et sens d'influence) :** Combine le classement des variables avec l'échelle de leurs valeurs (fortes vs faibles) pour révéler le sens de leur influence (hausse ou baisse de la prédiction).
* **Individual Breakdown Bar Plot (Explicabilité locale) :** Explique une prédiction individuelle en détaillant le passage de la valeur de référence moyenne $E[f(X)]$ à la prédiction finale $f(x)$ grâce aux contributions positives (+) et négatives (-) de chaque paramètre.

---

### B. Anatomie et Explication Détaillée du Script `src/explainability.py`

Pour accompagner l'acculturation de l'équipe Data Science aux bonnes pratiques MLOps, le module `src/explainability.py` a été conçu selon les principes de la programmation orientée objet (POO).

Voici le rôle détaillé de chaque composant du script :

1. **Patch de robustesse XGBoost / SHAP :** Intercepte et nettoie dynamiquement le paramètre `base_score` pour garantir une compatibilité absolue avec les versions récentes de XGBoost.
2. **Constructeur `__init__` :** Initialise les chemins, s'assure de l'existence du dossier de rapports `reports/figures/`, et charge le modèle sérialisé.
3. **Méthode `fit_explainer` :** Instancie le `TreeExplainer` de SHAP. Intègre un sous-échantillonnage dynamique si le jeu dépasse 300 lignes, avec un mode production rapide (`n_repeats=1`) ou un mode audit robuste par agrégation de tirages multiples (`n_repeats > 1`).
4. **Générateurs de graphiques (`generate_global_plots` & `generate_local_breakdown`) :** Produisent et exportent en haute définition (`dpi=300`) le Beeswarm plot, le Bar plot et l'explication séquentielle locale.
5. **Pont MLOps (`export_feature_weights`) :** Calcule les valeurs SHAP absolues moyennes, les convertit en poids relatifs normalisés (somme à 1.0) et les exporte vers `feature_weights.json` pour alimenter le calcul de dérive pondéré.

---

### C. Anatomie et Explication Détaillée du Script `src/monitoring.py`

Le module `src/monitoring.py` assure le contrôle continu de la santé statistique des données et des prédictions à l'aide d'Evidently AI.

Voici l'analyse détaillée de ses composants :

1. **Périmètre d'analyse (`FEATURE_COLS`) :** Isole strictement les 10 variables explicatives pour éviter que les métadonnées techniques ne faussent les calculs statistiques.
2. **Architecture Double Mode :** Capable de récupérer dynamiquement les données depuis PostgreSQL (mode autonome/CLI) ou de recevoir directement un DataFrame en mémoire (mode pipeline orchestré).
3. **Stratégie de Split Temporel :** Sépare les données chronologiquement (80% les plus anciens en Référence, 20% les plus récents en Actuel) pour simuler des conditions de production réalistes.
4. **Moteur Evidently AI :** Exécute le preset `DataDriftPreset()` pour appliquer automatiquement les tests statistiques adaptés à chaque type de variable (Wasserstein, Chi-Square, etc.) et sauvegarde un rapport HTML interactif (`reports/drift_report.html`).
5. **Score Pondéré, États et Alerting :** Multiplie les constats de dérive par les coefficients de criticité métier (issus de SHAP). Mappe le score final vers trois états (`STABLE`, `WARNING`, `CRITICAL`), écrit l'état dans `model_status.txt` pour l'API/Streamlit, et déclenche une alerte e-mail SMTP en cas de seuil critique dépassé.

---

## 4. Déroulement & Livrables de la Phase 05

1. **`notebooks/05_explainability.ipynb`** : Notebook d'expérimentation et de calibration.
2. **`src/explainability.py`** : Module automatisé d'extraction SHAP, de graphiques et d'export des poids de dérive.
3. **`src/monitoring.py`** : Script de surveillance Evidently AI avec gestion des alertes et rapports HTML/JSON.
4. **`reports/figures/`** et **`reports/drift_report.html`** : Artefacts visuels et rapports d'auditabilité.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>  

## 1. Phase Objectives
Phase 05 aims to ensure auditability (traceability), transparency, and long-term reliability for any Machine Learning model deployed in production.

It addresses two core challenges:
1. **Explainability:** Understanding the model's decision-making process at both global (most influential features) and local (individual prediction explanation) levels.
2. **Monitoring:** Detecting data drift and performance drift between training and production environments to anticipate model degradation.

---

## 2. Technical Choices Justification

### A. SHAP (*SHapley Additive exPlanations*) for Explainability

#### Why SHAP?
* **Solid Mathematical Foundation:** Grounded in game theory (Shapley values), SHAP guarantees fairness and consistency in allocating contributions to each feature.
* **Global and Local Explainability:** Provides both an overview of model behavior and an exact decomposition for a specific observation.
* **Optimization for Tree-Based Models:** The `TreeExplainer` algorithm allows exact and efficient calculation of SHAP values for Gradient Boosting models.

#### Compared Solutions and Reasons for Non-Selection:
* **Native Feature Importance:** Discarded because it is often biased toward high-cardinality features and does not indicate directionality.
* **LIME:** Discarded due to its local stochastic approximation approach, which conflicts with strict reproducibility requirements.

---

### B. Evidently AI for Monitoring (Data & Target Drift)

#### Why Evidently AI?
* **Drift Specialization:** Integrates automated statistical tests tailored to variable types (Kolmogorov-Smirnov, Wasserstein, Chi-Square, PSI).
* **Autonomy & Lightweight Design:** Generates interactive HTML reports for visual analysis and exports for automated pipeline integrations.
* **MLOps Interoperability:** Easily integrates into Python scripts and interfaces seamlessly with metrics infrastructure.

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
   * **Value:** Encapsulates the SHAP engine, integrates a robust compatibility fix for XGBoost >= 3.0 (`base_score`), exports cleaned static images (`.png`), and normalizes drift feature weights into a JSON artifact (`feature_weights.json`) consumed by monitoring.

3. **Monitoring Script (`src/monitoring.py`):**
   * **Role:** Continuous production monitoring pipeline.
   * **Value:** Compares incoming data against reference distributions via Evidently AI, computes a weighted drift score, manages state persistence in `model_status.txt` (three states: `STABLE`, `WARNING`, `CRITICAL`), and triggers SMTP email alerts upon critical risks.

---

## 3. SHAP Analysis Methodology & Production Implementation

### A. Role of the 3 SHAP Plots
* **Summary Bar Plot (Global Feature Importance):** Quantifies and ranks average absolute impact ($\vert{}SHAP\ value\vert$) of each feature across the dataset to identify primary model drivers.
* **Beeswarm Plot (Impact Distribution & Direction):** Combines feature ranking with feature value scales (high vs. low) to reveal directionality (positive or negative influence on predictions).
* **Individual Breakdown Bar Plot (Local Explainability):** Explains a single prediction by breaking down the shift from the average reference baseline $E[f(X)]$ to the final output $f(x)$ through positive (+) and negative (-) contributions.

---

### B. Anatomy and Detailed Explanation of `src/explainability.py`

To support the Data Science team's upskilling in MLOps best practices, the `src/explainability.py` module was designed following object-oriented programming (OOP) principles.

Here is the detailed role of each script component:

1. **XGBoost / SHAP Robustness Patch:** Dynamically intercepts and cleans the `base_score` parameter to guarantee absolute compatibility with recent versions of XGBoost.
2. **Constructor `__init__`:** Initializes paths, ensures the report destination directory (`reports/figures/`) exists, and loads the serialized model.
3. **`fit_explainer` Method:** Instantiates the SHAP `TreeExplainer`. Integrates dynamic subsampling if the dataset exceeds 300 rows, offering a fast production mode (`n_repeats=1`) or a robust audit mode through multi-pass aggregation (`n_repeats > 1`).
4. **Plot Generators (`generate_global_plots` & `generate_local_breakdown`):** Produce and export high-definition (`dpi=300`) Beeswarm plots, Bar plots, and sequential local breakdowns.
5. **MLOps Bridge (`export_feature_weights`):** Computes mean absolute SHAP values, converts them into normalized relative weights (summing to 1.0), and exports them to `feature_weights.json` to feed weighted drift calculations.

---

### C. Anatomy and Detailed Explanation of `src/monitoring.py`

The `src/monitoring.py` module ensures continuous control over the statistical health of input data and model outputs using Evidently AI.

Here is the detailed component breakdown:

1. **Analysis Scope (`FEATURE_COLS`):** Strictly isolates the 10 feature variables to prevent technical metadata from distorting statistical calculations.
2. **Dual-Mode Architecture:** Capable of dynamically retrieving data from PostgreSQL (standalone/CLI mode) or directly receiving an in-memory DataFrame (orchestrated pipeline mode).
3. **Temporal Split Strategy:** Separates data chronologically (oldest 80% as Reference, newest 20% as Current) to simulate realistic production conditions.
4. **Evidently AI Engine:** Executes the `DataDriftPreset()` to automatically apply tailored statistical tests to each variable type (Wasserstein, Chi-Square, etc.) and saves an interactive HTML report (`reports/drift_report.html`).
5. **Weighted Score, States & Alerting:** Multiplies drift findings by business criticality coefficients (derived from SHAP). Maps the final score to three states (`STABLE`, `WARNING`, `CRITICAL`), writes the status to `model_status.txt` for the API/Streamlit, and triggers an SMTP alert email if critical thresholds are breached.

---

## 4. Phase 05 Roadmap & Deliverables

1. **`notebooks/05_explainability.ipynb`**: Experimental and calibration notebook.
2. **`src/explainability.py`**: Automated module for SHAP extraction, plotting, and drift weight exports.
3. **`src/monitoring.py`**: Evidently AI monitoring script featuring alert handling and HTML/JSON reporting.
4. **`reports/figures/`** and **`reports/drift_report.html`**: Visual artifacts and auditability reports.

</details>