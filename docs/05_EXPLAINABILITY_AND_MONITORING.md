# Technical Documentation  
## Phase 5: Explainability & Monitoring  
**Project:** PharmaSupply-ML (Standard MLOps Framework)  
**File:** `docs/05_EXPLAINABILITY_AND_MONITORING.md`  
**Status:** Validated  

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>  

## 1. Objectifs de la Phase
La Phase 05 vise à garantir l'auditabilité (traçabilité), la transparence et la pérennité de tout modèle de Machine Learning destiné à la production[cite: 2].

Elle répond à deux enjeux majeurs :
1. **Explicabilité (Explainability) :** Comprendre le processus de décision du modèle au niveau global (variables les plus influentes) et local (explication d'une prédiction individuelle)[cite: 2].
2. **Surveillance (Monitoring) :** Détecter la dérive des données (*Data Drift*) et la dérive de performance (*Target / Prediction Drift*) entre l'entraînement et la production pour anticiper la dégradation du modèle[cite: 2].

---

## 2. Justification des Choix Techniques

### A. SHAP (*SHapley Additive exPlanations*) pour l'Explicabilité

#### Pourquoi SHAP ?
* **Fondement mathématique solide :** Basé sur la théorie des jeux (valeurs de Shapley), SHAP garantit l'équité et la consistance de la répartition des contributions de chaque variable[cite: 2].
* **Explicabilité globale et locale :** Offre à la fois une vue d'ensemble sur le comportement du modèle et une décomposition exacte pour une observation donnée[cite: 2].
* **Optimisation pour les modèles à base d'arbres :** L'algorithme `TreeExplainer` permet un calcul exact et performant des valeurs SHAP pour les modèles de type Gradient Boosting ou Random Forest[cite: 2].

#### Solutions comparées et motifs de non-rétention :
* **Feature Importance native des algorithmes :** Écartée car elle est souvent biaisée en faveur des variables ayant un grand nombre de valeurs distinctes (cardinalité) et n'indique pas le sens de l'impact (positif ou négatif)[cite: 2].
* **LIME :** Écarté en raison de son approche par approximation locale stochastique (instabilité des explications selon les tirages), incompatible avec les exigences de reproductibilité strictes[cite: 2].

---

### B. Evidently AI pour le Monitoring (Data & Target Drift)

#### Pourquoi Evidently AI ?
* **Spécialisation Drift :** Intègre des tests statistiques automatisés et adaptés à la nature des variables (tests Kolmogorov-Smirnov, Wasserstein, Chi-Square, PSI)[cite: 2].
* **Autonomie et Légèreté :** Génère des rapports HTML interactifs pour l'analyse visuelle et des exports JSON pour l'intégration automatisée dans les pipelines[cite: 2].
* **Interopérabilité MLOps :** S'intègre facilement dans les scripts Python et s'interface avec les outils de métriques (Prometheus, Grafana)[cite: 2].

#### Solutions comparées et motifs de non-rétention :
* **Great Expectations :** Très performant pour la validation de schéma et de qualité de données statiques, mais non optimisé pour mesurer la dérive statistique temporelle des distributions[cite: 3].
* **NannyML :** Très complet pour l'estimation de performance sans vérité terrain, mais présente une complexité d'infrastructure plus élevée au stade actuel[cite: 3].

---

### C. Rôle et complémentarité des livrables (Notebook vs Scripts `.py`)

L'architecture repose sur un découplage clair entre la phase d'expérimentation et l'industrialisation pour garantir la reproductibilité du projet[cite: 2] :

1. **Le Notebook d'exploration (`notebooks/05_explainability.ipynb`) :**
   * **Rôle :** Espace de recherche, de validation visuelle et de calibrage[cite: 2].
   * **Apport :** Permet d'analyser le comportement du modèle, de mesurer l'importance relative des variables, de déterminer la valeur de référence (*baseline*) et de déceler les contraintes d'affichage[cite: 2]. Il sert à fixer les règles métiers appliquées ensuite dans le code de production[cite: 2].

2. **Le Script d'Explicabilité (`src/explainability.py`) :**
   * **Rôle :** Module Python réutilisable pour le calcul d'explicabilité automatisé[cite: 2].
   * **Apport :** Encapsule le moteur SHAP, génère des visuels nettoyés au format image (`.png`), filtre l'affichage sur les variables principales et fournit des explications locales (*Waterfall Plots*) pour les prédictions atypiques[cite: 2].

3. **Le Script de Surveillance (`src/monitoring.py`) :**
   * **Rôle :** Pipeline de surveillance continue en production[cite: 2].
   * **Apport :** Compare les données entrantes aux distributions de référence et applique une stratégie d'alerte paramétrable en fonction du poids réel des variables[cite: 2].

---

## 3. Méthodologie d'Analyse SHAP et Implémentation Production

### A. Rôle des 3 graphiques SHAP
* **Summary Bar Plot (Global Feature Importance) :** Quantifie et classe l'impact moyen absolu ($\vert{}SHAP\ value\vert{}$) de chaque variable sur l'ensemble du jeu de données pour identifier les moteurs principaux du modèle[cite: 2].
* **Beeswarm Plot (Distribution et sens d'influence) :** Combine le classement des variables avec l'échelle de leurs valeurs (fortes vs faibles) pour révéler le sens de leur influence (hausse ou baisse de la prédiction)[cite: 2].
* **Waterfall Plot (Explicabilité locale) :** Explique une prédiction individuelle en détaillant le passage de la valeur de référence moyenne $E[f(X)]$ à la prédiction finale $f(x)$ grâce aux contributions positives (+) et négatives (-) de chaque paramètre[cite: 2].

---

### B. Anatomie et Explication Détaillée du Script `src/explainability.py`

Pour accompagner l'acculturation de l'équipe Data Science aux bonnes pratiques MLOps, le module `src/explainability.py` a été conçu selon les principes de la programmation orientée objet (POO), du typage strict et de l'optimisation des ressources serveur[cite: 2].

Voici le rôle détaillé de chaque composant du script[cite: 2] :

#### 1. En-tête et Stratégie d'Architecture (Docstring)
Le module débute par une documentation claire exposant les deux arbitrages MLOps majeurs[cite: 2] :
* **Échantillonnage dynamique (Subsampling) :** Sur des données volumineuses ($> 300$ lignes), le calcul SHAP complet devient lent et surcharge visuellement le *Beeswarm plot*[cite: 2]. Un sous-échantillonnage est donc déclenché automatiquement[cite: 2].
* **Mode Prod (`n_repeats=1`) vs Mode Audit (`n_repeats > 1`) :** 
  * En production (API REST / Streamlit), un seul tirage déterministe (`random_state=42`) garantit un temps de réponse $< 0.3$s[cite: 2].
  * En audit / rapport officiel, un calcul agrégé sur $N$ tirages (*Cross-Validation / Aggregated Subsampling*) élimine le biais de sous-échantillonnage[cite: 2].

#### 2. Constructeur `__init__(self, model_path, output_dir)`
* **Rôle :** Initialise l'instance du module d'explicabilité[cite: 2].
* **Fonctionnement :**
  * Convertit les chemins passés en paramètres en objets `Path` (gestion multiplateforme propre)[cite: 2].
  * Crée le dossier de destination des visuels (`reports/figures/`) s'il n'existe pas encore[cite: 2].
  * Déclenche le chargement du modèle sérialisé via la méthode privée `_load_model()`[cite: 2].
  * Initialise les attributs d'état (`explainer`, `shap_values`, `X_sample`) à `None`[cite: 2].

#### 3. Méthode privée `_load_model(self)`
* **Rôle :** Sécurise le chargement du fichier modèle `.joblib` depuis le disque[cite: 2].
* **Fonctionnement :**
  * Vérifie la présence physique du fichier sur le disque et lève une exception explicite (`FileNotFoundError`) si l'artefact est introuvable[cite: 2].
  * Désérialise et retourne l'objet modèle XGBoost via `joblib.load()`[cite: 2].

#### 4. Calculateur de valeurs SHAP `fit_explainer(self, X_sample, threshold=300, sample_size=200, n_repeats=1)`
* **Rôle :** Instancie un `shap.TreeExplainer` optimisé pour XGBoost et applique la logique de calcul[cite: 2].
* **Fonctionnement :**
  * Si la taille de `X_sample` est inférieure ou égale au seuil (`threshold`), le calcul est effectué sur l'intégralité du jeu de données[cite: 2].
  * Si le jeu dépasse le seuil[cite: 2] :
    * **Si `n_repeats == 1` (Fast Prod Mode) :** Extrait un échantillon unique de `sample_size` lignes (`random_state=42`) pour un calcul instantané[cite: 2].
    * **Si `n_repeats > 1` (Robust Audit Mode) :** Effectue `n_repeats` tirages indépendants, calcule les matrices SHAP associées et réalise la moyenne élément par élément (`np.mean`) pour stabiliser l'explicabilité[cite: 2].

#### 5. Générateur d'Explications Globales `generate_global_plots(self)`
* **Rôle :** Génère et sauvegarde les visuels d'interprétabilité globale du modèle[cite: 2].
* **Fonctionnement :**
  * Vérifie que `fit_explainer()` a bien été exécuté (sinon lève une `ValueError`)[cite: 2].
  * Génère le **Beeswarm Plot** (`shap_summary_beeswarm.png`) et le **Bar Plot** (`shap_summary_bar.png`)[cite: 2].
  * Exporte les images en haute définition (`dpi=300`) dans `reports/figures/` puis libère la mémoire RAM avec `plt.close()`[cite: 2].

#### 6. Générateur d'Explication Locale `generate_local_waterfall(self, sample_index=0, filename=...)`
* **Rôle :** Explique la prédiction individuelle d'une observation précise[cite: 2].
* **Fonctionnement :**
  * Génère un **Waterfall Plot** montrant pas à pas comment les caractéristiques d'une commande font dévier la prédiction à partir de la valeur moyenne de référence $E[f(X)]$[cite: 2].

#### 7. Bloc d'exécution autonome `if __name__ == "__main__":`
* **Rôle :** Permet de tester et de valider le script en ligne de commande (`python -m src.explainability`) sans impacter les imports de modules externes[cite: 2].

---

### C. Standardisation pour `src/monitoring.py` (Evidently AI)
* **Stratégie de Data Drift pondérée :**
  * **Seuils stricts (Alerte critique) :** Appliqués en priorité sur les variables identifiées comme majeures par SHAP[cite: 2]. Toute dérive statistique sur ces variables remet en cause la fiabilité des prédictions[cite: 2].
  * **Seuils tolérants (Alerte secondaire) :** Appliqués sur les variables à faible poids SHAP afin d'éviter les fausses alertes opérationnelles[cite: 2].
* **Suivi de la dérive de prédiction (*Prediction Drift*) :** Surveillance de la distribution des sorties du modèle par rapport à la distribution de référence pour détecter tout glissement du comportement global[cite: 2].

---

## 4. Déroulement & Livrables de la Phase 05

1. **`notebooks/05_explainability.ipynb`** : Notebook d'expérimentation pour l'analyse des graphiques SHAP, la définition de la *baseline* et la validation des tests de dérive[cite: 2].
2. **`src/explainability.py`** : Module Python automatisé d'extraction des valeurs SHAP et de génération des graphiques (`.png` / `.csv`)[cite: 2].
3. **`src/monitoring.py`** : Script de surveillance automatisé basé sur Evidently AI avec gestion des alertes et export des rapports HTML/JSON[cite: 2].
4. **`reports/figures/`** : Dossier de stockage des artefacts visuels d'explicabilité et des rapports de dérive pour l'auditabilité[cite: 2].

</details>

---

### [EN] English Version

## 1. Phase Objectives
Phase 05 aims to ensure auditability (traceability), transparency, and long-term reliability for any Machine Learning model deployed in production[cite: 2, 3].

It addresses two core challenges:
1. **Explainability:** Understanding the model's decision-making process at both global (most influential features) and local (individual prediction explanation) levels[cite: 3].
2. **Monitoring:** Detecting data drift (*Data Drift*) and performance drift (*Target / Prediction Drift*) between training and production environments to anticipate model degradation[cite: 3].

---

## 2. Technical Choices Justification

### A. SHAP (*SHapley Additive exPlanations*) for Explainability

#### Why SHAP?
* **Solid Mathematical Foundation:** Grounded in game theory (Shapley values), SHAP guarantees fairness and consistency in allocating contributions to each feature[cite: 3].
* **Global and Local Explainability:** Provides both an overview of model behavior and an exact decomposition for a specific observation[cite: 3].
* **Optimization for Tree-Based Models:** The `TreeExplainer` algorithm allows exact and efficient calculation of SHAP values for Gradient Boosting and Random Forest architectures[cite: 3].

#### Compared Solutions and Reasons for Non-Selection:
* **Native Feature Importance:** Discarded because it is often biased toward high-cardinality features (variables with many distinct values) and does not indicate directionality (positive or negative impact)[cite: 2, 3].
* **LIME:** Discarded due to its local stochastic approximation approach (instability of explanations across runs), which conflicts with strict reproducibility requirements[cite: 3].

---

### B. Evidently AI for Monitoring (Data & Target Drift)

#### Why Evidently AI?
* **Drift Specialization:** Integrates automated statistical tests tailored to variable types (Kolmogorov-Smirnov, Wasserstein, Chi-Square, PSI)[cite: 3].
* **Autonomy & Lightweight Design:** Generates interactive HTML reports for visual analysis and JSON exports for automated pipeline integrations[cite: 3].
* **MLOps Interoperability:** Easily integrates into Python scripts and interfaces seamlessly with metrics infrastructure (Prometheus, Grafana)[cite: 3].

#### Compared Solutions and Reasons for Non-Selection:
* **Great Expectations:** Excellent for schema validation and static data quality checks, but not optimized for measuring statistical temporal drift in distributions[cite: 3].
* **NannyML:** Comprehensive for estimating performance without ground truth, but involves higher infrastructure complexity at the current stage[cite: 3].

---

### C. Role and Complementarity of Deliverables (Notebook vs. `.py` Scripts)

The architecture relies on a clear decoupling between experimentation and industrialization to guarantee project reproducibility[cite: 3]:

1. **Exploration Notebook (`notebooks/05_explainability.ipynb`):**
   * **Role:** Research, visual validation, and calibration sandbox[cite: 3].
   * **Value:** Used to analyze model behavior, measure relative feature importance, set the baseline value, and identify rendering constraints[cite: 3]. It establishes the business logic applied later in production code[cite: 3].

2. **Explainability Script (`src/explainability.py`):**
   * **Role:** Reusable Python module for automated explainability computations[cite: 3].
   * **Value:** Encapsulates the SHAP engine, exports cleaned static images (`.png`), filters views to top features, and generates individual breakdown plots (*Waterfall Plots*) for atypical predictions[cite: 2, 3].

3. **Monitoring Script (`src/monitoring.py`):**
   * **Role:** Continuous production monitoring pipeline[cite: 3].
   * **Value:** Compares incoming data against reference distributions and enforces customizable alert strategies based on feature importance weights[cite: 3].

---

## 3. SHAP Analysis Methodology & Production Implementation

### A. Role of the 3 SHAP Plots
* **Summary Bar Plot (Global Feature Importance):** Quantifies and ranks average absolute impact ($\vert{}SHAP\ value\vert{}$) of each feature across the dataset to identify primary model drivers[cite: 3].
* **Beeswarm Plot (Impact Distribution & Direction):** Combines feature ranking with feature value scales (high vs. low) to reveal directionality (positive or negative influence on predictions)[cite: 3].
* **Waterfall Plot (Local Explainability):** Explains a single prediction by breaking down the shift from the average reference baseline $E[f(X)]$ to the final output $f(x)$ through positive (+) and negative (-) contributions[cite: 2, 3].

---

### B. Anatomy and Detailed Explanation of `src/explainability.py`

To support the Data Science team's upskilling in MLOps best practices, the `src/explainability.py` module was designed following object-oriented programming (OOP) principles, strict typing, and server resource optimization[cite: 2].

Here is the detailed role of each script component[cite: 2]:

#### 1. Header and Architectural Strategy (Docstring)
The module opens with clear documentation outlining two key MLOps trade-offs[cite: 2]:
* **Dynamic Subsampling:** On large datasets ($> 300$ rows), full SHAP computation becomes slow and visually cluttering for the *Beeswarm plot*[cite: 2]. Subsampling is therefore automatically triggered[cite: 2].
* **Prod Mode (`n_repeats=1`) vs. Audit Mode (`n_repeats > 1`):** 
  * In production (REST API / Streamlit), a single deterministic run (`random_state=42`) guarantees response times $< 0.3$s[cite: 2].
  * In audit / official reports, an aggregated calculation across $N$ runs (*Cross-Validation / Aggregated Subsampling*) eliminates subsampling bias[cite: 2].

#### 2. Constructor `__init__(self, model_path, output_dir)`
* **Role:** Initializes the explainability module instance[cite: 2].
* **How it works:**
  * Converts input paths into `Path` objects for clean cross-platform file handling[cite: 2].
  * Creates the output directory (`reports/figures/`) if it does not already exist[cite: 2].
  * Triggers model loading via the private `_load_model()` method[cite: 2].
  * Initializes state attributes (`explainer`, `shap_values`, `X_sample`) to `None`[cite: 2].

#### 3. Private Method `_load_model(self)`
* **Role:** Securely loads the `.joblib` model artifact from disk[cite: 2].
* **How it works:**
  * Checks physical file existence on disk and raises an explicit `FileNotFoundError` if the artifact is missing[cite: 2].
  * Deserializes and returns the XGBoost model object using `joblib.load()`[cite: 2].

#### 4. SHAP Values Calculator `fit_explainer(self, X_sample, threshold=300, sample_size=200, n_repeats=1)`
* **Role:** Instantiates a `shap.TreeExplainer` optimized for XGBoost and applies execution logic[cite: 2].
* **How it works:**
  * If `X_sample` size is below or equal to `threshold`, SHAP values are computed on the full dataset[cite: 2].
  * If the dataset exceeds the threshold[cite: 2]:
    * **If `n_repeats == 1` (Fast Prod Mode):** Extracts a single sample of `sample_size` rows (`random_state=42`) for near-instant computation[cite: 2].
    * **If `n_repeats > 1` (Robust Audit Mode):** Performs `n_repeats` independent random draws, computes corresponding SHAP matrices, and averages them element-wise (`np.mean`) to stabilize explanations[cite: 2].

#### 5. Global Explanations Generator `generate_global_plots(self)`
* **Role:** Generates and saves model global interpretability visualizations[cite: 2].
* **How it works:**
  * Verifies that `fit_explainer()` has been executed (raises a `ValueError` otherwise)[cite: 2].
  * Generates the **Beeswarm Plot** (`shap_summary_beeswarm.png`) and **Bar Plot** (`shap_summary_bar.png`)[cite: 2].
  * Exports high-resolution images (`dpi=300`) to `reports/figures/` and clears RAM using `plt.close()`[cite: 2].

#### 6. Local Explanation Generator `generate_local_waterfall(self, sample_index=0, filename=...)`
* **Role:** Explains individual predictions for a specific observation[cite: 2].
* **How it works:**
  * Generates a **Waterfall Plot** detailing step-by-step how an order's specific features push the prediction away from the mean reference baseline $E[f(X)]$[cite: 2].

#### 7. Autonomous Execution Block `if __name__ == "__main__":`
* **Role:** Allows direct CLI testing and validation (`python -m src.explainability`) without affecting external module imports[cite: 2].

---

### C. Standardization for `src/monitoring.py` (Evidently AI)
* **Weighted Data Drift Strategy:**
  * **Strict Thresholds (Critical Alerts):** Applied primarily to top SHAP features[cite: 3]. Statistical drift on these features invalidates model reliability[cite: 3].
  * **Tolerant Thresholds (Secondary Alerts):** Applied to low SHAP impact features to prevent operational false alarms[cite: 3].
* **Prediction Drift Tracking:** Monitors overall model output distributions against baseline runs to spot macro-level prediction shifts[cite: 3].

---

## 4. Phase 05 Roadmap & Deliverables

1. **`notebooks/05_explainability.ipynb`**: Experimental notebook for SHAP visual analysis, baseline configuration, and drift testing[cite: 3].
2. **`src/explainability.py`**: Automated Python module for SHAP extraction and figure/data export (`.png` / `.csv`)[cite: 3].
3. **`src/monitoring.py`**: Automated monitoring script built on Evidently AI with alert handling and HTML/JSON report generation[cite: 3].
4. **`reports/figures/`**: Storage directory for explainability artifacts and drift report audits[cite: 3].