# Technical Documentation  
## Phase 5: Explainability & Monitoring  
**Project:** PharmaSupply-ML (Standard MLOps Framework)  
**File:** `docs/05_EXPLAINABILITY_AND_MONITORING.md`  
**Status:** Validated  

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>  



## 1. Objectifs de la Phase
La Phase 05 vise à garantir l'auditabilité, la transparence et la pérennité de tout modèle de Machine Learning destiné à la production.

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

#### Comparatif et alternatives écartées :
* **Feature Importance native des algorithmes :** Écartée car elle est souvent biaisée en faveur des variables à forte cardinalité et n'indique pas le sens de l'impact (positif ou négatif).
* **LIME :** Écarté en raison de son approche par approximation locale stochastique (instabilité des explications selon les tirages), incompatible avec les exigences de reproductibilité strictes.

---

### B. Evidently AI pour le Monitoring (Data & Target Drift)

#### Pourquoi Evidently AI ?
* **Spécialisation Drift :** Intègre des tests statistiques automatisés et adaptés à la nature des variables (tests Kolmogorov-Smirnov, Wasserstein, Chi-Square, PSI).
* **Autonomie et Légèreté :** Génère des rapports HTML interactifs pour l'analyse visuelle et des exports JSON pour l'intégration automatisée dans les pipelines.
* **Interopérabilité MLOps :** S'intègre facilement dans les scripts Python et s'interface avec les outils de métriques (Prometheus, Grafana).

#### Comparatif et alternatives écartées :
* **Great Expectations :** Très performant pour la validation de schéma et de qualité de données statiques, mais non optimisé pour mesurer la dérive statistique temporelle des distributions.
* **NannyML :** Très complet pour l'estimation de performance sans vérité terrain, mais présente une complexité d'infrastructure plus élevée au stade actuel.

---

### C. Rôle et complémentarité des livrables (Notebook vs Scripts `.py`)

L'architecture repose sur un découplage clair entre la phase d'expérimentation et l'industrialisation pour garantir la reproductibilité du projet :

1. **Le Notebook d'exploration (`notebooks/05_explainability.ipynb`) :**
   * **Rôle :** Espace de recherche, de validation visuelle et de calibrage.
   * **Apport :** Permet d'analyser le comportement du modèle, de mesurer l'importance relative des variables, de déterminer la valeur de référence (*baseline*) et de déceler les contraintes d'affichage (ex. gestion des dépendances JavaScript). Il sert à fixer les règles métiers appliquées ensuite dans le code de production.

2. **Le Script d'Explicabilité (`src/explainability.py`) :**
   * **Rôle :** Module Python réutilisable pour le calcul d'explicabilité automatisé.
   * **Apport :** Encapsule le moteur SHAP, génère des visuels nettoyés au format image (`.png`), filtre l'affichage sur les variables principales et fournit des explications locales (*Waterfall Plots*) pour les prédictions atypiques.

3. **Le Script de Surveillance (`src/monitoring.py`) :**
   * **Rôle :** Pipeline de surveillance continue en production.
   * **Apport :** Compare les données entrantes aux distributions de référence et applique une stratégie d'alerte paramétrable en fonction du poids réel des variables.

---

## 3. Méthodologie d'Analyse SHAP et Implémentation Production

L'analyse réalisée lors de la phase d'exploration fournit la grille de paramétrage nécessaire pour automatiser les modules `explainability.py` et `monitoring.py` de façon générique.

### A. Rôle des 3 graphiques SHAP
* **Summary Bar Plot (Global Feature Importance) :** Quantifie et classe l'impact moyen absolu ($|SHAP\ value|$) de chaque variable sur l'ensemble du jeu de données pour identifier les moteurs principaux du modèle.
* **Beeswarm Plot (Distribution et sens d'influence) :** Combine le classement des variables avec l'échelle de leurs valeurs (fortes vs faibles) pour révéler le sens de leur influence (hausse ou baisse de la prédiction).
* **Individual Breakdown Bar Plot (Explicabilité locale) :** Explique une prédiction individuelle en détaillant le passage de la valeur de référence moyenne $E[f(X)]$ à la prédiction finale $f(x)$ grâce aux contributions positives (+) et négatives (-) de chaque paramètre.

### B. Standardisation pour `src/explainability.py`
* **Filtrage des variables :** Utilisation du paramètre `max_display` pour restreindre les exports visuels aux seules variables significatives et éliminer le bruit graphique.
* **Génération conditionnelle :** Implémentation d'une fonction déclenchant un graphique d'explication individuelle (Individual Breakdown Bar Plot) automatique dès qu'une prédiction dépasse un seuil d'écart-type prédéfini par rapport à la baseline.
* **Rendu statique & Reproductibilité :** Utilisation du backend Matplotlib (`matplotlib=True`) lors de la sauvegarde des figures afin d'éviter l'injection de scripts JavaScript interactifs bruts et garantir un rendu constant dans les rapports.
* **Exports tabulaires :** Sauvegarde de la matrice des valeurs SHAP au format `.csv` pour permettre l'alimentation de tableaux de bord sans re-calculer l'explicabilité.

### C. Standardisation pour `src/monitoring.py` (Evidently AI)
* **Stratégie de Data Drift pondérée :**
  * **Seuils stricts (Alerte critique) :** Appliqués en priorité sur les variables identifiées comme majeures par SHAP. Toute dérive statistique sur ces variables remet en cause la fiabilité des prédictions.
  * **Seuils tolérants (Alerte secondaire) :** Appliqués sur les variables à faible poids SHAP afin d'éviter les fausses alertes opérationnelles.
* **Suivi de la dérive de prédiction (*Prediction Drift*) :** Surveillance de la distribution des sorties du modèle par rapport à la distribution de référence pour détecter tout glissement du comportement global.

### D. Optimisation & Robustesse (Sampling & Cross-Validation)
* **Échantillonnage dynamique (Sampling) :**
  * Lors de l'ingestion d'un fichier utilisateur volumineux, extraction automatique d'un échantillon représentatif (ex. $N = 1\,000$) pour le calcul SHAP afin de garantir un temps de réponse rapide en production sans altérer la qualité des explications.
* **Validation croisée (Cross-Validation) :**
  * Validation de la stabilité de l'importance des variables et des seuils de dérive à travers plusieurs *folds* de cross-validation, garantissant la reproductibilité et l'insensibilité au découpage des données.

---

## 4. Déroulement & Livrables de la Phase 05

1. **`notebooks/05_explainability.ipynb`** : Notebook d'expérimentation pour l'analyse des graphiques SHAP, la définition de la *baseline* et la validation des tests de dérive.
2. **`src/explainability.py`** : Module Python automatisé d'extraction des valeurs SHAP et de génération des graphiques (`.png` / `.csv`).
3. **`src/monitoring.py`** : Script de surveillance automatisé basé sur Evidently AI avec gestion des alertes et export des rapports HTML/JSON.
4. **`reports/figures/`** : Dossier de stockage des artefacts visuels d'explicabilité et des rapports de dérive pour l'auditabilité.


</details>

---

### [EN] English Version


---

## 1. Phase Objectives
Phase 05 aims to ensure auditability, transparency, and long-term reliability for any Machine Learning model deployed in production[cite: 13].

It addresses two core challenges:
1. **Explainability:** Understanding the model's decision-making process at both global (most influential variables) and local (individual prediction explanation) levels[cite: 13].
2. **Monitoring:** Detecting data drift (*Data Drift*) and performance drift (*Target / Prediction Drift*) between training and production environments to anticipate model degradation[cite: 13].

---

## 2. Technical Choices Justification

### A. SHAP (*SHapley Additive exPlanations*) for Explainability

#### Why SHAP?
* **Solid Mathematical Foundation:** Grounded in game theory (Shapley values), SHAP guarantees fairness and consistency in allocating contributions to each feature[cite: 13].
* **Global and Local Explainability:** Provides both an overview of model behavior and an exact decomposition for a specific observation[cite: 13].
* **Optimization for Tree-Based Models:** The `TreeExplainer` algorithm allows exact and efficient calculation of SHAP values for Gradient Boosting and Random Forest architectures[cite: 13].

#### Comparison and Alternative Options Discarded:
* **Native Feature Importance:** Discarded because it is often biased toward high-cardinality features and does not indicate directionality (positive or negative impact)[cite: 13].
* **LIME:** Discarded due to its local stochastic approximation approach (instability of explanations across runs), which conflicts with strict reproducibility requirements[cite: 13].

---

### B. Evidently AI for Monitoring (Data & Target Drift)

#### Why Evidently AI?
* **Drift Specialization:** Integrates automated statistical tests tailored to variable types (Kolmogorov-Smirnov, Wasserstein, Chi-Square, PSI)[cite: 13].
* **Autonomy & Lightweight Design:** Generates interactive HTML reports for visual analysis and JSON exports for automated pipeline integrations[cite: 13].
* **MLOps Interoperability:** Easily integrates into Python scripts and interfaces seamlessly with metrics infrastructure (Prometheus, Grafana)[cite: 13].

#### Comparison and Alternative Options Discarded:
* **Great Expectations:** Excellent for schema validation and static data quality checks, but not optimized for measuring statistical temporal drift in distributions[cite: 13].
* **NannyML:** Comprehensive for estimating performance without ground truth, but involves higher infrastructure complexity at the current stage[cite: 13].

---

### C. Role and Complementarity of Deliverables (Notebook vs. `.py` Scripts)

The architecture relies on a clear decoupling between experimentation and industrialization to guarantee project reproducibility[cite: 13]:

1. **Exploration Notebook (`notebooks/05_explainability.ipynb`):**
   * **Role:** Research, visual validation, and calibration sandbox[cite: 13].
   * **Value:** Used to analyze model behavior, measure relative feature importance, set the baseline value, and identify rendering constraints (e.g., JavaScript dependency management)[cite: 13]. It establishes the business logic applied later in production code[cite: 13].

2. **Explainability Script (`src/explainability.py`):**
   * **Role:** Reusable Python module for automated explainability computations[cite: 13].
   * **Value:** Encapsulates the SHAP engine, exports cleaned static images (`.png`), filters views to top features, and generates individual breakdown plots for atypical predictions[cite: 13].

3. **Monitoring Script (`src/monitoring.py`):**
   * **Role:** Continuous production monitoring pipeline[cite: 13].
   * **Value:** Compares incoming data against reference distributions and enforces customizable alert strategies based on feature importance weights[cite: 13].

---

## 3. SHAP Analysis Methodology & Production Implementation

Exploration analysis provides the configuration framework required to make `explainability.py` and `monitoring.py` generic and automated[cite: 13].

### A. Role of the 3 SHAP Plots
* **Summary Bar Plot (Global Feature Importance):** Quantifies and ranks average absolute impact ($\vert{}SHAP\ value\vert{}$) of each feature across the dataset to identify primary model drivers[cite: 13].
* **Beeswarm Plot (Impact Distribution & Direction):** Combines feature ranking with feature value scales (high vs. low) to reveal directionality (positive or negative influence on predictions)[cite: 13].
* **Individual Breakdown Bar Plot (Local Explainability):** Explains a single prediction by breaking down the shift from the average reference baseline $E[f(X)]$ to the final output $f(x)$ through positive (+) and negative (-) contributions[cite: 13].

### B. Standardization for `src/explainability.py`
* **Feature Filtering:** Uses `max_display` to restrict visual exports to significant features and eliminate visual noise[cite: 13].
* **Conditional Generation:** Implements an automated trigger for individual breakdown bar plots whenever a prediction strays beyond a defined standard deviation threshold from the baseline[cite: 13].
* **Static Rendering & Reproducibility:** Uses the Matplotlib backend (`matplotlib=True`) when saving figures to prevent raw interactive JavaScript injection and ensure consistent document rendering[cite: 13].
* **Tabular Exports:** Saves the SHAP value matrix as a `.csv` file to feed downstream dashboards without recomputing explainability metrics[cite: 13].

### C. Standardization for `src/monitoring.py` (Evidently AI)
* **Weighted Data Drift Strategy:**
  * **Strict Thresholds (Critical Alerts):** Applied primarily to top SHAP features[cite: 13]. Statistical drift on these features invalidates model reliability[cite: 13].
  * **Tolerant Thresholds (Secondary Alerts):** Applied to low SHAP impact features to prevent operational false alarms[cite: 13].
* **Prediction Drift Tracking:** Monitors overall model output distributions against baseline runs to spot macro-level prediction shifts[cite: 13].

### D. Optimization & Robustness (Sampling & Cross-Validation)
* **Dynamic Sampling:**
  * Automatically draws a representative subset (e.g., $N = 1\,000$) when processing large input files to maintain rapid inference speeds without compromising explanation fidelity[cite: 13].
* **Cross-Validation:**
  * Assesses feature importance stability and drift thresholds across multiple cross-validation folds, guaranteeing reproducibility and resilience to data splits[cite: 13].

---

## 4. Phase 05 Roadmap & Deliverables

1. **`notebooks/05_explainability.ipynb`**: Experimental notebook for SHAP visual analysis, baseline configuration, and drift testing[cite: 13].
2. **`src/explainability.py`**: Automated Python module for SHAP extraction and figure/data export (`.png` / `.csv`)[cite: 13].
3. **`src/monitoring.py`**: Automated monitoring script built on Evidently AI with alert handling and HTML/JSON report generation[cite: 13].
4. **`reports/figures/`**: Storage directory for explainability artifacts and drift report audits[cite: 13].