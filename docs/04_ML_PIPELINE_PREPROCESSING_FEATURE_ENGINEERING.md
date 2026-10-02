# Technical Documentation
## Phase 4: ML Pipeline, Preprocessing & Feature Engineering
**Project:** PharmaSupply-ML  
**Framework:** Standard MLOps Blueprint  
**File:** `docs/04_ML_PIPELINE_PREPROCESSING_FEATURE_ENGINEERING.md` (`src/train.py`)  
**Status:** Validated  

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>  

## 1. Vue d'ensemble et objectifs

Cette étape met en œuvre le cœur Machine Learning du pipeline d'entraînement pour la prévision de la demande pharmaceutique. Le pipeline ingère les enregistrements quotidiens bruts standardisés depuis PostgreSQL, effectue l'ingénierie des variables (*feature engineering*) temporelles, entraîne un modèle **XGBoost Regressor** haute performance, évalue ses résultats à l'aide de métriques métier de la supply chain et persiste l'artefact entraîné pour le déploiement.

---

## 2. Pourquoi XGBoost pour la prévision de la demande ?

`XGBoost` (*eXtreme Gradient Boosting*) a été sélectionné comme modèle de référence selon les critères clés suivants :

* **Efficacité sur données tabulaires :** Les arbres de décision boostés par gradient surpassent systématiquement les modèles de deep learning sur les séries temporelles tabulaires structurées à échelle petite à moyenne.
* **Dynamiques non linéaires :** Capable de capturer des relations non linéaires, des pics saisonniers (ex. saison de la grippe vs été) et des délais d'approvisionnement sans nécessiter d'hypothèses strictes de stationnarité.
* **Interprétabilité :** Fournit un score natif d'importance des variables, permettant aux analystes de la supply chain de comprendre les facteurs clés de la demande.
* **Robustesse :** Résilient face aux valeurs aberrantes, aux données manquantes et aux anomalies opérationnelles à court terme.

---

## 3. Preprocessing, Feature Engineering & Reconstruction du Signal

Pour garantir une précision prédictive élevée tout en respectant les dynamiques réelles de la chaîne logistique, `src/train.py` exécute un pipeline d'ingénierie des variables en plusieurs étapes :

1. **Variables calendaires :**
   * `month` : Capture la saisonnalité annuelle.
   * `dayofweek` : Capture les variations de la demande au cours de la semaine.
   * `is_weekend` : Indicateur binaire pour les opérations du week-end.

2. **Reconstruction de la demande censurée (gestion des ruptures) :**
   * En cas de rupture de stock (`is_stockout == 1`), le volume de ventes (`sales_volume`) chute artificiellement à 0 ou à la limite du stock, masquant la véritable demande du marché.
   * Le pipeline remplace les ventes en rupture par `NaN` et applique une imputation par propagation (*forward/backward fill*) par `product_id` pour éviter de propager de faux zéros dans les statistiques de lag.

3. **Variables de retard (Lags) et moyennes glissantes :**
   * **Lags :** `lag_7`, `lag_14`, `lag_30` (capturent les historiques de demande à des intervalles de 1, 2 et 4 semaines).
   * **Moyennes glissantes :** `rolling_mean_7`, `rolling_mean_30` (décalées d'un jour pour éviter toute fuite de données).

---

## 4. Validation par découpage temporel (Train/Test Split)

Afin d'éviter la **fuite de données** (*data leakage*) temporelle, la validation croisée aléatoire traditionnelle a été remplacée par un **découpage temporel** séquentiel strict :

* **Ratio de découpage :** 80 % Entraînement / 20 % Test basé sur l'ordre chronologique des dates.
* **Période d'entraînement :** `18-10-2023` au `15-02-2026` (8 520 lignes).
* **Période de test :** `16-02-2026` au `16-09-2026` (2 130 lignes).

---

## 5. Métriques d'évaluation du modèle

Le modèle est évalué sur le jeu de données de test à l'aide de métriques statistiques standards (`scikit-learn`) et d'une métrique spécifique au domaine de la supply chain :

| Métrique | Valeur | Description |
| :--- | :--- | :--- |
| **MAE** | `8.80` | Erreur absolue moyenne (écart moyen en unités par prédiction) |
| **RMSE** | `11.43` | Racine de l'erreur quadratique moyenne (pénalise les grands écarts) |
| **WMAPE** | **`9.66%`** | **Weighted Absolute Percentage Error** (indicateur de référence en Supply Chain) |

> **Note de référence Supply Chain :** Un WAPE/WMAPE inférieur à **10 %** représente une excellente fiabilité prédictive pour la planification des stocks et le réapprovisionnement automatisé.

---

## 6. Persistance de l'artefact

Après un entraînement et une validation réussis, le modèle XGBoost entraîné est sérialisé et sauvegardé localement :

* **Chemin de l'artefact :** `models/xgboost_pharma_demand.joblib`
* **Utilisation :** Prêt à être chargé par les API d'inférence (ex. FastAPI) pour des prédictions en lot ou en temps réel.

---

## 7. Exécution

Pour exécuter le pipeline d'entraînement dans ton environnement Conda (`pharmasupply-ml`) :

~~~~bash
conda activate pharmasupply-ml
python src/train.py
~~~~

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

## 1. Overview & Objectives

This step implements the core Machine Learning training pipeline for pharmaceutical demand forecasting. The pipeline ingests standardized raw daily records from PostgreSQL, performs time-series feature engineering, trains a high-performance **XGBoost Regressor**, evaluates model performance using supply-chain domain metrics, and persists the trained artifact for downstream deployment.

---

## 2. Why XGBoost for Demand Forecasting?

`XGBoost` (*eXtreme Gradient Boosting*) was selected as the baseline model based on the following key criteria:

* **Tabular Efficiency:** Gradient-boosted decision trees consistently outperform deep learning models on structured tabular time-series data with small-to-medium dataset scales.
* **Non-Linear Dynamics:** Capable of capturing non-linear relationships, seasonal spikes (e.g., flu season vs. summer), and lead-time delays without strict stationarity requirements.
* **Interpretability:** Provides native feature importance scoring, allowing supply chain analysts to understand key drivers of demand.
* **Robustness:** Resilient to outliers, missing values, and short-term operational anomalies.

---

## 3. Preprocessing, Feature Engineering & Signal Reconstruction

To ensure high predictive accuracy while respecting real-world supply chain dynamics, `src/train.py` executes a multi-step feature engineering pipeline:

1. **Calendar Features:**
   * `month`: Captures annual seasonality.
   * `dayofweek`: Captures intra-week demand variations.
   * `is_weekend`: Binary flag for weekend operations.

2. **Censored Demand Reconstruction (Stockout Handling):**
   * During stockouts (`is_stockout == 1`), `sales_volume` artificially drops to 0 or stock limits, masking true underlying market demand.
   * The pipeline replaces stockout sales with `NaN` and applies forward/backward filling per `product_id` to prevent zero-propagation into lag statistics.

3. **Lag & Rolling Window Features:**
   * **Lags:** `lag_7`, `lag_14`, `lag_30` (captures historic demand patterns at 1, 2, and 4-week intervals).
   * **Rolling Averages:** `rolling_mean_7`, `rolling_mean_30` (shifted by 1 day to strictly prevent data leakage).

---

## 4. Time-Based Train/Test Validation Split

To prevent temporal **data leakage**, traditional random cross-validation was replaced with a strict sequential **Time-Based Split**:

* **Split Ratio:** 80% Train / 20% Test based on chronological date ordering.
* **Train Period:** `2023-10-18` to `2026-02-15` (8,520 records).
* **Test Period:** `2026-02-16` to `2026-09-16` (2,130 records).

---

## 5. Model Evaluation Metrics

The model is evaluated on the hold-out test set using both standard statistical metrics (`scikit-learn`) and a domain-specific supply chain metric:

| Metric | Value | Description |
| :--- | :--- | :--- |
| **MAE** | `8.80` | Mean Absolute Error (average unit error per prediction) |
| **RMSE** | `11.43` | Root Mean Squared Error (penalizes larger forecasting errors) |
| **WMAPE** | **`9.66%`** | **Weighted Absolute Percentage Error** (Supply Chain industry standard metric) |

> **Supply Chain Benchmark Note:** A WMAPE below **10%** represents excellent predictive reliability for inventory planning and automated replenishment.

---

## 6. Artifact Persistence

Upon successful training and validation, the trained XGBoost model is serialized and saved locally:

* **Artifact Path:** `models/xgboost_pharma_demand.joblib`
* **Utility:** Ready to be loaded by serve APIs (e.g., FastAPI) for batch or real-time inference.

---

## 7. How to Execute

To run the training pipeline within your Conda environment (`pharmasupply-ml`):

~~~~bash
conda activate pharmasupply-ml
python src/train.py
~~~~

</details>
