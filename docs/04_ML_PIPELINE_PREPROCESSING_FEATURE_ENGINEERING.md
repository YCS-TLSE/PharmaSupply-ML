# Technical Documentation
## Phase 4: ML Pipeline, Preprocessing & Feature Engineering
**Project:** PharmaSupply-ML  
**Framework:** Standard MLOps Blueprint  
**File:** `docs/04_ML_PIPELINE_PREPROCESSING_FEATURE_ENGINEERING.md`  
**Core Scripts & Files:** `src/train.py`  
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>  

## 1. Vue d'ensemble et objectifs

Cette étape met en œuvre le cœur Machine Learning du pipeline d'entraînement pour la prévision de la demande pharmaceutique. Le pipeline ingère les enregistrements quotidiens bruts standardisés depuis PostgreSQL (via la Quality Gate validée), effectue l'ingénierie des variables (*feature engineering*) temporelles et l'ingénierie de la cible, entraîne un modèle **XGBoost Regressor** haute performance avec fixation du `base_score` pour le moteur SHAP, évalue ses résultats à l'aide de métriques métier de la supply chain et persiste l'artefact entraîné sous `models/xgboost_pharma_demand.joblib`.

---

## 2. Pourquoi XGBoost pour la prévision de la demande ?

`XGBoost` (*eXtreme Gradient Boosting*) a été sélectionné comme modèle de référence selon les critères clés suivants :

* **Efficacité sur données tabulaires :** Les arbres de décision boostés par gradient surpassent systématiquement les modèles de deep learning sur les séries temporelles tabulaires structurées à échelle petite à moyenne.
* **Dynamiques non linéaires :** Capable de capturer des relations non linéaires, des pics saisonniers (ex. saison de la grippe vs été) et des délais d'approvisionnement sans nécessiter d'hypothèses strictes de stationnarité.
* **Interprétabilité :** Fournit un score natif d'importance des variables et s'intègre parfaitement avec SHAP pour l'explicabilité.
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
   * Le pipeline remplace les ventes en rupture par `NaN` et applique une imputation par propagation (*forward/backward fill*) par groupe (`store_id`, `product_id`) pour éviter de propager de faux zéros dans les statistiques de lag.

3. **Variables de retard (Lags) et moyennes glissantes :**
   * **Lags :** `lag_7`, `lag_14`, `lag_30` (capturent les historiques de demande à des intervalles de 1, 2 et 4 semaines).
   * **Moyennes glissantes :** `rolling_mean_7`, `rolling_mean_30` (décalées d'un jour pour éviter toute fuite de données).

4. **Recalcul de la Cible (Target Engineering) :**
   * La cible de prédiction `target_demand` est recalculée pour correspondre à la **demande cumulée sur une fenêtre glissante de 7 jours futurs** (`sum` avec une fenêtre de 7 décalée de -6), permettant d'anticiper les besoins logistiques à court terme.

---

## 4. Validation par découpage temporel (Train/Test Split)

Afin d'éviter la **fuite de données** (*data leakage*) temporelle, la validation croisée aléatoire traditionnelle a été remplacée par un **découpage temporel** séquentiel strict :

* **Ratio de découpage :** 80 % Entraînement / 20 % Test basé sur l'ordre chronologique des dates.
* **Période d'entraînement :** Du début de la série chronologique jusqu'au seuil des 80%.
* **Période de test :** Les 20 % de données les plus récentes pour évaluer la robustesse du modèle face à de futures observations.

---

## 5. Spécificité Technique : Compatibilité SHAP (`base_score`)

Le modèle XGBoost est instancié avec un paramètre explicite :
```python
XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    base_score=0.5,
    random_state=42,
)
```
Cette configuration du `base_score=0.5` est indispensable pour garantir une parfaite compatibilité avec le moteur `TreeExplainer` de SHAP (Phase 05), évitant ainsi les erreurs de sérialisation et de crash lors de l'interprétabilité des arbres.

---

## 6. Persistance de l'artefact

Après un entraînement et une validation réussis, le modèle XGBoost entraîné est sérialisé et sauvegardé localement :

* **Chemin de l'artefact :** `models/xgboost_pharma_demand.joblib`
* **Utilisation :** Prêt à être chargé par les API d'inférence (FastAPI) et les modules d'explicabilité/monitoring.

---

## 7. Exécution

Pour exécuter le pipeline d'entraînement dans l'environnement virtuel Conda du projet :

```bash
conda activate pharma-supply
python src/train.py
```

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

## 1. Overview & Objectives

This step implements the core Machine Learning training pipeline for pharmaceutical demand forecasting. The pipeline ingests standardized raw daily records from PostgreSQL (via the validated Quality Gate), performs time-series feature and target engineering, trains a high-performance **XGBoost Regressor** with an explicit `base_score` setup for SHAP compatibility, evaluates model performance using supply-chain domain metrics, and persists the trained artifact under `models/xgboost_pharma_demand.joblib`.

---

## 2. Why XGBoost for Demand Forecasting?

`XGBoost` (*eXtreme Gradient Boosting*) was selected as the baseline model based on the following key criteria:

* **Tabular Efficiency:** Gradient-boosted decision trees consistently outperform deep learning models on structured tabular time-series data with small-to-medium dataset scales.
* **Non-Linear Dynamics:** Capable of capturing non-linear relationships, seasonal spikes (e.g., flu season vs. summer), and lead-time delays without strict stationarity requirements.
* **Interpretability:** Provides native feature importance scoring and seamlessly integrates with SHAP for explainability.
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
   * The pipeline replaces stockout sales with `NaN` and applies forward/backward filling grouped by `store_id` and `product_id` to prevent zero-propagation into lag statistics.

3. **Lag & Rolling Window Features:**
   * **Lags:** `lag_7`, `lag_14`, `lag_30` (captures historic demand patterns at 1, 2, and 4-week intervals).
   * **Rolling Averages:** `rolling_mean_7`, `rolling_mean_30` (shifted by 1 day to strictly prevent data leakage).

4. **Target Recalculation (Target Engineering):**
   * The prediction target `target_demand` is re-engineered to represent the **cumulative demand over a future 7-day rolling window** (`sum` over a 7-period window shifted by -6), enabling accurate short-term operational inventory planning.

---

## 4. Time-Based Train/Test Validation Split

To prevent temporal **data leakage**, traditional random cross-validation was replaced with a strict sequential **Time-Based Split**:

* **Split Ratio:** 80% Train / 20% Test based on chronological date ordering.
* **Train Period:** From the beginning of the time-series up to the 80% threshold.
* **Test Period:** The most recent 20% of data to evaluate model generalization on unseen future observations.

---

## 5. Technical Specificity: SHAP Compatibility (`base_score`)

The XGBoost model is instantiated with an explicit parameter:
```python
XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    base_score=0.5,
    random_state=42,
)
```
Setting `base_score=0.5` is critical to ensure absolute compatibility with SHAP's `TreeExplainer` engine (Phase 05), preventing serialization bugs and runtime crashes during tree interpretation.

---

## 6. Artifact Persistence

Upon successful training and validation, the trained XGBoost model is serialized and saved locally:

* **Artifact Path:** `models/xgboost_pharma_demand.joblib`
* **Utility:** Ready to be loaded by inference APIs (FastAPI) and explainability/monitoring modules.

---

## 7. How to Execute

To run the training pipeline within the project's Conda environment:

```bash
conda activate pharma-supply
python src/train.py
```

</details>