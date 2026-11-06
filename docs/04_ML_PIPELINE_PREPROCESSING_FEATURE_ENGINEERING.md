# Technical Documentation
## Phase 04: ML Pipeline, Preprocessing and Feature Engineering
**Project:** PharmaSupply-ML
**Framework:** Standard MLOps Blueprint
**File:** `docs/04_ML_PIPELINE_PREPROCESSING_FEATURE_ENGINEERING.md`
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

*Un guide pour transformer une série temporelle en features fiables, éviter la fuite temporelle et tracer chaque entraînement.*

---

## Sommaire

1. [Du tableau brut aux features temporelles](#1-du-tableau-brut-aux-features-temporelles)
2. [Le piège du Train/Serve Skew](#2-le-piege-du-trainserve-skew)
3. [Validation temporelle et fuite de données](#3-validation-temporelle-et-fuite-de-donnees)
4. [Entraînement XGBoost et traçabilité MLflow](#4-entrainement-xgboost-et-tracabilite-mlflow)
5. [Tests et preuve de parité](#5-tests-et-preuve-de-parite)
6. [Annexes et checklist MLOps](#6-annexes-et-checklist-mlops)

---

<a id="1-du-tableau-brut-aux-features-temporelles"></a>
## 1. Du tableau brut aux features temporelles

### 1.1 Pourquoi un modèle tabulaire a besoin de mémoire

XGBoost ne comprend pas nativement une série temporelle. Il voit des lignes et des colonnes. Pour qu'il puisse anticiper une demande future, il faut transformer le passé en variables explicites : combien s'est-il vendu il y a sept jours, quelle est la moyenne des trente derniers jours, sommes-nous un week-end ?

C'est le rôle de `src/features.py`. Le module transforme un historique quotidien en une matrice de features tabulaires. Il ne se contente pas d'ajouter des colonnes : il définit le contrat exact que l'entraînement et l'inférence doivent respecter.

### 1.2 Les features calendaires

Les premières features sont simples :

```python
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
```

Un mois de décembre n'a pas la même demande qu'un mois de juillet. Un samedi n'a pas le même comportement qu'un mardi. Ces variables donnent au modèle un contexte temporel minimal.

### 1.3 La reconstruction de la demande censurée

Une rupture de stock fausse les ventes. Si le stock est vide, les ventes tombent à zéro même si la demande était forte. Utiliser directement `sales_volume` comme signal de demande apprendrait au modèle que la demande baisse pendant les ruptures.

Le module crée donc un signal de demande :

```python
df["demand_signal"] = np.where(
    df["is_stockout"] == 1, np.nan, df["sales_volume"]
)
df["demand_signal"] = (
    df.groupby(group_cols)["demand_signal"].ffill().bfill()
)
```

Pendant une rupture, la valeur devient `NaN`, puis elle est propagée depuis la dernière vente valide. Le `bfill()` couvre les cas où la série commence par une rupture. Cette approche est simple et efficace pour un projet pédagogique. Elle a une limite : le `bfill()` peut utiliser une valeur future pour combler un trou initial. Une modélisation plus rigoureuse de la demande censurée demanderait un modèle dédié, mais ce n'est pas nécessaire pour le niveau actuel.

### 1.4 Les lags

Les lags représentent la demande observée sept, quatorze et trente jours plus tôt.

```python
for lag in [7, 14, 30]:
    df[f"lag_{lag}"] = df.groupby(group_cols)[
        "demand_signal"
    ].shift(lag)
```

Le groupby est essentiel. Sans lui, le lag d'un produit pourrait être calculé à partir des ventes d'un autre produit. Les tests vérifient explicitement cette isolation.

Un lag de 7 correspond à la valeur observée sept jours avant la ligne courante. Ce n'est pas une somme sur sept jours. Cette distinction est importante pour interpréter les features et pour construire l'inférence.

### 1.5 Les moyennes glissantes

Les moyennes glissantes lissent le bruit et donnent une tendance locale.

```python
for window in [7, 30]:
    df[f"rolling_mean_{window}"] = df.groupby(
        group_cols
    )["demand_signal"].transform(
        lambda x: x.shift(1).rolling(window).mean()
    )
```

Le `shift(1)` évite d'utiliser la valeur du jour courant. Sans lui, la moyenne inclurait la cible ou une valeur contemporaine, ce qui introduirait une fuite d'information.

### 1.6 La cible cumulée à sept jours

Le modèle ne prédit pas la demande du jour. Il prédit la demande cumulée des sept prochains jours.

```python
df[TARGET_COL] = df.groupby(group_cols)[
    "demand_signal"
].transform(
    lambda x: x.rolling(7, min_periods=7).sum().shift(-6)
)
```

Le `rolling(7).sum()` calcule une somme sur sept jours. Le `shift(-6)` aligne cette somme sur la première journée de la fenêtre. La ligne courante contient donc la demande des jours `J` à `J+6`.

Les six dernières lignes n'ont pas de futur complet. Elles sont supprimées avec les premières lignes qui n'ont pas encore trente jours d'historique :

```python
df = df.dropna(
    subset=["lag_30", TARGET_COL]
).reset_index(drop=True)
```

Le dataset d'entraînement ne contient donc que des lignes où les features et la cible sont complètes.

### 1.7 Le contrat de features

La liste `FEATURE_COLS` vient de `src/config.py`. Elle est utilisée par l'entraînement, le serving et le monitoring. Les colonnes sont :

- `month`, `dayofweek`, `is_weekend` ;
- `stock_on_hand`, `supplier_lead_time` ;
- `lag_7`, `lag_14`, `lag_30` ;
- `rolling_mean_7`, `rolling_mean_30`.

Le modèle ne voit pas `store_id` ni `product_id`. Il apprend des relations globales à partir des lags et du contexte. C'est une limite de modélisation : un modèle par produit ou un encodage catégoriel pourrait capturer des effets spécifiques. Le choix actuel privilégie la simplicité et la robustesse sur un jeu synthétique.

---

<a id="2-le-piege-du-trainserve-skew"></a>
## 2. Le piège du Train/Serve Skew

### 2.1 Définition

Le train/serve skew est l'écart entre les features calculées pendant l'entraînement et celles calculées pendant l'inférence. Ce n'est pas une différence de modèle. C'est une différence de données d'entrée.

Un modèle peut être excellent et se dégrader en production simplement parce que l'API lui envoie des features qui ne ressemblent plus à celles qu'il a vues.

### 2.2 L'exemple qui a motivé la refonte

Dans une première version, l'API calculait :

```python
rolling_mean_7 = lag_7 / 7.0
```

L'entraînement, lui, calculait :

```python
rolling_mean_7 = demand.shift(1).rolling(7).mean()
```

Les deux formules produisent une moyenne, mais pas la même. La première divise un lag par sept. La seconde fait la moyenne des sept derniers jours observés, en excluant le jour courant. Le modèle recevait donc une distribution décalée.

Le problème est difficile à voir dans un notebook, car on ne fait pas tourner l'API. Il devient visible en production, lorsque les prédictions semblent étranges sans qu'aucune erreur ne soit levée.

### 2.3 La solution : un constructeur partagé

`src/features.py` expose `build_history_features()` :

```python
def build_history_features(
    sales_history: Sequence[float],
    as_of: Optional[pd.Timestamp] = None,
    min_history: int = 30,
) -> dict[str, float]:
    ...
```

La fonction prend un historique chronologique de ventes et une date de référence. Elle calcule les features du lendemain avec les mêmes formules que `create_features()` :

- `lag_7` correspond à la valeur observée sept jours avant la date cible ;
- `lag_14` et `lag_30` suivent la même logique ;
- `rolling_mean_7` est la moyenne des sept jours précédents ;
- `rolling_mean_30` est la moyenne des trente jours précédents ;
- les features calendaires viennent de `as_of`.

L'API utilise ce constructeur dès qu'un historique est fourni. Le router `predict` ne réimplémente pas les formules. Il appelle le service, qui appelle le constructeur.

### 2.4 Pourquoi un seul constructeur change tout

Un constructeur partagé réduit le nombre d'endroits où une formule peut diverger. Il rend aussi la parité testable : on peut comparer la sortie de `create_features()` et celle de `build_history_features()` sur la même série.

Le constructeur impose une contrainte de qualité : il exige au moins trente valeurs d'historique. Sans cette contrainte, les lags seraient incomplets et les moyennes glissantes seraient calculées sur une fenêtre partielle.

### 2.5 Le chemin legacy

Le contrat `/predict` accepte encore des lags pré-calculés sans historique. Ce chemin existe pour ne pas casser les clients existants. Il calcule les moyennes glissantes à partir des lags fournis, avec les limites que cela implique.

Ce chemin doit être considéré comme transitoire. La voie recommandée est de fournir `history`, car elle bénéficie de la parité testée. Un client qui utilise encore les lags seuls doit être identifié et migré.

---

<a id="3-validation-temporelle-et-fuite-de-donnees"></a>
## 3. Validation temporelle et fuite de données

### 3.1 Pourquoi un split aléatoire est interdit

Sur une série temporelle, un split aléatoire place des observations futures dans l'ensemble d'entraînement et des observations passées dans l'ensemble de test. Le modèle peut alors apprendre des motifs qui n'existaient pas au moment de la prédiction.

Le résultat est flatteur et faux. Le modèle semble performant parce qu'il a vu le futur. En production, il ne verra que le passé.

### 3.2 Le split chronologique de src/train.py

Le module `src/train.py` trie les dates uniques et coupe à 80 % :

```python
unique_dates = np.sort(df["date"].unique())
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

train_df = df[df["date"] < split_date]
test_df = df[df["date"] >= split_date]
```

Le split se fait sur les dates, pas sur les lignes. Toutes les observations d'une même date restent dans le même ensemble. Cette précaution évite qu'un produit soit dans l'entraînement et un autre dans le test pour la même journée.

Le test commence strictement après la dernière date d'entraînement. La frontière temporelle est nette.

### 3.3 La cible future et les dernières lignes

La cible est une somme sur les sept prochains jours. Les six dernières lignes du dataset n'ont pas de futur complet et sont supprimées par `create_features()`. Sans cette suppression, la cible contiendrait des `NaN` ou des valeurs partielles.

Le split hérite de cette préparation. Le jeu de test ne contient que des lignes dont la cible est complète.

### 3.4 Les métriques

Le modèle est évalué avec trois métriques :

```python
mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
wmape = (
    np.sum(np.abs(y_test - predictions))
    / np.sum(y_test)
    * 100
)
```

Le MAE donne l'erreur moyenne en unités. Le RMSE pénalise davantage les grosses erreurs. Le WMAPE rapporte l'erreur au volume total, ce qui est utile lorsque les produits n'ont pas la même échelle.

Sur le jeu synthétique du pipeline, le WMAPE tourne autour de quelques pourcents. Ce chiffre ne doit pas être extrapolé à des données réelles : il mesure la capacité du modèle à apprendre un générateur synthétique, pas sa performance en pharmacie réelle.

### 3.5 Ce que le split ne couvre pas encore

Le split actuel est un holdout chronologique unique. Il ne fait pas de validation croisée temporelle glissante (*walk-forward*). Une validation plus robuste entraînerait plusieurs modèles sur des fenêtres successives et mesurerait la stabilité des métriques.

Le projet reste volontairement simple. La priorité de la phase P1 était la parité et la traçabilité, pas l'optimisation exhaustive du protocole d'évaluation.

---

<a id="4-entrainement-xgboost-et-tracabilite-mlflow"></a>
## 4. Entraînement XGBoost et traçabilité MLflow

### 4.1 Le choix du modèle

XGBoost est un modèle d'arbres boostés, adapté aux données tabulaires et aux relations non linéaires. Il gère bien les interactions entre le stock, le délai fournisseur et les lags. Il est aussi rapide à entraîner sur un jeu de quelques dizaines de milliers de lignes.

Le projet utilise un seul modèle global, entraîné sur toutes les officines et tous les produits. Ce choix simplifie le serving et le monitoring. Il a une limite : le modèle ne peut pas apprendre un niveau de base spécifique à un produit en dehors de ce que les lags révèlent.

### 4.2 Les hyperparamètres

```python
model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    base_score=0.5,
    random_state=42,
)
```

`n_estimators=150` fixe le nombre d'arbres. `learning_rate=0.05` contrôle la contribution de chaque arbre. `max_depth=5` limite la complexité. `random_state=42` rend l'entraînement reproductible.

`base_score=0.5` est un choix de compatibilité avec SHAP. Il évite un crash de parsing sur certaines versions de XGBoost. Ce n'est pas un paramètre métier.

### 4.3 La sérialisation de l'artefact

Le modèle est sauvegardé avec `joblib` :

```python
model_path = Path(MODEL_PATH)
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
```

`MODEL_PATH` vient de `src/config.py`. Le chemin est absolu par défaut, ce qui évite les ambiguïtés entre le répertoire courant et la racine du projet.

L'API charge cet artefact au démarrage, dans le `lifespan`. Un seul chargement suffit pour toutes les requêtes.

### 4.4 MLflow en opt-in

MLflow est activé par `MLFLOW_ENABLED=true`. Le module `src/train.py` journalise :

- les paramètres d'entraînement ;
- les métriques MAE, RMSE et WMAPE ;
- une signature de modèle ;
- un exemple d'entrée ;
- l'artefact du modèle ;
- le nom du modèle enregistré dans le registry.

La signature est construite avec `infer_signature(sample, model.predict(sample))`. Elle décrit le schéma d'entrée et de sortie. C'est un contrat de plus, cette fois au niveau du modèle.

Le registry permet de versionner les modèles et de les promouvoir. Le projet utilise `MLFLOW_REGISTERED_MODEL_NAME` pour nommer l'entrée. Le chemin d'artefact est défini par `MLFLOW_ARTIFACT_PATH`.

### 4.5 Pourquoi MLflow reste optionnel

Un serveur de tracking peut être indisponible. Un registry peut être mal configuré. Si l'entraînement dépendait de MLflow, une panne externe bloquerait la production du modèle.

Le module `_log_to_mlflow()` capture les exceptions et journalise un avertissement. L'artefact local est déjà écrit avant l'appel. Le pipeline continue même si MLflow échoue.

Ce choix a un coût : un run peut exister localement sans être tracé dans MLflow. En production, il faut surveiller ce cas et alerter si le tracking échoue de manière répétée.

### 4.6 Le versioning du modèle

Le modèle local est écrasé à chaque entraînement. Le registry MLflow conserve l'historique des versions. En production, le registry doit être la source de vérité, et l'artefact local un cache de serving.

La réponse `/predict` expose `model_version`, alimenté par `MODEL_VERSION` dans `src/config.py`. Cette valeur est un label d'API. Elle ne remplace pas l'identifiant de version du registry.

### 4.7 Les paramètres et métriques journalisés

Le module construit un dictionnaire de paramètres :

```python
params = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 5,
    "base_score": 0.5,
    "random_state": 42,
    "feature_count": len(feature_cols),
}
```

Les métriques sont converties en `float` avant journalisation. Cette conversion évite les types NumPy que certains backends MLflow sérialisent mal.

Le run est identifié par `run_id`. Si MLflow est activé et que le run réussit, l'identifiant est ajouté au dictionnaire de métriques retourné par `train_xgboost`. Cela permet de relier un entraînement à son run.

---

<a id="5-tests-et-preuve-de-parite"></a>
## 5. Tests et preuve de parité

### 5.1 Le principe du test

`tests/test_parity.py` construit une série synthétique de 120 jours, applique `create_features()`, puis sélectionne une ligne. Il reconstruit l'historique disponible avant cette ligne et appelle `build_history_features()`.

Les deux résultats doivent être identiques sur :

- `month`, `dayofweek`, `is_weekend` ;
- `lag_7`, `lag_14`, `lag_30` ;
- `rolling_mean_7`, `rolling_mean_30`.

La comparaison utilise `pytest.approx` avec une tolérance très faible. Une divergence de formule fait échouer le test.

### 5.2 Le jeu de données du test

La série n'est pas constante. Elle mélange une tendance, un cycle hebdomadaire et un décalage :

```python
sales = [
    100.0 + (index % 7) * 3.0 + index * 0.5
    for index in range(days)
]
```

Une série constante aurait masqué les erreurs d'alignement. Une série variable oblige le test à vérifier la bonne position des lags et des fenêtres.

### 5.3 Le test d'historique insuffisant

Un second test vérifie qu'un historique de moins de trente valeurs lève une `ValueError`. Cette contrainte protège le serving : l'API ne peut pas calculer un lag de 30 avec dix valeurs.

Pydantic applique également `min_length=30` sur le champ `history`. La validation se fait donc à deux niveaux : contrat HTTP et constructeur de features.

### 5.4 Ce que le test ne couvre pas

Le test de parité compare les formules sur une série propre. Il ne teste pas :

- les trous de données ;
- les ruptures prolongées ;
- les valeurs extrêmes ;
- les produits avec des historiques très courts ;
- les changements de fréquence.

Ces cas devraient être ajoutés si le projet évolue vers des données réelles. La parité de formule est une condition nécessaire, pas une garantie de robustesse sur toutes les distributions.

### 5.5 Le rôle du test dans la CI

Le test de parité est exécuté avec la suite unitaire. Une modification de `src/features.py` qui casse la correspondance entre entraînement et inférence fait échouer la CI avant la fusion.

C'est exactement le type de régression difficile à détecter manuellement. Le test transforme une intuition en preuve exécutable.

---

<a id="6-annexes-et-checklist-mlops"></a>
## 6. Annexes et checklist MLOps

### 6.1 Commandes d'entraînement

```bash
# Entraînement standard
python -m src.train

# Entraînement avec MLflow
MLFLOW_ENABLED=true python -m src.train

# Interface MLflow locale
mlflow ui --backend-store-uri file:./mlruns

# Pipeline complet
python run_pipeline.py
```

### 6.2 Inspection des artefacts

```bash
# Vérifier la présence du modèle
ls -lh models/

# Charger le modèle et afficher son type
python - <<'PY'
import joblib
from src.config import MODEL_PATH
model = joblib.load(MODEL_PATH)
print(type(model).__name__)
print(getattr(model, "n_features_in_", "unknown"))
PY

# Vérifier les features attendues
python -c "from src.config import FEATURE_COLS; print(FEATURE_COLS)"
```

### 6.3 Checklist de revue d'un pipeline ML

- [ ] Les features calendaires sont calculées à partir de la date de référence.
- [ ] Les lags sont calculés par groupe `(store_id, product_id)`.
- [ ] Les moyennes glissantes utilisent `shift(1)` pour éviter la fuite.
- [ ] La cible est une somme future sur sept jours, sans `NaN`.
- [ ] Les lignes de warm-up et de fin de série sont supprimées.
- [ ] Le split est chronologique et se fait sur les dates, pas sur les lignes.
- [ ] Le jeu de test commence après la dernière date d'entraînement.
- [ ] `random_state=42` est fixé.
- [ ] Les métriques MAE, RMSE et WMAPE sont calculées sur le test.
- [ ] L'artefact est sérialisé dans `models/`.
- [ ] MLflow est optionnel et ne bloque pas l'entraînement.
- [ ] La signature du modèle est journalisée lorsque MLflow est actif.
- [ ] `tests/test_parity.py` passe.
- [ ] La couverture de tests reste au-dessus du seuil de 70 %.

### 6.4 Glossaire

| Terme | Définition |
|---|---|
| **Lag** | Valeur observée à un décalage temporel donné |
| **Moyenne glissante** | Moyenne calculée sur une fenêtre de jours précédents |
| **Demande censurée** | Demande réelle partiellement observée à cause d'une rupture |
| **Train/serve skew** | Écart de calcul des features entre entraînement et inférence |
| **Data leakage** | Utilisation d'information future dans l'entraînement |
| **Split chronologique** | Découpage qui respecte l'ordre temporel |
| **WMAPE** | Erreur absolue pondérée par le volume total |
| **Signature de modèle** | Description du schéma d'entrée et de sortie |
| **Model Registry** | Catalogue versionné des modèles |
| **Holdout** | Jeu de test conservé pour l'évaluation finale |

### 6.5 Fichiers de code à lire

- `src/features.py` : `create_features()` et `build_history_features()`.
- `src/train.py` : split, XGBoost, métriques, sérialisation et MLflow.
- `tests/test_parity.py` : preuve de parité train/serve.
- `tests/test_features.py` : tests des lags, moyennes, groupes et cible.
- `tests/test_train.py` : split temporel, métriques et sérialisation.
- `src/config.py` : `FEATURE_COLS`, `TARGET_COL`, `MODEL_PATH`, paramètres MLflow.
- `models/xgboost_pharma_demand.joblib` : artefact produit par l'entraînement.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *A guide to turning a time series into reliable features, avoiding temporal leakage and tracing every training run.*

---

## Table of Contents

1. [From Raw Table to Temporal Features](#1-from-raw-table-to-temporal-features)
2. [The Train/Serve Skew Trap](#2-the-trainserve-skew-trap)
3. [Temporal Validation and Data Leakage](#3-temporal-validation-and-data-leakage)
4. [XGBoost Training and MLflow Traceability](#4-xgboost-training-and-mlflow-traceability)
5. [Tests and Parity Proof](#5-tests-and-parity-proof)
6. [Appendices and MLOps Checklist](#6-appendices-and-mlops-checklist)

---

<a id="1-from-raw-table-to-temporal-features"></a>
## 1. From Raw Table to Temporal Features

### 1.1 Why a tabular model needs memory

XGBoost does not understand a time series natively. It sees rows and columns. To anticipate future demand, the past must be turned into explicit variables: how much was sold seven days ago, what is the thirty-day average, is it a weekend?

`src/features.py` performs this transformation. It does not merely add columns: it defines the exact contract that training and inference must follow.

### 1.2 Calendar features

The first features are simple:

```python
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
```

December does not have the same demand as July. Saturday does not behave like Tuesday. These variables give the model minimal temporal context.

### 1.3 Reconstructing censored demand

A stockout distorts sales. If stock is empty, sales drop to zero even when demand was high. Using `sales_volume` directly as a demand signal would teach the model that demand falls during stockouts.

The module therefore creates a demand signal:

```python
df["demand_signal"] = np.where(
    df["is_stockout"] == 1, np.nan, df["sales_volume"]
)
df["demand_signal"] = (
    df.groupby(group_cols)["demand_signal"].ffill().bfill()
)
```

During a stockout, the value becomes `NaN`, then is propagated from the last valid sale. `bfill()` covers series that start with a stockout. This is simple and effective for an educational project. It has a limitation: `bfill()` can use a future value to fill an initial gap. A more rigorous censored-demand model would be needed for real data, but it is not necessary at the current level.

### 1.4 Lags

Lags represent demand observed seven, fourteen and thirty days earlier.

```python
for lag in [7, 14, 30]:
    df[f"lag_{lag}"] = df.groupby(group_cols)[
        "demand_signal"
    ].shift(lag)
```

The groupby is essential. Without it, a product’s lag could be computed from another product’s sales. Tests explicitly verify this isolation.

A lag of 7 is the value observed seven days before the current row. It is not a seven-day sum. This distinction matters for interpreting features and building inference.

### 1.5 Rolling means

Rolling means smooth noise and provide a local trend.

```python
for window in [7, 30]:
    df[f"rolling_mean_{window}"] = df.groupby(
        group_cols
    )["demand_signal"].transform(
        lambda x: x.shift(1).rolling(window).mean()
    )
```

`shift(1)` avoids using the current day. Without it, the mean would include the target or a contemporary value, introducing leakage.

### 1.6 The seven-day cumulative target

The model does not predict today’s demand. It predicts cumulative demand over the next seven days.

```python
df[TARGET_COL] = df.groupby(group_cols)[
    "demand_signal"
].transform(
    lambda x: x.rolling(7, min_periods=7).sum().shift(-6)
)
```

`rolling(7).sum()` computes a seven-day sum. `shift(-6)` aligns that sum with the first day of the window. The current row therefore contains demand for days `J` through `J+6`.

The last six rows have no complete future. They are dropped together with rows that do not yet have thirty days of history:

```python
df = df.dropna(
    subset=["lag_30", TARGET_COL]
).reset_index(drop=True)
```

The training dataset therefore contains only rows with complete features and target.

### 1.7 The feature contract

`FEATURE_COLS` comes from `src/config.py`. It is used by training, serving and monitoring. The columns are:

- `month`, `dayofweek`, `is_weekend`;
- `stock_on_hand`, `supplier_lead_time`;
- `lag_7`, `lag_14`, `lag_30`;
- `rolling_mean_7`, `rolling_mean_30`.

The model does not see `store_id` or `product_id`. It learns global relationships from lags and context. This is a modeling limitation: a per-product model or categorical encoding could capture specific effects. The current choice favors simplicity and robustness on a synthetic dataset.

---

<a id="2-the-trainserve-skew-trap"></a>
## 2. The Train/Serve Skew Trap

### 2.1 Definition

Train/serve skew is the gap between features computed during training and those computed during inference. It is not a model difference. It is an input difference.

A model can be excellent and degrade in production simply because the API sends features that no longer resemble those seen during training.

### 2.2 The example that triggered the refactor

In an early version, the API computed:

```python
rolling_mean_7 = lag_7 / 7.0
```

Training computed:

```python
rolling_mean_7 = demand.shift(1).rolling(7).mean()
```

Both formulas produce an average, but not the same one. The first divides a lag by seven. The second averages the last seven observed days, excluding the current day. The model received a shifted distribution.

The problem is hard to see in a notebook because the API is not running. It becomes visible in production, when predictions look strange without any error being raised.

### 2.3 The shared builder

`src/features.py` exposes `build_history_features()`:

```python
def build_history_features(
    sales_history: Sequence[float],
    as_of: Optional[pd.Timestamp] = None,
    min_history: int = 30,
) -> dict[str, float]:
    ...
```

The function takes a chronological sales history and a reference date. It computes next-day features with the same formulas as `create_features()`:

- `lag_7` is the value observed seven days before the target date;
- `lag_14` and `lag_30` follow the same logic;
- `rolling_mean_7` is the mean of the previous seven days;
- `rolling_mean_30` is the mean of the previous thirty days;
- calendar features come from `as_of`.

The API uses this builder whenever a history is provided. The `predict` router does not reimplement the formulas. It calls the service, which calls the builder.

### 2.4 Why one builder changes everything

A shared builder reduces the number of places where a formula can diverge. It also makes parity testable: the output of `create_features()` and `build_history_features()` can be compared on the same series.

The builder enforces a quality constraint: it requires at least thirty history values. Without it, lags would be incomplete and rolling means would use a partial window.

### 2.5 The legacy path

The `/predict` contract still accepts pre-computed lags without history. This path exists to avoid breaking existing clients. It computes rolling means from the provided lags, with the limitations that implies.

This path should be considered transitional. The recommended route is to provide `history`, because it benefits from tested parity. A client still using lags alone must be identified and migrated.

---

<a id="3-temporal-validation-and-data-leakage"></a>
## 3. Temporal Validation and Data Leakage

### 3.1 Why a random split is forbidden

On a time series, a random split places future observations in training and past observations in test. The model can then learn patterns that did not exist at prediction time.

The result is flattering and false. The model looks performant because it has seen the future. In production, it will only see the past.

### 3.2 The chronological split in src/train.py

`src/train.py` sorts unique dates and cuts at 80%:

```python
unique_dates = np.sort(df["date"].unique())
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

train_df = df[df["date"] < split_date]
test_df = df[df["date"] >= split_date]
```

The split is on dates, not rows. All observations from the same date remain in the same set. This prevents one product from being in training and another in test for the same day.

The test set starts strictly after the last training date. The temporal boundary is clean.

### 3.3 The future target and the last rows

The target is a sum over the next seven days. The last six rows have no complete future and are removed by `create_features()`. Without that removal, the target would contain `NaN` or partial values.

The split inherits this preparation. The test set contains only rows with a complete target.

### 3.4 Metrics

The model is evaluated with three metrics:

```python
mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
wmape = (
    np.sum(np.abs(y_test - predictions))
    / np.sum(y_test)
    * 100
)
```

MAE gives the average error in units. RMSE penalizes large errors more. WMAPE reports error relative to total volume, which is useful when products have different scales.

On the synthetic pipeline dataset, WMAPE is around a few percent. That number must not be extrapolated to real data: it measures how well the model learns a synthetic generator, not its performance in a real pharmacy.

### 3.5 What the split does not cover yet

The current split is a single chronological holdout. It does not perform walk-forward validation. A more robust validation would train several models on successive windows and measure metric stability.

The project remains intentionally simple. P1 prioritized parity and traceability, not exhaustive evaluation protocol optimization.

---

<a id="4-xgboost-training-and-mlflow-traceability"></a>
## 4. XGBoost Training and MLflow Traceability

### 4.1 Model choice

XGBoost is a gradient-boosted tree model suited to tabular data and non-linear relationships. It handles interactions between stock, supplier lead time and lags well. It also trains quickly on a few tens of thousands of rows.

The project uses a single global model trained on all pharmacies and products. This simplifies serving and monitoring. It has a limitation: the model cannot learn a product-specific baseline outside what lags reveal.

### 4.2 Hyperparameters

```python
model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    base_score=0.5,
    random_state=42,
)
```

`n_estimators=150` sets the number of trees. `learning_rate=0.05` controls each tree’s contribution. `max_depth=5` limits complexity. `random_state=42` makes training reproducible.

`base_score=0.5` is a SHAP compatibility choice. It avoids a parsing crash on some XGBoost versions. It is not a business parameter.

### 4.3 Artifact serialization

The model is saved with `joblib`:

```python
model_path = Path(MODEL_PATH)
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
```

`MODEL_PATH` comes from `src/config.py`. The path is absolute by default, avoiding ambiguity between the current directory and the project root.

The API loads this artifact at startup, in the `lifespan`. One load is enough for all requests.

### 4.4 MLflow opt-in

MLflow is enabled by `MLFLOW_ENABLED=true`. `src/train.py` logs:

- training parameters;
- MAE, RMSE and WMAPE metrics;
- a model signature;
- an input example;
- the model artifact;
- the registered model name.

The signature is built with `infer_signature(sample, model.predict(sample))`. It describes the input and output schema. It is one more contract, this time at the model level.

The registry versions and promotes models. The project uses `MLFLOW_REGISTERED_MODEL_NAME` to name the entry. The artifact path is defined by `MLFLOW_ARTIFACT_PATH`.

### 4.5 Why MLflow remains optional

A tracking server can be unavailable. A registry can be misconfigured. If training depended on MLflow, an external outage would block model production.

`_log_to_mlflow()` catches exceptions and logs a warning. The local artifact is already written before the call. The pipeline continues even if MLflow fails.

This choice has a cost: a run can exist locally without being tracked in MLflow. In production, this case must be monitored and alerted if tracking fails repeatedly.

### 4.6 Model versioning

The local model is overwritten on every training run. MLflow registry keeps version history. In production, the registry should be the source of truth, and the local artifact a serving cache.

The `/predict` response exposes `model_version`, supplied by `MODEL_VERSION` in `src/config.py`. This value is an API label. It does not replace the registry version identifier.

### 4.7 Logged parameters and metrics

The module builds a parameter dictionary:

```python
params = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 5,
    "base_score": 0.5,
    "random_state": 42,
    "feature_count": len(feature_cols),
}
```

Metrics are converted to `float` before logging. This avoids NumPy types that some MLflow backends serialize poorly.

The run is identified by `run_id`. If MLflow is enabled and the run succeeds, the identifier is added to the metrics dictionary returned by `train_xgboost`. This links a training run to its MLflow record.

---

<a id="5-tests-and-parity-proof"></a>
## 5. Tests and Parity Proof

### 5.1 Test principle

`tests/test_parity.py` builds a 120-day synthetic series, applies `create_features()`, then selects a row. It reconstructs the available history before that row and calls `build_history_features()`.

Both results must be identical on:

- `month`, `dayofweek`, `is_weekend`;
- `lag_7`, `lag_14`, `lag_30`;
- `rolling_mean_7`, `rolling_mean_30`.

The comparison uses `pytest.approx` with a very small tolerance. Any formula divergence fails the test.

### 5.2 Test dataset

The series is not constant. It mixes a trend, a weekly cycle and an offset:

```python
sales = [
    100.0 + (index % 7) * 3.0 + index * 0.5
    for index in range(days)
]
```

A constant series would hide alignment errors. A variable series forces the test to verify the correct position of lags and windows.

### 5.3 Insufficient history test

A second test verifies that a history shorter than thirty values raises `ValueError`. This constraint protects serving: the API cannot compute a lag of 30 with ten values.

Pydantic also applies `min_length=30` on the `history` field. Validation therefore happens at two levels: HTTP contract and feature builder.

### 5.4 What the test does not cover

The parity test compares formulas on a clean series. It does not test:

- data gaps;
- prolonged stockouts;
- extreme values;
- products with very short histories;
- frequency changes.

These cases should be added if the project moves to real data. Formula parity is a necessary condition, not a robustness guarantee across all distributions.

### 5.5 Role in CI

The parity test runs with the unit suite. A change to `src/features.py` that breaks the training/inference correspondence fails CI before merge.

This is exactly the kind of regression that is hard to detect manually. The test turns an intuition into executable proof.

---

<a id="6-appendices-and-mlops-checklist"></a>
## 6. Appendices and MLOps Checklist

### 6.1 Training commands

```bash
# Standard training
python -m src.train

# Training with MLflow
MLFLOW_ENABLED=true python -m src.train

# Local MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Full pipeline
python run_pipeline.py
```

### 6.2 Artifact inspection

```bash
# Check the model
ls -lh models/

# Load the model and print its type
python - <<'PY'
import joblib
from src.config import MODEL_PATH
model = joblib.load(MODEL_PATH)
print(type(model).__name__)
print(getattr(model, "n_features_in_", "unknown"))
PY

# Check expected features
python -c "from src.config import FEATURE_COLS; print(FEATURE_COLS)"
```

### 6.3 ML pipeline review checklist

- [ ] Calendar features are computed from the reference date.
- [ ] Lags are computed per `(store_id, product_id)` group.
- [ ] Rolling means use `shift(1)` to avoid leakage.
- [ ] The target is a future seven-day sum without `NaN`.
- [ ] Warm-up and end-of-series rows are removed.
- [ ] The split is chronological and based on dates, not rows.
- [ ] The test set starts after the last training date.
- [ ] `random_state=42` is fixed.
- [ ] MAE, RMSE and WMAPE are computed on test data.
- [ ] The artifact is serialized in `models/`.
- [ ] MLflow is optional and does not block training.
- [ ] The model signature is logged when MLflow is active.
- [ ] `tests/test_parity.py` passes.
- [ ] Test coverage remains above the 70% threshold.

### 6.4 Glossary

| Term | Definition |
|---|---|
| **Lag** | Value observed at a given time offset |
| **Rolling mean** | Mean computed over a window of previous days |
| **Censored demand** | Real demand partially observed because of a stockout |
| **Train/serve skew** | Difference in feature computation between training and inference |
| **Data leakage** | Use of future information during training |
| **Chronological split** | Split that respects temporal order |
| **WMAPE** | Absolute error weighted by total volume |
| **Model signature** | Description of the input and output schema |
| **Model Registry** | Versioned catalogue of models |
| **Holdout** | Test set kept for final evaluation |

### 6.5 Code files to read

- `src/features.py`: `create_features()` and `build_history_features()`.
- `src/train.py`: split, XGBoost, metrics, serialization and MLflow.
- `tests/test_parity.py`: train/serve parity proof.
- `tests/test_features.py`: tests for lags, means, groups and target.
- `tests/test_train.py`: temporal split, metrics and serialization.
- `src/config.py`: `FEATURE_COLS`, `TARGET_COL`, `MODEL_PATH`, MLflow parameters.
- `models/xgboost_pharma_demand.joblib`: artifact produced by training.

</details>
