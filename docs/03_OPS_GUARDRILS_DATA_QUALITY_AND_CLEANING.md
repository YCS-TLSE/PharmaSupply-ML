# Technical Documentation
## Phase 03: Ops Guardrails, Data Quality and Cleaning
**Project:** PharmaSupply-ML
**Framework:** Standard MLOps Blueprint
**File:** `docs/03_OPS_GUARDRILS_DATA_QUALITY_AND_CLEANING.md`
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

*Un guide pour placer des garde-fous automatisés entre l'ingestion brute et le feature engineering, sans masquer les pannes ni inventer des données.*

---

## Sommaire

1. [Pourquoi un modèle a besoin d'un Quality Gate](#1-pourquoi-un-modele-a-besoin-dun-quality-gate)
2. [Hard Fail vs Soft Correction](#2-hard-fail-vs-soft-correction)
3. [Anatomie de src/validation.py](#3-anatomie-de-srcvalidationpy)
4. [Impact sur le pipeline E2E](#4-impact-sur-le-pipeline-e2e)
5. [Tests et non-régression de la qualité](#5-tests-et-non-regression-de-la-qualite)
6. [Annexes et checklist qualité](#6-annexes-et-checklist-qualite)

---

<a id="1-pourquoi-un-modele-a-besoin-dun-quality-gate"></a>
## 1. Pourquoi un modèle a besoin d'un Quality Gate

### 1.1 Le modèle ne voit pas la qualité des données

Un modèle XGBoost ne sait pas qu'un stock de -150 unités est impossible. Il apprend une relation mathématique entre des nombres. Si ces nombres sont faux, il apprend une relation fausse, avec une confiance qui peut être élevée.

Le problème est plus vicieux qu'il n'y paraît. Un modèle entraîné sur des données corrompues peut afficher un MAE correct sur un jeu de test lui aussi corrompu. La métrique rassure, mais la décision métier est mauvaise.

En pharmacie, les conséquences sont concrètes :

- un stock négatif peut fausser le calcul de réapprovisionnement ;
- une demande aberrante peut déclencher une commande excessive ;
- un MOQ invalide peut produire une quantité impossible à commander ;
- une date manquante peut casser la logique temporelle des lags ;
- une rupture de contrat de colonne peut faire échouer l'entraînement ou le serving.

Le quality gate sert à intercepter ces cas avant qu'ils n'atteignent le modèle.

### 1.2 Le bon emplacement dans le pipeline

Le filtre se place entre l'ingestion brute et le feature engineering.

```text
CSV ou source réelle
        │
        v
src/ingestion.py          ← normalisation, typage, UPSERT
        │
        v
src/validation.py         ← quality gate : hard fail / soft correction
        │
        v
src/features.py           ← lags, rolling means, cible
        │
        v
src/train.py              ← entraînement XGBoost
```

Placé plus tôt, le gate ne verrait pas les types définitifs. Placé plus tard, il laisserait des données corrompues entrer dans les transformations. L'emplacement actuel est le bon compromis : les colonnes sont normalisées, mais les features ne sont pas encore calculées.

### 1.3 Les trois familles de problèmes

Le module traite trois familles de problèmes.

La première est **structurelle** : une table vide, une colonne obligatoire absente, un schéma qui ne correspond plus au contrat. Ce type de problème ne se répare pas à la volée. Il doit bloquer.

La deuxième est **numérique** : un stock négatif, une demande négative, un délai fournisseur nul. Ces valeurs sont physiquement impossibles ou métierement invalides. Elles peuvent être corrigées selon une règle explicite.

La troisième est **logistique** : un MOQ inférieur à 1, une taille de lot nulle, une durée de conservation absente. Ces valeurs ont des bornes simples et des valeurs par défaut connues.

Le quality gate applique un traitement différent selon la famille. C'est ce qui évite de mélanger réparation et camouflage.

### 1.4 Ce que le gate ne doit pas faire

Un quality gate ne doit pas inventer une réalité. Il peut remplacer un MOQ manquant par 1, car la valeur par défaut est documentée. Il ne peut pas inventer une date, une officine ou une demande.

Il ne doit pas non plus absorber une panne d'ingestion. Si PostgreSQL est indisponible ou si le fichier source est vide, le pipeline doit s'arrêter avec un message clair. Une correction silencieuse donnerait l'illusion d'un run réussi.

La frontière est simple : on corrige une valeur aberrante isolée, on bloque une corruption systémique.

---

<a id="2-hard-fail-vs-soft-correction"></a>
## 2. Hard Fail vs Soft Correction

### 2.1 Deux philosophies, un seul module

`src/validation.py` applique deux modes.

Le **hard fail** arrête l'exécution. Il est réservé aux problèmes qui rendent le dataset inutilisable : table vide, colonnes obligatoires absentes.

La **soft correction** modifie la valeur et continue. Elle est réservée aux anomalies bornées et réparables : stock négatif ramené à zéro, MOQ nul ramené à 1.

Le choix entre les deux n'est pas une question de goût. Il dépend de la question suivante : peut-on reconstruire une valeur fiable à partir d'une règle métier explicite ?

Si la réponse est non, il faut bloquer. Si la réponse est oui, on peut corriger, à condition de tracer la correction.

### 2.2 Le tableau des décisions

| Situation | Mode | Règle appliquée | Justification |
|---|---|---|---|
| DataFrame vide | Hard fail | `ValueError` | Aucune observation exploitable |
| Colonne obligatoire absente | Hard fail | `ValueError` | Le contrat de données est rompu |
| `stock_on_hand < 0` | Soft correction | Ramené à 0 | Un stock physique ne peut pas être négatif |
| `target_demand < 0` | Soft correction | Ramené à 0 | Une demande négative n'a pas de sens |
| `sales_volume < 0` | Soft correction | Ramené à 0 | Une vente négative n'a pas de sens |
| `supplier_lead_time <= 0` | Soft correction | Remplacé par 3.0 | Délai strictement positif requis |
| `safety_threshold <= 0` | Soft correction | Remplacé par 150.0 | Seuil de sécurité strictement positif |
| `batch_size < 1` | Soft correction | Remplacé par 1 | Taille de lot au moins égale à 1 |
| `moq < 1` | Soft correction | Remplacé par 1 | MOQ au moins égal à 1 |
| `shelf_life_days < 1` | Soft correction | Remplacé par 365 | Durée de conservation positive |
| `critical_expiry_days < 0` | Soft correction | Ramené à 0 | Un délai restant ne peut pas être négatif |
| `sales_volume > target_demand` | Soft correction | Ventes plafonnées à la demande | Cohérence croisée |

Les valeurs par défaut viennent de `src/config.py` : `DEFAULT_SUPPLIER_LEAD_TIME`, `DEFAULT_SAFETY_THRESHOLD`, `DEFAULT_BATCH_SIZE`, `DEFAULT_MOQ`, `DEFAULT_SHELF_LIFE_DAYS`. Les dupliquer dans le module de validation aurait créé deux sources de vérité.

### 2.3 Les colonnes obligatoires

Le contrat minimum est le suivant :

```python
required_cols = [
    "date",
    "store_id",
    "product_id",
    "target_demand",
    "sales_volume",
    "stock_on_hand",
    "supplier_lead_time",
    "safety_threshold",
]
```

Si l'une de ces colonnes manque, le gate lève une `ValueError`. Il ne tente pas de la fabriquer. Une colonne absente signifie que le schéma source a changé ou que l'ingestion a échoué.

Les colonnes logistiques (`batch_size`, `moq`, `shelf_life_days`, `critical_expiry_days`) sont traitées séparément. Elles ne bloquent pas si elles manquent, car des valeurs par défaut métier existent. Cette distinction entre obligatoire et optionnel est essentielle dans un contrat de données.

### 2.4 Les valeurs manquantes

Le module applique une imputation simple :

- pour les colonnes numériques obligatoires, `fillna(0)` ;
- pour les colonnes texte obligatoires, `fillna("UNKNOWN")`.

Cette approche évite un crash immédiat. Elle a toutefois une limite qu'il faut connaître : elle peut masquer une panne en amont. Un `target_demand` manquant rempli à 0 n'est pas une observation valide ; c'est une valeur inventée.

L'imputation de `date` avec `"UNKNOWN"` est particulièrement sensible. En aval, `src/train.py` convertit la colonne avec `pd.to_datetime`. Une date `"UNKNOWN"` fera échouer l'entraînement. Une évolution raisonnable consiste à traiter une date manquante comme un hard fail, car sans date il n'y a ni lag ni split temporel.

L'imputation actuelle est donc un filet de sécurité de premier niveau, pas une stratégie statistique avancée. Elle doit être complétée par des contrôles de complétude et des logs exploitables.

### 2.5 Ce que la soft correction ne doit pas devenir

La soft correction est utile lorsqu'elle applique une règle métier stable. Elle devient dangereuse lorsqu'elle sert à faire passer un pipeline coûte que coûte.

Trois signaux doivent alerter :

- un taux de correction qui augmente fortement d'un run à l'autre ;
- une correction qui touche une colonne critique sans que personne ne la voie ;
- une valeur par défaut appliquée massivement, signe d'un problème de source plutôt que d'une anomalie isolée.

Un gate qui corrige 15 330 lignes sur 32 850 n'a pas seulement corrigé des valeurs. Il a révélé un problème de génération ou de contrat. Le pipeline doit continuer, mais l'équipe doit être informée.

---

<a id="3-anatomie-de-srcvalidationpy"></a>
## 3. Anatomie de src/validation.py

### 3.1 L'entrée et la sortie

La fonction publique est `run_validation_and_clean(input_df=None)`. Elle accepte deux modes :

- un DataFrame fourni, utilisé par les tests unitaires ;
- aucun DataFrame, auquel cas elle lit `raw_pharmaceutical_demand` via SQLAlchemy.

Le retour est un DataFrame nettoyé. Le module ne modifie pas la base. Il prépare les données pour les étapes suivantes.

Cette séparation permet de tester toutes les règles sans PostgreSQL. C'est un point important pour un Data Scientist : la logique de nettoyage doit être testable en isolation.

### 3.2 Les contrôles structurels

Le module commence par les contrôles bloquants.

```python
if len(df) == 0:
    raise ValueError("Data staging table is empty!")
```

Puis il vérifie les colonnes obligatoires :

```python
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(
        f"Missing mandatory columns in dataset: {missing_cols}"
    )
```

Ces deux contrôles suffisent à éviter les cas où le feature engineering travaillerait sur un dataset vide ou incomplet. Ils produisent un message explicite, ce qui facilite le diagnostic dans `run_pipeline.py`.

### 3.3 La complétude et l'imputation

Le module parcourt les colonnes obligatoires et compte les valeurs nulles. Pour chaque colonne concernée, il applique une imputation et affiche un avertissement.

```python
if df[col].dtype in ["int64", "float64"]:
    df[col] = df[col].fillna(0)
else:
    df[col] = df[col].fillna("UNKNOWN")
```

Cette étape est volontairement simple. Elle évite les erreurs de type en aval, mais elle ne distingue pas une valeur manquante isolée d'une absence systémique. Le message affiché est donc important : il donne une piste d'investigation.

### 3.4 Les colonnes logistiques optionnelles

Les colonnes `batch_size`, `moq`, `shelf_life_days` et `critical_expiry_days` sont ajoutées si elles manquent, avec des valeurs par défaut. Si elles existent, les valeurs nulles sont remplacées.

```python
if "batch_size" not in df.columns:
    df["batch_size"] = DEFAULT_BATCH_SIZE
else:
    df["batch_size"] = df["batch_size"].fillna(DEFAULT_BATCH_SIZE).astype(int)
```

Le même schéma s'applique au MOQ, à la durée de conservation et à l'expiration critique. Le choix de convertir en `int` évite de propager des flottants dans des colonnes qui représentent des unités ou des jours.

### 3.5 Les corrections numériques

Le module applique ensuite des corrections explicites, avec un compteur et un message pour chaque règle.

**Demande négative.**

```python
df.loc[df["target_demand"] < 0, "target_demand"] = 0
```

**Ventes négatives.**

```python
df.loc[df["sales_volume"] < 0, "sales_volume"] = 0
```

**Stock négatif.**

```python
df.loc[df["stock_on_hand"] < 0, "stock_on_hand"] = 0
```

**Délai fournisseur invalide.**

```python
df.loc[df["supplier_lead_time"] <= 0, "supplier_lead_time"] = 3
```

**Seuil de sécurité invalide.**

```python
df.loc[df["safety_threshold"] <= 0, "safety_threshold"] = 150.0
```

Les valeurs `3` et `150.0` sont remplacées dans le code par les constantes de `src/config.py`. Le document les présente ici pour rendre la règle lisible.

### 3.6 Les corrections logistiques

Les bornes logistiques sont simples :

```python
df.loc[df["batch_size"] < 1, "batch_size"] = 1
df.loc[df["moq"] < 1, "moq"] = 1
df.loc[df["shelf_life_days"] < 1, "shelf_life_days"] = 365
df.loc[df["critical_expiry_days"] < 0, "critical_expiry_days"] = 0
```

Une taille de lot ou un MOQ inférieur à 1 n'a pas de sens pour une commande. Une durée de conservation nulle ou négative non plus. L'expiration critique, en revanche, peut être nulle : un lot peut expirer aujourd'hui.

### 3.7 La cohérence croisée

La dernière règle compare deux colonnes :

```python
invalid_sales_mask = df["sales_volume"] > df["target_demand"]
df.loc[invalid_sales_mask, "sales_volume"] = df.loc[
    invalid_sales_mask, "target_demand"
]
```

L'idée est qu'une vente ne peut pas dépasser la demande cible. Cette règle est discutable si la demande cible est une prévision et non une demande observée. Dans le schéma actuel, `target_demand` est la cible de staging ; `src/features.py` recalcule ensuite une cible cumulée sur sept jours. La validation garantit donc une cohérence au niveau brut, avant transformation.

### 3.8 Les limites de l'implémentation

Le module utilise `print()` pour signaler les corrections. Dans un service de production, ces messages devraient être des logs structurés et des métriques : nombre de lignes corrigées, colonne concernée, run concerné.

L'imputation `fillna(0)` sur des colonnes obligatoires est un choix de résilience, pas un choix statistique. Elle peut déformer la distribution si elle est massive. Une évolution consisterait à distinguer :

- les colonnes à hard fail si nulles (`date`, `store_id`, `product_id`) ;
- les colonnes à imputation contrôlée (`safety_threshold`, `supplier_lead_time`) ;
- les colonnes à traitement métier spécifique (`stock_on_hand`, `sales_volume`).

Le module ne détecte pas encore les doublons sur la clé métier. Cette responsabilité est portée par la contrainte d'unicité PostgreSQL et par l'UPSERT de `src/ingestion.py`.

---

<a id="4-impact-sur-le-pipeline-e2e"></a>
## 4. Impact sur le pipeline E2E

### 4.1 Le gate dans run_pipeline.py

`run_pipeline.py` appelle `run_validation_and_clean()` après l'ingestion.

```python
df_raw = run_validation_and_clean()
df_raw["date"] = pd.to_datetime(df_raw["date"])
df_processed = create_features(df_raw)
```

Si le gate lève une `ValueError`, l'étape 2 échoue. Le pipeline capture l'exception, journalise l'erreur et appelle `sys.exit(1)`. Aucun entraînement n'est lancé sur des données invalides.

C'est le comportement attendu : un échec explicite vaut mieux qu'un modèle entraîné sur un dataset incomplet.

### 4.2 Le gate dans l'entraînement

`src/train.py` charge les données via `run_validation_and_clean()` dans `load_validated_data()`. Le gate est donc appliqué même lorsque l'entraînement est lancé seul, sans passer par l'orchestrateur.

Cette double utilisation est volontaire. Le gate doit être une propriété du chargement des données, pas une étape optionnelle que l'on peut oublier.

### 4.3 Pourquoi un échec explicite est préférable

Une correction silencieuse peut donner un pipeline vert et un modèle mauvais. Un échec explicite arrête la chaîne, mais il protège la décision.

Prenons un exemple. Si la colonne `date` disparaît du CSV, une imputation silencieuse remplirait la colonne avec une valeur factice. L'entraînement échouerait plus tard, ou pire, produirait un modèle sans logique temporelle. Un hard fail immédiat indique clairement que le contrat d'entrée est rompu.

Le même raisonnement s'applique à une table vide. Continuer avec zéro ligne n'a aucun sens. Le message d'erreur doit être visible dans les logs du pipeline.

### 4.4 Le cas des corrections massives

Le gate affiche des compteurs, par exemple :

```text
Auto-correction: Fixed 15330 invalid batch sizes (set to default: 1).
Auto-correction: Fixed 15330 invalid MOQ values (set to default: 1).
```

Ces chiffres ne sont pas anodins. Ils indiquent que la moitié du dataset a été corrigée. Le pipeline continue, mais l'équipe doit se demander pourquoi. Une valeur par défaut appliquée massivement peut signaler un problème de génération, un changement de source ou une colonne mal nommée.

Un gate n'est pas seulement un filtre. C'est aussi un instrument de mesure de la qualité de la chaîne d'ingestion.

---

<a id="5-tests-et-non-regression-de-la-qualite"></a>
## 5. Tests et non-régression de la qualité

### 5.1 Ce que couvre tests/test_validation.py

Le fichier `tests/test_validation.py` contient une dizaine de tests qui verrouillent les règles du gate.

| Test | Ce qu'il vérifie |
|---|---|
| `test_validation_nominal` | Un dataset valide passe sans modification de structure |
| `test_validation_hard_fail_missing_columns` | Une colonne obligatoire absente déclenche une `ValueError` |
| `test_validation_hard_fail_empty_dataframe` | Un DataFrame vide déclenche une `ValueError` |
| `test_validation_soft_correction_negative_stock` | Un stock négatif est ramené à 0 |
| `test_validation_soft_correction_negative_demand_and_sales` | Demande et ventes négatives ramenées à 0 |
| `test_validation_soft_correction_invalid_moq` | MOQ et batch size invalides corrigés à au moins 1 |
| `test_validation_soft_correction_invalid_lead_time_and_threshold` | Délai et seuil non positifs remplacés par les défauts |
| `test_validation_cross_column_sales_exceeding_demand` | Ventes plafonnées à la demande cible |
| `test_validation_completeness_null_imputation` | Valeurs nulles imputées selon le type |
| `test_validation_type_casting` | Les colonnes clés restent présentes et typées |

Ces tests sont rapides. Ils n'ont besoin ni de PostgreSQL ni de FastAPI. C'est exactement ce que l'on attend d'un test de règle métier.

### 5.2 Pourquoi ces tests sont indispensables

Les règles de validation sont faciles à casser sans le voir. Une refonte de l'ingestion peut renommer une colonne, une optimisation peut supprimer un `fillna`, une nouvelle valeur par défaut peut écraser une constante.

Sans tests, le pipeline peut continuer à tourner et produire des données silencieusement différentes. Les tests de validation empêchent cette dérive.

Ils jouent aussi un rôle de documentation. Un nouveau développeur peut lire `tests/test_validation.py` et comprendre les règles sans parcourir tout le module.

### 5.3 Les tests de non-régression dans la CI

Le workflow de CI exécute `ruff` puis `pytest` avec couverture. Le seuil de couverture est de 70 %, et la suite complète atteint 80,91 % sur le périmètre `src/`.

Les tests de validation font partie de cette suite. Une modification qui casse une règle de qualité fait échouer la CI avant la fusion.

### 5.4 Ce qui n'est pas encore couvert

Les tests actuels vérifient les règles, mais pas les volumes. Ils ne mesurent pas le temps de traitement sur 32 850 lignes ni le taux de correction par run. Ces métriques seraient utiles pour détecter une dérive de qualité.

Ils ne testent pas non plus l'imputation d'une date manquante. Comme indiqué plus haut, ce cas mérite probablement un hard fail plutôt qu'un `"UNKNOWN"`. Un test dédié permettrait de trancher et de verrouiller la décision.

---

<a id="6-annexes-et-checklist-qualite"></a>
## 6. Annexes et checklist qualité

### 6.1 Règles clés à retenir

| Règle | Comportement |
|---|---|
| Dataset vide | Hard fail |
| Colonne obligatoire absente | Hard fail |
| Valeur nulle dans une colonne obligatoire | Imputation contrôlée (0 ou `UNKNOWN`) |
| Stock, demande ou ventes négatifs | Ramenés à 0 |
| Délai fournisseur non positif | Remplacé par 3.0 |
| Seuil de sécurité non positif | Remplacé par 150.0 |
| MOQ ou batch size inférieur à 1 | Remplacé par 1 |
| Shelf life inférieure à 1 | Remplacée par 365 |
| Expiration critique négative | Ramenée à 0 |
| Ventes supérieures à la demande | Plafonnées à la demande |

### 6.2 Commandes de test

```bash
# Tests de validation uniquement
pytest -q tests/test_validation.py

# Suite unitaire complète avec couverture
pytest -q -m "not integration" --cov=src --cov-report=term-missing

# Lint
ruff check src tests run_pipeline.py alembic
```

### 6.3 Checklist de revue d'un quality gate

- [ ] Les colonnes obligatoires sont explicitement listées.
- [ ] Un dataset vide déclenche une erreur bloquante.
- [ ] Une colonne obligatoire absente déclenche une erreur bloquante.
- [ ] Les valeurs par défaut viennent d'une source unique (`src/config.py`).
- [ ] Les corrections numériques ont des bornes métier explicites.
- [ ] Les corrections logistiques respectent MOQ, batch size et shelf life.
- [ ] Les valeurs manquantes sont traitées sans masquer une panne systémique.
- [ ] Les corrections sont comptées et visibles dans les logs.
- [ ] Une date manquante est traitée comme un cas bloquant ou fortement signalé.
- [ ] Les règles sont couvertes par des tests unitaires.
- [ ] La CI exécute ces tests à chaque modification.
- [ ] Le gate est appelé à la fois par le pipeline et par l'entraînement.

### 6.4 Glossaire

| Terme | Définition |
|---|---|
| **Quality gate** | Filtre automatisé qui contrôle la qualité avant une étape critique |
| **Hard fail** | Erreur bloquante qui arrête le pipeline |
| **Soft correction** | Correction automatique d'une anomalie bornée, avec poursuite du pipeline |
| **Imputation** | Remplacement d'une valeur manquante par une valeur choisie |
| **Contrat de données** | Ensemble des colonnes, types et règles attendus en entrée |
| **Valeur aberrante** | Valeur hors des bornes métier plausibles |
| **Non-régression** | Garantie qu'une modification ne casse pas un comportement existant |
| **Cible cumulée** | Demande agrégée sur une fenêtre future, ici sept jours |

### 6.5 Fichiers de code à lire

- `src/validation.py` : quality gate, hard fail et soft correction.
- `src/config.py` : constantes de bornes et valeurs par défaut.
- `tests/test_validation.py` : tests de non-régression des règles.
- `src/ingestion.py` : normalisation et UPSERT en amont du gate.
- `src/features.py` : transformations appliquées après validation.
- `src/train.py` : chargement validé avant entraînement.
- `run_pipeline.py` : orchestration et arrêt explicite en cas d'échec.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *A guide to placing automated guardrails between raw ingestion and feature engineering, without hiding outages or inventing data.*

---

## Table of Contents

1. [Why a Model Needs a Quality Gate](#1-why-a-model-needs-a-quality-gate)
2. [Hard Fail vs Soft Correction](#2-hard-fail-vs-soft-correction)
3. [Anatomy of src/validation.py](#3-anatomy-of-srcvalidationpy)
4. [Impact on the End-to-End Pipeline](#4-impact-on-the-end-to-end-pipeline)
5. [Tests and Quality Non-Regression](#5-tests-and-quality-non-regression)
6. [Appendices and Quality Checklist](#6-appendices-and-quality-checklist)

---

<a id="1-why-a-model-needs-a-quality-gate"></a>
## 1. Why a Model Needs a Quality Gate

### 1.1 The model does not see data quality

An XGBoost model does not know that a stock of -150 units is impossible. It learns a mathematical relationship between numbers. If those numbers are wrong, it learns a wrong relationship, sometimes with high confidence.

The problem is subtler than it looks. A model trained on corrupted data can show a good MAE on a test set that is also corrupted. The metric reassures, but the business decision is wrong.

In pharmacy, the consequences are concrete:

- a negative stock can distort replenishment;
- an aberrant demand can trigger an excessive order;
- an invalid MOQ can produce an impossible quantity;
- a missing date can break the temporal logic of lags;
- a column contract break can fail training or serving.

The quality gate intercepts these cases before they reach the model.

### 1.2 The right place in the pipeline

The filter sits between raw ingestion and feature engineering.

```text
CSV or real source
        │
        v
src/ingestion.py          ← normalization, typing, UPSERT
        │
        v
src/validation.py         ← quality gate: hard fail / soft correction
        │
        v
src/features.py           ← lags, rolling means, target
        │
        v
src/train.py              ← XGBoost training
```

Earlier, the gate would not see final types. Later, it would let corrupted data enter transformations. The current position is the right trade-off: columns are normalized, but features are not computed yet.

### 1.3 Three families of problems

The module handles three families.

The first is **structural**: an empty table, a missing mandatory column, a schema that no longer matches the contract. This cannot be repaired on the fly. It must block.

The second is **numeric**: a negative stock, a negative demand, a zero supplier lead time. These values are physically impossible or business-invalid. They can be corrected through an explicit rule.

The third is **logistical**: a MOQ below 1, a zero batch size, a missing shelf life. These values have simple bounds and known defaults.

The quality gate applies a different treatment per family. That is what prevents repair from becoming camouflage.

### 1.4 What the gate must not do

A quality gate must not invent reality. It can replace a missing MOQ with 1 because the default is documented. It cannot invent a date, a pharmacy or a demand.

It must not absorb an ingestion outage either. If PostgreSQL is unavailable or the source file is empty, the pipeline must stop with a clear message. A silent correction would give the illusion of a successful run.

The boundary is simple: correct an isolated aberrant value, block systemic corruption.

---

<a id="2-hard-fail-vs-soft-correction"></a>
## 2. Hard Fail vs Soft Correction

### 2.1 Two philosophies, one module

`src/validation.py` applies two modes.

**Hard fail** stops execution. It is reserved for problems that make the dataset unusable: empty table, missing mandatory columns.

**Soft correction** changes the value and continues. It is reserved for bounded, repairable anomalies: negative stock set to zero, null MOQ set to 1.

The choice is not a matter of taste. It depends on one question: can we reconstruct a reliable value from an explicit business rule?

If the answer is no, block. If yes, correct, provided the correction is traced.

### 2.2 The decision table

| Situation | Mode | Applied rule | Rationale |
|---|---|---|---|
| Empty DataFrame | Hard fail | `ValueError` | No usable observation |
| Missing mandatory column | Hard fail | `ValueError` | Data contract is broken |
| `stock_on_hand < 0` | Soft correction | Set to 0 | Physical stock cannot be negative |
| `target_demand < 0` | Soft correction | Set to 0 | Negative demand is meaningless |
| `sales_volume < 0` | Soft correction | Set to 0 | Negative sales are meaningless |
| `supplier_lead_time <= 0` | Soft correction | Set to 3.0 | Strictly positive lead time required |
| `safety_threshold <= 0` | Soft correction | Set to 150.0 | Strictly positive safety stock required |
| `batch_size < 1` | Soft correction | Set to 1 | Batch size at least 1 |
| `moq < 1` | Soft correction | Set to 1 | MOQ at least 1 |
| `shelf_life_days < 1` | Soft correction | Set to 365 | Positive shelf life required |
| `critical_expiry_days < 0` | Soft correction | Set to 0 | Remaining time cannot be negative |
| `sales_volume > target_demand` | Soft correction | Cap sales at demand | Cross-column consistency |

Defaults come from `src/config.py`: `DEFAULT_SUPPLIER_LEAD_TIME`, `DEFAULT_SAFETY_THRESHOLD`, `DEFAULT_BATCH_SIZE`, `DEFAULT_MOQ`, `DEFAULT_SHELF_LIFE_DAYS`. Duplicating them in the validation module would create two sources of truth.

### 2.3 Mandatory columns

The minimum contract is:

```python
required_cols = [
    "date",
    "store_id",
    "product_id",
    "target_demand",
    "sales_volume",
    "stock_on_hand",
    "supplier_lead_time",
    "safety_threshold",
]
```

If one is missing, the gate raises a `ValueError`. It does not try to create it. A missing column means the source schema changed or ingestion failed.

Logistics columns (`batch_size`, `moq`, `shelf_life_days`, `critical_expiry_days`) are handled separately. They do not block when missing because business defaults exist. This distinction between mandatory and optional is essential in a data contract.

### 2.4 Missing values

The module applies a simple imputation:

- mandatory numeric columns: `fillna(0)`;
- mandatory text columns: `fillna("UNKNOWN")`.

This avoids an immediate crash. It has a limitation: it can hide an upstream outage. A missing `target_demand` filled with 0 is not a valid observation; it is an invented value.

Imputing `date` with `"UNKNOWN"` is especially sensitive. Downstream, `src/train.py` converts the column with `pd.to_datetime`. An `"UNKNOWN"` date will fail training. A reasonable evolution is to treat a missing date as a hard fail, because without a date there is no lag and no temporal split.

The current imputation is therefore a first-level safety net, not an advanced statistical strategy. It must be complemented by completeness checks and exploitable logs.

### 2.5 What soft correction must not become

Soft correction is useful when it applies a stable business rule. It becomes dangerous when it is used to make a pipeline pass at all costs.

Three signals should alert:

- a correction rate rising sharply between runs;
- a correction on a critical column without anyone seeing it;
- a default value applied massively, indicating a source problem rather than an isolated anomaly.

A gate that corrects 15,330 rows out of 32,850 did not only correct values. It revealed a generation or contract problem. The pipeline may continue, but the team must be informed.

---

<a id="3-anatomy-of-srcvalidationpy"></a>
## 3. Anatomy of src/validation.py

### 3.1 Input and output

The public function is `run_validation_and_clean(input_df=None)`. It supports two modes:

- a provided DataFrame, used by unit tests;
- no DataFrame, in which case it reads `raw_pharmaceutical_demand` through SQLAlchemy.

The return value is a cleaned DataFrame. The module does not modify the database. It prepares data for the next steps.

This separation allows every rule to be tested without PostgreSQL. That matters for a Data Scientist: cleaning logic must be testable in isolation.

### 3.2 Structural checks

The module starts with blocking checks.

```python
if len(df) == 0:
    raise ValueError("Data staging table is empty!")
```

Then it checks mandatory columns:

```python
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(
        f"Missing mandatory columns in dataset: {missing_cols}"
    )
```

These two checks prevent feature engineering from working on an empty or incomplete dataset. They produce an explicit message, which helps diagnosis in `run_pipeline.py`.

### 3.3 Completeness and imputation

The module iterates over mandatory columns and counts nulls. For each affected column, it applies imputation and prints a warning.

```python
if df[col].dtype in ["int64", "float64"]:
    df[col] = df[col].fillna(0)
else:
    df[col] = df[col].fillna("UNKNOWN")
```

This step is deliberately simple. It avoids downstream type errors, but it does not distinguish an isolated missing value from a systemic absence. The printed message is therefore important: it provides an investigation lead.

### 3.4 Optional logistics columns

`batch_size`, `moq`, `shelf_life_days` and `critical_expiry_days` are added when missing, with defaults. When present, null values are replaced.

```python
if "batch_size" not in df.columns:
    df["batch_size"] = DEFAULT_BATCH_SIZE
else:
    df["batch_size"] = df["batch_size"].fillna(DEFAULT_BATCH_SIZE).astype(int)
```

The same pattern applies to MOQ, shelf life and critical expiry. Casting to `int` avoids propagating floats in columns that represent units or days.

### 3.5 Numeric corrections

The module then applies explicit corrections, with a counter and a message per rule.

**Negative demand.**

```python
df.loc[df["target_demand"] < 0, "target_demand"] = 0
```

**Negative sales.**

```python
df.loc[df["sales_volume"] < 0, "sales_volume"] = 0
```

**Negative stock.**

```python
df.loc[df["stock_on_hand"] < 0, "stock_on_hand"] = 0
```

**Invalid lead time.**

```python
df.loc[df["supplier_lead_time"] <= 0, "supplier_lead_time"] = 3
```

**Invalid safety threshold.**

```python
df.loc[df["safety_threshold"] <= 0, "safety_threshold"] = 150.0
```

In the code, `3` and `150.0` are replaced by constants from `src/config.py`. They are shown here to make the rule readable.

### 3.6 Logistics corrections

Logistics bounds are simple:

```python
df.loc[df["batch_size"] < 1, "batch_size"] = 1
df.loc[df["moq"] < 1, "moq"] = 1
df.loc[df["shelf_life_days"] < 1, "shelf_life_days"] = 365
df.loc[df["critical_expiry_days"] < 0, "critical_expiry_days"] = 0
```

A batch size or MOQ below 1 makes no sense for an order. A zero or negative shelf life does not either. Critical expiry, however, can be zero: a batch may expire today.

### 3.7 Cross-column consistency

The last rule compares two columns:

```python
invalid_sales_mask = df["sales_volume"] > df["target_demand"]
df.loc[invalid_sales_mask, "sales_volume"] = df.loc[
    invalid_sales_mask, "target_demand"
]
```

The idea is that sales cannot exceed target demand. This rule is debatable if the target is a forecast rather than observed demand. In the current schema, `target_demand` is the staging target; `src/features.py` then recomputes a seven-day cumulative target. Validation therefore enforces consistency at the raw level, before transformation.

### 3.8 Implementation limits

The module uses `print()` to report corrections. In a production service, these messages should be structured logs and metrics: number of corrected rows, affected column, run identifier.

`fillna(0)` on mandatory columns is a resilience choice, not a statistical one. It can distort the distribution if applied massively. An evolution would distinguish:

- columns that should hard fail when null (`date`, `store_id`, `product_id`);
- columns with controlled imputation (`safety_threshold`, `supplier_lead_time`);
- columns with specific business treatment (`stock_on_hand`, `sales_volume`).

The module does not yet detect duplicates on the business key. That responsibility is carried by the PostgreSQL unique constraint and the UPSERT in `src/ingestion.py`.

---

<a id="4-impact-on-the-end-to-end-pipeline"></a>
## 4. Impact on the End-to-End Pipeline

### 4.1 The gate in run_pipeline.py

`run_pipeline.py` calls `run_validation_and_clean()` after ingestion.

```python
df_raw = run_validation_and_clean()
df_raw["date"] = pd.to_datetime(df_raw["date"])
df_processed = create_features(df_raw)
```

If the gate raises a `ValueError`, step 2 fails. The pipeline catches the exception, logs the error and calls `sys.exit(1)`. No training runs on invalid data.

This is the expected behavior: an explicit failure is better than a model trained on an incomplete dataset.

### 4.2 The gate in training

`src/train.py` loads data through `run_validation_and_clean()` in `load_validated_data()`. The gate is therefore applied even when training runs alone, without the orchestrator.

This dual use is intentional. The gate must be a property of data loading, not an optional step that can be forgotten.

### 4.3 Why explicit failure is preferable

A silent correction can produce a green pipeline and a bad model. An explicit failure stops the chain, but it protects the decision.

Consider an example. If the `date` column disappears from the CSV, silent imputation would fill it with a fake value. Training would fail later, or worse, produce a model without temporal logic. An immediate hard fail clearly indicates that the input contract is broken.

The same reasoning applies to an empty table. Continuing with zero rows makes no sense. The error message must be visible in pipeline logs.

### 4.4 The case of mass corrections

The gate prints counters, for example:

```text
Auto-correction: Fixed 15330 invalid batch sizes (set to default: 1).
Auto-correction: Fixed 15330 invalid MOQ values (set to default: 1).
```

These numbers are not trivial. They indicate that half the dataset was corrected. The pipeline continues, but the team must ask why. A default applied massively can signal a generation problem, a source change or a misnamed column.

A gate is not only a filter. It is also a measurement instrument for the quality of the ingestion chain.

---

<a id="5-tests-and-quality-non-regression"></a>
## 5. Tests and Quality Non-Regression

### 5.1 What tests/test_validation.py covers

`tests/test_validation.py` contains about ten tests that lock the gate rules.

| Test | What it verifies |
|---|---|
| `test_validation_nominal` | A valid dataset passes without structural change |
| `test_validation_hard_fail_missing_columns` | A missing mandatory column raises `ValueError` |
| `test_validation_hard_fail_empty_dataframe` | An empty DataFrame raises `ValueError` |
| `test_validation_soft_correction_negative_stock` | Negative stock is set to 0 |
| `test_validation_soft_correction_negative_demand_and_sales` | Negative demand and sales are set to 0 |
| `test_validation_soft_correction_invalid_moq` | Invalid MOQ and batch size are corrected to at least 1 |
| `test_validation_soft_correction_invalid_lead_time_and_threshold` | Non-positive lead time and threshold use defaults |
| `test_validation_cross_column_sales_exceeding_demand` | Sales are capped at target demand |
| `test_validation_completeness_null_imputation` | Null values are imputed according to type |
| `test_validation_type_casting` | Key columns remain present and typed |

These tests are fast. They need neither PostgreSQL nor FastAPI. That is exactly what a business-rule test should be.

### 5.2 Why these tests are indispensable

Validation rules are easy to break without noticing. An ingestion refactor can rename a column, an optimization can remove a `fillna`, a new default can override a constant.

Without tests, the pipeline can keep running and produce silently different data. Validation tests prevent this drift.

They also act as documentation. A new developer can read `tests/test_validation.py` and understand the rules without scanning the whole module.

### 5.3 Non-regression tests in CI

The CI workflow runs `ruff` then `pytest` with coverage. The coverage threshold is 70%, and the full suite reaches 80.91% on the `src/` scope.

Validation tests are part of this suite. A change that breaks a quality rule fails CI before merge.

### 5.4 What is not covered yet

Current tests verify rules, not volumes. They do not measure processing time on 32,850 rows or the correction rate per run. Such metrics would help detect quality drift.

They also do not test imputation of a missing date. As noted above, this case probably deserves a hard fail rather than `"UNKNOWN"`. A dedicated test would settle the decision and lock it.

---

<a id="6-appendices-and-quality-checklist"></a>
## 6. Appendices and Quality Checklist

### 6.1 Key rules

| Rule | Behavior |
|---|---|
| Empty dataset | Hard fail |
| Missing mandatory column | Hard fail |
| Null value in a mandatory column | Controlled imputation (0 or `UNKNOWN`) |
| Negative stock, demand or sales | Set to 0 |
| Non-positive supplier lead time | Replaced by 3.0 |
| Non-positive safety threshold | Replaced by 150.0 |
| MOQ or batch size below 1 | Replaced by 1 |
| Shelf life below 1 | Replaced by 365 |
| Negative critical expiry | Set to 0 |
| Sales above demand | Capped at demand |

### 6.2 Test commands

```bash
# Validation tests only
pytest -q tests/test_validation.py

# Full unit suite with coverage
pytest -q -m "not integration" --cov=src --cov-report=term-missing

# Lint
ruff check src tests run_pipeline.py alembic
```

### 6.3 Quality gate review checklist

- [ ] Mandatory columns are explicitly listed.
- [ ] An empty dataset triggers a blocking error.
- [ ] A missing mandatory column triggers a blocking error.
- [ ] Default values come from a single source (`src/config.py`).
- [ ] Numeric corrections have explicit business bounds.
- [ ] Logistics corrections respect MOQ, batch size and shelf life.
- [ ] Missing values are handled without hiding a systemic outage.
- [ ] Corrections are counted and visible in logs.
- [ ] A missing date is treated as blocking or strongly signalled.
- [ ] Rules are covered by unit tests.
- [ ] CI runs these tests on every change.
- [ ] The gate is called by both the pipeline and training.

### 6.4 Glossary

| Term | Definition |
|---|---|
| **Quality gate** | Automated filter that checks quality before a critical step |
| **Hard fail** | Blocking error that stops the pipeline |
| **Soft correction** | Automatic correction of a bounded anomaly, with pipeline continuation |
| **Imputation** | Replacing a missing value with a chosen value |
| **Data contract** | Expected columns, types and rules at input |
| **Aberrant value** | Value outside plausible business bounds |
| **Non-regression** | Guarantee that a change does not break existing behavior |
| **Cumulative target** | Demand aggregated over a future window, here seven days |

### 6.5 Code files to read

- `src/validation.py`: quality gate, hard fail and soft correction.
- `src/config.py`: bound constants and default values.
- `tests/test_validation.py`: non-regression tests for the rules.
- `src/ingestion.py`: normalization and UPSERT upstream of the gate.
- `src/features.py`: transformations applied after validation.
- `src/train.py`: validated loading before training.
- `run_pipeline.py`: orchestration and explicit stop on failure.

</details>
