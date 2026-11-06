# Technical Documentation
## Phase 05: Explainability and Monitoring
**Project:** PharmaSupply-ML
**Framework:** Standard MLOps Blueprint
**File:** `docs/05_EXPLAINABILITY_AND_MONITORING.md`
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

*Un guide pour expliquer les prédictions, mesurer la dérive hors ligne et en production, et alerter sans pénaliser l'API.*

---

## Sommaire

1. [Pourquoi un modèle de production doit s'expliquer](#1-pourquoi-un-modele-de-production-doit-sexpliquer)
2. [Explicabilité et export des poids](#2-explicabilite-et-export-des-poids)
3. [Le monitoring hors ligne avec Evidently](#3-le-monitoring-hors-ligne-avec-evidently)
4. [Le suivi de production sur logs JSONL](#4-le-suivi-de-production-sur-logs-jsonl)
5. [Statut, watcher et alertes asynchrones](#5-statut-watcher-et-alertes-asynchrones)
6. [Annexes et checklist de surveillance](#6-annexes-et-checklist-de-surveillance)

---

<a id="1-pourquoi-un-modele-de-production-doit-sexpliquer"></a>
## 1. Pourquoi un modèle de production doit s'expliquer

### 1.1 Une prédiction n'est pas une justification

Un modèle XGBoost peut répondre « 740 unités » sans dire pourquoi. En laboratoire, cela suffit souvent. En production pharmaceutique, non.

Une officine qui reçoit une recommandation de commande doit pouvoir comprendre ce qui l'a déclenchée : un stock bas, un lag élevé, une tendance récente, un délai fournisseur long. Sans cette explication, la recommandation est difficile à accepter, à contester ou à corriger.

L'explicabilité ne sert pas seulement à la pédagogie. Elle sert à l'audit, à la confiance et à la détection d'erreurs. Un modèle qui s'appuie sur une feature absurde peut être repéré avant de provoquer une mauvaise décision.

### 1.2 Global et local

L'explicabilité globale répond à la question : quelles variables comptent le plus pour le modèle en moyenne ? Elle sert à comprendre le comportement général, à documenter le modèle et à alimenter le monitoring.

L'explicabilité locale répond à une autre question : pourquoi cette prédiction précise a-t-elle cette valeur ? Elle sert à justifier une décision individuelle.

Les deux niveaux sont complémentaires. Une importance globale peut masquer un comportement local aberrant. Une explication locale peut être difficile à généraliser. Le projet produit les deux.

### 1.3 SHAP en quelques mots

SHAP (*SHapley Additive exPlanations*) attribue à chaque feature une contribution à la prédiction. La contribution est calculée par rapport à une valeur de référence, souvent la moyenne des prédictions.

Pour un modèle d'arbres, `TreeExplainer` calcule ces contributions de manière exacte et rapide. C'est adapté à XGBoost.

Le résultat se lit comme une décomposition :

```text
prédiction = valeur de base + somme des contributions des features
```

Une contribution positive pousse la prédiction vers le haut. Une contribution négative la tire vers le bas. La valeur de base représente le comportement moyen du modèle.

### 1.4 Ce que SHAP ne dit pas

SHAP explique le modèle, pas le monde. Une feature peut avoir une forte contribution sans être causalement responsable de la demande. Une corrélation apprise par le modèle n'est pas une preuve de causalité.

Il faut donc présenter SHAP comme un outil d'audit du comportement du modèle, pas comme une explication métier définitive. La validation métier reste nécessaire.

---

<a id="2-explicabilite-et-export-des-poids"></a>
## 2. Explicabilité et export des poids

### 2.1 Anatomie de src/explainability.py

Le module s'articule autour de la classe `ModelExplainer`.

```python
class ModelExplainer:
    def __init__(self, model_path: str, output_dir: str = "reports/figures"):
        ...
```

Le constructeur charge le modèle sérialisé et prépare le dossier de sortie. Les méthodes couvrent quatre besoins :

- calculer les valeurs SHAP ;
- générer les graphiques globaux et locaux ;
- exporter la matrice SHAP en CSV ;
- exporter les poids normalisés en JSON.

Le module contient aussi un patch de compatibilité pour XGBoost récent. Il nettoie la valeur `base_score` lorsqu'elle est sérialisée sous forme de chaîne entre crochets. Ce patch évite un crash de `TreeExplainer` sur certaines versions.

### 2.2 Le sous-échantillonnage

Calculer SHAP sur 32 850 lignes n'est pas nécessaire pour produire un rapport lisible. Le module applique donc une stratégie de sous-échantillonnage.

| Paramètre | Valeur par défaut | Rôle |
|---|---|---|
| `threshold` | 300 | Nombre de lignes au-delà duquel on échantillonne |
| `sample_size` | 200 | Nombre de lignes par tirage |
| `n_repeats` | 1 | Nombre de tirages indépendants |

Avec `n_repeats=1`, le module fait un seul tirage déterministe (`random_state=42`). C'est le mode rapide, adapté au pipeline et au dashboard.

Avec `n_repeats>1`, il fait plusieurs tirages et moyenne les valeurs SHAP. C'est le mode audit, plus stable mais plus lent. Ce mode réduit la variance d'échantillonnage pour un rapport officiel.

### 2.3 Les graphiques

Le module génère trois artefacts :

- `shap_summary_beeswarm.png` : distribution et sens d'influence des features ;
- `shap_summary_bar.png` : importance globale moyenne ;
- `shap_individual_breakdown.png` : décomposition d'une prédiction individuelle.

Les figures sont écrites dans `reports/figures/`. Elles servent à la revue métier et à la documentation du modèle.

### 2.4 L'export CSV

La matrice SHAP est exportée dans `reports/figures/shap_values.csv`. Chaque ligne correspond à une observation échantillonnée, chaque colonne à une feature. Ce fichier permet une analyse plus fine, par exemple pour identifier les observations où une feature domine.

### 2.5 L'export des poids

La méthode `export_feature_weights()` calcule la moyenne des valeurs SHAP absolues par feature :

```python
mean_abs_shap = np.mean(np.abs(self.shap_values.values), axis=0)
```

Puis elle normalise les poids pour que leur somme vaille 1.0. Le résultat est écrit dans `reports/feature_weights.json`.

Un exemple de structure :

```json
{
  "lag_7": 0.25,
  "stock_on_hand": 0.20,
  "lag_14": 0.15,
  "lag_30": 0.10,
  "rolling_mean_7": 0.10,
  "rolling_mean_30": 0.10,
  "supplier_lead_time": 0.05,
  "month": 0.02,
  "dayofweek": 0.02,
  "is_weekend": 0.01
}
```

Ces poids ne sont pas décoratifs. Ils alimentent le calcul du drift pondéré dans `src/monitoring.py`. Une dérive sur `lag_7` compte plus qu'une dérive sur `is_weekend`.

### 2.6 Le repli sur DEFAULT_FEATURE_WEIGHTS

Le fichier `reports/feature_weights.json` peut être absent, corrompu ou vide. Dans ce cas, `src/monitoring.py` utilise `DEFAULT_FEATURE_WEIGHTS`, défini dans `src/config.py`.

Ce repli garantit que le monitoring continue de fonctionner même si l'export SHAP a échoué. Il a une limite : les poids par défaut sont des estimations de domaine, pas les poids réels du modèle. Il faut donc surveiller la présence et la fraîcheur du fichier JSON.

Un poids manquant ne doit pas arrêter une alerte de drift. Mais un repli permanent doit être visible dans les logs.

---

<a id="3-le-monitoring-hors-ligne-avec-evidently"></a>
## 3. Le monitoring hors ligne avec Evidently

### 3.1 À quoi sert le holdout chronologique

`run_drift_monitoring()` compare deux sous-ensembles d'un même dataset historique :

- une référence, correspondant aux 80 % les plus anciens ;
- un courant, correspondant aux 20 % les plus récents.

Le split se fait sur les dates, comme pour l'entraînement. Il ne mélange pas passé et futur.

Cette analyse répond à une question limitée : les distributions récentes du dataset historique diffèrent-elles des distributions anciennes ? Elle ne dit pas ce qui se passe en production. Elle sert de contrôle de régression et de premier signal.

### 3.2 Le rapport Evidently

Le module utilise `DataDriftPreset` :

```python
report = Report(metrics=[DataDriftPreset()])
evaluation = report.run(
    reference_data=reference_df,
    current_data=current_df,
)
```

Le rapport HTML est écrit dans `reports/drift_report.html`. Il contient les tests statistiques par colonne et le résumé global.

Les colonnes comparées sont `FEATURE_COLS` plus `TARGET_COL`. La cible est incluse pour détecter une dérive de distribution de la variable à prédire, même si Evidently ne mesure pas ici la performance du modèle.

### 3.3 L'extraction des drapeaux de drift

Evidently renvoie une structure imbriquée. `_extract_drift_by_column()` cherche la métrique `DatasetDriftMetric` et récupère `drift_by_columns`.

Chaque colonne contient notamment un booléen `drift_detected`. Le module s'appuie sur ce booléen, pas sur la valeur de la statistique.

Cette approche est simple, mais elle dépend du format interne d'Evidently. Un changement de version peut casser le parsing. Le module capture l'erreur et retourne un dictionnaire vide. Il faut donc surveiller les logs : un parsing vide produit un score de 0 et un statut `STABLE`, ce qui peut masquer une dérive.

### 3.4 Le score pondéré

`compute_weighted_drift_score()` calcule la part des poids associés à des features en dérive :

```python
drifting_weight = 0.0
for column, weight in feature_weights.items():
    if drift_by_column.get(column, {}).get("drift_detected", False):
        drifting_weight += weight
return drifting_weight / total_weight
```

Le score est donc une proportion pondérée, comprise entre 0 et 1. Il ne mesure pas l'ampleur de la dérive, seulement sa présence sur les features importantes.

### 3.5 Les seuils de statut

Les seuils viennent de `src/config.py` :

| Seuil | Valeur | Statut |
|---|---:|---|
| `DRIFT_WARNING_THRESHOLD` | 0.15 | `WARNING` |
| `DRIFT_CRITICAL_THRESHOLD` | 0.30 | `CRITICAL` |
| En dessous de 0.15 | — | `STABLE` |

Un score de 0.20 signifie que 20 % du poids des features est en dérive. Ce n'est pas un pourcentage de lignes affectées. C'est une synthèse pondérée par l'importance des features.

### 3.6 La persistance atomique du statut

Le statut est écrit dans `model_status.txt`. L'écriture utilise un fichier temporaire puis `os.replace()` :

```python
tmp_path = status_path.with_name(status_path.name + ".tmp")
tmp_path.write_text(status_value, encoding="utf-8")
os.replace(tmp_path, status_path)
```

Cette précaution évite qu'un lecteur voie un fichier partiellement écrit. Le module refuse aussi d'écraser un répertoire au même chemin, ce qui protège contre une erreur de configuration.

### 3.7 Les limites du monitoring hors ligne

Le holdout compare deux périodes du même dataset. Il ne voit pas les requêtes réelles. Une dérive peut apparaître en production sans être visible dans le holdout.

Le monitoring de production répond à cette limite, mais il dépend de la qualité du journal de prédiction.

---

<a id="4-le-suivi-de-production-sur-logs-jsonl"></a>
## 4. Le suivi de production sur logs JSONL

### 4.1 Le journal de prédiction

Chaque appel à `/predict` planifie une tâche d'arrière-plan qui écrit une ligne JSON dans `reports/prediction_log.jsonl`.

La ligne contient :

- l'horodatage ;
- l'officine et le produit ;
- la demande prédite ;
- l'indicateur de réapprovisionnement ;
- les features du modèle.

Le journal ne contient pas de donnée personnelle. Il ne contient pas non plus le payload brut ni la réponse complète. C'est un choix de minimisation.

### 4.2 L'écriture asynchrone

L'écriture est confiée à `BackgroundTasks` dans le router `predict`. La réponse HTTP n'attend pas l'écriture disque.

```python
background_tasks.add_task(
    record_prediction,
    features,
    predicted_demand,
    decision.reorder_needed,
    payload.store_id,
    payload.product_id,
)
```

Ce découplage protège la latence de l'API. Une écriture lente ne doit pas retarder une prédiction.

Le module `src/observability.py` protège l'écriture avec un verrou et capture les erreurs d'écriture. Une panne de disque ne doit pas casser l'inférence.

### 4.3 La lecture tolérante

`read_prediction_log()` lit le fichier ligne par ligne. Les lignes vides sont ignorées. Les lignes JSON invalides sont signalées et ignorées.

Cette tolérance est importante : un fichier partiellement écrit ou corrompu ne doit pas faire échouer tout le batch de monitoring.

### 4.4 Le minimum d'échantillons

`run_production_drift_monitoring()` exige un nombre minimum de lignes, configuré par `PRODUCTION_DRIFT_MIN_SAMPLES` (30 par défaut).

En dessous de ce seuil, la fonction retourne `UNKNOWN` sans écrire de nouveau statut. Un échantillon trop petit produirait un rapport instable, voire trompeur.

### 4.5 La comparaison avec la référence

Si le nombre de lignes est suffisant, le module construit un DataFrame courant à partir du JSONL. Il vérifie que toutes les colonnes de `FEATURE_COLS` sont présentes.

La référence vient de `create_features(load_validated_data())` ou d'un DataFrame fourni. Le rapport Evidently compare les distributions de features. La cible n'est pas disponible dans le journal de production, donc la comparaison porte sur les features.

Le rapport HTML est écrit dans `reports/production_drift_report.html`. Le statut est calculé avec les mêmes seuils que le monitoring hors ligne.

### 4.6 Les limites du suivi de production

Le journal contient les features envoyées à l'API. Si un client appelle `/predict` avec le chemin legacy (lags pré-calculés), les features enregistrées reflètent ce chemin, pas le constructeur partagé. Il faut donc privilégier le champ `history` pour que le journal soit représentatif.

Le journal n'est pas un échantillon aléatoire de la population. Il reflète les appels reçus. Un biais de sélection est possible si certains produits ou officines sont surreprésentés.

Enfin, la comparaison mesure la dérive des features, pas la performance réelle. Sans vérité terrain, on ne peut pas calculer un WMAPE de production. La dérive reste un signal indirect.

---

<a id="5-statut-watcher-et-alertes-asynchrones"></a>
## 5. Statut, watcher et alertes asynchrones

### 5.1 Les états du modèle

Le fichier `model_status.txt` peut contenir :

| Valeur | Signification |
|---|---|
| `STABLE` | Le modèle est considéré fiable |
| `WARNING` | Dérive modérée, vigilance requise |
| `CRITICAL` | Dérive forte, action corrective nécessaire |
| `UNKNOWN` | Statut illisible, absent ou invalide |

`UNKNOWN` n'est pas une valeur écrite par le monitoring. C'est une valeur de repli produite par `read_model_status()` lorsqu'il ne peut pas déterminer l'état.

Cette distinction est volontaire. Un fichier absent ne doit pas être présenté comme `STABLE`.

### 5.2 La lecture du statut

`read_model_status()` lit le fichier et applique trois règles :

- `"OK"` est accepté comme alias de `"STABLE"` pour compatibilité ;
- une valeur de `MODEL_STATUS_VALUES` est acceptée telle quelle ;
- toute autre valeur ou erreur de lecture produit `UNKNOWN`.

La fonction ne déclenche aucun e-mail. Elle est utilisée par `/model/health`, `/alerts` et `/predict`. Elle doit rester sans effet de bord.

### 5.3 Le watcher de transitions

`watch_model_status()` est une tâche asyncio démarrée dans le `lifespan` de l'API.

Elle lit le statut à intervalle régulier (`MODEL_STATUS_POLL_SECONDS`, 30 secondes par défaut) et envoie un e-mail uniquement lors d'une transition vers `WARNING` ou `CRITICAL`.

```python
if current_status in {"WARNING", "CRITICAL"}:
    if current_status != last_alerted_status:
        await asyncio.to_thread(send_support_email, current_status)
        last_alerted_status = current_status
else:
    last_alerted_status = None
```

Le watcher se souvient du dernier statut alerté. Un `WARNING` stable ne provoque pas un e-mail à chaque polling. Un retour à `STABLE` réinitialise la mémoire, ce qui permet une nouvelle alerte si le modèle dérive à nouveau.

### 5.4 Pourquoi le watcher est hors du chemin HTTP

Envoyer un e-mail SMTP peut prendre plusieurs centaines de millisecondes, voire plusieurs secondes. Si l'envoi était déclenché par `/predict`, chaque prédiction paierait ce coût.

Le watcher déplace l'alerte dans une tâche de fond. La requête HTTP lit simplement le statut et répond. La latence reste stable.

Le watcher est annulé proprement au shutdown de l'application. Une tâche de fond qui survivrait au processus serait une fuite.

### 5.5 L'envoi SMTP

`send_support_email()` construit un message différent selon le statut :

- `CRITICAL` : sujet d'alerte critique, action sous 24 heures ;
- `WARNING` : sujet de vigilance modérée.

Le serveur, le port et le timeout viennent de `src/config.py`. Le mot de passe vient de `SMTP_PASSWORD`.

Si le mot de passe est absent, la fonction journalise un avertissement et s'arrête. Elle ne lève pas d'exception. Une erreur SMTP est capturée et journalisée.

### 5.6 Les limites du dispositif d'alerte

Le watcher n'a pas d'outbox persistante. Si le processus redémarre, il peut réémettre une alerte pour un statut encore en `WARNING`. Ce comportement est acceptable pour un projet pédagogique, mais il faudrait une déduplication persistante en production.

Le watcher ne réessaie pas un envoi échoué. Une panne SMTP peut faire perdre une alerte. Un système de production devrait ajouter une file de messages et une politique de retry.

Le watcher lit un fichier local. Dans un déploiement multi-instances, chaque instance aurait son propre watcher et pourrait envoyer la même alerte. Une coordination externe serait nécessaire.

---

<a id="6-annexes-et-checklist-de-surveillance"></a>
## 6. Annexes et checklist de surveillance

### 6.1 Commandes d'exécution

```bash
# Explicabilité SHAP et export des poids
python -m src.explainability

# Monitoring hors ligne
python -m src.monitoring

# Pipeline complet (SHAP + drift offline + smoke tests)
python run_pipeline.py

# Tests ciblés
pytest -q tests/test_explainability.py
pytest -q tests/test_monitoring.py
pytest -q tests/test_observability.py
pytest -q tests/test_model_status.py
```

### 6.2 Inspection des rapports

```bash
# Rapport de drift hors ligne
ls -lh reports/drift_report.html

# Rapport de drift production
ls -lh reports/production_drift_report.html

# Poids SHAP
cat reports/feature_weights.json

# Journal de prédiction
wc -l reports/prediction_log.jsonl
head -n 2 reports/prediction_log.jsonl

# Figures SHAP
ls -lh reports/figures/
```

### 6.3 Checklist de surveillance

- [ ] Les figures SHAP globales et locales sont générées.
- [ ] `reports/feature_weights.json` existe et sa somme vaut 1.0.
- [ ] Le repli `DEFAULT_FEATURE_WEIGHTS` est journalisé s'il est utilisé.
- [ ] Le rapport Evidently hors ligne est généré.
- [ ] Le score pondéré est calculé avec les poids SHAP.
- [ ] Les seuils WARNING et CRITICAL sont ceux de `src/config.py`.
- [ ] `model_status.txt` est écrit de manière atomique.
- [ ] Le journal de prédiction contient toutes les features du contrat.
- [ ] Le nombre minimum d'échantillons de production est respecté.
- [ ] Le rapport de drift production est généré.
- [ ] Le statut `UNKNOWN` est correctement traité par l'API et le dashboard.
- [ ] Le watcher alerte uniquement sur transition.
- [ ] L'envoi SMTP possède un timeout.
- [ ] Une panne SMTP ne casse pas l'API.
- [ ] Le watcher est annulé au shutdown.

### 6.4 Glossaire

| Terme | Définition |
|---|---|
| **SHAP** | Méthode d'attribution des contributions des features à une prédiction |
| **Valeur de base** | Prédiction moyenne de référence utilisée par SHAP |
| **Importance globale** | Moyenne des contributions absolues sur un échantillon |
| **Explication locale** | Décomposition d'une prédiction individuelle |
| **Data drift** | Changement de distribution des données d'entrée |
| **Holdout** | Sous-ensemble conservé pour l'évaluation |
| **JSONL** | Format texte avec un objet JSON par ligne |
| **Watcher** | Tâche de fond qui surveille un état et déclenche une action |
| **Outbox** | File persistante de messages à envoyer |
| **Fail-safe** | Comportement prudent en cas d'incertitude, ici `UNKNOWN` |

### 6.5 Fichiers de code et artefacts à lire

- `src/explainability.py` : SHAP, graphiques, export CSV et poids JSON.
- `src/monitoring.py` : drift hors ligne, drift production, score pondéré et statut.
- `src/observability.py` : journal de prédiction JSONL et lecture tolérante.
- `src/services/model_status.py` : lecture du statut, watcher et SMTP.
- `src/config.py` : seuils, chemins, poids par défaut et paramètres SMTP.
- `src/api.py` : démarrage et arrêt du watcher dans le lifespan.
- `src/api_routers/predict.py` : planification de l'écriture asynchrone.
- `reports/feature_weights.json` : poids SHAP normalisés.
- `reports/prediction_log.jsonl` : journal agrégé de production.
- `reports/drift_report.html` et `reports/production_drift_report.html` : rapports Evidently.
- `tests/test_explainability.py`, `tests/test_monitoring.py`, `tests/test_observability.py`, `tests/test_model_status.py` : non-régression.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *A guide to explaining predictions, measuring offline and production drift, and alerting without slowing down the API.*

---

## Table of Contents

1. [Why a Production Model Must Explain Itself](#1-why-a-production-model-must-explain-itself)
2. [Explainability and Weight Export](#2-explainability-and-weight-export)
3. [Offline Monitoring with Evidently](#3-offline-monitoring-with-evidently)
4. [Production Tracking on JSONL Logs](#4-production-tracking-on-jsonl-logs)
5. [Status, Watcher and Asynchronous Alerts](#5-status-watcher-and-asynchronous-alerts)
6. [Appendices and Monitoring Checklist](#6-appendices-and-monitoring-checklist)

---

<a id="1-why-a-production-model-must-explain-itself"></a>
## 1. Why a Production Model Must Explain Itself

### 1.1 A prediction is not a justification

An XGBoost model can answer “740 units” without saying why. In a lab, that is often enough. In pharmaceutical production, it is not.

A pharmacy receiving a replenishment recommendation must understand what triggered it: low stock, a high lag, a recent trend, a long supplier lead time. Without that explanation, the recommendation is hard to accept, challenge or correct.

Explainability is not only pedagogical. It supports audit, trust and error detection. A model relying on an absurd feature can be spotted before it causes a bad decision.

### 1.2 Global and local

Global explainability answers: which variables matter most to the model on average? It helps understand overall behavior, document the model and feed monitoring.

Local explainability answers: why did this specific prediction take this value? It helps justify an individual decision.

The two levels are complementary. A global importance can hide an aberrant local behavior. A local explanation can be hard to generalize. The project produces both.

### 1.3 SHAP in a few words

SHAP (*SHapley Additive exPlanations*) assigns each feature a contribution to the prediction. The contribution is computed relative to a reference value, often the average prediction.

For tree models, `TreeExplainer` computes these contributions exactly and quickly. It is suited to XGBoost.

The result reads as a decomposition:

```text
prediction = base value + sum of feature contributions
```

A positive contribution pushes the prediction up. A negative contribution pulls it down. The base value represents the model’s average behavior.

### 1.4 What SHAP does not say

SHAP explains the model, not the world. A feature can have a strong contribution without being causally responsible for demand. A correlation learned by the model is not proof of causality.

SHAP must therefore be presented as an audit tool for model behavior, not as a definitive business explanation. Business validation remains necessary.

---

<a id="2-explainability-and-weight-export"></a>
## 2. Explainability and Weight Export

### 2.1 Anatomy of src/explainability.py

The module is organized around the `ModelExplainer` class.

```python
class ModelExplainer:
    def __init__(self, model_path: str, output_dir: str = "reports/figures"):
        ...
```

The constructor loads the serialized model and prepares the output directory. Methods cover four needs:

- compute SHAP values;
- generate global and local plots;
- export the SHAP matrix as CSV;
- export normalized weights as JSON.

The module also contains a compatibility patch for recent XGBoost. It cleans the `base_score` value when serialized as a bracketed string. This patch avoids a `TreeExplainer` crash on some versions.

### 2.2 Subsampling

Computing SHAP on 32,850 rows is not necessary for a readable report. The module applies a subsampling strategy.

| Parameter | Default | Role |
|---|---|---|
| `threshold` | 300 | Row count above which sampling starts |
| `sample_size` | 200 | Rows per draw |
| `n_repeats` | 1 | Number of independent draws |

With `n_repeats=1`, the module makes one deterministic draw (`random_state=42`). This is the fast mode, suitable for the pipeline and dashboard.

With `n_repeats>1`, it makes several draws and averages SHAP values. This is the audit mode, more stable but slower. It reduces sampling variance for an official report.

### 2.3 Plots

The module generates three artifacts:

- `shap_summary_beeswarm.png`: feature distribution and direction of influence;
- `shap_summary_bar.png`: average global importance;
- `shap_individual_breakdown.png`: decomposition of one individual prediction.

Figures are written to `reports/figures/`. They support business review and model documentation.

### 2.4 CSV export

The SHAP matrix is exported to `reports/figures/shap_values.csv`. Each row is a sampled observation, each column a feature. This file allows deeper analysis, for example identifying observations where one feature dominates.

### 2.5 Weight export

`export_feature_weights()` computes the mean absolute SHAP value per feature:

```python
mean_abs_shap = np.mean(np.abs(self.shap_values.values), axis=0)
```

It then normalizes weights so their sum equals 1.0. The result is written to `reports/feature_weights.json`.

Example structure:

```json
{
  "lag_7": 0.25,
  "stock_on_hand": 0.20,
  "lag_14": 0.15,
  "lag_30": 0.10,
  "rolling_mean_7": 0.10,
  "rolling_mean_30": 0.10,
  "supplier_lead_time": 0.05,
  "month": 0.02,
  "dayofweek": 0.02,
  "is_weekend": 0.01
}
```

These weights are not decorative. They feed the weighted drift calculation in `src/monitoring.py`. Drift on `lag_7` counts more than drift on `is_weekend`.

### 2.6 Fallback on DEFAULT_FEATURE_WEIGHTS

`reports/feature_weights.json` can be missing, corrupted or empty. In that case, `src/monitoring.py` uses `DEFAULT_FEATURE_WEIGHTS`, defined in `src/config.py`.

This fallback keeps monitoring running even if the SHAP export failed. It has a limitation: default weights are domain estimates, not the model’s real weights. The presence and freshness of the JSON file must therefore be monitored.

A missing weight must not stop a drift alert. But a permanent fallback must be visible in logs.

---

<a id="3-offline-monitoring-with-evidently"></a>
## 3. Offline Monitoring with Evidently

### 3.1 What the chronological holdout is for

`run_drift_monitoring()` compares two subsets of the same historical dataset:

- a reference, the oldest 80%;
- a current set, the most recent 20%.

The split is on dates, as for training. It does not mix past and future.

This analysis answers a limited question: do recent distributions in the historical dataset differ from older ones? It does not say what happens in production. It serves as a regression check and a first signal.

### 3.2 The Evidently report

The module uses `DataDriftPreset`:

```python
report = Report(metrics=[DataDriftPreset()])
evaluation = report.run(
    reference_data=reference_df,
    current_data=current_df,
)
```

The HTML report is written to `reports/drift_report.html`. It contains per-column statistical tests and a global summary.

Compared columns are `FEATURE_COLS` plus `TARGET_COL`. The target is included to detect distribution drift in the variable to predict, even though Evidently does not measure model performance here.

### 3.3 Extracting drift flags

Evidently returns a nested structure. `_extract_drift_by_column()` looks for the `DatasetDriftMetric` and retrieves `drift_by_columns`.

Each column contains a `drift_detected` boolean. The module relies on that boolean, not on the statistical value.

This is simple but depends on Evidently’s internal format. A version change can break parsing. The module catches the error and returns an empty dictionary. Logs must therefore be monitored: empty parsing produces a score of 0 and a `STABLE` status, which can hide drift.

### 3.4 Weighted score

`compute_weighted_drift_score()` computes the share of weights associated with drifting features:

```python
drifting_weight = 0.0
for column, weight in feature_weights.items():
    if drift_by_column.get(column, {}).get("drift_detected", False):
        drifting_weight += weight
return drifting_weight / total_weight
```

The score is therefore a weighted proportion between 0 and 1. It does not measure drift magnitude, only its presence on important features.

### 3.5 Status thresholds

Thresholds come from `src/config.py`:

| Threshold | Value | Status |
|---|---:|---|
| `DRIFT_WARNING_THRESHOLD` | 0.15 | `WARNING` |
| `DRIFT_CRITICAL_THRESHOLD` | 0.30 | `CRITICAL` |
| Below 0.15 | — | `STABLE` |

A score of 0.20 means 20% of feature weight is drifting. It is not a percentage of affected rows. It is a weighted summary based on feature importance.

### 3.6 Atomic status persistence

Status is written to `model_status.txt`. Writing uses a temporary file then `os.replace()`:

```python
tmp_path = status_path.with_name(status_path.name + ".tmp")
tmp_path.write_text(status_value, encoding="utf-8")
os.replace(tmp_path, status_path)
```

This prevents a reader from seeing a partially written file. The module also refuses to overwrite a directory at the same path, protecting against a configuration error.

### 3.7 Limits of offline monitoring

The holdout compares two periods of the same dataset. It does not see real requests. Drift can appear in production without being visible in the holdout.

Production monitoring addresses this limitation, but it depends on the quality of the prediction log.

---

<a id="4-production-tracking-on-jsonl-logs"></a>
## 4. Production Tracking on JSONL Logs

### 4.1 The prediction log

Every `/predict` call schedules a background task that writes one JSON line to `reports/prediction_log.jsonl`.

The line contains:

- the timestamp;
- the pharmacy and product;
- the predicted demand;
- the reorder flag;
- the model features.

The log contains no personal data. It also contains neither the raw payload nor the full response. This is a minimization choice.

### 4.2 Asynchronous writing

Writing is delegated to `BackgroundTasks` in the `predict` router. The HTTP response does not wait for disk I/O.

```python
background_tasks.add_task(
    record_prediction,
    features,
    predicted_demand,
    decision.reorder_needed,
    payload.store_id,
    payload.product_id,
)
```

This decoupling protects API latency. A slow write must not delay a prediction.

`src/observability.py` protects writing with a lock and catches write errors. A disk failure must not break inference.

### 4.3 Tolerant reading

`read_prediction_log()` reads the file line by line. Empty lines are ignored. Invalid JSON lines are reported and skipped.

This tolerance matters: a partially written or corrupted file must not fail the whole monitoring batch.

### 4.4 Minimum sample count

`run_production_drift_monitoring()` requires a minimum number of lines, configured by `PRODUCTION_DRIFT_MIN_SAMPLES` (30 by default).

Below that threshold, the function returns `UNKNOWN` without writing a new status. Too small a sample would produce an unstable, possibly misleading report.

### 4.5 Comparison with the reference

If enough lines exist, the module builds a current DataFrame from the JSONL. It checks that every `FEATURE_COLS` column is present.

The reference comes from `create_features(load_validated_data())` or a provided DataFrame. The Evidently report compares feature distributions. The target is not available in the production log, so comparison focuses on features.

The HTML report is written to `reports/production_drift_report.html`. Status uses the same thresholds as offline monitoring.

### 4.6 Limits of production tracking

The log contains the features sent to the API. If a client calls `/predict` through the legacy path (pre-computed lags), the logged features reflect that path, not the shared builder. The `history` field should therefore be preferred so the log is representative.

The log is not a random population sample. It reflects received calls. Selection bias is possible if some products or pharmacies are overrepresented.

Finally, comparison measures feature drift, not actual performance. Without ground truth, production WMAPE cannot be computed. Drift remains an indirect signal.

---

<a id="5-status-watcher-and-asynchronous-alerts"></a>
## 5. Status, Watcher and Asynchronous Alerts

### 5.1 Model states

`model_status.txt` can contain:

| Value | Meaning |
|---|---|
| `STABLE` | The model is considered reliable |
| `WARNING` | Moderate drift, vigilance required |
| `CRITICAL` | Strong drift, corrective action required |
| `UNKNOWN` | Unreadable, missing or invalid status |

`UNKNOWN` is not written by monitoring. It is a fallback produced by `read_model_status()` when it cannot determine the state.

This distinction is deliberate. A missing file must not be presented as `STABLE`.

### 5.2 Reading status

`read_model_status()` reads the file and applies three rules:

- `"OK"` is accepted as an alias for `"STABLE"` for compatibility;
- a value in `MODEL_STATUS_VALUES` is accepted as-is;
- any other value or read error produces `UNKNOWN`.

The function triggers no email. It is used by `/model/health`, `/alerts` and `/predict`. It must remain side-effect free.

### 5.3 The transition watcher

`watch_model_status()` is an asyncio task started in the API `lifespan`.

It reads status at a regular interval (`MODEL_STATUS_POLL_SECONDS`, 30 seconds by default) and sends an email only on a transition to `WARNING` or `CRITICAL`.

```python
if current_status in {"WARNING", "CRITICAL"}:
    if current_status != last_alerted_status:
        await asyncio.to_thread(send_support_email, current_status)
        last_alerted_status = current_status
else:
    last_alerted_status = None
```

The watcher remembers the last alerted status. A stable `WARNING` does not trigger an email on every poll. A return to `STABLE` resets memory, allowing a new alert if the model drifts again.

### 5.4 Why the watcher is outside the HTTP path

Sending an SMTP email can take hundreds of milliseconds or several seconds. If sending were triggered by `/predict`, every prediction would pay that cost.

The watcher moves alerts to a background task. The HTTP request simply reads status and responds. Latency remains stable.

The watcher is cancelled cleanly on application shutdown. A background task surviving the process would be a leak.

### 5.5 SMTP sending

`send_support_email()` builds a different message per status:

- `CRITICAL`: critical alert subject, action within 24 hours;
- `WARNING`: moderate vigilance subject.

Server, port and timeout come from `src/config.py`. The password comes from `SMTP_PASSWORD`.

If the password is missing, the function logs a warning and stops. It does not raise. SMTP errors are captured and logged.

### 5.6 Alerting limitations

The watcher has no persistent outbox. If the process restarts, it may re-emit an alert for a status still in `WARNING`. This is acceptable for an educational project, but production needs persistent deduplication.

The watcher does not retry a failed send. An SMTP outage can lose an alert. A production system should add a message queue and a retry policy.

The watcher reads a local file. In a multi-instance deployment, each instance would have its own watcher and could send the same alert. External coordination would be required.

---

<a id="6-appendices-and-monitoring-checklist"></a>
## 6. Appendices and Monitoring Checklist

### 6.1 Execution commands

```bash
# SHAP explainability and weight export
python -m src.explainability

# Offline monitoring
python -m src.monitoring

# Full pipeline (SHAP + offline drift + smoke tests)
python run_pipeline.py

# Targeted tests
pytest -q tests/test_explainability.py
pytest -q tests/test_monitoring.py
pytest -q tests/test_observability.py
pytest -q tests/test_model_status.py
```

### 6.2 Report inspection

```bash
# Offline drift report
ls -lh reports/drift_report.html

# Production drift report
ls -lh reports/production_drift_report.html

# SHAP weights
cat reports/feature_weights.json

# Prediction log
wc -l reports/prediction_log.jsonl
head -n 2 reports/prediction_log.jsonl

# SHAP figures
ls -lh reports/figures/
```

### 6.3 Monitoring checklist

- [ ] Global and local SHAP figures are generated.
- [ ] `reports/feature_weights.json` exists and its sum equals 1.0.
- [ ] The `DEFAULT_FEATURE_WEIGHTS` fallback is logged when used.
- [ ] The offline Evidently report is generated.
- [ ] The weighted score uses SHAP weights.
- [ ] WARNING and CRITICAL thresholds match `src/config.py`.
- [ ] `model_status.txt` is written atomically.
- [ ] The prediction log contains every contract feature.
- [ ] The minimum production sample count is respected.
- [ ] The production drift report is generated.
- [ ] The `UNKNOWN` status is handled by the API and dashboard.
- [ ] The watcher alerts only on transition.
- [ ] SMTP sending has a timeout.
- [ ] An SMTP failure does not break the API.
- [ ] The watcher is cancelled on shutdown.

### 6.4 Glossary

| Term | Definition |
|---|---|
| **SHAP** | Method for attributing feature contributions to a prediction |
| **Base value** | Average reference prediction used by SHAP |
| **Global importance** | Average absolute contribution over a sample |
| **Local explanation** | Decomposition of an individual prediction |
| **Data drift** | Change in the distribution of input data |
| **Holdout** | Subset kept for evaluation |
| **JSONL** | Text format with one JSON object per line |
| **Watcher** | Background task monitoring a state and triggering an action |
| **Outbox** | Persistent queue of messages to send |
| **Fail-safe** | Prudent behavior under uncertainty, here `UNKNOWN` |

### 6.5 Code files and artifacts to read

- `src/explainability.py`: SHAP, plots, CSV export and JSON weights.
- `src/monitoring.py`: offline drift, production drift, weighted score and status.
- `src/observability.py`: JSONL prediction log and tolerant reading.
- `src/services/model_status.py`: status reading, watcher and SMTP.
- `src/config.py`: thresholds, paths, default weights and SMTP settings.
- `src/api.py`: watcher startup and shutdown in the lifespan.
- `src/api_routers/predict.py`: asynchronous write scheduling.
- `reports/feature_weights.json`: normalized SHAP weights.
- `reports/prediction_log.jsonl`: aggregated production log.
- `reports/drift_report.html` and `reports/production_drift_report.html`: Evidently reports.
- `tests/test_explainability.py`, `tests/test_monitoring.py`, `tests/test_observability.py`, `tests/test_model_status.py`: non-regression.

</details>
