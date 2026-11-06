# Technical Documentation
## Blueprint: From ML Notebook to MLOps
**Project:** PharmaSupply-ML
**Framework:** Standard MLOps Blueprint
**File:** `docs/00_MLOPS_BLUEPRINT.md`
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

*Un blueprint pédagogique pour faire passer un Data Scientist du notebook à un système MLOps industrialisé, en s'appuyant sur l'architecture réelle de PharmaSupply-ML.*

---

## Sommaire

1. [Vision & Cadrage MLOps](#1-vision--cadrage-mlops)
2. [Cartographie de l'Architecture](#2-cartographie-de-larchitecture)
3. [Grille de Maturité MLOps](#3-grille-de-maturite-mlops)
4. [Principes d'ingénierie pour apprenants](#4-principes-dingenierie-pour-apprenants)
5. [Passer du laboratoire à la production](#5-passer-du-laboratoire-a-la-production)
6. [Annexes](#6-annexes)

---

<a id="1-vision--cadrage-mlops"></a>
## 1. Vision & Cadrage MLOps

### 1.1 Ce qui casse quand on sort du notebook

Un notebook sert à chercher. On charge un CSV, on essaie trois modèles, on regarde les métriques, on garde celui qui semble le meilleur. Ce mode de travail est utile, mais il repose sur des hypothèses fragiles : les cellules sont exécutées dans un ordre choisi à la main, les données vivent dans la mémoire du processus, les dépendances ne sont pas figées et personne ne sait ce qui se passe si le modèle reçoit une requête réelle.

Le passage en production ne se heurte pas d'abord à un problème de modèle. Il se heurte à un problème de système.

| Ce que l'on observe en production | Ce qui se cache derrière |
|---|---|
| Les prédictions changent sans qu'on ait touché au modèle | Les features sont recalculées autrement en ligne |
| Impossible de rejouer un résultat | Dépendances non épinglées, ordre manuel, `random_state` oublié |
| Une panne passe inaperçue | Un `try/except` renvoie une valeur par défaut |
| Personne ne sait si le modèle est encore fiable | Aucune métrique, aucun log, aucun suivi de dérive |
| Les données sont perdues ou dupliquées | Ingestion non idempotente, `TRUNCATE` ou écrasement de fichiers |
| Le service tombe dès qu'on le sollicite | Script monolithique, non testé, non conteneurisé |
| L'API est exposée à n'importe qui | Pas d'authentification, pas de quota, secrets dans le code |

Le notebook reste un excellent atelier. Le MLOps commence quand on décide que le travail doit survivre au départ du Data Scientist qui l'a écrit.

### 1.2 Les exigences d'un système ML en production

Un service de prévision ne se juge pas uniquement à son MAE. Six propriétés conditionnent sa durée de vie.

La **reproductibilité** d'abord : le même code, les mêmes données et le même environnement doivent produire le même résultat. Ensuite, les **contrats** : les entrées et sorties sont typées, les règles métier sont explicites et les erreurs ont un sens. La **séparation des responsabilités** évite qu'une route HTTP finisse par contenir du SQL et des règles de réapprovisionnement. L'**observabilité** permet de comprendre un incident sans relancer le système à l'aveugle. La **sécurité** protège l'accès, les secrets et la chaîne de build. L'**évolutivité**, enfin, rend possibles le réentraînement, la migration et le retour arrière.

Ces six exigences ne sont pas des options « entreprise ». Elles sont la différence entre une démonstration et un outil que l'on peut laisser tourner.

### 1.3 Le terrain de jeu : PharmaSupply-ML

Le projet prévoit la demande pharmaceutique à sept jours pour plusieurs officines et plusieurs produits. La prédiction ne s'arrête pas à un chiffre : elle sert à calculer une quantité de réapprovisionnement qui doit respecter le stock de sécurité, le délai fournisseur, le MOQ, la taille de lot, la durée de conservation et la logique FEFO (*First Expired, First Out*).

Ce contexte est particulièrement formateur parce qu'il mélange plusieurs difficultés :

- des séries temporelles avec saisonnalité, week-ends et ruptures ;
- des règles métier qui ne se déduisent pas du modèle ;
- des données imparfaites, avec valeurs manquantes et anomalies volontaires ;
- des conséquences opérationnelles directes : une erreur peut provoquer une rupture ou immobiliser du stock inutilement.

Un modèle qui prédit bien mais qui ignore le MOQ n'est pas un bon modèle pour ce métier. Un service qui prédit juste mais qui renvoie des valeurs fictives quand la base est indisponible n'est pas un service fiable.

### 1.4 Trois piliers, une seule chaîne

```text
┌────────────────────┐      ┌────────────────────┐      ┌────────────────────┐
│ DATA ENGINEERING   │ ---> │ MODEL ENGINEERING  │ ---> │ OPERATIONS &       │
│ ingestion, qualité │      │ features, training │      │ GOVERNANCE         │
│ features, contrats │      │ SHAP, MLflow       │      │ API, monitoring,   │
│                    │      │                    │      │ sécurité, CI/CD    │
└────────────────────┘      └────────────────────┘      └────────────────────┘
```

Le dépôt applique cette chaîne sans mélanger les étages :

- la donnée entre par `src/data/generate_data.py`, `src/ingestion.py`, `src/validation.py` et `src/features.py` ;
- le modèle se construit dans `src/train.py`, s'explique dans `src/explainability.py` et se surveille dans `src/monitoring.py` ;
- le service vit dans `src/api.py`, `src/api_routers/`, `src/services/`, `src/repositories/`, `src/security.py` et `src/observability.py` ;
- l'interface métier est assurée par `src/dashboard.py`.

Chaque dossier a une raison d'être. Si vous hésitez sur l'emplacement d'un nouveau code, demandez-vous quelle couche serait responsable en cas d'incident.

---

<a id="2-cartographie-de-larchitecture"></a>
## 2. Cartographie de l'Architecture

### 2.1 Le trajet d'une ligne de données

```text
┌───────────────────────┐
│ Génération synthétique│  src/data/generate_data.py
│ 3 sites × 10 produits │  saisonnalité, ruptures, FEFO
└──────────┬────────────┘
           │ CSV
           v
┌───────────────────────┐
│ Ingestion PostgreSQL  │  src/ingestion.py
│ Alembic + UPSERT      │  clé métier (store_id, product_id, date)
└──────────┬────────────┘
           │
           v
┌───────────────────────┐
│ Quality Gate          │  src/validation.py
│ hard fail / soft fix  │  bornes métier, imputation, cohérence
└──────────┬────────────┘
           │ DataFrame propre
           v
┌───────────────────────┐
│ Feature Engineering   │  src/features.py
│ lags, rolling, target │  build_history_features() pour l'inférence
└──────────┬────────────┘
           │
           ├──────────────► Entraînement XGBoost ──► MLflow (opt-in)
           │                 src/train.py             tracking + registry
           │
           ├──────────────► Explicabilité SHAP ────► reports/feature_weights.json
           │                 src/explainability.py
           │
           └──────────────► Monitoring Evidently ──► reports/drift_report.html
                             src/monitoring.py        model_status.txt
                                      │
                                      v
┌───────────────────────┐      ┌───────────────────────────┐
│ FastAPI Serving       │ <─── │ Journal de prédiction     │
│ api_routers/services/ │      │ reports/prediction_log.jsonl
│ repositories/schemas  │      └───────────────────────────┘
└──────────┬────────────┘
           │
           ├── Observabilité : logs JSON, Prometheus /metrics, /livez, /readyz
           ├── Sécurité : X-API-Key, CORS, slowapi
           v
┌───────────────────────┐
│ Streamlit Dashboard   │  src/dashboard.py
│ thin client API       │  sélection par exception, simulation FEFO
└───────────────────────┘
```

Le point important n'est pas le dessin. C'est la direction du flux. Une donnée brute ne doit jamais arriver directement dans une route HTTP sans passer par la validation et le feature engineering. Une prédiction ne doit jamais être produite sans laisser une trace exploitable.

### 2.2 Qui fait quoi dans le dépôt

| Dossier ou fichier | Responsabilité | Ce qu'il faut y observer |
|---|---|---|
| `src/data/generate_data.py` | Génère un jeu multi-sites réaliste | Reproductibilité, limites du jeu synthétique |
| `src/ingestion.py` | Charge le CSV dans PostgreSQL | Normalisation, typage, UPSERT |
| `src/validation.py` | Filtre la donnée avant le ML | Erreur bloquante vs anomalie réparable |
| `src/features.py` | Calcule les features d'entraînement et d'inférence | Définition unique des features |
| `src/train.py` | Split temporel, XGBoost, métriques, MLflow | Absence de fuite temporelle |
| `src/explainability.py` | SHAP global et local, poids normalisés | Auditabilité du modèle |
| `src/monitoring.py` | Drift offline et drift de production | Différence entre holdout et réel |
| `src/api.py` | Assemble l'application FastAPI | Composition root, lifespan |
| `src/api_routers/` | Routes HTTP | Contrats, codes d'erreur, auth |
| `src/services/` | Règles métier pures | Testabilité sans base ni HTTP |
| `src/repositories/` | Accès SQL et pool | Séparation persistance/transport |
| `src/schemas.py` | Contrats Pydantic | Frontières explicites |
| `src/security.py` | API key, CORS, rate limiting | Défense en profondeur |
| `src/observability.py` | Logs JSON, métriques, journal de prédiction | Instrumentation |
| `src/dashboard.py` | Interface Streamlit | Client mince, pas de logique dupliquée |
| `scripts/` | Nettoyage et conversion d'environnement | Automatisation d'exploitation |
| `alembic/` | Migrations de schéma | Évolution sans perte de données |
| `tests/` | Tests unitaires et d'intégration | Non-régression |
| `sql/schema.sql` | DDL de bootstrap | Clé métier et contraintes |
| `models/`, `reports/` | Artefacts générés | Traçabilité des sorties |

### 2.3 La API en couches

La phase P1 a sorti la logique du fichier `api.py`. Aujourd'hui, une requête traverse quatre couches.

```text
Requête HTTP
    │
    v
src/api_routers/        ← transport : validation, codes HTTP, dépendances
    │
    v
src/services/           ← logique métier pure : alertes, MOQ, FEFO, features
    │
    v
src/repositories/       ← SQL, pool de connexions, curseurs
    │
    v
PostgreSQL 16
```

| Couche | Fichiers | Rôle | Ce qu'elle ne doit pas faire |
|---|---|---|---|
| Transport | `src/api_routers/` | Routes, codes HTTP, auth, quotas | SQL, calculs métier |
| Métier | `src/services/` | Règles, calculs, décisions | Accès HTTP ou SQL |
| Persistance | `src/repositories/` | Requêtes, pool, curseurs | Règles métier |
| Contrats | `src/schemas.py` | Validation Pydantic, OpenAPI | Logique métier |
| Configuration | `src/config.py` | Constantes, chemins, seuils | Effets de bord |

Un router qui commence à calculer un MOQ ou à écrire une requête SQL est un signal d'alerte. La logique doit descendre dans `services/` ou `repositories/`. Cette discipline paraît contraignante au début. Elle devient très rentable dès qu'il faut tester une règle sans lancer PostgreSQL ni FastAPI.

### 2.4 Le socle de données

Le stockage repose sur PostgreSQL 16, SQLAlchemy avec un pool de connexions (`pool_pre_ping`, taille bornée) et Alembic pour les migrations.

La table `raw_pharmaceutical_demand` possède une clé métier naturelle : `(store_id, product_id, date)`. Cette clé n'est pas décorative. Elle permet à l'ingestion de faire un UPSERT :

```sql
INSERT INTO raw_pharmaceutical_demand (...)
VALUES (...)
ON CONFLICT (store_id, product_id, date)
DO UPDATE SET ...;
```

Le choix de l'UPSERT change tout pour un pipeline que l'on relance souvent.

| Approche | Comportement | Conséquence |
|---|---|---|
| `TRUNCATE + APPEND` | Vide la table puis réinsère | Historique perdu, fenêtre d'indisponibilité |
| `UPSERT` sur clé métier | Met à jour ou insère ligne par ligne | Rejeu sûr, historique conservé |

Alembic joue un rôle complémentaire. Le fichier `sql/schema.sql` initialise un volume Docker neuf. Les migrations `alembic/` prennent le relais sur une base existante. Les deux chemins doivent rester cohérents, notamment sur la contrainte d'unicité.

### 2.5 Qualité, observabilité et sécurité

Ces trois sujets sont souvent traités après coup. Ici, ils sont branchés dès la conception.

| Préoccupation | Implémentation | Effet concret |
|---|---|---|
| Qualité | `src/validation.py` | Bloque les corruptions, répare les anomalies mineures |
| Logs | `src/logging_config.py` | Lignes JSON exploitables par un collecteur |
| Corrélation | `X-Request-ID` dans `src/observability.py` | Relier un incident à une requête |
| Métriques | Prometheus `GET /metrics` | Latence, volume, erreurs |
| Liveness | `GET /livez` | Le processus répond |
| Readiness | `GET /readyz` | Modèle chargé et base joignable |
| Authentification | `X-API-Key` dans `src/security.py` | Pas d'accès anonyme en production |
| CORS | `CORS_ORIGINS` | Seul le dashboard autorisé appelle l'API |
| Quotas | slowapi sur `/predict` et `/alerts` | Protection contre les rafales |
| Images | non-root, dépendances épinglées, Trivy | Surface d'attaque réduite |

Une sonde `/readyz` qui renvoie `200` alors que le modèle n'est pas chargé est un faux sentiment de sécurité. C'est pour cette raison que la readiness vérifie deux choses : le modèle en mémoire et une requête triviale vers PostgreSQL.

### 2.6 Industrialisation ML

| Brique | Implémentation | Ce qu'il faut comprendre |
|---|---|---|
| Tracking | MLflow avec `MLFLOW_ENABLED=true` | Retrouver paramètres, métriques et artefacts |
| Registry | MLflow Model Registry | Promouvoir un modèle sans écraser l'ancien |
| Signature | `infer_signature` | Vérifier la conformité des entrées |
| Parité | `build_history_features()` et `tests/test_parity.py` | Mêmes formules en training et en serving |
| Explicabilité | SHAP et `reports/feature_weights.json` | Comprendre et pondérer le drift |
| Drift offline | Evidently sur holdout chronologique | Détecter une dérive historique |
| Drift production | `run_production_drift_monitoring()` sur JSONL | Comparer le réel à la référence |
| Statut | `model_status.txt` et watcher | Alerter sans bloquer les requêtes |

MLflow reste optionnel. Le pipeline d'entraînement ne doit pas s'arrêter parce qu'un serveur de tracking est indisponible. C'est un choix d'ingénierie : la traçabilité est importante, mais elle ne doit pas devenir un point de défaillance unique.

---

<a id="3-grille-de-maturite-mlops"></a>
## 3. Grille de Maturité MLOps

### 3.1 Les paliers

| Niveau | Nom | Ce que l'on observe |
|---|---|---|
| **0** | Expérimental | Notebooks, exécution manuelle, pas de tests ni de monitoring |
| **1** | Pipeline automatisé | Code modularisé, artefacts traçables, validation, monitoring |
| **1+** | Socle stabilisé et structuré | Erreurs explicites, API modulaire, sécurité, observabilité, tests |
| **2** | Continuous Training | Réentraînement piloté par le drift, promotion automatisée, SLOs |
| **3** | Plateforme industrialisée | Multi-équipes, gouvernance, self-service, coûts pilotés |

### 3.2 Grille détaillée

| Dimension | Niveau 0 | Niveau 1 | Niveau 1+ | Niveau 2 |
|---|---|---|---|---|
| Reproductibilité | Manuelle | Partielle | Environnements épinglés, split déterministe | Runs entièrement rejouables |
| Données | Aucun versioning | CSV copié à la main | Clé métier et migrations | Data versioning dédié |
| Modularité | Monolithe | Modules `src/` | Routers/services/repositories | Plateforme de composants |
| Tests | Aucun | Quelques tests | 92 tests, 80,91 %, intégration | Charge et chaos engineering |
| CI/CD | Aucun | Script manuel | Lint, tests, build, Trivy | Déploiement continu avec gate modèle |
| Observabilité | `print()` | Logs basiques | JSON, Prometheus, probes | Traces, SLOs, alerting |
| Sécurité | Aucune | Secrets dans `.env` | API key, CORS, quotas, non-root | OAuth2/OIDC, rotation, WAF |
| Monitoring | Aucun | Rapport ponctuel | Drift offline et production | Réentraînement déclenché |
| Gouvernance | Aucune | `.joblib` | MLflow tracking et registry | Promotion et rollback formalisés |

### 3.3 Où en est PharmaSupply-ML

Les phases P0 et P1 sont clôturées et validées.

La P0 a stabilisé le socle : e-mail SMTP hors du chemin de requête, erreurs explicites, statut `UNKNOWN` au lieu d'un faux `STABLE`, contrat `/features` typé, pool PostgreSQL, constantes centralisées, poids SHAP consommés, plancher MOQ corrigé.

La P1 a structuré le service : découpage en routers, services et repositories, parité train/serve, MLflow en opt-in, Alembic et UPSERT, logs JSON, métriques Prometheus, sondes, API key, CORS, rate limiting, Dockerfiles non-root et scan Trivy, drift de production.

Les preuves sont mesurables : 92 tests au vert, dont 91 unitaires et 1 test d'intégration Testcontainers, pour 80,91 % de couverture. Le seuil de 70 % est configuré dans `pyproject.toml`. Le lint Ruff passe. La configuration Docker et la génération SQL Alembic ont été vérifiées.

Le projet se situe donc à un niveau 1+ consolidé. Les fondations du niveau 2 sont présentes, mais la boucle de réentraînement automatique n'est pas encore fermée.

### 3.4 Ce qui reste pour le niveau 2

| Chantier | Pourquoi le faire | Indicateur de réussite |
|---|---|---|
| Réentraînement piloté par le drift | Réduire la dérive sans intervention manuelle | Un run MLflow est créé et promu après un seuil critique |
| Endpoint batch et cache distribué | Servir de gros catalogues sans saturer l'API | Latence p95 stable |
| Model server dédié | Isoler l'inférence du service métier | Scalabilité indépendante |
| Partitionnement et dimensions | Garder des requêtes rapides à grande échelle | Temps de requête borné |
| Cloud et SLOs | Industrialiser la disponibilité | Tableaux de bord et alertes |
| Copilote RAG/LLM | Aider à la décision en cas de rupture | Recommandations tracées et évaluées |

---

<a id="4-principes-dingenierie-pour-apprenants"></a>
## 4. Principes d'ingénierie pour apprenants

### 4.1 Parité train/serve

La parité train/serve signifie que les features calculées pendant l'entraînement et celles calculées pendant l'inférence sont identiques. Pas « à peu près identiques ». Identiques.

Le piège classique est de croire que la formule importe peu. Un exemple vécu dans ce projet : l'API calculait la moyenne glissante à sept jours comme `lag_7 / 7.0`, alors que l'entraînement utilisait `demand.shift(1).rolling(7).mean()`. Les deux valeurs se ressemblent, mais elles ne décrivent pas la même chose. Le modèle recevait donc une distribution décalée.

```python
# Entraînement
rolling_mean_7 = demand.shift(1).rolling(7).mean()

# Ancienne inférence, incorrecte
rolling_mean_7 = lag_7 / 7.0
```

La correction tient dans un constructeur unique, `build_history_features()` dans `src/features.py`. L'API l'utilise dès qu'un historique de ventes est fourni. Le test `tests/test_parity.py` compare les deux implémentations sur une série synthétique.

La règle à retenir est simple : une feature doit avoir une seule définition dans le code. Si vous avez une version « training » et une version « serving », vous avez déjà un bug latent.

### 4.2 Idempotence

Une opération est idempotente si la relancer produit le même état final. Pas de doublon, pas de perte, pas de surprise.

Dans ce projet, l'ingestion utilise la clé `(store_id, product_id, date)` et un UPSERT. Vous pouvez relancer `python run_pipeline.py` autant de fois que nécessaire : les lignes existantes sont mises à jour, les nouvelles sont insérées, l'historique reste en place.

L'ancien comportement, `TRUNCATE` puis `APPEND`, avait l'apparence de la simplicité. En réalité, il ouvrait une fenêtre pendant laquelle la table était vide et il effaçait toute donnée dont le CSV ne parlait plus.

L'idempotence n'est pas un détail d'ingénieur. C'est ce qui permet à un pipeline d'être rejoué après un incident sans craindre d'aggraver la situation.

### 4.3 Fail-safe et fail-explicit

Ces deux modes de défaillance répondent à des besoins différents.

Le **fail-safe** protège l'utilisateur lorsqu'on ne sait pas. Si le fichier `model_status.txt` est absent ou illisible, le statut devient `UNKNOWN`, jamais `STABLE`. On préfère afficher une incertitude plutôt qu'une fausse bonne nouvelle.

Le **fail-explicit** protège la décision métier lorsqu'une dépendance est cassée. Si PostgreSQL est indisponible, `/stores`, `/features` ou `/alerts` renvoient `503`. Si aucun historique n'existe pour un couple store/produit, `/features` renvoie `404`. On ne fabrique pas une liste de magasins fictive ni des lags par défaut.

Le projet a supprimé les fallbacks silencieux de la première version. C'était pourtant confortable en apparence : l'API répondait toujours `200`. Mais elle répondait avec des données inventées. En pharmacie, ce confort est un risque.

Un `try/except` qui masque une panne n'est pas de la robustesse. C'est une dette qui se paiera au pire moment.

### 4.4 Traçabilité

La traçabilité répond à trois questions : quel modèle a produit cette prédiction, avec quelles données, et dans quel environnement ?

Le projet combine plusieurs mécanismes. MLflow enregistre les paramètres, les métriques, la signature et l'artefact lorsque `MLFLOW_ENABLED=true`. La réponse `/predict` expose `model_version`. Les logs JSON contiennent un `request_id`, la méthode, le chemin, le statut et la durée. Le journal `reports/prediction_log.jsonl` conserve les features agrégées et la prédiction, sans donnée personnelle. Alembic garde l'historique du schéma. `src/config.py` centralise les seuils et les chemins.

Sans ces traces, un incident devient une enquête sans preuves. Vous savez qu'il y a eu un problème, mais vous ne pouvez pas dire quand, ni pourquoi, ni avec quelle version du modèle.

### 4.5 Observabilité

Observer un système, ce n'est pas ajouter des `print()`. C'est produire des signaux exploitables.

Les logs JSON répondent à la question « que s'est-il passé à cet instant ? ». Les métriques Prometheus répondent à « le service est-il lent, saturé, en erreur ? ». Les sondes répondent à « le service peut-il recevoir du trafic ? ».

Le pipeline utilise cette distinction. Après avoir reconstruit l'image API, il interroge `/readyz` et attend que le modèle soit chargé et la base joignable. Sans cette sonde, le smoke test pouvait tomber sur un conteneur encore en démarrage et échouer avec un `Connection reset by peer`.

Une métrique utile n'est pas une métrique décorative. Elle doit servir à une décision : alerter, scaler, revenir en arrière ou corriger.

### 4.6 Sécurité

La sécurité d'une API ne se résume pas à un mot de passe. Elle couvre l'accès, les secrets, les quotas et la chaîne de build.

`X-API-Key` bloque les accès anonymes lorsque `API_KEY` est défini. CORS limite les origines autorisées. slowapi applique des quotas sur `/predict` et `/alerts`. Les secrets vivent dans `.env`, jamais dans le code, et `.env.example` documente les variables attendues. Les images Docker tournent avec un utilisateur non-root. Les dépendances sont épinglées. Trivy scanne les images en CI.

Aucune de ces mesures n'est suffisante seule. Ensemble, elles réduisent la surface d'attaque et rendent les incidents plus faciles à contenir.

### 4.7 Contrats et validation

Un contrat explicite décrit ce qu'un composant accepte et ce qu'il retourne. En Python, Pydantic v2 joue ce rôle pour l'API. Dans le projet, `PredictionInput` impose des bornes (`ge=0`, `ge=1`) et une longueur minimale de 30 valeurs pour `history`. Un stock négatif ou un historique trop court est rejeté en `422` avant d'atteindre le modèle.

Le contrat ne s'arrête pas à l'API. `FeaturesOutput` inclut `critical_expiry_days` et `sales_history`. Le quality gate de `src/validation.py` bloque les corruptions structurelles et répare les anomalies mineures. Le schéma SQL impose des colonnes non nulles, une clé métier unique et un index.

Un contrat clair vaut mieux qu'un commentaire perdu dans un notebook. Il est vérifiable, testable et visible dans OpenAPI.

### 4.8 Tests

Les tests sont la mémoire du comportement attendu. Ils ne servent pas à prouver que le code est parfait, mais à éviter qu'une modification casse silencieusement une règle métier.

La suite couvre plusieurs niveaux. Les tests unitaires de `tests/test_services.py`, `tests/test_features.py` et `tests/test_validation.py` vérifient les règles et les transformations. `tests/test_api.py` contrôle les codes HTTP, les schémas et les erreurs explicites. `tests/test_parity.py` compare les features d'entraînement et d'inférence. `tests/test_ingestion_integration.py` exécute un vrai PostgreSQL via Testcontainers. `tests/test_security.py` et `tests/test_observability.py` couvrent l'authentification, le CORS et le journal de prédiction.

Le résultat actuel est de 92 tests au vert pour 80,91 % de couverture. Le seuil de 70 % est configuré dans `pyproject.toml`. Ce chiffre n'est pas une fin en soi, mais il empêche les régressions les plus grossières.

### 4.9 Modularité et séparation des responsabilités

Un module doit avoir une seule raison de changer. C'est la formulation la plus utile de la modularité.

Avant la phase P1, `api.py` concentrait le transport, le SQL, les règles métier et les schémas. Le fichier faisait plus de 800 lignes. Aujourd'hui, `src/api.py` assemble l'application, `src/api_routers/` gère le transport, `src/services/` porte les règles, `src/repositories/` porte le SQL et `src/schemas.py` porte les contrats.

Ce découpage a un coût : plus de fichiers, plus d'imports, une navigation moins immédiate. Il a un bénéfice : on peut tester une règle de réapprovisionnement sans lancer FastAPI ni PostgreSQL, et on peut changer une requête SQL sans toucher aux routes.

Si vous devez modifier trois fichiers pour changer une seule règle, la modularité est insuffisante. Si vous devez modifier un seul fichier pour changer une règle, vous êtes probablement au bon endroit.

### 4.10 Reproductibilité

Un résultat est reproductible si le même code, les mêmes données et le même environnement produisent le même résultat. Cette propriété ne s'obtient pas par hasard.

Le projet épingle les dépendances dans `environment.yml`, `requirements.txt`, `requirements.api.txt` et `requirements.dashboard.txt`. XGBoost et les échantillonnages SHAP utilisent `random_state=42`. Le split est temporel et déterministe. `.env.example` documente les variables. MLflow enregistre les paramètres et la signature. Alembic versionne le schéma. La CI exécute le lint, les tests, le build et le scan.

« Ça marche chez moi » n'est pas une propriété d'ingénierie. C'est une anecdote.

### 4.11 Checklist de passage du laboratoire à la production

- [ ] Les features d'entraînement et d'inférence ont une définition unique.
- [ ] L'ingestion est idempotente, avec une clé métier et un UPSERT.
- [ ] Les erreurs sont explicites et ne renvoient pas de fausses données.
- [ ] Les entrées et sorties sont typées.
- [ ] Les règles métier sont testées sans base ni HTTP.
- [ ] Les secrets sont hors du code.
- [ ] L'API est authentifiée et limitée en débit.
- [ ] Les logs sont structurés et corrélables.
- [ ] Les métriques et les sondes de santé existent.
- [ ] Le modèle est versionné et signé.
- [ ] Le drift offline et le drift de production sont mesurés.
- [ ] Les images sont non-root et scannées.

---

<a id="5-passer-du-laboratoire-a-la-production"></a>
## 5. Passer du laboratoire à la production

### 5.1 Trois réflexes à prendre dès le prochain projet

Le premier réflexe est de définir les contrats avant d'écrire le modèle. Un schéma d'entrée, un schéma de sortie et une règle de validation évitent des semaines de débogage.

Le deuxième est de versionner les données et les artefacts, pas seulement le code. Un modèle sans dataset ni environnement est un objet difficile à reproduire. MLflow, Alembic et les fichiers de dépendances épinglés répondent à ce besoin.

Le troisième est d'instrumenter avant d'avoir un incident. Logs JSON, métriques, sondes et journal de prédiction coûtent peu au départ. Ils coûtent très cher lorsqu'on les ajoute après une panne.

### 5.2 Un plan de migration réaliste

Commencez par extraire la logique métier du notebook dans des fonctions pures. Testez ces fonctions avec des exemples métier, pas seulement avec des formes de données.

Ensuite, séparez l'accès aux données du transport. Un repository ou un module de requêtes doit pouvoir être testé avec un curseur simulé. Cela évite de dépendre d'une base live pour chaque test.

Puis ajoutez un quality gate. Une donnée vide, une colonne manquante ou un stock négatif doivent être traités avant le feature engineering.

Enfin, branchez le serving et l'observabilité. Une route `/predict` sans validation, sans logs et sans sonde n'est pas prête pour la production, même si le modèle est excellent.

### 5.3 Ce qui doit vous alerter

Quelques signaux doivent déclencher une revue immédiate :

- une route qui contient du SQL ou des règles de réapprovisionnement ;
- un `try/except` qui renvoie une valeur par défaut sans le dire ;
- une feature calculée différemment dans `src/features.py` et dans le router ;
- une ingestion qui vide la table avant de recharger ;
- un statut de santé qui affiche `STABLE` alors que le fichier de statut est absent ;
- des tests qui acceptent deux issues possibles pour ne jamais échouer ;
- des secrets présents dans un fichier versionné.

Ces signaux ne sont pas théoriques. Ils ont tous existé dans ce projet avant les phases P0 et P1.

---

<a id="6-annexes"></a>
## 6. Annexes

### 6.1 Glossaire

| Terme | Définition |
|---|---|
| **Idempotence** | Propriété d'une opération qui peut être rejouée sans changer le résultat final |
| **UPSERT** | `INSERT ... ON CONFLICT DO UPDATE` : insère ou met à jour selon une clé |
| **Train/serve skew** | Écart de traitement des features entre entraînement et inférence |
| **Fail-safe** | Comportement prudent en cas d'incertitude, par exemple le statut `UNKNOWN` |
| **Fail-explicit** | Erreur claire et immédiate en cas de panne, par exemple un `503` |
| **Drift** | Dérive de distribution des données ou de la performance du modèle |
| **FEFO** | *First Expired, First Out* : prioriser les lots les plus proches de la péremption |
| **MOQ** | *Minimum Order Quantity* : quantité minimale de commande |
| **SLO** | *Service Level Objective* : objectif mesurable de niveau de service |
| **Registry** | Catalogue versionné des modèles prêts à être déployés |

### 6.2 Commandes utiles

```bash
# Environnement
conda env create -f environment.yml
conda activate pharma-supply
conda env update -f environment.yml --prune

# Infrastructure
docker compose up -d --build

# Migrations
alembic upgrade head
alembic upgrade head --sql

# Pipeline complet
python run_pipeline.py

# Tests et lint
ruff check src tests run_pipeline.py alembic
pytest -q -m "not integration" --cov=src --cov-report=term-missing
pytest -q -m integration

# MLflow (opt-in)
MLFLOW_ENABLED=true python -m src.train
mlflow ui --backend-store-uri file:./mlruns
```

### 6.3 Fichiers de code à lire en premier

- `src/api.py` : composition root, lifespan et chargement du modèle.
- `src/api_routers/predict.py` : contrat d'inférence et orchestration des règles.
- `src/services/inventory.py` : MOQ, FEFO, alertes et construction des features.
- `src/repositories/database.py` : pool, curseurs et gestion des erreurs.
- `src/features.py` : `create_features()` et `build_history_features()`.
- `tests/test_parity.py` : preuve de parité train/serve.
- `alembic/versions/0001_initial_schema.py` : clé métier et contraintes SQL.
- `src/config.py` : source de vérité des constantes et variables d'environnement.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *A pedagogical blueprint for moving from notebook experiments to an industrialized MLOps system, grounded in the real PharmaSupply-ML architecture.*

---

## Table of Contents

1. [Vision & MLOps Scoping](#1-vision--mlops-scoping)
2. [Architecture Map](#2-architecture-map)
3. [MLOps Maturity Grid](#3-mlops-maturity-grid)
4. [Engineering Principles for Learners](#4-engineering-principles-for-learners)
5. [From Lab to Production](#5-from-lab-to-production)
6. [Appendices](#6-appendices)

---

<a id="1-vision--mlops-scoping"></a>
## 1. Vision & MLOps Scoping

### 1.1 What breaks outside the notebook

A notebook is for exploration. You load a CSV, try three models, inspect metrics, and keep the one that looks best. That workflow relies on fragile assumptions: cells run in an order chosen by hand, data lives in process memory, dependencies are not pinned, and nobody knows what happens when the model receives a real request.

Production does not fail first because of the model. It fails because of the system around it.

| What production shows | What is really happening |
|---|---|
| Predictions change without touching the model | Features are recomputed differently online |
| A result cannot be reproduced | Unpinned dependencies, manual order, forgotten `random_state` |
| An outage goes unnoticed | A `try/except` returns a default value |
| Nobody knows whether the model is still reliable | No metrics, no logs, no drift tracking |
| Data is lost or duplicated | Non-idempotent ingestion, `TRUNCATE` or file overwrite |
| The service collapses under load | Monolithic, untested, non-containerized script |
| The API is open to anyone | No authentication, no quota, secrets in code |

The notebook remains a good workshop. MLOps starts when the work must survive the departure of the Data Scientist who wrote it.

### 1.2 What a production ML system actually requires

A forecasting service is not judged solely by its MAE. Six properties determine its lifespan.

**Reproducibility** comes first: the same code, data and environment must produce the same result. Then **contracts**: inputs and outputs are typed, business rules are explicit, and errors have meaning. **Separation of concerns** prevents an HTTP route from eventually containing SQL and replenishment rules. **Observability** makes it possible to understand an incident without rerunning the system blindly. **Security** protects access, secrets and the build chain. **Scalability** makes retraining, migration and rollback possible.

These six requirements are not optional “enterprise” features. They are the difference between a demo and a tool that can run unattended.

### 1.3 The PharmaSupply-ML use case

The project forecasts pharmaceutical demand seven days ahead for several pharmacies and products. The prediction does not stop at a number: it drives a replenishment quantity that must respect safety stock, supplier lead time, MOQ, batch size, shelf life and the FEFO rule (*First Expired, First Out*).

This context mixes several difficulties:

- time series with seasonality, weekends and stockouts;
- business rules that cannot be inferred from the model;
- imperfect data with missing values and deliberate anomalies;
- direct operational consequences: an error can cause a stockout or immobilize inventory.

A model that predicts well but ignores MOQ is not a good model for this business. A service that predicts well but returns fake values when the database is down is not a reliable service.

### 1.4 Three pillars, one chain

```text
┌────────────────────┐      ┌────────────────────┐      ┌────────────────────┐
│ DATA ENGINEERING   │ ---> │ MODEL ENGINEERING  │ ---> │ OPERATIONS &       │
│ ingestion, quality │      │ features, training │      │ GOVERNANCE         │
│ features, contracts│      │ SHAP, MLflow       │      │ API, monitoring,   │
│                    │      │                    │      │ security, CI/CD    │
└────────────────────┘      └────────────────────┘      └────────────────────┘
```

The repository applies this chain without mixing layers:

- data enters through `src/data/generate_data.py`, `src/ingestion.py`, `src/validation.py` and `src/features.py`;
- the model is built in `src/train.py`, explained in `src/explainability.py` and monitored in `src/monitoring.py`;
- the service lives in `src/api.py`, `src/api_routers/`, `src/services/`, `src/repositories/`, `src/security.py` and `src/observability.py`;
- the business interface is provided by `src/dashboard.py`.

Each folder has a reason to exist. When you hesitate about where new code belongs, ask which layer would be responsible in an incident.

---

<a id="2-architecture-map"></a>
## 2. Architecture Map

### 2.1 The journey of a data row

```text
┌───────────────────────┐
│ Synthetic generation  │  src/data/generate_data.py
│ 3 stores × 10 products│  seasonality, stockouts, FEFO
└──────────┬────────────┘
           │ CSV
           v
┌───────────────────────┐
│ PostgreSQL ingestion  │  src/ingestion.py
│ Alembic + UPSERT      │  natural key (store_id, product_id, date)
└──────────┬────────────┘
           │
           v
┌───────────────────────┐
│ Quality Gate          │  src/validation.py
│ hard fail / soft fix  │  business bounds, imputation, consistency
└──────────┬────────────┘
           │ clean DataFrame
           v
┌───────────────────────┐
│ Feature Engineering   │  src/features.py
│ lags, rolling, target │  build_history_features() for inference
└──────────┬────────────┘
           │
           ├──────────────► XGBoost training ──► MLflow (opt-in)
           │                 src/train.py          tracking + registry
           │
           ├──────────────► SHAP explainability ──► reports/feature_weights.json
           │                 src/explainability.py
           │
           └──────────────► Evidently monitoring ──► reports/drift_report.html
                             src/monitoring.py        model_status.txt
                                      │
                                      v
┌───────────────────────┐      ┌───────────────────────────┐
│ FastAPI Serving       │ <─── │ Prediction log            │
│ api_routers/services/ │      │ reports/prediction_log.jsonl
│ repositories/schemas  │      └───────────────────────────┘
└──────────┬────────────┘
           │
           ├── Observability: JSON logs, Prometheus /metrics, /livez, /readyz
           ├── Security: X-API-Key, CORS, slowapi
           v
┌───────────────────────┐
│ Streamlit Dashboard   │  src/dashboard.py
│ thin API client       │  exception-based selection, FEFO simulation
└───────────────────────┘
```

The important part is not the drawing. It is the direction of the flow. Raw data must never reach an HTTP route without validation and feature engineering. A prediction must never be produced without leaving an exploitable trace.

### 2.2 Who does what in the repository

| Folder or file | Responsibility | What to look for |
|---|---|---|
| `src/data/generate_data.py` | Generate a realistic multi-site dataset | Reproducibility, limits of synthetic data |
| `src/ingestion.py` | Load the CSV into PostgreSQL | Normalization, typing, UPSERT |
| `src/validation.py` | Filter data before ML | Blocking error vs repairable anomaly |
| `src/features.py` | Compute training and inference features | Single feature definition |
| `src/train.py` | Temporal split, XGBoost, metrics, MLflow | No temporal leakage |
| `src/explainability.py` | Global and local SHAP, normalized weights | Model auditability |
| `src/monitoring.py` | Offline and production drift | Difference between holdout and live data |
| `src/api.py` | Assemble the FastAPI application | Composition root, lifespan |
| `src/api_routers/` | HTTP routes | Contracts, error codes, auth |
| `src/services/` | Pure business rules | Testability without DB or HTTP |
| `src/repositories/` | SQL access and pool | Persistence/transport separation |
| `src/schemas.py` | Pydantic contracts | Explicit boundaries |
| `src/security.py` | API key, CORS, rate limiting | Defense in depth |
| `src/observability.py` | JSON logs, metrics, prediction log | Instrumentation |
| `src/dashboard.py` | Streamlit interface | Thin client, no duplicated logic |
| `scripts/` | Cleanup and environment conversion | Operations automation |
| `alembic/` | Schema migrations | Evolution without data loss |
| `tests/` | Unit and integration tests | Non-regression |
| `sql/schema.sql` | Bootstrap DDL | Natural key and constraints |
| `models/`, `reports/` | Generated artifacts | Pipeline output traceability |

### 2.3 The layered API

Phase P1 moved logic out of `api.py`. A request now crosses four layers.

```text
HTTP request
    │
    v
src/api_routers/        ← transport: validation, HTTP codes, dependencies
    │
    v
src/services/           ← pure business rules: alerts, MOQ, FEFO, features
    │
    v
src/repositories/       ← SQL, connection pool, cursors
    │
    v
PostgreSQL 16
```

| Layer | Files | Role | What it must not do |
|---|---|---|---|
| Transport | `src/api_routers/` | Routes, HTTP codes, auth, quotas | SQL, business calculations |
| Business | `src/services/` | Rules, calculations, decisions | HTTP or SQL access |
| Persistence | `src/repositories/` | Queries, pool, cursors | Business rules |
| Contracts | `src/schemas.py` | Pydantic validation, OpenAPI | Business logic |
| Configuration | `src/config.py` | Constants, paths, thresholds | Side effects |

A router that starts calculating MOQ or writing SQL is a warning sign. The logic belongs in `services/` or `repositories/`.

### 2.4 The data foundation

Storage relies on PostgreSQL 16, SQLAlchemy with a bounded connection pool (`pool_pre_ping`) and Alembic for migrations.

`raw_pharmaceutical_demand` has a natural business key: `(store_id, product_id, date)`. This key enables an UPSERT:

```sql
INSERT INTO raw_pharmaceutical_demand (...)
VALUES (...)
ON CONFLICT (store_id, product_id, date)
DO UPDATE SET ...;
```

The choice changes everything for a pipeline that is run repeatedly.

| Approach | Behavior | Consequence |
|---|---|---|
| `TRUNCATE + APPEND` | Empty the table, then insert | History lost, availability gap |
| `UPSERT` on natural key | Update or insert row by row | Safe replay, history preserved |

Alembic complements the bootstrap file. `sql/schema.sql` initializes a fresh Docker volume. Migrations under `alembic/` handle existing databases. Both paths must stay consistent, especially on the unique constraint.

### 2.5 Quality, observability and security

| Concern | Implementation | Concrete effect |
|---|---|---|
| Quality | `src/validation.py` | Blocks corruption, repairs minor anomalies |
| Logs | `src/logging_config.py` | JSON lines consumable by a collector |
| Correlation | `X-Request-ID` in `src/observability.py` | Link an incident to a request |
| Metrics | Prometheus `GET /metrics` | Latency, volume, errors |
| Liveness | `GET /livez` | The process answers |
| Readiness | `GET /readyz` | Model loaded and database reachable |
| Authentication | `X-API-Key` in `src/security.py` | No anonymous access in production |
| CORS | `CORS_ORIGINS` | Only the allowed dashboard calls the API |
| Quotas | slowapi on `/predict` and `/alerts` | Protection against bursts |
| Images | non-root, pinned dependencies, Trivy | Reduced attack surface |

A `/readyz` that returns `200` while the model is not loaded is a false sense of security. Readiness checks both the in-memory model and a trivial PostgreSQL query.

### 2.6 ML industrialization

| Brick | Implementation | What to understand |
|---|---|---|
| Tracking | MLflow with `MLFLOW_ENABLED=true` | Retrieve parameters, metrics and artifacts |
| Registry | MLflow Model Registry | Promote a model without overwriting the previous one |
| Signature | `infer_signature` | Verify input conformity |
| Parity | `build_history_features()` and `tests/test_parity.py` | Same formulas in training and serving |
| Explainability | SHAP and `reports/feature_weights.json` | Understand and weight drift |
| Offline drift | Evidently on a chronological holdout | Detect historical drift |
| Production drift | `run_production_drift_monitoring()` on JSONL | Compare live data with the reference |
| Status | `model_status.txt` and watcher | Alert without blocking requests |

MLflow remains optional. Training must not stop because a tracking server is unavailable. Traceability matters, but it must not become a single point of failure.

---

<a id="3-mlops-maturity-grid"></a>
## 3. MLOps Maturity Grid

### 3.1 Levels

| Level | Name | What you observe |
|---|---|---|
| **0** | Experimental | Notebooks, manual execution, no tests or monitoring |
| **1** | Automated pipeline | Modular code, traceable artifacts, validation, monitoring |
| **1+** | Stabilized and structured core | Explicit errors, modular API, security, observability, tests |
| **2** | Continuous Training | Drift-triggered retraining, automated promotion, SLOs |
| **3** | Industrialized platform | Multi-team, governance, self-service, managed costs |

### 3.2 Detailed grid

| Dimension | Level 0 | Level 1 | Level 1+ | Level 2 |
|---|---|---|---|---|
| Reproducibility | Manual | Partial | Pinned environments, deterministic split | Fully replayable runs |
| Data | No versioning | Manually copied CSV | Natural key and migrations | Dedicated data versioning |
| Modularity | Monolith | `src/` modules | Routers/services/repositories | Component platform |
| Tests | None | A few tests | 92 tests, 80.91%, integration | Load and chaos testing |
| CI/CD | None | Manual script | Lint, tests, build, Trivy | Continuous deployment with model gate |
| Observability | `print()` | Basic logs | JSON, Prometheus, probes | Traces, SLOs, alerting |
| Security | None | Secrets in `.env` | API key, CORS, quotas, non-root | OAuth2/OIDC, rotation, WAF |
| Monitoring | None | Occasional report | Offline and production drift | Triggered retraining |
| Governance | None | `.joblib` | MLflow tracking and registry | Formal promotion and rollback |

### 3.3 Where PharmaSupply-ML stands

P0 and P1 are closed and validated.

P0 stabilized the core: SMTP out of the request path, explicit errors, `UNKNOWN` instead of a false `STABLE`, typed `/features` contract, PostgreSQL pool, centralized constants, consumed SHAP weights, fixed MOQ floor.

P1 structured the service: routers, services and repositories, train/serve parity, MLflow opt-in, Alembic and UPSERT, JSON logs, Prometheus metrics, probes, API key, CORS, rate limiting, non-root Dockerfiles and Trivy scan, production drift.

The evidence is measurable: 92 passing tests, including 91 unit tests and 1 Testcontainers integration test, for 80.91% coverage. The 70% threshold is configured in `pyproject.toml`. Ruff passes. Docker configuration and Alembic SQL generation have been verified.

The project sits at a consolidated level 1+. The foundations for level 2 are present, but the automatic retraining loop is not closed yet.

### 3.4 What remains for level 2

| Workstream | Why | Success indicator |
|---|---|---|
| Drift-triggered retraining | Reduce drift without manual intervention | An MLflow run is created and promoted after a critical threshold |
| Batch endpoint and distributed cache | Serve large catalogues without saturating the API | Stable p95 latency |
| Dedicated model server | Isolate inference from the business service | Independent scalability |
| Partitioning and dimensions | Keep queries fast at scale | Bounded query time |
| Cloud and SLOs | Industrialize availability | Dashboards and alerts |
| RAG/LLM copilot | Assist decisions during stockouts | Traceable and evaluated recommendations |

---

<a id="4-engineering-principles-for-learners"></a>
## 4. Engineering Principles for Learners

### 4.1 Train/serve parity

Train/serve parity means that features computed during training and during inference are identical. Not “close enough”. Identical.

A classic trap is assuming the formula does not matter. One example from this project: the API computed the seven-day rolling mean as `lag_7 / 7.0`, while training used `demand.shift(1).rolling(7).mean()`. The values look similar, but they describe different things. The model received a shifted distribution.

```python
# Training
rolling_mean_7 = demand.shift(1).rolling(7).mean()

# Old inference, incorrect
rolling_mean_7 = lag_7 / 7.0
```

The fix is a single builder, `build_history_features()` in `src/features.py`. The API uses it whenever a sales history is provided. `tests/test_parity.py` compares both implementations on a synthetic series.

The rule is simple: a feature must have one definition in the code. If you have a “training” version and a “serving” version, you already have a latent bug.

### 4.2 Idempotence

An operation is idempotent if running it again produces the same final state. No duplicate, no loss, no surprise.

In this project, ingestion uses the key `(store_id, product_id, date)` and an UPSERT. You can run `python run_pipeline.py` as many times as needed: existing rows are updated, new rows are inserted, history remains.

The old behavior, `TRUNCATE` then `APPEND`, looked simple. In reality, it opened a window during which the table was empty and erased any data no longer present in the CSV.

Idempotence is not an engineering detail. It is what allows a pipeline to be replayed after an incident without making things worse.

### 4.3 Fail-safe and fail-explicit

These two failure modes serve different needs.

**Fail-safe** protects the user when the system does not know. If `model_status.txt` is missing or unreadable, the status becomes `UNKNOWN`, never `STABLE`. An uncertainty is better than false good news.

**Fail-explicit** protects the business decision when a dependency is broken. If PostgreSQL is unavailable, `/stores`, `/features` or `/alerts` return `503`. If no history exists for a store/product pair, `/features` returns `404`. No fake store list and no default lags.

The project removed the silent fallbacks from the first version. They were comfortable: the API always returned `200`. But it returned invented data. In pharmacy, that comfort is a risk.

A `try/except` that hides an outage is not robustness. It is debt that will be paid at the worst possible time.

### 4.4 Traceability

Traceability answers three questions: which model produced this prediction, with which data, and in which environment?

The project combines several mechanisms. MLflow records parameters, metrics, signature and artifact when `MLFLOW_ENABLED=true`. The `/predict` response exposes `model_version`. JSON logs contain a `request_id`, method, path, status and duration. `reports/prediction_log.jsonl` keeps aggregated features and predictions, without personal data. Alembic keeps the schema history. `src/config.py` centralizes thresholds and paths.

Without these traces, an incident becomes an investigation without evidence. You know something went wrong, but you cannot say when, why, or with which model version.

### 4.5 Observability

Observing a system is not adding `print()`. It is producing exploitable signals.

JSON logs answer “what happened at this instant?”. Prometheus metrics answer “is the service slow, saturated, failing?”. Probes answer “can the service receive traffic?”.

The pipeline uses this distinction. After rebuilding the API image, it polls `/readyz` and waits until the model is loaded and the database is reachable. Without that probe, the smoke test could hit a container still starting and fail with `Connection reset by peer`.

A useful metric is not decorative. It must drive a decision: alert, scale, roll back or fix.

### 4.6 Security

API security is not just a password. It covers access, secrets, quotas and the build chain.

`X-API-Key` blocks anonymous access when `API_KEY` is set. CORS restricts allowed origins. slowapi applies quotas to `/predict` and `/alerts`. Secrets live in `.env`, never in code, and `.env.example` documents expected variables. Docker images run as a non-root user. Dependencies are pinned. Trivy scans images in CI.

None of these measures is sufficient alone. Together, they reduce the attack surface and make incidents easier to contain.

### 4.7 Contracts and validation

An explicit contract describes what a component accepts and returns. In Python, Pydantic v2 plays this role for the API. `PredictionInput` enforces bounds (`ge=0`, `ge=1`) and a minimum history length of 30 values. A negative stock or a too-short history is rejected with `422` before reaching the model.

The contract does not stop at the API. `FeaturesOutput` includes `critical_expiry_days` and `sales_history`. The quality gate in `src/validation.py` blocks structural corruption and repairs minor anomalies. The SQL schema enforces non-null columns, a natural unique key and an index.

A clear contract is better than a comment lost in a notebook. It is verifiable, testable and visible in OpenAPI.

### 4.8 Tests

Tests are the memory of expected behavior. They do not prove the code is perfect; they prevent a change from silently breaking a business rule.

The suite covers several levels. Unit tests in `tests/test_services.py`, `tests/test_features.py` and `tests/test_validation.py` verify rules and transformations. `tests/test_api.py` checks HTTP codes, schemas and explicit errors. `tests/test_parity.py` compares training and inference features. `tests/test_ingestion_integration.py` runs a real PostgreSQL through Testcontainers. `tests/test_security.py` and `tests/test_observability.py` cover authentication, CORS and the prediction log.

The current result is 92 passing tests for 80.91% coverage. The 70% threshold is configured in `pyproject.toml`. The number is not an end in itself, but it prevents the most obvious regressions.

### 4.9 Modularity and separation of concerns

A module should have one reason to change. That is the most useful definition of modularity.

Before P1, `api.py` concentrated transport, SQL, business rules and schemas. The file was over 800 lines. Today, `src/api.py` assembles the application, `src/api_routers/` handles transport, `src/services/` carries rules, `src/repositories/` carries SQL, and `src/schemas.py` carries contracts.

The split has a cost: more files, more imports, less immediate navigation. It has a benefit: a replenishment rule can be tested without FastAPI or PostgreSQL, and a SQL query can change without touching routes.

If you must modify three files to change one rule, modularity is insufficient. If you must modify one file to change a rule, you are probably in the right place.

### 4.10 Reproducibility

A result is reproducible if the same code, data and environment produce the same result. This property does not happen by accident.

The project pins dependencies in `environment.yml`, `requirements.txt`, `requirements.api.txt` and `requirements.dashboard.txt`. XGBoost and SHAP sampling use `random_state=42`. The split is temporal and deterministic. `.env.example` documents variables. MLflow records parameters and signature. Alembic versions the schema. CI runs lint, tests, build and scan.

“It works on my machine” is not an engineering property. It is an anecdote.

### 4.11 Checklist for moving from lab to production

- [ ] Training and inference features have a single definition.
- [ ] Ingestion is idempotent, with a natural key and an UPSERT.
- [ ] Errors are explicit and never return fake data.
- [ ] Inputs and outputs are typed.
- [ ] Business rules are tested without a database or HTTP.
- [ ] Secrets are outside the code.
- [ ] The API is authenticated and rate-limited.
- [ ] Logs are structured and correlatable.
- [ ] Metrics and health probes exist.
- [ ] The model is versioned and signed.
- [ ] Offline and production drift are measured.
- [ ] Images are non-root and scanned.

---

<a id="5-from-lab-to-production"></a>
## 5. From Lab to Production

### 5.1 Three reflexes for your next project

The first reflex is to define contracts before writing the model. An input schema, an output schema and a validation rule save weeks of debugging.

The second is to version data and artifacts, not only code. A model without its dataset and environment is hard to reproduce. MLflow, Alembic and pinned dependency files address this need.

The third is to instrument before an incident. JSON logs, metrics, probes and a prediction log cost little at the start. They cost a lot when added after an outage.

### 5.2 A realistic migration plan

Start by extracting business logic from the notebook into pure functions. Test them with business examples, not only with data shapes.

Then separate data access from transport. A repository or query module must be testable with a stub cursor. This avoids depending on a live database for every test.

Add a quality gate next. An empty dataset, a missing column or a negative stock must be handled before feature engineering.

Finally, wire serving and observability. A `/predict` route without validation, logs and probes is not production-ready, even if the model is excellent.

### 5.3 Warning signs

Some signals should trigger an immediate review:

- a route that contains SQL or replenishment rules;
- a `try/except` that returns a default value without saying so;
- a feature computed differently in `src/features.py` and in the router;
- an ingestion that empties the table before reloading;
- a health status that shows `STABLE` while the status file is missing;
- tests that accept two possible outcomes so they never fail;
- secrets present in a versioned file.

These signals are not theoretical. They all existed in this project before P0 and P1.

---

<a id="6-appendices"></a>
## 6. Appendices

### 6.1 Glossary

| Term | Definition |
|---|---|
| **Idempotence** | Property of an operation that can be replayed without changing the final result |
| **UPSERT** | `INSERT ... ON CONFLICT DO UPDATE`: insert or update based on a key |
| **Train/serve skew** | Difference in feature processing between training and inference |
| **Fail-safe** | Prudent behavior under uncertainty, for example the `UNKNOWN` status |
| **Fail-explicit** | Clear and immediate error on failure, for example a `503` |
| **Drift** | Change in data distribution or model performance |
| **FEFO** | *First Expired, First Out*: prioritize batches closest to expiry |
| **MOQ** | *Minimum Order Quantity*: minimum order quantity |
| **SLO** | *Service Level Objective*: measurable service-level target |
| **Registry** | Versioned catalogue of models ready for deployment |

### 6.2 Useful commands

```bash
# Environment
conda env create -f environment.yml
conda activate pharma-supply
conda env update -f environment.yml --prune

# Infrastructure
docker compose up -d --build

# Migrations
alembic upgrade head
alembic upgrade head --sql

# Full pipeline
python run_pipeline.py

# Tests and lint
ruff check src tests run_pipeline.py alembic
pytest -q -m "not integration" --cov=src --cov-report=term-missing
pytest -q -m integration

# MLflow (opt-in)
MLFLOW_ENABLED=true python -m src.train
mlflow ui --backend-store-uri file:./mlruns
```

### 6.3 Code files to read first

- `src/api.py`: composition root, lifespan and model loading.
- `src/api_routers/predict.py`: inference contract and rule orchestration.
- `src/services/inventory.py`: MOQ, FEFO, alerts and feature construction.
- `src/repositories/database.py`: pool, cursors and error handling.
- `src/features.py`: `create_features()` and `build_history_features()`.
- `tests/test_parity.py`: train/serve parity proof.
- `alembic/versions/0001_initial_schema.py`: natural key and SQL constraints.
- `src/config.py`: single source of truth for constants and environment variables.

</details>
