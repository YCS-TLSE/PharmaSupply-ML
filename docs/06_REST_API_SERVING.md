# Technical Documentation
## Phase 06: REST API Serving
**Project:** PharmaSupply-ML
**Framework:** Standard MLOps Blueprint
**File:** `docs/06_REST_API_SERVING.md`
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

*Un guide pour découper une API monolithique en couches testables, sécuriser les routes et rendre le serving observable sans masquer les pannes.*

---

## Sommaire

1. [Pourquoi découper un monolithe d'API](#1-pourquoi-decouper-un-monolithe-dapi)
2. [L'architecture en couches de PharmaSupply-ML](#2-larchitecture-en-couches-de-pharmasupply-ml)
3. [Les contrats et schémas Pydantic v2](#3-les-contrats-et-schemas-pydantic-v2)
4. [Sécurisation de l'API](#4-securisation-de-lapi)
5. [Observabilité, métriques et sondes](#5-observabilite-metriques-et-sondes)
6. [Annexes et checklist de revue API](#6-annexes-et-checklist-de-revue-api)

---

<a id="1-pourquoi-decouper-un-monolithe-dapi"></a>
## 1. Pourquoi découper un monolithe d'API

### 1.1 L'analogie du grand hôpital

Imaginez un hôpital où la réception pose les diagnostics, où les archives réalisent les actes médicaux et où le standard téléphonique décide des stocks de pharmacie. Le bâtiment fonctionne peut-être, mais personne ne sait qui est responsable en cas d'erreur.

Une route FastAPI monolithique ressemble à cet hôpital. Elle reçoit la requête, valide le JSON, écrit une requête SQL, applique une règle de MOQ, appelle le modèle et formate la réponse. Au début, c'est rapide à écrire. Au premier incident, c'est impossible à diagnostiquer.

### 1.2 Les symptômes d'une route trop chargée

Une route qui contient du SQL et des règles métier produit plusieurs symptômes.

Le premier est la **difficulté de test**. Pour vérifier un calcul de MOQ, il faut lancer une base, un serveur et un modèle. Le test devient lent et fragile.

Le deuxième est la **duplication**. Une deuxième route a besoin du même calcul ; elle le réécrit, souvent avec une variante. Les deux versions divergent.

Le troisième est le **couplage au transport**. Une règle métier change parce qu'un champ JSON a changé. Le métier devient dépendant de la forme de la requête.

Le quatrième est l'**observabilité floue**. Une erreur peut venir de la base, du modèle, de la validation ou de la règle métier. Les logs ne permettent pas de distinguer la couche responsable.

### 1.3 Le bon réflexe

Une route doit orchestrer, pas tout faire. Elle reçoit une requête valide, appelle un service, traduit le résultat en réponse HTTP et laisse les détails ailleurs.

Cette discipline a un coût initial : plus de fichiers, plus d'imports, une navigation moins directe. Elle se rembourse dès le premier test unitaire et dès le premier incident.

---

<a id="2-larchitecture-en-couches-de-pharmasupply-ml"></a>
## 2. L'architecture en couches de PharmaSupply-ML

### 2.1 Le flux d'une requête

```text
Requête HTTP
    │
    v
src/api_routers/          ← transport : validation, codes HTTP, auth, quotas
    │
    v
src/services/             ← règles métier : MOQ, FEFO, alertes, features
    │
    v
src/repositories/         ← SQL, pool, curseurs, erreurs de base
    │
    v
PostgreSQL 16
```

La réponse remonte le chemin inverse. Chaque couche ne connaît que la suivante. Un router ne sait pas comment fonctionne le pool. Un repository ne sait pas qu'un quota slowapi existe.

### 2.2 La composition root

`src/api.py` assemble l'application. Il ne contient pas de route métier. Il fait quatre choses :

- configure le logging JSON ;
- crée l'application FastAPI ;
- branche la sécurité et l'observabilité ;
- inclut les routers.

Le `lifespan` charge le modèle XGBoost au démarrage et démarre le watcher de statut. Au shutdown, il annule le watcher et libère la référence au modèle.

Cette séparation permet de comprendre le démarrage du service sans lire toutes les routes.

### 2.3 Le transport : src/api_routers/

Les routers traduisent HTTP en appels de service.

| Router | Routes | Rôle |
|---|---|---|
| `health.py` | `/model/health`, `/livez`, `/readyz` | Santé et sondes |
| `data.py` | `/stores`, `/stores/{store_id}/products` | Référentiels |
| `alerts.py` | `/stores/{store_id}/alerts` | Audit et alertes |
| `features.py` | `/features/{store_id}/{product_id}` | Features et historique |
| `predict.py` | `/predict` | Inférence et réapprovisionnement |

Un router peut :

- déclarer un `response_model` ;
- dépendre de `require_api_key` ;
- appliquer un quota slowapi ;
- traduire une exception métier en code HTTP.

Il ne doit pas :

- écrire de SQL ;
- calculer un MOQ ;
- appeler directement psycopg2 ;
- contenir une formule de feature.

### 2.4 Les règles métier : src/services/

`src/services/inventory.py` contient les décisions métier pures.

| Fonction | Responsabilité |
|---|---|
| `normalize_store_id()` | Générer les variantes d'identifiant d'officine |
| `compute_reorder()` | Calculer la quantité commandée avec MOQ, lot et shelf life |
| `compute_fefo_loss()` | Estimer la perte des lots proches de la péremption |
| `build_alerts()` | Classer les alertes par niveau et sévérité |
| `features_from_rows()` | Construire la réponse `/features` |
| `build_prediction_features()` | Construire le vecteur de features d'inférence |

Ces fonctions ne dépendent ni de FastAPI ni de PostgreSQL. Elles peuvent être testées avec des dictionnaires et des DataFrames.

Le calcul de réapprovisionnement illustre le bénéfice. Une route pourrait appeler `compute_reorder()` sans connaître la formule. Un test peut vérifier le plancher MOQ sans lancer de serveur.

### 2.5 La persistance : src/repositories/

`src/repositories/database.py` gère l'engine, le pool et les curseurs. `src/repositories/inventory.py` contient les requêtes read-only du serving.

Les routers appellent les repositories, puis passent les lignes aux services. Cette séparation permet de remplacer un curseur par un stub dans les tests.

`DatabaseUnavailableError` est traduit en `503` par le router. Une panne de base ne doit pas produire une réponse `200` avec des données fictives.

### 2.6 Le cas de /predict

Le flux de `/predict` résume l'architecture.

1. Pydantic valide `PredictionInput`.
2. La dépendance `require_api_key` vérifie la clé.
3. Le limiteur slowapi applique le quota.
4. Le router récupère le modèle via `get_loaded_model`.
5. Le service `build_prediction_features` construit les features.
6. Le modèle produit une prédiction.
7. `compute_reorder` applique les règles MOQ, lot et shelf life.
8. `compute_fefo_loss` estime la perte.
9. Une tâche d'arrière-plan écrit le journal de prédiction.
10. Le router retourne `PredictionOutput`.

Chaque étape a une responsabilité claire. Si une prédiction est fausse, on peut isoler la validation, les features, le modèle ou les règles.

---

<a id="3-les-contrats-et-schemas-pydantic-v2"></a>
## 3. Les contrats et schémas Pydantic v2

### 3.1 À quoi sert un schéma

Un schéma décrit ce que l'API accepte et ce qu'elle retourne. Il ne documente pas seulement : il valide avant l'exécution du code métier.

Sans schéma, une route peut recevoir un stock négatif, un historique vide ou un identifiant manquant. Le modèle reçoit alors une entrée invalide et produit une réponse qui ne veut rien dire.

### 3.2 Les modèles d'entrée

`PredictionInput` décrit le payload de `/predict`.

| Champ | Type | Contrainte |
|---|---|---|
| `store_id` | `int` ou `str` | Requis |
| `product_id` | `int` ou `str` | Requis |
| `stock_on_hand` | `float` | `ge=0` |
| `supplier_lead_time` | `float` | `ge=0` |
| `safety_threshold` | `float` | `ge=0` |
| `batch_size` | `int` | `ge=1`, défaut 1 |
| `moq` | `int` | `ge=1`, défaut 1 |
| `shelf_life_days` | `int` | `ge=1`, défaut 365 |
| `lag_7`, `lag_14`, `lag_30` | `float` | `ge=0` |
| `history` | liste de `float` | optionnelle, `min_length=30` |
| `batches` | liste de `BatchItem` | optionnelle |

`BatchItem` impose une quantité positive ou nulle et une expiration positive ou nulle.

Un payload invalide est rejeté en `422` avant d'atteindre le modèle. C'est le premier garde-fou du serving.

### 3.3 Les modèles de sortie

Les réponses sont typées :

- `PredictionOutput` : demande prédite, stock projeté, indicateur de commande, quantité, perte FEFO, statut et version du modèle ;
- `StoresOutput` et `ProductsOutput` : listes de référentiels ;
- `FeaturesOutput` : stock, délai, seuil, MOQ, lot, shelf life, expiration critique, lags et historique de ventes ;
- `StoreAlertsOutput` : liste d'`AlertItem` ;
- `ModelHealthOutput`, `LivezOutput`, `ReadyzOutput` : santé et sondes.

Le `response_model` ne sert pas qu'à la documentation OpenAPI. Il filtre les champs inattendus et garantit que la réponse respecte le contrat.

### 3.4 Le champ history

`history` est le champ qui active la parité train/serve. Il contient les ventes quotidiennes dans l'ordre chronologique, de la plus ancienne à la plus récente.

La contrainte `min_length=30` empêche un client d'envoyer cinq valeurs et d'obtenir des lags incomplets. Le constructeur `build_history_features()` applique la même contrainte côté service.

Le chemin sans `history` reste accepté pour compatibilité. Il utilise les lags fournis. La voie recommandée reste `history`.

### 3.5 Les codes d'erreur

| Code | Situation | Exemple |
|---|---|---|
| `401` | Clé API absente ou invalide | `X-API-Key` incorrect |
| `422` | Payload invalide | Stock négatif, historique trop court |
| `404` | Ressource absente | Aucun historique pour un couple store/produit |
| `429` | Quota dépassé | Trop d'appels à `/predict` |
| `503` | Dépendance indisponible | Modèle non chargé, base inaccessible |
| `400` | Erreur d'inférence inattendue | Exception non prévue dans le calcul |

Un code explicite vaut mieux qu'un `200` ambigu. Un client peut réagir à un `503`, il ne peut pas réagir à une réponse plausible mais fausse.

### 3.6 Les limites du contrat actuel

Les schémas ne interdisent pas encore les valeurs `NaN` ou `inf` dans tous les champs flottants. Pydantic peut les accepter selon la configuration. Une évolution consisterait à ajouter `allow_inf_nan=False` sur les champs concernés.

Le contrat ne vérifie pas non plus la cohérence entre `lag_7`, `lag_14` et `lag_30`. Un client pourrait envoyer des valeurs incohérentes. Le chemin `history` réduit ce risque, puisqu'il recalcule les lags à partir de la série.

---

<a id="4-securisation-de-lapi"></a>
## 4. Sécurisation de l'API

### 4.1 Authentification par X-API-Key

`src/security.py` définit une dépendance `require_api_key()`.

```python
api_key_header = APIKeyHeader(
    name=config.API_KEY_HEADER, auto_error=False
)
```

Si `API_KEY` est vide, l'authentification est désactivée pour le développement local. Si `API_KEY` est défini, une requête sans clé ou avec une clé incorrecte reçoit `401`.

Les routers métier déclarent la dépendance au niveau du router :

```python
router = APIRouter(
    tags=["Inference"],
    dependencies=[Depends(require_api_key)],
)
```

Cette approche protège toutes les routes du router, y compris celles ajoutées plus tard. Elle évite d'oublier une route.

### 4.2 Ce que la clé API ne couvre pas

Une clé statique est un socle, pas une solution complète. Elle ne gère pas :

- la rotation ;
- les rôles et permissions ;
- l'expiration ;
- la révocation fine ;
- l'authentification utilisateur.

Pour une exposition publique, il faut passer à OAuth2/OIDC, avec un fournisseur d'identité et une gestion de secrets adaptée.

### 4.3 CORS

Le middleware CORS limite les origines autorisées.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

Par défaut, seul le dashboard local est autorisé. Une origine inconnue ne peut pas appeler l'API depuis un navigateur.

CORS ne protège pas contre un appel serveur à serveur. Il protège le navigateur. La clé API reste nécessaire pour les appels directs.

### 4.4 Rate limiting avec slowapi

Le limiteur est appliqué aux routes coûteuses.

```python
@router.post("/predict", response_model=PredictionOutput)
@limiter.limit(config.RATE_LIMIT_PREDICT)
def predict(...):
    ...
```

Les valeurs par défaut sont `60/minute` pour `/predict` et `120/minute` pour `/alerts`. Elles sont configurables.

Le dépassement produit `429`, avec un message explicite. Le handler personnalisé évite de laisser fuiter la réponse par défaut de la bibliothèque.

### 4.5 Les limites du rate limiting actuel

Le limiteur est en mémoire, par processus. Avec plusieurs instances, chaque instance possède son propre compteur. Un client pourrait dépasser le quota global en répartissant ses appels.

Une production multi-instances devrait utiliser un backend partagé, par exemple Redis.

### 4.6 Les images et les secrets

Les images Docker utilisent un utilisateur non-root. Les dépendances sont épinglées. Les secrets viennent de l'environnement, pas du code. Trivy scanne les images en CI.

Ces mesures ne remplacent pas une revue de sécurité, mais elles réduisent la surface d'attaque.

---

<a id="5-observabilite-metriques-et-sondes"></a>
## 5. Observabilité, métriques et sondes

### 5.1 Les logs JSON

`src/logging_config.py` configure un formateur JSON. Chaque ligne contient :

- un horodatage UTC ;
- le niveau ;
- le logger ;
- le message ;
- éventuellement un `request_id`, la méthode, le chemin, le statut et la durée.

Les logs sont lisibles par un humain et par un collecteur. Un `print()` dans une route ne fournit ni niveau, ni corrélation, ni structure.

### 5.2 Le X-Request-ID

`ObservabilityMiddleware` lit l'en-tête `X-Request-ID` ou génère un identifiant. Il l'ajoute à la réponse et aux logs.

Ce détail change la vie lors d'un incident. Un utilisateur signale une erreur à un instant donné. Le support retrouve la requête exacte dans les logs, avec sa durée et son statut.

### 5.3 Les métriques Prometheus

`src/observability.py` expose quatre instruments :

| Métrique | Type | Usage |
|---|---|---|
| `pharma_api_requests_total` | Compteur | Volume par méthode, chemin et statut |
| `pharma_api_request_duration_seconds` | Histogramme | Latence des requêtes |
| `pharma_api_predicted_demand` | Histogramme | Distribution des prédictions |
| `pharma_api_reorders_total` | Compteur | Nombre de recommandations de commande |

Les métriques sont exposées sur `GET /metrics`. Prometheus peut les scraper. Grafana peut les visualiser.

### 5.4 Les sondes de santé

| Route | Question | Réponse |
|---|---|---|
| `/livez` | Le processus répond-il ? | `200 {"status":"alive"}` |
| `/readyz` | Le modèle est-il chargé et la base joignable ? | `200` ou `503` avec le détail |
| `/model/health` | Quel est le statut de dérive ? | `STABLE`, `WARNING`, `CRITICAL` ou `UNKNOWN` |

`/readyz` vérifie deux dépendances. Si le modèle n'est pas chargé ou si PostgreSQL ne répond pas, la route renvoie `503`. Le pipeline s'en sert avant de lancer le smoke test.

`/model/health` est sans effet de bord. Il ne déclenche pas d'e-mail. Les alertes sont gérées par le watcher de fond.

### 5.5 Le journal de prédiction

Chaque prédiction planifie une écriture JSONL dans `reports/prediction_log.jsonl`. Le contenu est agrégé : features, prédiction, indicateur de commande. Aucune donnée personnelle n'est enregistrée.

L'écriture passe par `BackgroundTasks`. Elle ne bloque pas la réponse HTTP.

### 5.6 Ce que l'observabilité ne couvre pas encore

Le projet n'a pas de traces distribuées, de tableaux de bord Grafana versionnés ni d'alertes SLO. Les métriques existent, mais l'exploitation doit encore être branchée.

Le endpoint `/metrics` n'est pas authentifié. C'est courant pour un scrape interne, mais une exposition publique devrait le protéger ou le restreindre au réseau de monitoring.

---

<a id="6-annexes-et-checklist-de-revue-api"></a>
## 6. Annexes et checklist de revue API

### 6.1 Lancer l'API

```bash
# Développement local
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Via Docker
docker compose up -d --build api

# Vérifier l'état
docker compose ps
docker compose logs api --tail=50
```

### 6.2 Appels cURL

```bash
# Santé
curl http://localhost:8000/livez
curl http://localhost:8000/readyz
curl http://localhost:8000/model/health

# Métriques
curl http://localhost:8000/metrics

# Référentiels
curl http://localhost:8000/stores
curl http://localhost:8000/stores/1/products

# Features
curl http://localhost:8000/features/1/MED_001

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "1",
    "product_id": "MED_001",
    "stock_on_hand": 500.0,
    "supplier_lead_time": 3.0,
    "safety_threshold": 150.0,
    "batch_size": 50,
    "moq": 100,
    "shelf_life_days": 365,
    "lag_1": 100.0,
    "lag_7": 100.0,
    "lag_14": 100.0,
    "lag_30": 100.0,
    "history": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
  }'

# Avec authentification
curl -H "X-API-Key: $API_KEY" http://localhost:8000/stores
```

### 6.3 Tests API

```bash
# Tests API et sécurité
pytest -q tests/test_api.py tests/test_security.py

# Suite unitaire complète avec couverture
pytest -q -m "not integration" --cov=src --cov-report=term-missing

# Lint
ruff check src tests run_pipeline.py alembic
```

### 6.4 Checklist de revue d'une route API

- [ ] La route ne contient ni SQL ni règle métier complexe.
- [ ] Le payload d'entrée est un modèle Pydantic.
- [ ] La réponse possède un `response_model`.
- [ ] Les champs critiques ont des bornes (`ge`, `min_length`).
- [ ] Les erreurs sont traduites en codes HTTP explicites.
- [ ] La route métier est protégée par `require_api_key`.
- [ ] Les routes coûteuses ont un quota slowapi.
- [ ] La route n'envoie pas d'e-mail dans le chemin de requête.
- [ ] Les logs contiennent un `request_id`.
- [ ] Une panne de base produit `503`, pas une réponse factice.
- [ ] Un historique absent produit `404`, pas des valeurs par défaut.
- [ ] Le modèle absent produit `503`.
- [ ] Les tests couvrent le cas nominal et au moins un cas d'erreur.
- [ ] La route est documentée dans OpenAPI par ses modèles.

### 6.5 Glossaire

| Terme | Définition |
|---|---|
| **Composition root** | Point d'assemblage de l'application |
| **Router** | Groupe de routes HTTP |
| **Service** | Couche de règles métier pures |
| **Repository** | Couche d'accès aux données |
| **Pydantic** | Bibliothèque de validation et de sérialisation |
| **X-API-Key** | En-tête d'authentification par clé |
| **CORS** | Politique d'accès entre origines navigateur |
| **Rate limiting** | Limitation du nombre d'appels sur une période |
| **Liveness** | Sonde qui vérifie que le processus répond |
| **Readiness** | Sonde qui vérifie que le service peut recevoir du trafic |
| **Prometheus** | Système de collecte de métriques |
| **Request ID** | Identifiant de corrélation d'une requête |

### 6.6 Fichiers de code et tests à lire

- `src/api.py` : composition root, lifespan et assemblage.
- `src/api_routers/health.py` : santé, liveness et readiness.
- `src/api_routers/data.py` : référentiels et gestion du `503`.
- `src/api_routers/alerts.py` : alertes et rate limiting.
- `src/api_routers/features.py` : contrat `/features` et distinction `404`/`503`.
- `src/api_routers/predict.py` : inférence, MOQ, FEFO et journal asynchrone.
- `src/services/inventory.py` : règles métier pures.
- `src/repositories/database.py` et `src/repositories/inventory.py` : persistance.
- `src/schemas.py` : contrats Pydantic v2.
- `src/security.py` : API key, CORS et slowapi.
- `src/observability.py` : logs JSON, métriques et journal de prédiction.
- `src/logging_config.py` : formatage JSON.
- `tests/test_api.py` et `tests/test_security.py` : non-régression du serving.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *A guide to splitting a monolithic API into testable layers, securing routes, and making serving observable without hiding failures.*

---

## Table of Contents

1. [Why split an API monolith](#1-why-split-an-api-monolith)
2. [The layered architecture of PharmaSupply-ML](#2-the-layered-architecture-of-pharmasupply-ml)
3. [Pydantic v2 contracts and schemas](#3-pydantic-v2-contracts-and-schemas)
4. [Securing the API](#4-securing-the-api)
5. [Observability, metrics and probes](#5-observability-metrics-and-probes)
6. [Appendices and API review checklist](#6-appendices-and-api-review-checklist)

---

<a id="1-why-split-an-api-monolith"></a>
## 1. Why split an API monolith

### 1.1 The large hospital analogy

Imagine a hospital where the front desk makes diagnoses, where the records office performs medical procedures and where the switchboard decides pharmacy stock levels. The building may work, but nobody knows who is responsible when something goes wrong.

A monolithic FastAPI route looks like that hospital. It receives the request, validates the JSON, writes a SQL query, applies an MOQ rule, calls the model and formats the response. At first, it is quick to write. At the first incident, it is impossible to diagnose.

### 1.2 The symptoms of an overloaded route

A route that contains SQL and business rules produces several symptoms.

The first is **testing difficulty**. To check an MOQ calculation, you have to start a database, a server and a model. The test becomes slow and fragile.

The second is **duplication**. A second route needs the same calculation; it rewrites it, often with a variant. The two versions drift apart.

The third is **coupling to the transport**. A business rule changes because a JSON field changed. The business logic becomes dependent on the shape of the request.

The fourth is **blurry observability**. An error can come from the database, the model, the validation or the business rule. The logs do not let you identify the responsible layer.

### 1.3 The right reflex

A route must orchestrate, not do everything. It receives a valid request, calls a service, translates the result into an HTTP response and leaves the details elsewhere.

This discipline has an initial cost: more files, more imports, less direct navigation. It pays off at the first unit test and the first incident.

---

<a id="2-the-layered-architecture-of-pharmasupply-ml"></a>
## 2. The layered architecture of PharmaSupply-ML

### 2.1 The flow of a request

```text
HTTP request
    │
    v
src/api_routers/          ← transport: validation, HTTP codes, auth, quotas
    │
    v
src/services/             ← business rules: MOQ, FEFO, alerts, features
    │
    v
src/repositories/         ← SQL, pool, cursors, database errors
    │
    v
PostgreSQL 16
```

The response travels back up the same path. Each layer knows only the next one. A router does not know how the pool works. A repository does not know that a slowapi quota exists.

### 2.2 The composition root

`src/api.py` assembles the application. It contains no business route. It does four things:

- configures JSON logging;
- creates the FastAPI application;
- wires security and observability;
- includes the routers.

The `lifespan` loads the XGBoost model at startup and starts the status watcher. On shutdown, it cancels the watcher and releases the model reference.

This separation lets you understand the service startup without reading every route.

### 2.3 Transport: src/api_routers/

The routers translate HTTP into service calls.

| Router | Routes | Role |
|---|---|---|
| `health.py` | `/model/health`, `/livez`, `/readyz` | Health and probes |
| `data.py` | `/stores`, `/stores/{store_id}/products` | Reference data |
| `alerts.py` | `/stores/{store_id}/alerts` | Audit and alerts |
| `features.py` | `/features/{store_id}/{product_id}` | Features and history |
| `predict.py` | `/predict` | Inference and reordering |

A router may:

- declare a `response_model`;
- depend on `require_api_key`;
- apply a slowapi quota;
- translate a business exception into an HTTP code.

It must not:

- write SQL;
- compute an MOQ;
- call psycopg2 directly;
- contain a feature formula.

### 2.4 Business rules: src/services/

`src/services/inventory.py` contains the pure business decisions.

| Function | Responsibility |
|---|---|
| `normalize_store_id()` | Generate pharmacy identifier variants |
| `compute_reorder()` | Compute the order quantity with MOQ, batch and shelf life |
| `compute_fefo_loss()` | Estimate the loss of batches close to expiry |
| `build_alerts()` | Rank alerts by level and severity |
| `features_from_rows()` | Build the `/features` response |
| `build_prediction_features()` | Build the inference feature vector |

These functions depend on neither FastAPI nor PostgreSQL. They can be tested with dictionaries and DataFrames.

The reorder calculation illustrates the benefit. A route could call `compute_reorder()` without knowing the formula. A test can check the MOQ floor without starting a server.

### 2.5 Persistence: src/repositories/

`src/repositories/database.py` manages the engine, the pool and the cursors. `src/repositories/inventory.py` contains the read-only serving queries.

The routers call the repositories, then pass the rows to the services. This separation lets you replace a cursor with a stub in tests.

`DatabaseUnavailableError` is translated into `503` by the router. A database outage must not produce a `200` response with fictitious data.

### 2.6 The /predict case

The `/predict` flow summarizes the architecture.

1. Pydantic validates `PredictionInput`.
2. The `require_api_key` dependency checks the key.
3. The slowapi limiter applies the quota.
4. The router retrieves the model via `get_loaded_model`.
5. The `build_prediction_features` service builds the features.
6. The model produces a prediction.
7. `compute_reorder` applies the MOQ, batch and shelf life rules.
8. `compute_fefo_loss` estimates the loss.
9. A background task writes the prediction log.
10. The router returns `PredictionOutput`.

Each step has a clear responsibility. If a prediction is wrong, you can isolate the validation, the features, the model or the rules.

---

<a id="3-pydantic-v2-contracts-and-schemas"></a>
## 3. Pydantic v2 contracts and schemas

### 3.1 What a schema is for

A schema describes what the API accepts and what it returns. It does not only document: it validates before the business code runs.

Without a schema, a route can receive a negative stock, an empty history or a missing identifier. The model then receives an invalid input and produces a meaningless response.

### 3.2 Input models

`PredictionInput` describes the `/predict` payload.

| Field | Type | Constraint |
|---|---|---|
| `store_id` | `int` or `str` | Required |
| `product_id` | `int` or `str` | Required |
| `stock_on_hand` | `float` | `ge=0` |
| `supplier_lead_time` | `float` | `ge=0` |
| `safety_threshold` | `float` | `ge=0` |
| `batch_size` | `int` | `ge=1`, default 1 |
| `moq` | `int` | `ge=1`, default 1 |
| `shelf_life_days` | `int` | `ge=1`, default 365 |
| `lag_7`, `lag_14`, `lag_30` | `float` | `ge=0` |
| `history` | list of `float` | optional, `min_length=30` |
| `batches` | list of `BatchItem` | optional |

`BatchItem` enforces a non-negative quantity and a non-negative expiry.

An invalid payload is rejected with `422` before reaching the model. This is the first serving guardrail.

### 3.3 Output models

Responses are typed:

- `PredictionOutput`: predicted demand, projected stock, order indicator, quantity, FEFO loss, status and model version;
- `StoresOutput` and `ProductsOutput`: reference lists;
- `FeaturesOutput`: stock, lead time, threshold, MOQ, batch, shelf life, critical expiry, lags and sales history;
- `StoreAlertsOutput`: list of `AlertItem`;
- `ModelHealthOutput`, `LivezOutput`, `ReadyzOutput`: health and probes.

The `response_model` is not only for OpenAPI documentation. It filters unexpected fields and guarantees that the response respects the contract.

### 3.4 The history field

`history` is the field that enables train/serve parity. It contains the daily sales in chronological order, from oldest to most recent.

The `min_length=30` constraint prevents a client from sending five values and getting incomplete lags. The `build_history_features()` builder applies the same constraint on the service side.

The path without `history` remains accepted for compatibility. It uses the supplied lags. The recommended path remains `history`.

### 3.5 Error codes

| Code | Situation | Example |
|---|---|---|
| `401` | Missing or invalid API key | Incorrect `X-API-Key` |
| `422` | Invalid payload | Negative stock, history too short |
| `404` | Missing resource | No history for a store/product pair |
| `429` | Quota exceeded | Too many calls to `/predict` |
| `503` | Dependency unavailable | Model not loaded, database unreachable |
| `400` | Unexpected inference error | Unforeseen exception in the calculation |

An explicit code is better than an ambiguous `200`. A client can react to a `503`; it cannot react to a plausible but wrong response.

### 3.6 Limits of the current contract

The schemas do not yet forbid `NaN` or `inf` values in all float fields. Pydantic may accept them depending on the configuration. One improvement would be to add `allow_inf_nan=False` on the affected fields.

The contract also does not check consistency between `lag_7`, `lag_14` and `lag_30`. A client could send inconsistent values. The `history` path reduces this risk, since it recalculates the lags from the series.

---

<a id="4-securing-the-api"></a>
## 4. Securing the API

### 4.1 X-API-Key authentication

`src/security.py` defines a `require_api_key()` dependency.

```python
api_key_header = APIKeyHeader(
    name=config.API_KEY_HEADER, auto_error=False
)
```

If `API_KEY` is empty, authentication is disabled for local development. If `API_KEY` is set, a request without a key or with an incorrect key receives `401`.

The business routers declare the dependency at the router level:

```python
router = APIRouter(
    tags=["Inference"],
    dependencies=[Depends(require_api_key)],
)
```

This approach protects all the routes of the router, including those added later. It avoids forgetting a route.

### 4.2 What the API key does not cover

A static key is a foundation, not a complete solution. It does not handle:

- rotation;
- roles and permissions;
- expiration;
- fine-grained revocation;
- user authentication.

For a public exposure, you need to move to OAuth2/OIDC, with an identity provider and appropriate secret management.

### 4.3 CORS

The CORS middleware limits the allowed origins.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

By default, only the local dashboard is allowed. An unknown origin cannot call the API from a browser.

CORS does not protect against a server-to-server call. It protects the browser. The API key remains necessary for direct calls.

### 4.4 Rate limiting with slowapi

The limiter is applied to costly routes.

```python
@router.post("/predict", response_model=PredictionOutput)
@limiter.limit(config.RATE_LIMIT_PREDICT)
def predict(...):
    ...
```

The defaults are `60/minute` for `/predict` and `120/minute` for `/alerts`. They are configurable.

Exceeding the limit produces `429`, with an explicit message. The custom handler avoids leaking the library's default response.

### 4.5 Limits of the current rate limiting

The limiter is in-memory, per process. With several instances, each instance has its own counter. A client could exceed the global quota by spreading its calls.

A multi-instance production should use a shared backend, for example Redis.

### 4.6 Images and secrets

The Docker images use a non-root user. Dependencies are pinned. Secrets come from the environment, not the code. Trivy scans the images in CI.

These measures do not replace a security review, but they reduce the attack surface.

---

<a id="5-observability-metrics-and-probes"></a>
## 5. Observability, metrics and probes

### 5.1 JSON logs

`src/logging_config.py` configures a JSON formatter. Each line contains:

- a UTC timestamp;
- the level;
- the logger;
- the message;
- optionally a `request_id`, the method, the path, the status and the duration.

Logs are readable by a human and by a collector. A `print()` in a route provides neither level, nor correlation, nor structure.

### 5.2 X-Request-ID

`ObservabilityMiddleware` reads the `X-Request-ID` header or generates an identifier. It adds it to the response and to the logs.

This detail changes everything during an incident. A user reports an error at a given moment. Support finds the exact request in the logs, with its duration and status.

### 5.3 Prometheus metrics

`src/observability.py` exposes four instruments:

| Metric | Type | Usage |
|---|---|---|
| `pharma_api_requests_total` | Counter | Volume by method, path and status |
| `pharma_api_request_duration_seconds` | Histogram | Request latency |
| `pharma_api_predicted_demand` | Histogram | Distribution of predictions |
| `pharma_api_reorders_total` | Counter | Number of order recommendations |

The metrics are exposed on `GET /metrics`. Prometheus can scrape them. Grafana can visualize them.

### 5.4 Health probes

| Route | Question | Response |
|---|---|---|
| `/livez` | Is the process responding? | `200 {"status":"alive"}` |
| `/readyz` | Is the model loaded and the database reachable? | `200` or `503` with details |
| `/model/health` | What is the drift status? | `STABLE`, `WARNING`, `CRITICAL` or `UNKNOWN` |

`/readyz` checks two dependencies. If the model is not loaded or if PostgreSQL does not respond, the route returns `503`. The pipeline uses it before running the smoke test.

`/model/health` has no side effect. It does not trigger an email. Alerts are handled by the background watcher.

### 5.5 The prediction log

Each prediction schedules a JSONL write in `reports/prediction_log.jsonl`. The content is aggregated: features, prediction, order indicator. No personal data is recorded.

The write goes through `BackgroundTasks`. It does not block the HTTP response.

### 5.6 What observability does not cover yet

The project has no distributed tracing, no versioned Grafana dashboards and no SLO alerts. The metrics exist, but operations still need to be wired.

The `/metrics` endpoint is not authenticated. This is common for an internal scrape, but a public exposure should protect it or restrict it to the monitoring network.

---

<a id="6-appendices-and-api-review-checklist"></a>
## 6. Appendices and API review checklist

### 6.1 Starting the API

```bash
# Local development
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Via Docker
docker compose up -d --build api

# Check the state
docker compose ps
docker compose logs api --tail=50
```

### 6.2 cURL calls

```bash
# Health
curl http://localhost:8000/livez
curl http://localhost:8000/readyz
curl http://localhost:8000/model/health

# Metrics
curl http://localhost:8000/metrics

# Reference data
curl http://localhost:8000/stores
curl http://localhost:8000/stores/1/products

# Features
curl http://localhost:8000/features/1/MED_001

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "1",
    "product_id": "MED_001",
    "stock_on_hand": 500.0,
    "supplier_lead_time": 3.0,
    "safety_threshold": 150.0,
    "batch_size": 50,
    "moq": 100,
    "shelf_life_days": 365,
    "lag_1": 100.0,
    "lag_7": 100.0,
    "lag_14": 100.0,
    "lag_30": 100.0,
    "history": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
  }'

# With authentication
curl -H "X-API-Key: $API_KEY" http://localhost:8000/stores
```

### 6.3 API tests

```bash
# API and security tests
pytest -q tests/test_api.py tests/test_security.py

# Full unit suite with coverage
pytest -q -m "not integration" --cov=src --cov-report=term-missing

# Lint
ruff check src tests run_pipeline.py alembic
```

### 6.4 API route review checklist

- [ ] The route contains neither SQL nor complex business rules.
- [ ] The input payload is a Pydantic model.
- [ ] The response has a `response_model`.
- [ ] Critical fields have bounds (`ge`, `min_length`).
- [ ] Errors are translated into explicit HTTP codes.
- [ ] The business route is protected by `require_api_key`.
- [ ] Costly routes have a slowapi quota.
- [ ] The route does not send email in the request path.
- [ ] The logs contain a `request_id`.
- [ ] A database outage produces `503`, not a fake response.
- [ ] A missing history produces `404`, not default values.
- [ ] A missing model produces `503`.
- [ ] Tests cover the nominal case and at least one error case.
- [ ] The route is documented in OpenAPI through its models.

### 6.5 Glossary

| Term | Definition |
|---|---|
| **Composition root** | Assembly point of the application |
| **Router** | Group of HTTP routes |
| **Service** | Layer of pure business rules |
| **Repository** | Data access layer |
| **Pydantic** | Validation and serialization library |
| **X-API-Key** | Key-based authentication header |
| **CORS** | Cross-origin access policy for browsers |
| **Rate limiting** | Limiting the number of calls over a period |
| **Liveness** | Probe that checks the process responds |
| **Readiness** | Probe that checks the service can receive traffic |
| **Prometheus** | Metrics collection system |
| **Request ID** | Correlation identifier of a request |

### 6.6 Code files and tests to read

- `src/api.py`: composition root, lifespan and assembly.
- `src/api_routers/health.py`: health, liveness and readiness.
- `src/api_routers/data.py`: reference data and `503` handling.
- `src/api_routers/alerts.py`: alerts and rate limiting.
- `src/api_routers/features.py`: `/features` contract and `404`/`503` distinction.
- `src/api_routers/predict.py`: inference, MOQ, FEFO and asynchronous log.
- `src/services/inventory.py`: pure business rules.
- `src/repositories/database.py` and `src/repositories/inventory.py`: persistence.
- `src/schemas.py`: Pydantic v2 contracts.
- `src/security.py`: API key, CORS and slowapi.
- `src/observability.py`: JSON logs, metrics and prediction log.
- `src/logging_config.py`: JSON formatting.
- `tests/test_api.py` and `tests/test_security.py`: serving non-regression.

</details>
