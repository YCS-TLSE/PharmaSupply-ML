# Technical Documentation
## Phase 01: Initialization and Scoping
**Project:** PharmaSupply-ML
**Framework:** Standard MLOps Blueprint
**File:** `docs/01_INITIALIZATION_AND_SCOPING.md`
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

*Un guide pragmatique pour cadrer un projet MLOps, structurer le dépôt, isoler l'environnement et démarrer PostgreSQL sur des bases saines.*

---

## Sommaire

1. [Cadrage du projet et objectifs métier](#1-cadrage-du-projet-et-objectifs-metier)
2. [Structure et organisation du dépôt](#2-structure-et-organisation-du-depot)
3. [Environnement de développement et parité](#3-environnement-de-developpement-et-parite)
4. [Gestion des configurations et des secrets](#4-gestion-des-configurations-et-des-secrets)
5. [Bootstrap de l'infrastructure locale](#5-bootstrap-de-linfrastructure-locale)
6. [Annexes et commandes clés](#6-annexes-et-commandes-cles)

---

<a id="1-cadrage-du-projet-et-objectifs-metier"></a>
## 1. Cadrage du projet et objectifs métier

### 1.1 Le problème à résoudre

PharmaSupply-ML répond à une question simple en apparence : combien d'unités d'un médicament une officine va-t-elle vendre dans les sept prochains jours ? La réponse ne sert pas à remplir un tableau de bord. Elle déclenche une commande.

Le système doit donc produire une prévision exploitable par une décision de réapprovisionnement. Une prévision trop basse crée une rupture. Une prévision trop haute immobilise du capital et expose à la péremption. Le modèle statistique n'est qu'une partie du problème ; la qualité de la décision dépend aussi des contraintes logistiques, de la fraîcheur des données et de la manière dont le service réagit aux pannes.

Le périmètre fonctionnel couvre :

- plusieurs officines et plusieurs produits ;
- un historique quotidien de ventes, de stock et de ruptures ;
- une prévision de demande à sept jours ;
- un calcul de quantité de commande ;
- des alertes de paramétrage, de rupture et de péremption ;
- un suivi de la fiabilité du modèle.

### 1.2 Les contraintes qui changent tout

Trois contraintes métier empêchent de traiter ce projet comme une simple régression.

La première est le **MOQ** (*Minimum Order Quantity*). Le fournisseur ne livre pas une unité. Si le MOQ est de 100 et que le besoin calculé est de 40, la commande doit être portée à 100, puis arrondie à un multiple de la taille de lot.

La deuxième est le **FEFO** (*First Expired, First Out*). Les lots les plus proches de la péremption doivent être consommés en premier. Un lot qui expire dans moins de sept jours peut représenter une perte, même si le stock global semble suffisant.

La troisième est le **stock de sécurité**. Il ne se déduit pas uniquement du modèle. Il dépend du délai fournisseur, de la variabilité de la demande et de la tolérance au risque de l'officine. Le système doit donc combiner une prévision et des règles métier explicites.

Ces contraintes ont une conséquence d'ingénierie : les règles ne doivent pas vivre dans un notebook. Elles doivent être testables, documentées et exécutées de la même manière en batch et en ligne.

### 1.3 Du laboratoire au MLOps

Un Data Scientist qui arrive du laboratoire a l'habitude de chercher le meilleur score. En MLOps, le score reste important, mais il ne suffit pas. Il faut aussi pouvoir rejouer l'expérience, expliquer une prédiction, détecter une dérive, redémarrer le service et revenir à une version précédente.

Le cadrage initial doit donc fixer quatre choses :

1. **Le problème métier et sa mesure de succès.** Ici, une prévision utile et une commande conforme aux contraintes, pas seulement un MAE faible.
2. **Les frontières du système.** Ce que le pipeline prend en entrée, ce qu'il produit, et ce qu'il ne fait pas.
3. **Les contraintes non fonctionnelles.** Reproductibilité, latence, sécurité, observabilité, coût.
4. **Les responsabilités.** Qui maintient les données, qui valide le modèle, qui exploite l'API.

Un cadrage flou produit toujours le même symptôme : un modèle performant que personne ne sait déployer.

### 1.4 Critères de réussite et limites assumées

Le projet considère qu'une phase d'initialisation est réussie lorsque :

- le dépôt est structuré et compréhensible par un nouvel arrivant ;
- l'environnement est reproductible sur une autre machine ;
- les secrets ne sont pas dans le code ;
- PostgreSQL démarre avec un schéma connu ;
- le pipeline peut être lancé sans intervention manuelle cachée ;
- les tests et le lint peuvent être exécutés en une commande.

Les limites sont assumées : les données sont synthétiques, le modèle est un XGBoost unique, l'infrastructure reste locale et l'authentification est une clé statique. Ces choix sont adaptés à un projet pédagogique et à une montée progressive vers la production. Ils ne doivent pas être confondus avec une architecture cloud multi-tenant.

---

<a id="2-structure-et-organisation-du-depot"></a>
## 2. Structure et organisation du dépôt

### 2.1 Vue d'ensemble

```text
PharmaSupply-ML/
├── src/
│   ├── api.py
│   ├── api_routers/
│   ├── services/
│   ├── repositories/
│   ├── schemas.py
│   ├── config.py
│   ├── security.py
│   ├── observability.py
│   ├── data/
│   ├── features.py
│   ├── train.py
│   ├── validation.py
│   ├── ingestion.py
│   ├── monitoring.py
│   ├── explainability.py
│   └── dashboard.py
├── tests/
├── scripts/
├── alembic/
│   └── versions/
├── sql/
│   └── schema.sql
├── models/
├── reports/
├── notebooks/
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── environment.yml
├── requirements.txt
├── requirements.api.txt
├── requirements.dashboard.txt
├── pyproject.toml
└── .env.example
```

Le dépôt sépare trois préoccupations : le code exécutable (`src/`), les preuves (`tests/`) et l'exploitation (`scripts/`, `alembic/`, `sql/`, `Dockerfile*`, `docker-compose.yml`). Cette séparation n'est pas cosmétique. Elle indique immédiatement où chercher lorsqu'un problème apparaît.

### 2.2 Rôle des dossiers et fichiers racine

| Élément | Rôle | Remarque pédagogique |
|---|---|---|
| `src/` | Code applicatif | Un module par responsabilité, pas un fichier fourre-tout |
| `src/api.py` | Composition root FastAPI | Assemble les routers et le lifespan |
| `src/api_routers/` | Routes HTTP | Transport, codes d'erreur, auth |
| `src/services/` | Règles métier pures | Testables sans base ni HTTP |
| `src/repositories/` | Accès SQL et pool | Séparation persistance/transport |
| `src/schemas.py` | Contrats Pydantic | Frontières explicites |
| `src/config.py` | Constantes et variables | Source de vérité unique |
| `src/data/generate_data.py` | Génération synthétique | Reproductibilité du jeu de démonstration |
| `src/ingestion.py` | Chargement PostgreSQL | UPSERT et normalisation |
| `src/validation.py` | Quality gate | Erreur bloquante vs anomalie réparable |
| `src/features.py` | Features training et serving | Définition unique des transformations |
| `src/train.py` | Entraînement XGBoost | Split temporel et MLflow opt-in |
| `src/explainability.py` | SHAP et poids | Auditabilité du modèle |
| `src/monitoring.py` | Drift offline et production | Evidently et journal JSONL |
| `src/dashboard.py` | Interface Streamlit | Client mince de l'API |
| `tests/` | Tests unitaires et intégration | Non-régression |
| `scripts/` | Nettoyage, conversion | Automatisation d'exploitation |
| `alembic/` | Migrations | Évolution du schéma sans perte |
| `sql/schema.sql` | DDL de bootstrap | Schéma initial du volume Docker |
| `models/` | Artefact `.joblib` | Sortie d'entraînement |
| `reports/` | Rapports et figures | Sorties de monitoring et SHAP |
| `notebooks/` | Exploration | Laboratoire, pas production |
| `Dockerfile.api` | Image API | Dépendances épinglées, non-root |
| `Dockerfile.dashboard` | Image dashboard | Client Streamlit minimal |
| `docker-compose.yml` | Orchestration locale | PostgreSQL, API, dashboard |
| `environment.yml` | Environnement Conda | Reproductibilité du développement |
| `requirements.txt` | Dépendances pip | Développement et CI |
| `requirements.api.txt` | Dépendances API | Image de production allégée |
| `requirements.dashboard.txt` | Dépendances dashboard | Image Streamlit minimale |
| `pyproject.toml` | Ruff, pytest, coverage | Qualité et seuil de couverture |
| `.env.example` | Modèle de configuration | Aucun secret réel |

### 2.3 Ce qui doit rester hors du dépôt

Le fichier `.gitignore` exclut les éléments qui ne doivent pas être versionnés : environnements virtuels, caches Python, fichiers de couverture, données CSV locales, fichiers `.env`, suivi MLflow local et journal de prédiction.

La règle est simple : un clone du dépôt doit pouvoir démarrer avec un fichier `.env` reconstruit à partir de `.env.example`. Si un secret est nécessaire au démarrage, il doit être fourni par l'environnement d'exécution, pas par Git.

Les artefacts de modèle et les rapports générés méritent une politique explicite. Dans ce projet, le modèle est écrit dans `models/` et les rapports dans `reports/`. En production, ces artefacts devraient être publiés dans un stockage versionné ou un registry plutôt que commités à chaque entraînement.

### 2.4 Nommage et lisibilité

Quelques conventions structurent le dépôt :

- les modules d'infrastructure portent un nom explicite (`db_config.py`, `logging_config.py`, `observability.py`) ;
- les routers portent le nom de leur ressource (`health.py`, `data.py`, `alerts.py`, `features.py`, `predict.py`) ;
- les services portent le nom du domaine (`inventory.py`, `model_status.py`) ;
- les repositories portent le nom de la couche (`database.py`, `inventory.py`) ;
- les tests suivent le nom du module testé (`test_features.py`, `test_parity.py`, `test_security.py`).

Un nouveau venu doit pouvoir deviner où se trouve une responsabilité sans ouvrir tous les fichiers. Si ce n'est pas le cas, le découpage est à revoir.

---

<a id="3-environnement-de-developpement-et-parite"></a>
## 3. Environnement de développement et parité

### 3.1 Pourquoi WSL2 et Linux

Le projet est développé sous WSL2 avec un shell Bash/Zsh et Miniconda. Ce choix n'est pas une préférence personnelle. Il réduit les écarts entre le poste de développement, les conteneurs Docker et les serveurs de production.

Sous Windows natif, certains chemins, permissions et bibliothèques C/C++ se comportent différemment. XGBoost, SHAP et psycopg2 reposent sur des dépendances compilées. Travailler dans un environnement Linux dès le départ évite les surprises du type « le conteneur fonctionne, mais pas le poste local ».

La parité ne s'arrête pas au système. Elle concerne aussi les versions de Python et les dépendances.

### 3.2 Conda et Pip, chacun son rôle

Le projet utilise deux fichiers pour deux usages différents.

`environment.yml` décrit l'environnement Conda de développement. Il fixe Python 3.11 et les dépendances scientifiques lourdes (pandas, numpy, XGBoost, scikit-learn, SHAP, matplotlib). Conda gère mieux certains binaires et certains conflits de compilation.

`requirements.txt` sert aux installations pip, à la CI et aux environnements sans Conda. Il reprend les mêmes bibliothèques avec des bornes de version.

Les images de production ne doivent pas embarquer tout l'environnement de développement. C'est pourquoi deux fichiers séparés existent :

- `requirements.api.txt` pour l'image FastAPI ;
- `requirements.dashboard.txt` pour l'image Streamlit.

Cette séparation évite d'installer MLflow, Alembic, les notebooks et les outils de test dans une image qui n'en a pas besoin. Une image plus petite est plus rapide à construire, plus facile à scanner et moins exposée.

### 3.3 Épinglage et reproductibilité

Une dépendance non épinglée est une promesse de build différent demain. Le projet utilise deux niveaux :

- des versions exactes pour les composants sensibles (`fastapi==0.162.0`, `pydantic==2.10.3`, `mlflow==2.19.0`) ;
- des bornes pour les bibliothèques qui évoluent rapidement (`xgboost>=2.1,<3`, `sqlalchemy>=2.0,<2.1`).

Le `pyproject.toml` fixe aussi la version cible de Python (`py311`) et la longueur de ligne pour Ruff. Le but n'est pas de figer pour figer. Le but est de rendre un incident reproductible.

Lorsqu'une dépendance change, la démarche est toujours la même : mettre à jour le fichier de dépendances, reconstruire l'environnement, puis relancer les tests. La commande de mise à jour Conda utilisée dans ce projet est :

```bash
conda env update -f environment.yml --prune
```

L'option `--prune` retire les paquets qui ne sont plus déclarés. C'est utile pour éviter qu'un environnement de développement accumule des dépendances fantômes.

### 3.4 Docker comme deuxième niveau de parité

Docker intervient après l'environnement local. Il ne remplace pas Conda : il vérifie que le code fonctionne dans une image reproductible.

`Dockerfile.api` et `Dockerfile.dashboard` installent des dépendances épinglées, copient le code, créent un utilisateur non-root et définissent un healthcheck. `docker-compose.yml` orchestre PostgreSQL 16, l'API et le dashboard.

Un piège fréquent consiste à modifier une dépendance puis à lancer `docker compose restart`. Le conteneur redémarre avec l'ancienne image. Le pipeline corrige ce piège en reconstruisant l'image API avant de tester le serving.

---

<a id="4-gestion-des-configurations-et-des-secrets"></a>
## 4. Gestion des configurations et des secrets

### 4.1 Le rôle de src/config.py

`src/config.py` est la source de vérité des valeurs partagées. Il centralise :

- les chemins (`MODEL_PATH`, `STATUS_FILE_PATH`, `PREDICTION_LOG_PATH`) ;
- le contrat de features (`FEATURE_COLS`) ;
- les valeurs métier par défaut (stock de sécurité, MOQ, shelf life, lead time) ;
- les seuils opérationnels (FEFO, buffer de délai, seuils d'alerte) ;
- les seuils de drift ;
- les paramètres MLflow, sécurité et observabilité ;
- les messages destinés à l'interface.

Ce module ne contient pas de logique métier. Il lit l'environnement et expose des constantes. Cette séparation évite les valeurs magiques dispersées dans les routers, les services et les scripts.

Un exemple concret : `FEATURE_COLS` est utilisé par l'entraînement, le serving et le monitoring. Si la liste était dupliquée dans trois fichiers, un ajout de feature pourrait être oublié dans l'un des trois. Le modèle recevrait alors une matrice incomplète.

### 4.2 Le rôle de src/db_config.py

`src/db_config.py` construit l'URL PostgreSQL. Il accepte deux stratégies :

- une variable `DATABASE_URL` explicite ;
- une construction à partir de `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST` et `POSTGRES_PORT`.

L'utilisateur, le mot de passe et le nom de base sont obligatoires. L'hôte et le port ont des valeurs par défaut, car ce ne sont pas des secrets. La fonction échoue au moment de la connexion, pas à l'import, ce qui permet aux tests de charger les modules sans base live.

Cette distinction est importante : une erreur de configuration doit être visible au moment où l'on en a besoin, avec un message actionnable, plutôt que de produire une URL silencieusement incorrecte.

### 4.3 Le fichier .env et le fichier .env.example

Le fichier `.env` contient les valeurs locales : identifiants PostgreSQL, clé API, paramètres SMTP, activation MLflow, chemins de logs. Il n'est jamais versionné.

Le fichier `.env.example` joue le rôle de contrat. Il liste les variables attendues, avec des valeurs d'exemple et des commentaires. Un nouveau développeur copie le fichier, remplit les secrets, puis démarre.

```bash
cp .env.example .env
```

Les variables principales sont :

| Variable | Usage |
|---|---|
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Connexion PostgreSQL |
| `POSTGRES_HOST`, `POSTGRES_PORT` | Localisation de la base |
| `DATABASE_URL` | URL complète, prioritaire si définie |
| `API_KEY`, `API_KEY_HEADER` | Authentification de l'API |
| `CORS_ORIGINS` | Origines autorisées |
| `RATE_LIMIT_PREDICT`, `RATE_LIMIT_ALERTS` | Quotas slowapi |
| `MLFLOW_ENABLED`, `MLFLOW_TRACKING_URI` | Tracking et registry |
| `PREDICTION_LOG_PATH` | Journal utilisé par le drift production |
| `SMTP_PASSWORD`, `SMTP_SERVER`, `SMTP_PORT` | Notifications de drift |

### 4.4 Absence de secrets en dur

Le code ne doit jamais contenir de mot de passe, de clé API ou de jeton SMTP. Cette règle paraît évidente, mais elle est souvent contournée pour gagner du temps en développement.

Le projet applique trois garde-fous :

- `.gitignore` exclut `.env` et ses variantes ;
- `.env.example` ne contient que des valeurs factices ;
- `src/db_config.py` lève une erreur explicite si les variables obligatoires manquent.

En production, les secrets doivent être fournis par le gestionnaire de secrets de la plateforme (Vault, AWS Secrets Manager, variables d'environnement injectées par le orchestrateur). Un fichier `.env` copié sur un serveur n'est pas une stratégie de sécurité.

### 4.5 Priorité des sources de configuration

L'ordre de résolution est volontairement simple :

1. variable d'environnement du processus ;
2. fichier `.env` chargé au démarrage ;
3. valeur par défaut du code, uniquement pour les paramètres non sensibles.

Cet ordre permet de surcharger une valeur en CI ou en production sans modifier le fichier local. Il permet aussi de garder des valeurs de développement confortables sans les confondre avec des secrets.

---

<a id="5-bootstrap-de-linfrastructure-locale"></a>
## 5. Bootstrap de l'infrastructure locale

### 5.1 Premier lancement

La séquence d'initialisation est volontairement courte. Elle doit tenir en quelques commandes et ne pas dépendre d'une manipulation manuelle dans un terminal PostgreSQL.

```bash
git clone https://github.com/YCS-TLSE/PharmaSupply-ML.git
cd PharmaSupply-ML

conda env create -f environment.yml
conda activate pharma-supply

cp .env.example .env
# Renseigner les valeurs PostgreSQL dans .env

docker compose up -d --build
```

Le `--build` n'est pas décoratif. Il garantit que l'image API contient les dépendances actuelles. Un `docker compose up -d` seul peut réutiliser une image ancienne et produire des erreurs d'import difficiles à diagnostiquer.

### 5.2 Les trois services

`docker-compose.yml` déclare trois services.

| Service | Image ou build | Rôle | Port |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | Stockage relationnel | `5432` |
| `api` | `Dockerfile.api` | Serving FastAPI | `8000` |
| `dashboard` | `Dockerfile.dashboard` | Interface Streamlit | `8501` |

PostgreSQL possède un healthcheck `pg_isready`. L'API dépend de ce healthcheck, ce qui évite de démarrer avant que la base accepte les connexions. Le dashboard dépend de l'API.

Les volumes partagés jouent un rôle précis : le modèle entraîné, le fichier de statut et les rapports sont visibles à la fois sur l'hôte et dans les conteneurs. Cette synchronisation permet au pipeline local de mettre à jour le service sans reconstruire l'image à chaque entraînement. La reconstruction reste nécessaire lorsqu'une dépendance change.

### 5.3 Le schéma PostgreSQL

Le fichier `sql/schema.sql` crée la table `raw_pharmaceutical_demand`. Il contient les colonnes nécessaires à la prévision et aux règles métier :

| Colonne | Rôle |
|---|---|
| `id` | Clé technique auto-incrémentée |
| `date` | Dimension temporelle |
| `store_id` | Officine |
| `product_id` | Produit |
| `target_demand` | Cible de demande |
| `sales_volume` | Ventes observées |
| `stock_on_hand` | Stock physique |
| `is_stockout` | Indicateur de rupture |
| `supplier_lead_time` | Délai fournisseur |
| `safety_threshold` | Stock de sécurité |
| `batch_size` | Taille de lot |
| `moq` | Quantité minimale de commande |
| `shelf_life_days` | Durée de conservation |
| `critical_expiry_days` | Jours restants du lot le plus urgent |
| `created_at` | Horodatage d'ingestion |

La clé métier `(store_id, product_id, date)` est déclarée comme contrainte d'unicité. Sans elle, l'UPSERT ne peut pas fonctionner. L'index `idx_pharma_store_product_date` accélère les requêtes par officine, produit et date.

Le script commence par un `DROP TABLE IF EXISTS ... CASCADE`. C'est acceptable pour un volume Docker neuf, pas pour une base de production. Sur une base existante, il faut utiliser Alembic.

### 5.4 Alembic et évolution du schéma

Alembic prend le relais lorsque le schéma doit évoluer. La migration `alembic/versions/0001_initial_schema.py` crée la même table, la même contrainte et le même index. `alembic/env.py` récupère l'URL via `src.db_config.get_database_url()` pour éviter de dupliquer les identifiants.

Deux commandes suffisent pour vérifier la migration :

```bash
alembic upgrade head
alembic upgrade head --sql
```

La seconde génère le SQL sans connexion live. Elle est utile en revue de code et en CI, car elle permet de relire les changements de schéma avant de les appliquer.

### 5.5 Vérifications après démarrage

Une infrastructure qui démarre n'est pas forcément une infrastructure prête. Quelques vérifications évitent les faux positifs.

```bash
# État des conteneurs
docker compose ps

# Logs PostgreSQL
docker compose logs postgres --tail=50

# Santé de l'API
curl http://localhost:8000/livez
curl http://localhost:8000/readyz

# Schéma de la table
docker compose exec postgres psql \
  -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "\d raw_pharmaceutical_demand"

# Contrat de features chargé par le code
python -c "from src.config import FEATURE_COLS; print(FEATURE_COLS)"
```

`/readyz` ne renvoie `200` que si le modèle est chargé et si la base répond. C'est une vérification plus utile qu'un simple test de port.

### 5.6 Pièges classiques au premier lancement

Le premier lancement concentre la plupart des erreurs d'initialisation.

**Le fichier `.env` est absent.** PostgreSQL et l'API ne trouvent pas les identifiants. Le message d'erreur de `src/db_config.py` indique les variables manquantes.

**Le port 5432 est déjà utilisé.** Un PostgreSQL local ou un autre conteneur occupe le port. Il faut arrêter le service concurrent ou changer `POSTGRES_PORT`.

**Le volume PostgreSQL existe déjà.** `sql/schema.sql` n'est exécuté qu'à la création du volume. Si le schéma a changé, il faut appliquer Alembic ou recréer le volume en connaissance de cause.

**L'image API est obsolète.** Une dépendance ajoutée dans `requirements.api.txt` n'est pas présente dans le conteneur. Il faut relancer `docker compose up -d --build api`.

**Le fichier de statut est monté comme un dossier.** Docker crée un répertoire si le fichier source n'existe pas. Le pipeline vérifie ce cas avant de démarrer.

**Le dashboard ne joint pas l'API.** À l'intérieur de Compose, l'URL est `http://api:8000`, pas `http://localhost:8000`.

---

<a id="6-annexes-et-commandes-cles"></a>
## 6. Annexes et commandes clés

### 6.1 Commandes d'installation

```bash
# Clonage
git clone https://github.com/YCS-TLSE/PharmaSupply-ML.git
cd PharmaSupply-ML

# Environnement Conda
conda env create -f environment.yml
conda activate pharma-supply

# Mise à jour après changement de dépendances
conda env update -f environment.yml --prune

# Variante Pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Commandes d'infrastructure

```bash
# Démarrage avec reconstruction des images
docker compose up -d --build

# État et logs
docker compose ps
docker compose logs api --tail=100
docker compose logs postgres --tail=100

# Arrêt
docker compose down
```

### 6.3 Commandes de schéma

```bash
# Appliquer les migrations
alembic upgrade head

# Générer le SQL sans connexion
alembic upgrade head --sql

# Inspecter la table
docker compose exec postgres psql \
  -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "\d raw_pharmaceutical_demand"
```

### 6.4 Commandes de qualité

```bash
# Lint
ruff check src tests run_pipeline.py alembic

# Tests unitaires avec couverture
pytest -q -m "not integration" --cov=src --cov-report=term-missing

# Tests d'intégration Testcontainers
pytest -q -m integration
```

### 6.5 Checklist d'initialisation

- [ ] Le dépôt est cloné et le répertoire de travail est la racine du projet.
- [ ] L'environnement Conda ou le venv est créé et activé.
- [ ] Python 3.11 est utilisé.
- [ ] `.env` est créé à partir de `.env.example`.
- [ ] Les identifiants PostgreSQL sont renseignés.
- [ ] `docker compose up -d --build` démarre les trois services.
- [ ] `docker compose ps` montre des conteneurs sains.
- [ ] `sql/schema.sql` a créé la table et la contrainte d'unicité.
- [ ] `alembic upgrade head` s'applique sans erreur sur une base existante.
- [ ] `GET /livez` renvoie `200`.
- [ ] `GET /readyz` renvoie `200` avec le modèle chargé.
- [ ] `ruff check` passe.
- [ ] `pytest -m "not integration"` passe.
- [ ] Aucun secret n'est présent dans un fichier versionné.

### 6.6 Glossaire d'initialisation

| Terme | Définition |
|---|---|
| **Environnement reproductible** | Environnement recréable à partir de fichiers versionnés |
| **Parité Dev/Prod** | Réduction des écarts entre poste local, conteneurs et production |
| **Secret** | Information sensible qui ne doit pas être versionnée |
| **Bootstrap** | Ensemble des opérations qui rendent un système utilisable pour la première fois |
| **UPSERT** | Insertion ou mise à jour selon une clé métier |
| **Migration** | Évolution versionnée du schéma de base de données |
| **Healthcheck** | Vérification automatique qu'un service est opérationnel |

### 6.7 Fichiers à lire pour approfondir

- `environment.yml` et `requirements.txt` : dépendances de développement.
- `requirements.api.txt` et `requirements.dashboard.txt` : dépendances de production.
- `pyproject.toml` : lint, tests et couverture.
- `.env.example` : contrat des variables d'environnement.
- `src/config.py` : constantes et lecture de l'environnement.
- `src/db_config.py` : résolution de l'URL PostgreSQL.
- `docker-compose.yml` : orchestration locale.
- `sql/schema.sql` : schéma initial et contrainte d'unicité.
- `alembic/env.py` et `alembic/versions/0001_initial_schema.py` : migrations.
- `tests/test_db_config.py` : vérification de la résolution des identifiants.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *A pragmatic guide to scoping an MLOps project, structuring the repository, isolating the environment and bootstrapping PostgreSQL on sound foundations.*

---

## Table of Contents

1. [Project Scoping and Business Objectives](#1-project-scoping-and-business-objectives)
2. [Repository Structure and Organization](#2-repository-structure-and-organization)
3. [Development Environment and Parity](#3-development-environment-and-parity)
4. [Configuration and Secrets Management](#4-configuration-and-secrets-management)
5. [Local Infrastructure Bootstrap](#5-local-infrastructure-bootstrap)
6. [Appendices and Key Commands](#6-appendices-and-key-commands)

---

<a id="1-project-scoping-and-business-objectives"></a>
## 1. Project Scoping and Business Objectives

### 1.1 The problem to solve

PharmaSupply-ML answers a simple question on the surface: how many units of a medicine will a pharmacy sell over the next seven days? The answer does not fill a dashboard. It triggers an order.

The system must therefore produce a forecast that a replenishment decision can use. A forecast that is too low creates a stockout. A forecast that is too high immobilizes capital and exposes inventory to expiry. The statistical model is only one part of the problem; decision quality also depends on logistics constraints, data freshness and how the service reacts to failures.

The functional scope covers:

- several pharmacies and several products;
- a daily history of sales, stock and stockouts;
- a seven-day demand forecast;
- an order quantity calculation;
- alerts for parameter issues, stockouts and expiry;
- model reliability tracking.

### 1.2 Constraints that change everything

Three business constraints prevent this project from being treated as a simple regression.

The first is **MOQ** (*Minimum Order Quantity*). The supplier does not deliver a single unit. If the MOQ is 100 and the calculated need is 40, the order must be raised to 100, then rounded to a batch multiple.

The second is **FEFO** (*First Expired, First Out*). Batches closest to expiry must be consumed first. A batch expiring in less than seven days may represent a loss even if total stock looks sufficient.

The third is **safety stock**. It cannot be derived from the model alone. It depends on supplier lead time, demand variability and the pharmacy’s risk tolerance. The system must combine a forecast with explicit business rules.

These constraints have an engineering consequence: rules must not live in a notebook. They must be testable, documented and executed the same way in batch and online.

### 1.3 From lab to MLOps

A Data Scientist coming from a lab is used to chasing the best score. In MLOps, the score still matters, but it is not enough. The team must also replay the experiment, explain a prediction, detect drift, restart the service and roll back to a previous version.

Initial scoping therefore fixes four things:

1. **The business problem and its success measure.** Here, a useful forecast and a compliant order, not only a low MAE.
2. **System boundaries.** What goes in, what comes out, and what the system does not do.
3. **Non-functional constraints.** Reproducibility, latency, security, observability, cost.
4. **Responsibilities.** Who maintains data, who validates the model, who operates the API.

Vague scoping always produces the same symptom: a high-performing model that nobody can deploy.

### 1.4 Success criteria and assumed limits

Initialization is successful when:

- the repository is structured and understandable by a newcomer;
- the environment is reproducible on another machine;
- secrets are not in the code;
- PostgreSQL starts with a known schema;
- the pipeline can run without hidden manual steps;
- tests and lint run with one command.

Limits are explicit: data is synthetic, the model is a single XGBoost regressor, infrastructure is local and authentication is a static key. These choices fit an educational project and a progressive path to production. They must not be confused with a multi-tenant cloud architecture.

---

<a id="2-repository-structure-and-organization"></a>
## 2. Repository Structure and Organization

### 2.1 Overview

```text
PharmaSupply-ML/
├── src/
│   ├── api.py
│   ├── api_routers/
│   ├── services/
│   ├── repositories/
│   ├── schemas.py
│   ├── config.py
│   ├── security.py
│   ├── observability.py
│   ├── data/
│   ├── features.py
│   ├── train.py
│   ├── validation.py
│   ├── ingestion.py
│   ├── monitoring.py
│   ├── explainability.py
│   └── dashboard.py
├── tests/
├── scripts/
├── alembic/
│   └── versions/
├── sql/
│   └── schema.sql
├── models/
├── reports/
├── notebooks/
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── environment.yml
├── requirements.txt
├── requirements.api.txt
├── requirements.dashboard.txt
├── pyproject.toml
└── .env.example
```

The repository separates three concerns: executable code (`src/`), evidence (`tests/`) and operations (`scripts/`, `alembic/`, `sql/`, `Dockerfile*`, `docker-compose.yml`). This separation is not cosmetic. It tells you where to look when something goes wrong.

### 2.2 Role of folders and root files

| Item | Role | Teaching note |
|---|---|---|
| `src/` | Application code | One module per responsibility, not a catch-all file |
| `src/api.py` | FastAPI composition root | Assembles routers and lifespan |
| `src/api_routers/` | HTTP routes | Transport, error codes, auth |
| `src/services/` | Pure business rules | Testable without DB or HTTP |
| `src/repositories/` | SQL access and pool | Persistence/transport separation |
| `src/schemas.py` | Pydantic contracts | Explicit boundaries |
| `src/config.py` | Constants and environment | Single source of truth |
| `src/data/generate_data.py` | Synthetic generation | Reproducibility of the demo dataset |
| `src/ingestion.py` | PostgreSQL loading | UPSERT and normalization |
| `src/validation.py` | Quality gate | Blocking error vs repairable anomaly |
| `src/features.py` | Training and serving features | Single feature definition |
| `src/train.py` | XGBoost training | Temporal split and MLflow opt-in |
| `src/explainability.py` | SHAP and weights | Model auditability |
| `src/monitoring.py` | Offline and production drift | Evidently and JSONL log |
| `src/dashboard.py` | Streamlit interface | Thin API client |
| `tests/` | Unit and integration tests | Non-regression |
| `scripts/` | Cleanup, conversion | Operations automation |
| `alembic/` | Migrations | Schema evolution without loss |
| `sql/schema.sql` | Bootstrap DDL | Initial Docker volume schema |
| `models/` | `.joblib` artifact | Training output |
| `reports/` | Reports and figures | SHAP and monitoring outputs |
| `notebooks/` | Exploration | Lab, not production |
| `Dockerfile.api` | API image | Pinned dependencies, non-root |
| `Dockerfile.dashboard` | Dashboard image | Minimal Streamlit client |
| `docker-compose.yml` | Local orchestration | PostgreSQL, API, dashboard |
| `environment.yml` | Conda environment | Development reproducibility |
| `requirements.txt` | Pip dependencies | Development and CI |
| `requirements.api.txt` | API dependencies | Lean production image |
| `requirements.dashboard.txt` | Dashboard dependencies | Minimal Streamlit image |
| `pyproject.toml` | Ruff, pytest, coverage | Quality and coverage threshold |
| `.env.example` | Configuration template | No real secrets |

### 2.3 What must stay out of the repository

`.gitignore` excludes files that must not be versioned: virtual environments, Python caches, coverage files, local CSV data, `.env` files, local MLflow tracking and the prediction log.

The rule is simple: a fresh clone must start with an `.env` rebuilt from `.env.example`. If a secret is required at startup, it must be provided by the runtime environment, not by Git.

Model artifacts and generated reports need an explicit policy. In this project, the model is written to `models/` and reports to `reports/`. In production, these artifacts should be published to versioned storage or a registry rather than committed on every training run.

### 2.4 Naming and readability

A few conventions structure the repository:

- infrastructure modules have explicit names (`db_config.py`, `logging_config.py`, `observability.py`);
- routers are named after their resource (`health.py`, `data.py`, `alerts.py`, `features.py`, `predict.py`);
- services are named after the domain (`inventory.py`, `model_status.py`);
- repositories are named after the layer (`database.py`, `inventory.py`);
- tests follow the module they test (`test_features.py`, `test_parity.py`, `test_security.py`).

A newcomer should guess where a responsibility lives without opening every file. If not, the split needs review.

---

<a id="3-development-environment-and-parity"></a>
## 3. Development Environment and Parity

### 3.1 Why WSL2 and Linux

The project is developed under WSL2 with Bash/Zsh and Miniconda. This is not a personal preference. It reduces the gap between the developer machine, Docker containers and production servers.

On native Windows, paths, permissions and C/C++ libraries behave differently. XGBoost, SHAP and psycopg2 rely on compiled dependencies. Working in a Linux environment from the start avoids “the container works, but the local machine does not” surprises.

Parity does not stop at the operating system. It also covers Python and dependency versions.

### 3.2 Conda and Pip, each with a role

The project uses two files for two purposes.

`environment.yml` describes the Conda development environment. It pins Python 3.11 and heavy scientific dependencies (pandas, numpy, XGBoost, scikit-learn, SHAP, matplotlib). Conda handles some binaries and compilation conflicts better.

`requirements.txt` supports pip installations, CI and environments without Conda. It mirrors the same libraries with version bounds.

Production images must not embed the entire development environment. That is why two separate files exist:

- `requirements.api.txt` for the FastAPI image;
- `requirements.dashboard.txt` for the Streamlit image.

This separation avoids installing MLflow, Alembic, notebooks and test tooling in an image that does not need them. A smaller image builds faster, scans more easily and exposes less.

### 3.3 Pinning and reproducibility

An unpinned dependency is a promise of a different build tomorrow. The project uses two levels:

- exact versions for sensitive components (`fastapi==0.162.0`, `pydantic==2.10.3`, `mlflow==2.19.0`);
- bounds for fast-moving libraries (`xgboost>=2.1,<3`, `sqlalchemy>=2.0,<2.1`).

`pyproject.toml` also fixes the Python target (`py311`) and the Ruff line length. The goal is not to freeze for the sake of freezing. The goal is to make an incident reproducible.

When a dependency changes, the process is always the same: update the dependency file, rebuild the environment, rerun the tests. The Conda update command used in this project is:

```bash
conda env update -f environment.yml --prune
```

`--prune` removes packages that are no longer declared. It prevents a development environment from accumulating ghost dependencies.

### 3.4 Docker as a second parity level

Docker comes after the local environment. It does not replace Conda: it verifies that the code runs in a reproducible image.

`Dockerfile.api` and `Dockerfile.dashboard` install pinned dependencies, copy code, create a non-root user and define a healthcheck. `docker-compose.yml` orchestrates PostgreSQL 16, the API and the dashboard.

A common trap is to change a dependency and run `docker compose restart`. The container restarts with the old image. The pipeline prevents this by rebuilding the API image before testing serving.

---

<a id="4-configuration-and-secrets-management"></a>
## 4. Configuration and Secrets Management

### 4.1 The role of src/config.py

`src/config.py` is the single source of truth for shared values. It centralizes:

- paths (`MODEL_PATH`, `STATUS_FILE_PATH`, `PREDICTION_LOG_PATH`);
- the feature contract (`FEATURE_COLS`);
- business defaults (safety stock, MOQ, shelf life, lead time);
- operational thresholds (FEFO, lead-time buffer, alert thresholds);
- drift thresholds;
- MLflow, security and observability settings;
- operator-facing messages.

The module contains no business logic. It reads the environment and exposes constants. This separation prevents magic values from spreading across routers, services and scripts.

A concrete example: `FEATURE_COLS` is used by training, serving and monitoring. If the list were duplicated in three files, a new feature could be forgotten in one of them. The model would then receive an incomplete matrix.

### 4.2 The role of src/db_config.py

`src/db_config.py` builds the PostgreSQL URL. It supports two strategies:

- an explicit `DATABASE_URL`;
- construction from `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST` and `POSTGRES_PORT`.

User, password and database name are mandatory. Host and port have defaults because they are not secrets. The function fails at connection time, not at import, so tests can load modules without a live database.

This distinction matters: a configuration error must be visible when needed, with an actionable message, rather than silently producing an incorrect URL.

### 4.3 The .env and .env.example files

`.env` contains local values: PostgreSQL credentials, API key, SMTP settings, MLflow activation, log paths. It is never versioned.

`.env.example` acts as a contract. It lists expected variables with example values and comments. A new developer copies the file, fills in secrets, then starts.

```bash
cp .env.example .env
```

The main variables are:

| Variable | Purpose |
|---|---|
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | PostgreSQL connection |
| `POSTGRES_HOST`, `POSTGRES_PORT` | Database location |
| `DATABASE_URL` | Full URL, takes precedence when defined |
| `API_KEY`, `API_KEY_HEADER` | API authentication |
| `CORS_ORIGINS` | Allowed origins |
| `RATE_LIMIT_PREDICT`, `RATE_LIMIT_ALERTS` | slowapi quotas |
| `MLFLOW_ENABLED`, `MLFLOW_TRACKING_URI` | Tracking and registry |
| `PREDICTION_LOG_PATH` | Log used by production drift |
| `SMTP_PASSWORD`, `SMTP_SERVER`, `SMTP_PORT` | Drift notifications |

### 4.4 No hardcoded secrets

Code must never contain a password, API key or SMTP token. The rule sounds obvious, yet it is often bypassed to save time during development.

The project applies three safeguards:

- `.gitignore` excludes `.env` and its variants;
- `.env.example` contains only fake values;
- `src/db_config.py` raises an explicit error when mandatory variables are missing.

In production, secrets must be provided by the platform secret manager (Vault, AWS Secrets Manager, environment variables injected by the orchestrator). A `.env` file copied to a server is not a security strategy.

### 4.5 Configuration precedence

Resolution order is intentionally simple:

1. process environment variable;
2. `.env` file loaded at startup;
3. code default, only for non-sensitive parameters.

This order lets CI or production override a value without editing the local file. It also keeps comfortable development defaults without confusing them with secrets.

---

<a id="5-local-infrastructure-bootstrap"></a>
## 5. Local Infrastructure Bootstrap

### 5.1 First launch

The initialization sequence is short. It should fit in a few commands and require no manual PostgreSQL terminal work.

```bash
git clone https://github.com/YCS-TLSE/PharmaSupply-ML.git
cd PharmaSupply-ML

conda env create -f environment.yml
conda activate pharma-supply

cp .env.example .env
# Fill in PostgreSQL values in .env

docker compose up -d --build
```

`--build` is not decorative. It guarantees the API image contains current dependencies. A plain `docker compose up -d` may reuse an old image and produce hard-to-diagnose import errors.

### 5.2 The three services

`docker-compose.yml` declares three services.

| Service | Image or build | Role | Port |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | Relational storage | `5432` |
| `api` | `Dockerfile.api` | FastAPI serving | `8000` |
| `dashboard` | `Dockerfile.dashboard` | Streamlit interface | `8501` |

PostgreSQL has a `pg_isready` healthcheck. The API depends on it, which avoids starting before the database accepts connections. The dashboard depends on the API.

Shared volumes have a precise role: the trained model, status file and reports are visible on both the host and inside containers. This synchronization lets the local pipeline refresh the service without rebuilding the image on every training run. A rebuild is still required when a dependency changes.

### 5.3 The PostgreSQL schema

`sql/schema.sql` creates `raw_pharmaceutical_demand`. It contains the columns needed for forecasting and business rules:

| Column | Role |
|---|---|
| `id` | Auto-incremented technical key |
| `date` | Temporal dimension |
| `store_id` | Pharmacy |
| `product_id` | Product |
| `target_demand` | Demand target |
| `sales_volume` | Observed sales |
| `stock_on_hand` | Physical stock |
| `is_stockout` | Stockout indicator |
| `supplier_lead_time` | Supplier lead time |
| `safety_threshold` | Safety stock |
| `batch_size` | Batch size |
| `moq` | Minimum order quantity |
| `shelf_life_days` | Shelf life |
| `critical_expiry_days` | Remaining days of the most urgent batch |
| `created_at` | Ingestion timestamp |

The natural key `(store_id, product_id, date)` is declared as a unique constraint. Without it, the UPSERT cannot work. The `idx_pharma_store_product_date` index speeds up queries by pharmacy, product and date.

The script starts with `DROP TABLE IF EXISTS ... CASCADE`. This is acceptable for a fresh Docker volume, not for production. Existing databases should use Alembic.

### 5.4 Alembic and schema evolution

Alembic takes over when the schema must evolve. Migration `alembic/versions/0001_initial_schema.py` creates the same table, constraint and index. `alembic/env.py` resolves the URL through `src/db_config.get_database_url()` to avoid duplicating credentials.

Two commands are enough to verify the migration:

```bash
alembic upgrade head
alembic upgrade head --sql
```

The second generates SQL without a live connection. It is useful in code review and CI because schema changes can be inspected before being applied.

### 5.5 Post-start verification

An infrastructure that starts is not necessarily ready. A few checks avoid false positives.

```bash
# Container status
docker compose ps

# PostgreSQL logs
docker compose logs postgres --tail=50

# API health
curl http://localhost:8000/livez
curl http://localhost:8000/readyz

# Table schema
docker compose exec postgres psql \
  -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "\d raw_pharmaceutical_demand"

# Feature contract loaded by the code
python -c "from src.config import FEATURE_COLS; print(FEATURE_COLS)"
```

`/readyz` returns `200` only when the model is loaded and the database responds. That is more useful than a simple port check.

### 5.6 Common first-launch pitfalls

**Missing `.env`.** PostgreSQL and the API cannot find credentials. The error from `src/db_config.py` lists missing variables.

**Port 5432 already in use.** A local PostgreSQL or another container holds the port. Stop the competing service or change `POSTGRES_PORT`.

**Existing PostgreSQL volume.** `sql/schema.sql` runs only when the volume is created. If the schema changed, apply Alembic or recreate the volume knowingly.

**Stale API image.** A dependency added to `requirements.api.txt` is missing in the container. Run `docker compose up -d --build api`.

**Status file mounted as a directory.** Docker creates a directory if the source file does not exist. The pipeline checks this case before starting.

**Dashboard cannot reach the API.** Inside Compose, the URL is `http://api:8000`, not `http://localhost:8000`.

---

<a id="6-appendices-and-key-commands"></a>
## 6. Appendices and Key Commands

### 6.1 Installation commands

```bash
# Clone
git clone https://github.com/YCS-TLSE/PharmaSupply-ML.git
cd PharmaSupply-ML

# Conda environment
conda env create -f environment.yml
conda activate pharma-supply

# Update after dependency changes
conda env update -f environment.yml --prune

# Pip alternative
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Infrastructure commands

```bash
# Start with image rebuild
docker compose up -d --build

# Status and logs
docker compose ps
docker compose logs api --tail=100
docker compose logs postgres --tail=100

# Stop
docker compose down
```

### 6.3 Schema commands

```bash
# Apply migrations
alembic upgrade head

# Generate SQL without a connection
alembic upgrade head --sql

# Inspect the table
docker compose exec postgres psql \
  -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "\d raw_pharmaceutical_demand"
```

### 6.4 Quality commands

```bash
# Lint
ruff check src tests run_pipeline.py alembic

# Unit tests with coverage
pytest -q -m "not integration" --cov=src --cov-report=term-missing

# Testcontainers integration tests
pytest -q -m integration
```

### 6.5 Initialization checklist

- [ ] The repository is cloned and the working directory is the project root.
- [ ] The Conda environment or venv is created and activated.
- [ ] Python 3.11 is used.
- [ ] `.env` is created from `.env.example`.
- [ ] PostgreSQL credentials are filled in.
- [ ] `docker compose up -d --build` starts the three services.
- [ ] `docker compose ps` shows healthy containers.
- [ ] `sql/schema.sql` created the table and unique constraint.
- [ ] `alembic upgrade head` applies cleanly on an existing database.
- [ ] `GET /livez` returns `200`.
- [ ] `GET /readyz` returns `200` with the model loaded.
- [ ] `ruff check` passes.
- [ ] `pytest -m "not integration"` passes.
- [ ] No secret is present in a versioned file.

### 6.6 Initialization glossary

| Term | Definition |
|---|---|
| **Reproducible environment** | Environment rebuildable from versioned files |
| **Dev/Prod parity** | Reduced gap between local machine, containers and production |
| **Secret** | Sensitive information that must not be versioned |
| **Bootstrap** | Operations that make a system usable for the first time |
| **UPSERT** | Insert or update based on a natural key |
| **Migration** | Versioned evolution of the database schema |
| **Healthcheck** | Automated verification that a service is operational |

### 6.7 Files to read for deeper understanding

- `environment.yml` and `requirements.txt`: development dependencies.
- `requirements.api.txt` and `requirements.dashboard.txt`: production dependencies.
- `pyproject.toml`: lint, tests and coverage.
- `.env.example`: environment variable contract.
- `src/config.py`: constants and environment loading.
- `src/db_config.py`: PostgreSQL URL resolution.
- `docker-compose.yml`: local orchestration.
- `sql/schema.sql`: initial schema and unique constraint.
- `alembic/env.py` and `alembic/versions/0001_initial_schema.py`: migrations.
- `tests/test_db_config.py`: credential resolution verification.

</details>
