# Technical Documentation  
## Phase 2: Data Infrastructure & Ingestion  
**Project:** PharmaSupply-ML  
**Framework:** Standard MLOps Blueprint  
**File:** `docs/02_DATA_INFRASTRUCTURE_AND_INGESTION.md`  
**Core Scripts & Files:** `sql/schema.sql`, `src/ingestion.py`, `docker-compose.yml`, `data/pharmaceutical_demand_row.csv`  
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

### *Ce document détaille l'infrastructure de stockage des données conteneurisée sous Docker (PostgreSQL), la définition explicite du schéma DDL (`sql/schema.sql`) couplée à son initialisation automatique, ainsi que le pipeline d'ingestion robuste et normalisé (`src/ingestion.py`).*

---

## 1. Contexte & Architecture MLOps

L'objectif de cette étape est d'isoler l'infrastructure de stockage des données pour garantir la **parité Dev/Prod** (*Dev-Prod Parity*). Dans notre environnement sous Linux (WSL2), PostgreSQL est instancié via Docker Compose pour offrir une base de données relationnelle isolée, versionnée et reproductible, interconnectée avec les services API et Dashboard.

```text
PharmaSupply-ML/
├── .env                          # Credentials PostgreSQL (non versionné)
├── .env.example                  # Gabarit des variables d'environnement
├── docker-compose.yml            # Services PostgreSQL, API & Dashboard + Volumes
├── sql/
│   └── schema.sql                # Schéma DDL officiel et indexation multi-colonnes
├── data/
│   └── pharmaceutical_demand_row.csv # Jeu de données brut (10 950 lignes)
├── src/
│   └── ingestion.py              # Pipeline automatisé d'ingestion et vérification conteneur
└── docs/
    └── 02_DATA_INFRASTRUCTURE_AND_INGESTION.md
```

---

## 2. Infrastructure Conteneurisée & Initialisation Automatique du Schéma

### 2.1. Fichier `docker-compose.yml`
La base de données PostgreSQL 16 est configurée avec persistance des données sur volume dédié et initialisation automatique du schéma DDL au premier démarrage via le point d'entrée de la base :

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: pharmasupply-db
    restart: always
    env_file:
      - .env
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/schema.sql:/docker-entrypoint-initdb.d/01_schema.sql
    healthcheck:
      test: [ "CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}" ]
      interval: 5s
      timeout: 5s
      retries: 5
```

### 2.2. Gestion du Schéma SQL (`sql/schema.sql`)
La création de la table `raw_pharmaceutical_demand` intègre les dimensions temporelles, les identifiants de magasins/produits, ainsi que les contraintes logistiques et de péremption (*FEFO - First Expired, First Out*). Un index composite de performance (`idx_pharma_store_product_date`) garantit l'optimisation des requêtes de séries temporelles :

```sql
CREATE TABLE raw_pharmaceutical_demand (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    store_id VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    target_demand INT NOT NULL,
    sales_volume INT NOT NULL,
    stock_on_hand INT NOT NULL,
    is_stockout INT NOT NULL DEFAULT 0,
    supplier_lead_time INT NOT NULL,
    safety_threshold NUMERIC(10, 2) NOT NULL DEFAULT 150.0,
    batch_size INT NOT NULL DEFAULT 1,
    moq INT NOT NULL DEFAULT 1,
    shelf_life_days INT NOT NULL DEFAULT 365,
    critical_expiry_days INT NOT NULL DEFAULT 365,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pharma_store_product_date ON raw_pharmaceutical_demand (store_id, product_id, date);
```

### 2.3. Commandes de Gestion du Conteneur
Toutes les commandes s'exécutent depuis la racine du projet :

| Action | Commande | Description |
| :--- | :--- | :--- |
| **Démarrage** | `docker compose up -d` | Télécharge l'image, crée le volume, initialise le schéma et lance le service. |
| **Vérification** | `docker compose ps` | Confirme l'état de santé (`healthy`). |
| **Consultation Logs** | `docker compose logs -f` | Affiche le flux de journaux PostgreSQL en temps réel. |
| **Arrêt Sécurisé** | `docker compose stop` | Arrête le conteneur en préservant les données dans le volume. |
| **Purge Complète** | `docker compose down -v` | Supprime le conteneur et purge le volume de données. |

### 2.4. Guide d'Interrogation de la Base de Données (CLI)

Pour explorer ou déboguer les données directement depuis le terminal via `psql` :

#### A. Session interactive (Ligne de commande SQL)
```bash
# Ouvrir le terminal interactif PostgreSQL
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db
```
*Commandes utiles dans le shell `psql` :*
* `\dt` : Lister toutes les tables.
* `\d raw_pharmaceutical_demand` : Afficher la structure détaillée d'une table.
* `\q` : Quitter le shell `psql`.

#### B. Requêtes rapides ponctuelles
```bash
# Compter le nombre de lignes
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT COUNT(*) FROM raw_pharmaceutical_demand;"

# Inspecter les 5 premières lignes
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT * FROM raw_pharmaceutical_demand LIMIT 5;"
```

---

## 3. Pipeline d'Ingestion & Normalisation (`src/ingestion.py`)

Le script `src/ingestion.py` automatise le chargement des données depuis le fichier source (`data/pharmaceutical_demand_row.csv`) vers la table de staging PostgreSQL.

### 3.1. Fonctionnement Détaillé du Script
Le script s'articule autour de fonctions clés garantissant la résilience et l'intégrité des flux :
1. **Vérification du cycle de vie du conteneur (`ensure_postgres_running`) :** 
   * Exécute préventivement `docker compose up -d` via `subprocess` pour s'assurer que le service PostgreSQL est bien actif avant toute tentative de connexion, prévenant ainsi les erreurs de type `OperationalError`.
2. **Normalisation rigoureuse des identifiants :**
   * Convertit explicitement les colonnes `store_id` et `product_id` en types textuels (`str`), garantissant l'absence d'ambiguïté en cas d'identifiants numériques ou mixtes.
   * Assure un repli automatique par défaut (`store_id = "1"`, `product_id = "MED_001"`) si les colonnes venaient à manquer dans le jeu de données source.
3. **Propagation des paramètres logistiques & FEFO :**
   * Vérifie et intègre dynamiquement les seuils de sécurité (`safety_threshold` par défaut à 150.0), les tailles de lots (`batch_size` par défaut à 1), les quantités minimales de commande (`moq` par défaut à 1), ainsi que les durées de conservation (`shelf_life_days` à 365) et la traçabilité de péremption critique (`critical_expiry_days` alignée sur le cycle de vie).
4. **Persistance relationnelle via SQLAlchemy (`get_db_engine`) :**
   * Établit une connexion sécurisée à l'aide des variables d'environnement lues par `python-dotenv` et charge le DataFrame normalisé dans la base de données via la méthode `.to_sql()` en mode écrasement ou ajout (`if_exists`).

---

## 4. Exécution & Validation

### 4.1. Lancement du Pipeline
```bash
python src/ingestion.py
```

### 4.2. Vérification de la Base de Données
```bash
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT COUNT(*) FROM raw_pharmaceutical_demand;"
```

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *This document details the containerized data storage infrastructure using Docker (PostgreSQL), explicit DDL schema management coupled with automated initialization (`sql/schema.sql`), and the robust ingestion pipeline (`src/ingestion.py`).*

---

## 1. Context & MLOps Architecture

The primary goal of this phase is to isolate the data storage layer to guarantee **Dev-Prod Parity**. Under Linux (WSL2), PostgreSQL is instantiated using Docker Compose to provide an isolated, versioned, and reproducible relational database interconnected with API and Dashboard microservices.

```text
PharmaSupply-ML/
├── .env                          # PostgreSQL credentials (ignored by Git)
├── .env.example                  # Environment variable template
├── docker-compose.yml            # PostgreSQL, API & Dashboard services + Volumes
├── sql/
│   └── schema.sql                # Official DDL schema and multi-column indexing
├── data/
│   └── pharmaceutical_demand_row.csv # Raw dataset (10,950 rows)
├── src/
│   └── ingestion.py              # Automated ingestion and container check pipeline
└── docs/
    └── 02_DATA_INFRASTRUCTURE_AND_INGESTION.md
```

---

## 2. Containerized Infrastructure & Automatic Schema Initialization

### 2.1. `docker-compose.yml` File
The PostgreSQL 16 database is configured with volume persistence and automated DDL schema bootstrapping upon initial startup via the entrypoint directory:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: pharmasupply-db
    restart: always
    env_file:
      - .env
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/schema.sql:/docker-entrypoint-initdb.d/01_schema.sql
    healthcheck:
      test: [ "CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}" ]
      interval: 5s
      timeout: 5s
      retries: 5
```

### 2.2. SQL Schema Management (`sql/schema.sql`)
The `raw_pharmaceutical_demand` table schema explicitly manages time-series dimensions, store/product identifiers, logistical constraints, and FEFO expiry tracking. Performance is optimized via a composite index (`idx_pharma_store_product_date`):

```sql
CREATE TABLE raw_pharmaceutical_demand (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    store_id VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    target_demand INT NOT NULL,
    sales_volume INT NOT NULL,
    stock_on_hand INT NOT NULL,
    is_stockout INT NOT NULL DEFAULT 0,
    supplier_lead_time INT NOT NULL,
    safety_threshold NUMERIC(10, 2) NOT NULL DEFAULT 150.0,
    batch_size INT NOT NULL DEFAULT 1,
    moq INT NOT NULL DEFAULT 1,
    shelf_life_days INT NOT NULL DEFAULT 365,
    critical_expiry_days INT NOT NULL DEFAULT 365,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pharma_store_product_date ON raw_pharmaceutical_demand (store_id, product_id, date);
```

### 2.3. Container Management Commands
All commands should be executed from the project root:

| Action | Command | Description |
| :--- | :--- | :--- |
| **Start** | `docker compose up -d` | Pulls image, creates volume, initializes schema, and starts container. |
| **Verify** | `docker compose ps` | Confirms container health status (`healthy`). |
| **View Logs** | `docker compose logs -f` | Streams live PostgreSQL output logs. |
| **Stop (Safe)** | `docker compose stop` | Stops container while preserving data in the volume. |
| **Full Cleanup** | `docker compose down -v` | Removes container and purges the data volume. |

### 2.4. Database Querying Guide (CLI)

To inspect or debug data directly from your terminal using `psql`:

#### A. Interactive Session (SQL Shell)
```bash
# Connect to interactive PostgreSQL shell
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db
```
*Common meta-commands:*
* `\dt`: List all tables.
* `\d raw_pharmaceutical_demand`: Display detailed table schema.
* `\q`: Exit `psql`.

#### B. Quick One-Line Queries
```bash
# Row count check
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT COUNT(*) FROM raw_pharmaceutical_demand;"

# Inspect first 5 records
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT * FROM raw_pharmaceutical_demand LIMIT 5;"
```

---

## 3. Ingestion Pipeline & Normalization (`src/ingestion.py`)

The `src/ingestion.py` script automates loading records from the source CSV file (`data/pharmaceutical_demand_row.csv`) into the PostgreSQL staging table.

### 3.1. Detailed Script Breakdown
The script relies on dedicated functions ensuring robustness and data integrity:
1. **Container Lifecycle Verification (`ensure_postgres_running`):**
   * Preventively executes `docker compose up -d` via `subprocess` to ensure the PostgreSQL service is running prior to any connection attempts, preventing operational errors.
2. **Rigorous Identifier Normalization:**
   * Explicitly casts `store_id` and `product_id` columns to string types (`str`), avoiding potential type mismatch issues with mixed or numeric identifiers.
   * Enforces automatic default fallbacks (`store_id = "1"`, `product_id = "MED_001"`) if expected columns are missing from the raw dataset.
3. **Logistics & FEFO Parameter Propagation:**
   * Checks and dynamically injects default or custom fallback values for safety thresholds (`safety_threshold` defaulted to 150.0), batch sizes (`batch_size` defaulted to 1), minimum order quantities (`moq` defaulted to 1), shelf-life constraints (`shelf_life_days` to 365), and critical expiration tracking (`critical_expiry_days` aligned with the lifecycle).
4. **Relational Persistence via SQLAlchemy (`get_db_engine`):**
   * Establishes a secure connection using environment variables loaded via `python-dotenv` and loads the normalized DataFrame into the database using `.to_sql()` with overwrite or append modes (`if_exists`).

---

## 4. Execution & Validation

### 4.1. Run Pipeline
```bash
python src/ingestion.py
```

### 4.2. Query Staging Table
```bash
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT COUNT(*) FROM raw_pharmaceutical_demand;"
```

</details>