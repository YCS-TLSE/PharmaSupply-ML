# Technical Documentation  
## Phase 2: Data Infrastructure & Ingestion  
**Project:** PharmaSupply-ML  
**Framework:** Standard MLOps Blueprint  
**File:** `docs/02_DATA_INFRASTRUCTURE_AND_INGESTION.md`  
**Status:** Validated  

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

### *Ce document détaille l'infrastructure de stockage des données conteneurisée sous Docker (PostgreSQL), la définition explicite du schéma DDL (`sql/schema.sql`) ainsi que le pipeline d'ingestion automatisé (`src/ingestion.py`).*

---

## 1. Contexte & Architecture MLOps

L'objectif de cette étape est d'isoler l'infrastructure de stockage des données pour garantir la **parité Dev/Prod** (*Dev-Prod Parity*). Dans notre environnement sous Linux (WSL2), PostgreSQL est instancié via Docker Compose pour offrir une base de données relationnelle isolée, versionnée et reproductible.

~~~~text
PharmaSupply-ML/
├── .env                          # Credentials PostgreSQL (non versionné)
├── .env.example                  # Gabarit des variables d'environnement
├── docker-compose.yml            # Services PostgreSQL & volumes
├── sql/
│   └── schema.sql                # Schéma DDL officiel et indexation
├── data/
│   └── pharmaceutical_demand.csv   # Jeu de données brut (10 950 lignes)
├── src/
│   └── ingestion.py              # Pipeline automatisé d'ingestion
└── docs/
    └── 02_DATA_INFRASTRUCTURE_AND_INGESTION.md
~~~~

---

## 2. Infrastructure Conteneurisée & Schéma DDL Explicite

### 2.1. Fichier `docker-compose.yml`
La base de données PostgreSQL 16 (image Alpine ultra-légère) est configurée avec persistance des données sur un volume dédié :

~~~~yaml
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
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
    driver: local
~~~~

### 2.2. Gestion du Schéma SQL (`sql/schema.sql`)
Pour respecter les normes de production et MLOps, la création de la table `raw_pharmaceutical_demand` est gérée de manière **explicite** via un fichier SQL DDL. Cela remplace la génération dynamique automatique de Pandas pour offrir un meilleur contrôle sur les typages de données et pour ajouter un **index composite de performance** (`idx_pharma_date_product`).

**Application du schéma DDL dans la base :**
~~~~bash
docker compose cp sql/schema.sql postgres:/tmp/schema.sql
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -f /tmp/schema.sql
~~~~

### 2.3. Commandes de Gestion du Conteneur
Toutes les commandes s'exécutent depuis la racine du projet :

| Action | Commande | Description |
| :--- | :--- | :--- |
| **Démarrage** | `docker compose up -d` | Télécharge l'image, crée le volume et lance le service. |
| **Vérification** | `docker compose ps` | Confirme l'état de santé (`healthy`). |
| **Consultation Logs** | `docker compose logs -f` | Affiche le flux de journaux PostgreSQL en temps réel. |
| **Arrêt Sécurisé** | `docker compose stop` | Arrête le conteneur en préservant les données dans le volume. |
| **Purge Complète** | `docker compose down -v` | Supprime le conteneur et purge le volume de données. |

### 2.4. Guide d'Interrogation de la Base de Données (CLI)

Pour explorer ou déboguer les données directement depuis le terminal via `psql` :

#### A. Session interactive (Ligne de commande SQL)
~~~~bash
# Ouvrir le terminal interactif PostgreSQL
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db
~~~~
*Une fois dans le shell `psql` :*
* `\dt` : Lister toutes les tables.
* `\d raw_pharmaceutical_demand` : Afficher la structure détaillée d'une table (colonnes, types, index).
* `\q` : Quitter le shell `psql`.

#### B. Requêtes rapides ponctuelles (Command-line one-liners)
~~~~bash
# Compter le nombre de lignes
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT COUNT(*) FROM raw_pharmaceutical_demand;"

# Inspecter les 5 premières lignes
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT * FROM raw_pharmaceutical_demand LIMIT 5;"

# Obtenir la liste des produits uniques et leur volume d'enregistrements
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT product_id, COUNT(*) FROM raw_pharmaceutical_demand GROUP BY product_id;"
~~~~

---

## 3. Pipeline d'Ingestion (`src/ingestion.py`)

Le script `src/ingestion.py` extrait les données brutes du fichier CSV local (`data/pharmaceutical_demand.csv`) et les charge dans la table de staging `raw_pharmaceutical_demand` pré-structurée par le DDL.

### 3.1. Évolution MLOps : Du Dynamique vers l'Explicite
* **Ancienne approche (Pandas `to_sql`) :** Pandas créait dynamiquement la table au premier lancement, sans typage strict ni indexation.
* **Nouvelle approche (DDL Explicite) :** La table et ses index sont initialisés via `sql/schema.sql`. Le pipeline Python insère ensuite les enregistrements dans la structure existante via `if_exists="append"`, garantissant l'intégrité du schéma et des performances optimales sur les séries temporelles.

---

## 4. Exécution & Validation

### 4.1. Lancement du Pipeline
~~~~bash
python src/ingestion.py
~~~~

### 4.2. Vérification de la Base de Données
~~~~bash
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT COUNT(*) FROM raw_pharmaceutical_demand;"
~~~~

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *This document details the containerized data storage infrastructure using Docker (PostgreSQL), explicit DDL schema management (`sql/schema.sql`), and the automated data ingestion pipeline (`src/ingestion.py`).*

---

## 1. Context & MLOps Architecture

The primary goal of this phase is to isolate the data storage layer to guarantee **Dev-Prod Parity**. Under Linux (WSL2), PostgreSQL is instantiated using Docker Compose to provide an isolated, versioned, and reproducible relational database.

~~~~text
PharmaSupply-ML/
├── .env                          # PostgreSQL credentials (ignored by Git)
├── .env.example                  # Environment variable template
├── docker-compose.yml            # PostgreSQL services & persistent volumes
├── sql/
│   └── schema.sql                # Official DDL schema and performance indexing
├── data/
│   └── pharmaceutical_demand.csv   # Raw dataset (10,950 rows)
├── src/
│   └── ingestion.py              # Automated ingestion pipeline
└── docs/
    └── 02_DATA_INFRASTRUCTURE_AND_INGESTION.md
~~~~

---

## 2. Containerized Infrastructure & Explicit DDL Schema

### 2.1. `docker-compose.yml` File
The PostgreSQL 16 database (lightweight Alpine image) is configured with volume persistence:

~~~~yaml
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
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
    driver: local
~~~~

### 2.2. SQL Schema Management (`sql/schema.sql`)
To adhere to production and MLOps best practices, table creation (`raw_pharmaceutical_demand`) is **explicitly** defined in a DDL script. This replaces Pandas' implicit table generation to ensure strict column types and to introduce a **composite performance index** (`idx_pharma_date_product`).

**Applying DDL schema to database:**
~~~~bash
docker compose cp sql/schema.sql postgres:/tmp/schema.sql
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -f /tmp/schema.sql
~~~~

### 2.3. Container Management Commands
All commands should be executed from the project root:

| Action | Command | Description |
| :--- | :--- | :--- |
| **Start** | `docker compose up -d` | Pulls image, creates volume, and starts container in detached mode. |
| **Verify** | `docker compose ps` | Confirms container health status (`healthy`). |
| **View Logs** | `docker compose logs -f` | Streams live PostgreSQL output logs. |
| **Stop (Safe)** | `docker compose stop` | Stops container while preserving data in the volume. |
| **Full Cleanup** | `docker compose down -v` | Removes container and **purges the data volume**. |

### 2.4. Database Querying Guide (CLI)

To inspect or debug data directly from your terminal using `psql`:

#### A. Interactive Session (SQL Shell)
~~~~bash
# Connect to interactive PostgreSQL shell
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db
~~~~
*Common `psql` meta-commands:*
* `\dt`: List all tables in current database.
* `\d raw_pharmaceutical_demand`: Display detailed table schema (columns, types, indexes).
* `\q`: Exit `psql` shell.

#### B. Quick One-Line Queries
~~~~bash
# Row count check
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT COUNT(*) FROM raw_pharmaceutical_demand;"

# Inspect first 5 records
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT * FROM raw_pharmaceutical_demand LIMIT 5;"

# List unique products with record count
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT product_id, COUNT(*) FROM raw_pharmaceutical_demand GROUP BY product_id;"
~~~~

---

## 3. Ingestion Pipeline (`src/ingestion.py`)

The `src/ingestion.py` script extracts raw data from the local CSV file (`data/pharmaceutical_demand.csv`) and loads it into the `raw_pharmaceutical_demand` staging table pre-structured by DDL.

### 3.1. MLOps Transition: From Dynamic to Explicit
* **Legacy Approach (Pandas `to_sql`):** Pandas dynamically created the table on execution without explicit indexing or strict typing.
* **Modern Approach (Explicit DDL):** The table and indices are initialized via `sql/schema.sql`. The Python script appends records (`if_exists="append"`) into the defined structure, ensuring schema integrity and optimal time-series query performance.

---

## 4. Execution & Validation

### 4.1. Run Pipeline
~~~~bash
python src/ingestion.py
~~~~

### 4.2. Query Staging Table
~~~~bash
docker compose exec postgres psql -U ycs_admin -d pharmasupply_db -c "SELECT COUNT(*) FROM raw_pharmaceutical_demand;"
~~~~

</details>

