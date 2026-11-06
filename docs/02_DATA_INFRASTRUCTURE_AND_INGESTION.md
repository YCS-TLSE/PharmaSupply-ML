# Technical Documentation
## Phase 02: Data Infrastructure and Ingestion
**Project:** PharmaSupply-ML
**Framework:** Standard MLOps Blueprint
**File:** `docs/02_DATA_INFRASTRUCTURE_AND_INGESTION.md`
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

*Un guide pratique pour modéliser, ingérer et faire évoluer des données pharmaceutiques sans perte ni duplication.*

---

## Sommaire

1. [Modélisation et architecture des données](#1-modelisation-et-architecture-des-donnees)
2. [Ingestion et idempotence](#2-ingestion-et-idempotence)
3. [Accès aux données et gestion des connexions](#3-acces-aux-donnees-et-gestion-des-connexions)
4. [Migrations de schéma avec Alembic](#4-migrations-de-schema-avec-alembic)
5. [Génération de données et découpage Plug & Play](#5-generation-de-donnees-et-decoupage-plug--play)
6. [Annexes et checklist d'ingénierie](#6-annexes-et-checklist-dingenierie)

---

<a id="1-modelisation-et-architecture-des-donnees"></a>
## 1. Modélisation et architecture des données

### 1.1 Une table de faits, pas un fichier plat

Dans un notebook, les données vivent souvent dans un CSV unique. En production, il faut un contrat de stockage : des colonnes typées, des contraintes et une clé qui identifie une observation.

PharmaSupply-ML utilise une table de faits, `raw_pharmaceutical_demand`. Chaque ligne représente une observation quotidienne pour un couple officine/produit : la demande, les ventes, le stock, le délai fournisseur et les contraintes logistiques.

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_pharma_store_product_date
        UNIQUE (store_id, product_id, date)
);

CREATE INDEX idx_pharma_store_product_date
    ON raw_pharmaceutical_demand (store_id, product_id, date);
```

Cette table n'est pas entièrement normalisée. Les informations produit et officine sont répétées à chaque ligne. C'est un choix assumé pour un projet pédagogique et pour un premier niveau de maturité. Une évolution naturelle consisterait à introduire des tables de dimension `dim_store` et `dim_product`, puis une table de faits plus étroite.

### 1.2 Le rôle des colonnes métier

Chaque colonne existe pour une décision précise.

| Colonne | Type | Rôle métier |
|---|---|---|
| `date` | `DATE` | Dimension temporelle, base des lags et du split chronologique |
| `store_id` | `VARCHAR(50)` | Identifiant de l'officine |
| `product_id` | `VARCHAR(50)` | Identifiant du produit |
| `target_demand` | `INT` | Demande observée ou cible d'apprentissage |
| `sales_volume` | `INT` | Ventes réellement réalisées |
| `stock_on_hand` | `INT` | Stock physique disponible |
| `is_stockout` | `INT` | Indique une rupture, donc une demande censurée |
| `supplier_lead_time` | `INT` | Délai de livraison fournisseur |
| `safety_threshold` | `NUMERIC(10,2)` | Seuil de stock de sécurité |
| `batch_size` | `INT` | Taille de lot pour l'arrondi de commande |
| `moq` | `INT` | Quantité minimale de commande |
| `shelf_life_days` | `INT` | Durée de conservation |
| `critical_expiry_days` | `INT` | Jours restants avant péremption du lot le plus urgent |
| `created_at` | `TIMESTAMP` | Horodatage d'ingestion |

Les colonnes `batch_size`, `moq`, `shelf_life_days` et `critical_expiry_days` ne servent pas directement au modèle XGBoost. Elles servent à la décision de réapprovisionnement et aux alertes FEFO. C'est une distinction importante : la table ne stocke pas seulement des features, elle stocke le contexte métier nécessaire à l'action.

### 1.3 La clé métier naturelle

La clé `(store_id, product_id, date)` est la clé métier. Elle répond à une question simple : que s'est-il passé pour ce produit, dans cette officine, ce jour-là ?

La contrainte d'unicité correspondante est déclarée dans `sql/schema.sql` et dans la migration `alembic/versions/0001_initial_schema.py`. Dans le code, elle porte le nom `uq_pharma_store_product_date` ; elle matérialise la clé métier parfois désignée plus courtement par `uq_store_product_date`. Sans elle, l'UPSERT ne peut pas cibler un conflit et l'ingestion ne peut pas être idempotente.

La colonne `id` reste une clé technique pour PostgreSQL. Elle ne doit pas être utilisée comme identifiant métier. Un jour, le pipeline pourrait changer de base ou fusionner des sources ; la clé métier, elle, doit rester stable.

### 1.4 L'index et les requêtes de serving

L'index `idx_pharma_store_product_date` accélère les requêtes filtrées par officine, produit et date. C'est exactement le motif utilisé par :

- `src/repositories/inventory.py` pour lister les produits d'une officine ;
- `src/repositories/inventory.py` pour récupérer la dernière ligne par produit ;
- `src/repositories/inventory.py` pour récupérer les 30 derniers jours d'un couple officine/produit.

Un index n'est pas gratuit. Il ralentit légèrement les écritures et occupe de l'espace. Ici, le compromis est pertinent : les lectures de serving sont fréquentes, les écritures sont principalement des ingestions planifiées.

### 1.5 Les limites du schéma actuel

Le schéma actuel ne gère pas encore :

- les dimensions séparées pour les officines et les produits ;
- l'historique des changements de paramètres logistiques ;
- les lots physiques multiples avec leurs quantités et dates de péremption ;
- le partitionnement par date ;
- les contraintes `CHECK` sur les plages de valeurs.

Ces limites sont connues. Elles correspondent à des chantiers d'industrialisation avancée, pas à des oublis de la phase d'ingestion.

---

<a id="2-ingestion-et-idempotence"></a>
## 2. Ingestion et idempotence

### 2.1 Le problème du TRUNCATE

Une première version de l'ingestion utilisait `TRUNCATE TABLE ... RESTART IDENTITY` puis `APPEND`. Le comportement semblait simple : vider la table, recharger le CSV, repartir proprement.

En pratique, cette approche pose plusieurs problèmes.

| Problème | Conséquence |
|---|---|
| Perte de l'historique | Les lignes absentes du CSV disparaissent |
| Fenêtre d'indisponibilité | La table est vide entre le truncate et le rechargement |
| Rejeu risqué | Un incident pendant le chargement laisse la base partielle |
| Traçabilité | Impossible de savoir ce qui a été mis à jour ou supprimé |
| Concurrence | Une lecture pendant le rechargement peut voir un état incohérent |

Un pipeline qui doit tourner tous les jours ne peut pas reposer sur la destruction de la donnée.

### 2.2 L'UPSERT comme contrat d'idempotence

L'ingestion actuelle utilise un UPSERT :

```sql
INSERT INTO raw_pharmaceutical_demand (
    date, store_id, product_id, target_demand, sales_volume,
    stock_on_hand, is_stockout, supplier_lead_time, safety_threshold,
    batch_size, moq, shelf_life_days, critical_expiry_days
)
VALUES %s
ON CONFLICT (store_id, product_id, date)
DO UPDATE SET
    target_demand = EXCLUDED.target_demand,
    sales_volume = EXCLUDED.sales_volume,
    stock_on_hand = EXCLUDED.stock_on_hand,
    is_stockout = EXCLUDED.is_stockout,
    supplier_lead_time = EXCLUDED.supplier_lead_time,
    safety_threshold = EXCLUDED.safety_threshold,
    batch_size = EXCLUDED.batch_size,
    moq = EXCLUDED.moq,
    shelf_life_days = EXCLUDED.shelf_life_days,
    critical_expiry_days = EXCLUDED.critical_expiry_days;
```

Le résultat est simple : relancer l'ingestion ne duplique rien et ne supprime rien. Les lignes existantes sont mises à jour avec les valeurs du CSV. Les nouvelles lignes sont insérées. L'état final est le même, que la table soit vide ou déjà remplie.

C'est la définition pratique de l'idempotence.

### 2.3 Anatomie de src/ingestion.py

Le module `src/ingestion.py` suit une séquence stable.

**Lecture du CSV.** Le fichier est chargé dans un DataFrame. Si le fichier est absent, le module s'arrête avec un message clair.

**Normalisation des identifiants.** `store_id` et `product_id` sont convertis en chaînes. Si une colonne manque, une valeur par défaut est appliquée. Cette normalisation évite les mélanges entre `1`, `"1"` et `STORE_001`.

**Application des valeurs par défaut.** Les colonnes `safety_threshold`, `batch_size`, `moq`, `shelf_life_days` et `critical_expiry_days` reçoivent des valeurs par défaut si elles sont absentes. Les valeurs nulles sont également traitées.

**Sécurisation du nom de table.** `_assert_safe_table_name()` refuse tout identifiant qui ne contient pas uniquement des lettres, des chiffres et des underscores. Le nom de table est interpolé dans le SQL, donc cette garde est nécessaire.

**Inspection du schéma.** Le module vérifie que la table existe et récupère ses colonnes. Seules les colonnes présentes à la fois dans le CSV et dans la table sont insérées. Les colonnes techniques `id` et `created_at` sont laissées à PostgreSQL.

**Alignement des types.** La date est convertie en objet `date`. Les colonnes entières sont converties en `int`. `safety_threshold` est converti en `float`. Ce point évite les erreurs de binding psycopg2.

**Création défensive de l'index.** `CREATE UNIQUE INDEX IF NOT EXISTS uq_pharma_store_product_date` garantit que la cible du conflit existe, même sur une base créée avec un ancien schéma. Cette sécurité complète la migration Alembic, elle ne la remplace pas.

**Écriture par lots.** `execute_values(..., page_size=1000)` envoie les lignes par paquets. Cette approche est plus rapide qu'un `to_sql` ligne par ligne et plus simple qu'une construction SQL manuelle.

**Gestion des erreurs.** Les erreurs SQLAlchemy et psycopg2 sont capturées et affichées. La connexion brute est fermée dans un `finally`, ce qui provoque un rollback si le commit n'a pas eu lieu.

### 2.4 Gestion des conflits

Le conflit est ciblé sur les trois colonnes métier. PostgreSQL utilise l'index unique pour retrouver la ligne existante et applique le `DO UPDATE`.

Deux détails méritent l'attention.

Le premier est que la cible du `ON CONFLICT` doit correspondre exactement à une contrainte ou à un index unique. Une faute de nom ou une colonne manquante fait échouer l'UPSERT.

Le second est que les colonnes de clé ne sont pas mises à jour. On ne réécrit pas `store_id`, `product_id` ou `date` dans le `SET`. Ce serait un changement d'identité, pas une mise à jour.

### 2.5 Rejeu et exploitation

Un pipeline idempotent change la manière de travailler.

Vous pouvez relancer l'ingestion après une correction de CSV sans vider la base. Vous pouvez rejouer un lot après un incident réseau. Vous pouvez comparer le résultat de deux exécutions sans craindre un doublon.

Le prix à payer est une contrainte de qualité : la clé métier doit être fiable. Si deux lignes du CSV partagent le même `(store_id, product_id, date)`, PostgreSQL signalera un conflit dans le même lot. Ce cas doit être traité en amont par la validation ou par une règle de déduplication.

---

<a id="3-acces-aux-donnees-et-gestion-des-connexions"></a>
## 3. Accès aux données et gestion des connexions

### 3.1 Pourquoi séparer les repositories

Les routers HTTP ne doivent pas connaître SQL. Les services métier ne doivent pas dépendre de psycopg2. Les repositories isolent cette responsabilité.

Le dépôt contient deux modules :

- `src/repositories/database.py` : engine, pool, curseurs, erreurs et health check ;
- `src/repositories/inventory.py` : requêtes read-only utilisées par l'API.

Cette séparation permet de tester les routes avec un curseur simulé, sans lancer PostgreSQL. Elle permet aussi de changer une requête sans toucher au transport.

### 3.2 Le pool SQLAlchemy

`src/repositories/database.py` crée un engine SQLAlchemy unique pour le processus. Les paramètres viennent de `src/config.py` :

- `DB_POOL_SIZE` : nombre de connexions maintenues ;
- `DB_MAX_OVERFLOW` : connexions supplémentaires autorisées ;
- `DB_POOL_TIMEOUT` : attente maximale pour obtenir une connexion.

L'option `pool_pre_ping=True` vérifie qu'une connexion est encore vivante avant de la réutiliser. Sans elle, un redémarrage PostgreSQL peut laisser des connexions mortes dans le pool et provoquer des erreurs intermittentes.

Le pool évite d'ouvrir une connexion à chaque requête. Sur une API qui reçoit du trafic, cette différence est loin d'être négligeable.

### 3.3 Le contexte db_cursor

Toutes les requêtes passent par `db_cursor()`. Ce context manager :

1. emprunte une connexion au pool ;
2. crée un curseur standard ou un `DictCursor` ;
3. yield le curseur ;
4. ferme la connexion dans un `finally`, ce qui la rend au pool.

Le `DictCursor` est utilisé pour les endpoints qui lisent des lignes par nom de colonne (`row["product_id"]`). Le curseur standard suffit pour les listes scalaires.

Le fait de centraliser ce cycle de vie évite les fuites de connexions. Une connexion oubliée finit par saturer le pool et provoquer des timeouts.

### 3.4 Résilience aux pannes de base

`db_cursor()` convertit les erreurs de connexion et les erreurs psycopg2/SQLAlchemy en `DatabaseUnavailableError`. Les routers capturent cette exception et renvoient `503`.

Ce choix est volontaire. Une API qui renvoie `200` avec une liste de magasins vide alors que la base est down trompe le client. Une API qui renvoie `503` indique clairement que la dépendance est indisponible.

Le health check `check_database_health()` exécute un `SELECT 1`. Il est utilisé par `/readyz`. Il ne remonte pas les détails de connexion au client, mais il permet à l'orchestrateur de savoir si le service peut recevoir du trafic.

### 3.5 Requêtes read-only de serving

`src/repositories/inventory.py` contient quatre requêtes.

| Fonction | Usage | Particularité |
|---|---|---|
| `fetch_store_ids()` | Lister les officines | `DISTINCT`, tri alphabétique |
| `fetch_product_ids()` | Lister les produits d'une officine | Variantes d'identifiants via `normalize_store_id` |
| `fetch_alert_rows()` | Alimenter les alertes | `DISTINCT ON (product_id)` pour garder la dernière ligne |
| `fetch_feature_rows()` | Alimenter `/features` | `LIMIT 30` pour reconstruire l'historique récent |

Ces requêtes sont en lecture seule. Elles ne modifient jamais la base. Cette propriété simplifie l'audit et réduit le risque d'effet de bord dans le serving.

---

<a id="4-migrations-de-schema-avec-alembic"></a>
## 4. Migrations de schéma avec Alembic

### 4.1 Pourquoi Alembic

`sql/schema.sql` initialise un volume Docker neuf. Il ne gère pas les évolutions d'une base existante. Si vous ajoutez une colonne ou une contrainte, un simple redémarrage ne l'appliquera pas.

Alembic apporte un historique versionné des changements de schéma. Chaque migration possède un identifiant, une révision parente et deux fonctions : `upgrade()` et `downgrade()`.

En production, ce n'est pas un confort. C'est la différence entre une évolution maîtrisée et une intervention manuelle sur une base critique.

### 4.2 Structure du dossier alembic/

```text
alembic/
├── env.py
├── script.py.mako
└── versions/
    └── 0001_initial_schema.py
```

`alembic.ini` configure le chemin des scripts et laisse l'URL vide. `env.py` la remplit à partir de `src/db_config.get_database_url()`. Cette approche évite de dupliquer les identifiants dans plusieurs fichiers.

`script.py.mako` est le modèle utilisé par `alembic revision`. Il génère les métadonnées de révision et les fonctions `upgrade` / `downgrade`.

### 4.3 La migration 0001

`alembic/versions/0001_initial_schema.py` crée :

- la table `raw_pharmaceutical_demand` ;
- les colonnes et leurs types ;
- la contrainte `uq_pharma_store_product_date` ;
- l'index `idx_pharma_store_product_date`.

La migration est volontairement explicite. Elle ne dépend pas d'un modèle ORM ni d'un autogenerate. Ce choix rend la revue plus simple : le SQL attendu est lisible directement dans le fichier Python.

### 4.4 Vérifier le SQL sans base live

La commande suivante génère le SQL de migration sans se connecter à PostgreSQL :

```bash
alembic upgrade head --sql
```

Cette vérification est utile en revue de code et en CI. Elle permet de relire les `CREATE TABLE`, `ALTER TABLE` et `CREATE INDEX` avant de les appliquer. Elle permet aussi de détecter une migration qui dépend d'un état absent.

Pour appliquer réellement les migrations :

```bash
alembic upgrade head
```

### 4.5 Bonnes pratiques de migration

Quelques règles évitent les migrations douloureuses.

Ne modifiez pas une migration déjà appliquée en production. Créez une nouvelle révision. Une migration est un fait historique, pas un brouillon.

Séparez les changements destructifs en plusieurs étapes : ajouter une colonne nullable, remplir les données, puis ajouter la contrainte. Un `DROP COLUMN` direct peut casser une version précédente du service.

Testez la migration sur une copie de base avant de l'appliquer. `alembic upgrade head --sql` aide à relire, mais il ne remplace pas un test sur un volume réel.

### 4.6 Cohérence entre schema.sql et Alembic

Deux chemins créent le schéma : `sql/schema.sql` pour un volume neuf, Alembic pour une base existante. Ils doivent rester alignés.

La table, la contrainte et l'index doivent avoir les mêmes noms et les mêmes types dans les deux fichiers. Une divergence crée des environnements différents selon la manière dont la base a été initialisée.

L'ingestion ajoute une création défensive d'index unique. Cette sécurité évite un échec d'UPSERT sur une base ancienne, mais elle ne dispense pas de maintenir la migration. À terme, Alembic doit rester l'autorité unique du schéma.

---

<a id="5-generation-de-donnees-et-decoupage-plug--play"></a>
## 5. Génération de données et découpage Plug & Play

### 5.1 Pourquoi générer des données

Les données pharmaceutiques réelles sont sensibles. Elles sont couvertes par le secret commercial, le RGPD et parfois des réglementations sectorielles. Un projet pédagogique ne peut pas embarquer un historique réel.

`src/data/generate_data.py` produit un jeu synthétique multi-sites et multi-produits. Par défaut, il génère 3 officines, 10 produits et 1 095 jours, soit 32 850 lignes.

Le générateur ne cherche pas à imiter parfaitement la réalité. Il cherche à produire des motifs suffisamment riches pour exercer le pipeline : saisonnalité, week-ends, ruptures, délais variables, contraintes logistiques et anomalies.

### 5.2 Saisonnalité, ruptures et délais

La demande de base est tirée aléatoirement par produit. Elle est ensuite modulée par :

- une tendance annuelle légère ;
- une saisonnalité hivernale et printanière ;
- une réduction des ventes le week-end ;
- une vérification de stock qui peut provoquer une rupture.

Le délai fournisseur est tiré parmi plusieurs valeurs avec des probabilités réalistes. Le stock est réapprovisionné lorsqu'il descend sous un seuil. Ces règles produisent des séries qui ne sont pas de simples bruits blancs.

### 5.3 Anomalies injectées volontairement

Le générateur introduit des cas de test :

- `MED_003` reçoit un seuil de sécurité trop bas ;
- `MED_008` reçoit un seuil trop haut et une durée de conservation plus courte ;
- certains produits ont des `batch_size` ou des `moq` manquants ;
- `critical_expiry_days` simule la proximité de péremption du lot le plus urgent.

Ces anomalies ne sont pas des bugs. Elles permettent de vérifier que le quality gate, les alertes et les règles de réapprovisionnement réagissent correctement.

### 5.4 Isolation vis-à-vis d'un futur connecteur réel

Le générateur produit un CSV. L'ingestion lit ce CSV et écrit dans PostgreSQL. Le reste du pipeline ne sait pas d'où viennent les données.

Cette frontière est volontaire. Un connecteur réel (ERP, entrepôt de données, API fournisseur) devra produire le même contrat de colonnes et écrire dans la même table de staging. Les modules de validation, de feature engineering et de serving n'auront pas à changer.

Le point de couplage est le schéma de `raw_pharmaceutical_demand`, pas le fichier CSV. C'est ce qui rend le découpage « Plug & Play » réaliste.

### 5.5 Limites du jeu synthétique

Un jeu synthétique ne remplace pas une validation métier. Il ne contient pas les effets de promotion, les ruptures fournisseur exceptionnelles, les retours produit ni les changements de catalogue.

La date de fin est ancrée sur `datetime.now()`. Deux générations à des dates différentes produisent donc des index temporels différents. Le seed fixe les valeurs aléatoires, pas le calendrier.

Ces limites doivent être explicites dans toute présentation des résultats. Un modèle entraîné sur des données synthétiques ne doit pas être présenté comme validé sur le terrain.

---

<a id="6-annexes-et-checklist-dingenierie"></a>
## 6. Annexes et checklist d'ingénierie

### 6.1 Commandes d'ingestion et de migration

```bash
# Ingestion directe du CSV
python -m src.ingestion

# Pipeline complet (génération, ingestion, validation, entraînement)
python run_pipeline.py

# Application des migrations
alembic upgrade head

# Vérification du SQL sans base live
alembic upgrade head --sql
```

### 6.2 Requêtes d'inspection

```sql
-- Nombre de lignes
SELECT COUNT(*) FROM raw_pharmaceutical_demand;

-- Doublons sur la clé métier
SELECT store_id, product_id, date, COUNT(*)
FROM raw_pharmaceutical_demand
GROUP BY store_id, product_id, date
HAVING COUNT(*) > 1;

-- Plage temporelle
SELECT MIN(date), MAX(date) FROM raw_pharmaceutical_demand;

-- Contraintes et index
SELECT conname, contype
FROM pg_constraint
WHERE conrelid = 'raw_pharmaceutical_demand'::regclass;

-- Valeurs négatives ou invalides
SELECT COUNT(*) FROM raw_pharmaceutical_demand WHERE stock_on_hand < 0;
SELECT COUNT(*) FROM raw_pharmaceutical_demand WHERE target_demand < 0;
SELECT COUNT(*) FROM raw_pharmaceutical_demand WHERE moq < 1;
```

### 6.3 Checklist de validation des données

- [ ] Le nombre de lignes correspond au produit `jours × produits × officines`.
- [ ] La clé `(store_id, product_id, date)` ne contient aucun doublon.
- [ ] Les colonnes `store_id`, `product_id` et `date` ne sont pas nulles.
- [ ] La plage de dates est continue et cohérente.
- [ ] `target_demand` et `sales_volume` sont positifs ou nuls.
- [ ] `stock_on_hand` est positif ou nul.
- [ ] `supplier_lead_time` est strictement positif.
- [ ] `safety_threshold` est strictement positif.
- [ ] `batch_size` et `moq` sont supérieurs ou égaux à 1.
- [ ] `shelf_life_days` est supérieur ou égal à 1.
- [ ] `critical_expiry_days` est positif ou nul.
- [ ] L'index unique `uq_pharma_store_product_date` existe.
- [ ] L'index `idx_pharma_store_product_date` existe.
- [ ] La migration Alembic s'applique sans erreur.

### 6.4 Glossaire

| Terme | Définition |
|---|---|
| **Clé métier** | Ensemble de colonnes qui identifie une observation de manière stable |
| **UPSERT** | Insertion ou mise à jour selon une clé, via `ON CONFLICT` |
| **Idempotence** | Propriété d'une opération rejouable sans changer l'état final |
| **Pool de connexions** | Réserve de connexions réutilisables pour éviter les ouvertures répétées |
| **Migration** | Changement de schéma versionné et réversible |
| **Staging** | Table intermédiaire qui reçoit les données brutes avant transformation |
| **FEFO** | *First Expired, First Out* : prioriser les lots les plus proches de la péremption |
| **MOQ** | *Minimum Order Quantity* : quantité minimale de commande |

### 6.5 Fichiers de code à lire

- `src/data/generate_data.py` : génération synthétique, saisonnalité et anomalies.
- `src/ingestion.py` : normalisation, alignement de types et UPSERT.
- `src/repositories/database.py` : engine, pool, curseurs et health check.
- `src/repositories/inventory.py` : requêtes read-only de serving.
- `sql/schema.sql` : DDL initial, clé métier et index.
- `alembic/env.py` : résolution de l'URL et exécution des migrations.
- `alembic/versions/0001_initial_schema.py` : migration initiale.
- `tests/test_ingestion.py` : tests unitaires de normalisation et d'ingestion.
- `tests/test_ingestion_integration.py` : test Testcontainers sur PostgreSQL réel.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *A practical guide to modeling, ingesting and evolving pharmaceutical data without loss or duplication.*

---

## Table of Contents

1. [Data Modeling and Architecture](#1-data-modeling-and-architecture)
2. [Ingestion and Idempotence](#2-ingestion-and-idempotence)
3. [Data Access and Connection Management](#3-data-access-and-connection-management)
4. [Schema Migrations with Alembic](#4-schema-migrations-with-alembic)
5. [Data Generation and Plug & Play Boundaries](#5-data-generation-and-plug--play-boundaries)
6. [Appendices and Engineering Checklist](#6-appendices-and-engineering-checklist)

---

<a id="1-data-modeling-and-architecture"></a>
## 1. Data Modeling and Architecture

### 1.1 A fact table, not a flat file

In a notebook, data often lives in a single CSV. In production, you need a storage contract: typed columns, constraints and a key that identifies an observation.

PharmaSupply-ML uses a fact table, `raw_pharmaceutical_demand`. Each row represents a daily observation for one pharmacy/product pair: demand, sales, stock, supplier lead time and logistics constraints.

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_pharma_store_product_date
        UNIQUE (store_id, product_id, date)
);

CREATE INDEX idx_pharma_store_product_date
    ON raw_pharmaceutical_demand (store_id, product_id, date);
```

The table is not fully normalized. Product and pharmacy information is repeated on every row. This is an intentional choice for an educational project at an early maturity level. A natural evolution would introduce `dim_store` and `dim_product` dimension tables and a narrower fact table.

### 1.2 The role of business columns

Each column exists for a precise decision.

| Column | Type | Business role |
|---|---|---|
| `date` | `DATE` | Temporal dimension, basis for lags and chronological split |
| `store_id` | `VARCHAR(50)` | Pharmacy identifier |
| `product_id` | `VARCHAR(50)` | Product identifier |
| `target_demand` | `INT` | Observed demand or training target |
| `sales_volume` | `INT` | Actual sales |
| `stock_on_hand` | `INT` | Physical stock available |
| `is_stockout` | `INT` | Stockout flag, therefore censored demand |
| `supplier_lead_time` | `INT` | Supplier delivery time |
| `safety_threshold` | `NUMERIC(10,2)` | Safety stock threshold |
| `batch_size` | `INT` | Batch size for order rounding |
| `moq` | `INT` | Minimum order quantity |
| `shelf_life_days` | `INT` | Shelf life |
| `critical_expiry_days` | `INT` | Remaining days before the most urgent batch expires |
| `created_at` | `TIMESTAMP` | Ingestion timestamp |

`batch_size`, `moq`, `shelf_life_days` and `critical_expiry_days` are not direct model features. They support replenishment decisions and FEFO alerts. This distinction matters: the table stores the business context needed for action, not only model features.

### 1.3 The natural business key

The key `(store_id, product_id, date)` is the business key. It answers a simple question: what happened for this product, in this pharmacy, on this day?

The corresponding unique constraint, `uq_pharma_store_product_date`, is declared in `sql/schema.sql` and in `alembic/versions/0001_initial_schema.py`. In the code it is sometimes referred to by the shorter name `uq_store_product_date`. Without it, the UPSERT cannot target a conflict and ingestion cannot be idempotent.

`id` remains a PostgreSQL technical key. It must not be used as a business identifier. The pipeline may change database or merge sources one day; the business key must remain stable.

### 1.4 Index and serving queries

`idx_pharma_store_product_date` speeds up queries filtered by pharmacy, product and date. This is exactly the pattern used by:

- `src/repositories/inventory.py` to list products for a pharmacy;
- `src/repositories/inventory.py` to fetch the latest row per product;
- `src/repositories/inventory.py` to fetch the last 30 days for a pharmacy/product pair.

An index is not free. It slightly slows writes and uses space. Here the trade-off is sound: serving reads are frequent, writes are mostly scheduled ingestions.

### 1.5 Limits of the current schema

The current schema does not yet handle:

- separate dimensions for pharmacies and products;
- history of logistics parameter changes;
- multiple physical batches with quantities and expiry dates;
- date partitioning;
- `CHECK` constraints on value ranges.

These limits are known. They belong to advanced industrialization work, not to an oversight in the ingestion phase.

---

<a id="2-ingestion-and-idempotence"></a>
## 2. Ingestion and Idempotence

### 2.1 The TRUNCATE problem

An early ingestion version used `TRUNCATE TABLE ... RESTART IDENTITY` followed by `APPEND`. It looked simple: empty the table, reload the CSV, start clean.

In practice, this approach has several problems.

| Problem | Consequence |
|---|---|
| History loss | Rows absent from the CSV disappear |
| Availability gap | The table is empty between truncate and reload |
| Risky replay | A failure during loading leaves a partial database |
| Traceability | Impossible to know what was updated or deleted |
| Concurrency | A reader during reload may see an inconsistent state |

A pipeline that runs daily cannot rely on destroying data.

### 2.2 UPSERT as an idempotence contract

Current ingestion uses an UPSERT:

```sql
INSERT INTO raw_pharmaceutical_demand (
    date, store_id, product_id, target_demand, sales_volume,
    stock_on_hand, is_stockout, supplier_lead_time, safety_threshold,
    batch_size, moq, shelf_life_days, critical_expiry_days
)
VALUES %s
ON CONFLICT (store_id, product_id, date)
DO UPDATE SET
    target_demand = EXCLUDED.target_demand,
    sales_volume = EXCLUDED.sales_volume,
    stock_on_hand = EXCLUDED.stock_on_hand,
    is_stockout = EXCLUDED.is_stockout,
    supplier_lead_time = EXCLUDED.supplier_lead_time,
    safety_threshold = EXCLUDED.safety_threshold,
    batch_size = EXCLUDED.batch_size,
    moq = EXCLUDED.moq,
    shelf_life_days = EXCLUDED.shelf_life_days,
    critical_expiry_days = EXCLUDED.critical_expiry_days;
```

The result is simple: rerunning ingestion duplicates nothing and deletes nothing. Existing rows are updated from the CSV. New rows are inserted. The final state is the same whether the table was empty or already populated.

That is the practical definition of idempotence.

### 2.3 Anatomy of src/ingestion.py

The module follows a stable sequence.

**CSV reading.** The file is loaded into a DataFrame. If missing, the module stops with a clear message.

**Identifier normalization.** `store_id` and `product_id` are converted to strings. Missing columns receive defaults. This avoids mixing `1`, `"1"` and `STORE_001`.

**Default values.** `safety_threshold`, `batch_size`, `moq`, `shelf_life_days` and `critical_expiry_days` receive defaults when absent. Null values are also handled.

**Table name safety.** `_assert_safe_table_name()` rejects identifiers containing anything other than letters, digits and underscores. The table name is interpolated into SQL, so this guard is required.

**Schema inspection.** The module checks that the table exists and retrieves its columns. Only columns present in both the CSV and the table are inserted. Technical columns `id` and `created_at` are left to PostgreSQL.

**Type alignment.** Dates become `date` objects. Integer columns become `int`. `safety_threshold` becomes `float`. This avoids psycopg2 binding errors.

**Defensive index creation.** `CREATE UNIQUE INDEX IF NOT EXISTS uq_pharma_store_product_date` ensures the conflict target exists, even on a database created from an older schema. This safety net complements Alembic; it does not replace it.

**Batch writing.** `execute_values(..., page_size=1000)` sends rows in chunks. This is faster than row-by-row `to_sql` and simpler than manual SQL construction.

**Error handling.** SQLAlchemy and psycopg2 errors are captured and displayed. The raw connection is closed in a `finally`, which rolls back if commit did not happen.

### 2.4 Conflict handling

The conflict targets the three business columns. PostgreSQL uses the unique index to find the existing row and applies `DO UPDATE`.

Two details matter.

First, the `ON CONFLICT` target must exactly match a unique constraint or index. A wrong name or missing column makes the UPSERT fail.

Second, key columns are not updated. `store_id`, `product_id` and `date` are not rewritten in `SET`. That would be an identity change, not an update.

### 2.5 Replay and operations

An idempotent pipeline changes how you work.

You can rerun ingestion after fixing a CSV without emptying the database. You can replay a batch after a network incident. You can compare the result of two runs without fearing duplicates.

The price is a quality constraint: the business key must be reliable. If two CSV rows share the same `(store_id, product_id, date)`, PostgreSQL will report a conflict within the same batch. That case must be handled upstream by validation or a deduplication rule.

---

<a id="3-data-access-and-connection-management"></a>
## 3. Data Access and Connection Management

### 3.1 Why separate repositories

HTTP routers must not know SQL. Business services must not depend on psycopg2. Repositories isolate this responsibility.

The repository contains two modules:

- `src/repositories/database.py`: engine, pool, cursors, errors and health check;
- `src/repositories/inventory.py`: read-only queries used by the API.

This separation allows routes to be tested with a stub cursor, without PostgreSQL. It also allows a query to change without touching transport.

### 3.2 The SQLAlchemy pool

`src/repositories/database.py` creates a single SQLAlchemy engine per process. Parameters come from `src/config.py`:

- `DB_POOL_SIZE`: number of maintained connections;
- `DB_MAX_OVERFLOW`: additional connections allowed;
- `DB_POOL_TIMEOUT`: maximum wait for a connection.

`pool_pre_ping=True` checks that a connection is still alive before reuse. Without it, a PostgreSQL restart can leave dead connections in the pool and cause intermittent errors.

The pool avoids opening a connection for every request. On an API receiving traffic, the difference is significant.

### 3.3 The db_cursor context

Every query goes through `db_cursor()`. This context manager:

1. borrows a connection from the pool;
2. creates a standard or `DictCursor`;
3. yields the cursor;
4. closes the connection in a `finally`, returning it to the pool.

`DictCursor` is used by endpoints that read rows by column name (`row["product_id"]`). The standard cursor is enough for scalar lists.

Centralizing this lifecycle prevents connection leaks. A forgotten connection eventually saturates the pool and causes timeouts.

### 3.4 Resilience to database failures

`db_cursor()` converts connection, psycopg2 and SQLAlchemy errors into `DatabaseUnavailableError`. Routers catch it and return `503`.

This is deliberate. An API that returns `200` with an empty store list while the database is down misleads the client. An API that returns `503` clearly indicates an unavailable dependency.

`check_database_health()` runs `SELECT 1`. It is used by `/readyz`. It does not expose connection details, but it lets the orchestrator know whether the service can receive traffic.

### 3.5 Read-only serving queries

`src/repositories/inventory.py` contains four queries.

| Function | Usage | Particularity |
|---|---|---|
| `fetch_store_ids()` | List pharmacies | `DISTINCT`, alphabetical order |
| `fetch_product_ids()` | List products for a pharmacy | Identifier variants through `normalize_store_id` |
| `fetch_alert_rows()` | Feed alerts | `DISTINCT ON (product_id)` to keep the latest row |
| `fetch_feature_rows()` | Feed `/features` | `LIMIT 30` to rebuild recent history |

These queries are read-only. They never modify the database. This simplifies audit and reduces side effects in serving.

---

<a id="4-schema-migrations-with-alembic"></a>
## 4. Schema Migrations with Alembic

### 4.1 Why Alembic

`sql/schema.sql` initializes a fresh Docker volume. It does not handle existing database changes. If you add a column or constraint, a simple restart will not apply it.

Alembic provides a versioned history of schema changes. Each migration has an identifier, a parent revision and two functions: `upgrade()` and `downgrade()`.

In production, this is not a convenience. It is the difference between a controlled evolution and a manual intervention on a critical database.

### 4.2 The alembic/ folder structure

```text
alembic/
├── env.py
├── script.py.mako
└── versions/
    └── 0001_initial_schema.py
```

`alembic.ini` configures the script path and leaves the URL empty. `env.py` fills it from `src/db_config.get_database_url()`. This avoids duplicating credentials across files.

`script.py.mako` is the template used by `alembic revision`. It generates revision metadata and the `upgrade` / `downgrade` functions.

### 4.3 Migration 0001

`alembic/versions/0001_initial_schema.py` creates:

- the `raw_pharmaceutical_demand` table;
- columns and types;
- the `uq_pharma_store_product_date` constraint;
- the `idx_pharma_store_product_date` index.

The migration is intentionally explicit. It does not depend on an ORM model or autogenerate. This makes review simpler: the expected SQL is readable directly in the Python file.

### 4.4 Verifying SQL without a live database

The following command generates migration SQL without connecting to PostgreSQL:

```bash
alembic upgrade head --sql
```

This check is useful in code review and CI. It allows `CREATE TABLE`, `ALTER TABLE` and `CREATE INDEX` statements to be inspected before application. It also detects a migration that depends on a missing state.

To actually apply migrations:

```bash
alembic upgrade head
```

### 4.5 Migration best practices

A few rules prevent painful migrations.

Do not modify a migration already applied in production. Create a new revision. A migration is a historical fact, not a draft.

Split destructive changes into steps: add a nullable column, backfill data, then add the constraint. A direct `DROP COLUMN` can break a previous service version.

Test the migration on a database copy before applying it. `alembic upgrade head --sql` helps review, but it does not replace a test on a real volume.

### 4.6 Consistency between schema.sql and Alembic

Two paths create the schema: `sql/schema.sql` for a fresh volume, Alembic for an existing database. They must stay aligned.

Table, constraint and index must have the same names and types in both files. A divergence creates different environments depending on how the database was initialized.

Ingestion adds a defensive unique index creation. This prevents an UPSERT failure on an old database, but it does not remove the need to maintain the migration. Eventually, Alembic must remain the single schema authority.

---

<a id="5-data-generation-and-plug--play-boundaries"></a>
## 5. Data Generation and Plug & Play Boundaries

### 5.1 Why generate data

Real pharmaceutical data is sensitive. It is covered by trade secrecy, GDPR and sometimes sector-specific regulations. An educational project cannot embed a real history.

`src/data/generate_data.py` produces a synthetic multi-site, multi-product dataset. By default it generates 3 pharmacies, 10 products and 1,095 days, or 32,850 rows.

The generator does not try to imitate reality perfectly. It aims to produce patterns rich enough to exercise the pipeline: seasonality, weekends, stockouts, variable lead times, logistics constraints and anomalies.

### 5.2 Seasonality, stockouts and lead times

Base demand is drawn randomly per product. It is then modulated by:

- a slight annual trend;
- winter and spring seasonality;
- reduced weekend sales;
- a stock check that can trigger a stockout.

Supplier lead time is drawn from several values with realistic probabilities. Stock is replenished below a threshold. These rules produce series that are not pure white noise.

### 5.3 Deliberately injected anomalies

The generator introduces test cases:

- `MED_003` receives a too-low safety threshold;
- `MED_008` receives a too-high threshold and a shorter shelf life;
- some products have missing `batch_size` or `moq`;
- `critical_expiry_days` simulates the imminent expiry of the most urgent batch.

These anomalies are not bugs. They verify that the quality gate, alerts and replenishment rules react correctly.

### 5.4 Isolation from a future real connector

The generator produces a CSV. Ingestion reads it and writes to PostgreSQL. The rest of the pipeline does not know where the data came from.

This boundary is intentional. A real connector (ERP, data warehouse, supplier API) must produce the same column contract and write to the same staging table. Validation, feature engineering and serving modules will not need to change.

The coupling point is the `raw_pharmaceutical_demand` schema, not the CSV file. That is what makes the Plug & Play boundary realistic.

### 5.5 Limits of synthetic data

Synthetic data does not replace business validation. It lacks promotions, exceptional supplier disruptions, product returns and catalogue changes.

The end date is anchored to `datetime.now()`. Two generations on different dates therefore produce different temporal indexes. The seed fixes random values, not the calendar.

These limits must be explicit in any result presentation. A model trained on synthetic data must not be presented as field-validated.

---

<a id="6-appendices-and-engineering-checklist"></a>
## 6. Appendices and Engineering Checklist

### 6.1 Ingestion and migration commands

```bash
# Direct CSV ingestion
python -m src.ingestion

# Full pipeline (generation, ingestion, validation, training)
python run_pipeline.py

# Apply migrations
alembic upgrade head

# Verify SQL without a live database
alembic upgrade head --sql
```

### 6.2 Inspection queries

```sql
-- Row count
SELECT COUNT(*) FROM raw_pharmaceutical_demand;

-- Duplicates on the business key
SELECT store_id, product_id, date, COUNT(*)
FROM raw_pharmaceutical_demand
GROUP BY store_id, product_id, date
HAVING COUNT(*) > 1;

-- Time range
SELECT MIN(date), MAX(date) FROM raw_pharmaceutical_demand;

-- Constraints and indexes
SELECT conname, contype
FROM pg_constraint
WHERE conrelid = 'raw_pharmaceutical_demand'::regclass;

-- Negative or invalid values
SELECT COUNT(*) FROM raw_pharmaceutical_demand WHERE stock_on_hand < 0;
SELECT COUNT(*) FROM raw_pharmaceutical_demand WHERE target_demand < 0;
SELECT COUNT(*) FROM raw_pharmaceutical_demand WHERE moq < 1;
```

### 6.3 Data validation checklist

- [ ] Row count matches `days × products × pharmacies`.
- [ ] The `(store_id, product_id, date)` key contains no duplicates.
- [ ] `store_id`, `product_id` and `date` are not null.
- [ ] The date range is continuous and consistent.
- [ ] `target_demand` and `sales_volume` are non-negative.
- [ ] `stock_on_hand` is non-negative.
- [ ] `supplier_lead_time` is strictly positive.
- [ ] `safety_threshold` is strictly positive.
- [ ] `batch_size` and `moq` are greater than or equal to 1.
- [ ] `shelf_life_days` is greater than or equal to 1.
- [ ] `critical_expiry_days` is non-negative.
- [ ] The `uq_pharma_store_product_date` unique index exists.
- [ ] The `idx_pharma_store_product_date` index exists.
- [ ] The Alembic migration applies without error.

### 6.4 Glossary

| Term | Definition |
|---|---|
| **Business key** | Set of columns that stably identifies an observation |
| **UPSERT** | Insert or update based on a key, through `ON CONFLICT` |
| **Idempotence** | Property of an operation that can be replayed without changing the final state |
| **Connection pool** | Reusable connection reserve that avoids repeated openings |
| **Migration** | Versioned and reversible schema change |
| **Staging** | Intermediate table receiving raw data before transformation |
| **FEFO** | *First Expired, First Out*: prioritize batches closest to expiry |
| **MOQ** | *Minimum Order Quantity*: minimum order quantity |

### 6.5 Code files to read

- `src/data/generate_data.py`: synthetic generation, seasonality and anomalies.
- `src/ingestion.py`: normalization, type alignment and UPSERT.
- `src/repositories/database.py`: engine, pool, cursors and health check.
- `src/repositories/inventory.py`: read-only serving queries.
- `sql/schema.sql`: initial DDL, business key and index.
- `alembic/env.py`: URL resolution and migration execution.
- `alembic/versions/0001_initial_schema.py`: initial migration.
- `tests/test_ingestion.py`: unit tests for normalization and ingestion.
- `tests/test_ingestion_integration.py`: Testcontainers test on real PostgreSQL.

</details>
