# Technical Documentation  
## Phase 3: Ops Guardrails, Data Quality & Cleaning  
**Project:** PharmaSupply-ML  
**Framework:** Standard MLOps Blueprint  
**File:** `docs/03_OPS_GUARDRAILS_DATA_QUALITY_AND_CLEANING.md`  
**Status:** Validated  

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

### *Ce document détaille la stratégie de qualité des données, les garde-fous opérationnels (Ops Guardrails) et le fonctionnement du module de validation (`src/validation.py`).*

---

## 1. Contexte & Stratégie des Données Synthétiques

Dans le cadre du projet **PharmaSupply-ML**, les données de demande pharmaceutique ont été générées de manière synthétique pour modéliser des scénarios réalistes (saisonnalité, ruptures de stock, variabilité des commandes).

* **Absence de nettoyage lourd (Data Cleaning) :** Les règles de génération ayant été maîtrisées à la source, aucun traitement complexe d'imputation ou de filtrage de bruit n'a été requis.
* **Priorité à la Qualité & aux Guardrails :** L'accent est mis sur la vérification automatique des contraintes de structure et de cohérence métier avant l'injection des données dans le pipeline de Machine Learning.

---

## 2. Garde-fous Opérationnels (Ops Guardrails) & Validation

Le module `src/validation.py` agit comme une **porte de contrôle (Quality Gate)** automatisée. Il effectue trois niveaux d'assertions strictes sur la table `raw_pharmaceutical_demand` :

1. **Test de non-vacuité :**
   * Vérifie que la table contient au moins une ligne enregistrée.
2. **Conformité du schéma et valeurs nulles (NOT NULL) :**
   * S'assure de l'absence totale de valeurs manquantes (`NaN`/`NULL`) sur les colonnes critiques : `date`, `product_id`, `target_demand`, `sales_volume` et `stock_on_hand`.
3. **Assertions de cohérence métier (Plages logiques) :**
   * **`target_demand` :** Doit être supérieur ou égal à 0 (pas de demande négative).
   * **`stock_on_hand` :** Doit être supérieur ou égal à 0 (pas de stock négatif).

---

## 3. Comportement en cas d'Échec

Les vérifications s'appuient sur des instructions `assert` en Python. Si un seul des contrôles échoue :
* Une exception `AssertionError` est levée avec un message explicite.
* L'exécution du script s'arrête immédiatement, empêchant toute propagation de données invalides vers la suite du pipeline.

---

## 4. Exécution du Pipeline de Qualité

Le script de validation s'exécute dans l'environnement virtuel du projet :

~~~~bash
conda activate pharmasupply-ml
python src/validation.py
~~~~

---

## 5. Statut & Alignement

* **État :** Validé. Les données brutes respectent l'ensemble des garde-fous.
* **Prochaine étape :** Passage à la **Phase 4 : ML Pipeline, Preprocessing & Feature Engineering**.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *This document details the data quality strategy, operational guardrails (Ops Guardrails), and the execution of the validation module (`src/validation.py`).*

---

## 1. Context & Synthetic Data Strategy

Within the **PharmaSupply-ML** project framework, pharmaceutical demand data was artificially generated to model realistic supply chain dynamics (seasonality, stockouts, order variability).

* **No Heavy Data Cleaning Required:** Because generation rules were controlled at source, no complex noise removal or missing value imputation was needed.
* **Focus on Quality & Guardrails:** Focus is placed on automated assertion checks covering schema integrity and business rules prior to ingestion by downstream ML pipelines.

---

## 2. Operational Guardrails & Data Validation

The `src/validation.py` module acts as an automated **Quality Gate**, enforcing three levels of strict assertion checks on the `raw_pharmaceutical_demand` table:

1. **Non-Empty Table Check:**
   * Verifies that the target database table contains at least one record.
2. **Schema Compliance & Null Value Check:**
   * Ensures the complete absence of missing values (`NaN`/`NULL`) across critical columns: `date`, `product_id`, `target_demand`, `sales_volume`, and `stock_on_hand`.
3. **Business Logic Assertions (Valid Ranges):**
   * **`target_demand`:** Must be non-negative ($\ge 0$).
   * **`stock_on_hand`:** Must be non-negative ($\ge 0$).

---

## 3. Failure Behavior

Validation logic relies on standard Python `assert` statements. If any check fails:
* An `AssertionError` exception is raised with an explicit diagnostic message.
* Execution terminates immediately, preventing invalid or corrupted data from propagating to downstream machine learning steps.

---

## 4. Quality Pipeline Execution

The validation script is executed directly within the project virtual environment:

~~~~bash
conda activate pharmasupply-ml
python src/validation.py
~~~~

---

## 5. Status & Next Steps

* **Status:** Validated. Raw records strictly adhere to operational guardrails.
* **Next Step:** Proceed to **Phase 4: ML Pipeline, Preprocessing & Feature Engineering**.

</details>
