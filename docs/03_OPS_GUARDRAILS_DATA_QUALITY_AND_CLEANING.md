# Technical Documentation  
## Phase 3: Ops Guardrails, Data Quality & Cleaning  
**Project:** PharmaSupply-ML  
**Framework:** Standard MLOps Blueprint  
**File:** `docs/03_OPS_GUARDRAILS_DATA_QUALITY_AND_CLEANING.md`  
**Core Scripts & Files:** `src/validation.py`  
**Status:** Pending final approval (Living document, subject to iterative updates throughout the project lifecycle)

---

<details>
<summary>🇫🇷 <b>Version Française (Cliquez pour dérouler)</b></summary>

### *Ce document détaille la stratégie de qualité des données, la porte de contrôle autonome (Quality Gate) et le fonctionnement du module de validation et d'auto-correction (`src/validation.py`).*

---

## 1. Contexte & Stratégie des Données Synthétiques

Dans le cadre du projet **PharmaSupply-ML**, les données de demande pharmaceutique modélisent des scénarios de chaîne logistique réalistes (saisonnalité, ruptures de stock, variabilité des commandes).

* **Philosophie Bi-mode :** Le pipeline de validation n'est pas seulement un filtre binaire bloquant ; il combine des contrôles structurels stricts et de l'auto-correction intelligente pour garantir la continuité opérationnelle.
* **Préparation Logistique (FEFO) :** Intégration native des paramètres de traçabilité et de péremption pour alimenter les logiques de gestion d'inventaire en aval (First Expired, First Out).

---

## 2. Garde-fous Opérationnels (Ops Guardrails) & Auto-Correction

Le module `src/validation.py` agit comme une **porte de contrôle (Quality Gate)** autonome structurée en quatre étapes majeures :

1. **Contrôles structurels critiques (Blocage) :**
   * Vérification de la non-vacuité de la table `raw_pharmaceutical_demand`.
   * Validation de la présence de l'ensemble des colonnes obligatoires (`date`, `store_id`, `product_id`, `target_demand`, `sales_volume`, `stock_on_hand`, `supplier_lead_time`, `safety_threshold`). Une absence déclenche une erreur critique `ValueError`.
2. **Complétude et Imputation Intelligente :**
   * Détection des valeurs manquantes (`NaN`/`NULL`) sur les colonnes requises avec imputation de repli automatique (0 pour les variables numériques, `"UNKNOWN"` pour les catégorielles).
3. **Champs logistiques & FEFO optionnels :**
   * Gestion des fallbacks pour les attributs de dimensionnement et de péremption s'ils sont absents (`batch_size = 1`, `moq = 1`, `shelf_life_days = 365`, `critical_expiry_days` aligné sur la durée de conservation).
4. **Auto-Correction Métier (Soft Fixes) :**
   * Correction automatique des valeurs aberrantes ou négatives sur la demande (`target_demand`), les volumes de vente (`sales_volume`) et les stocks (`stock_on_hand`) en les ramenant à zéro.
   * Normalisation des délais d'approvisionnement (`supplier_lead_time` par défaut à 3 jours) et des seuils de sécurité (`safety_threshold` par défaut à 150.0) si non valides.
   * Vérification de la cohérence croisée : les ventes (`sales_volume`) ne peuvent excéder la demande globale (`target_demand`), avec écrêtage automatique en cas d'incohérence.

---

## 3. Comportement du Pipeline

* **Arrêt critique (Hard Fail) :** Déclenché uniquement en cas de corruption structurelle majeure (table vide ou colonnes obligatoires manquantes).
* **Résilience (Soft Auto-Correction) :** Les anomalies mineures sont corrigées dynamiquement avec journalisation de traçabilité textuelle (`print`), permettant de renvoyer un DataFrame propre et prêt pour le Feature Engineering.

---

## 4. Exécution du Pipeline de Qualité

Le script s'exécute directement dans l'environnement virtuel du projet :

```bash
conda activate pharma-supply
python src/validation.py
```

---

## 5. Statut & Alignement

* **État :** Synchronisé avec le code source de la Quality Gate et évoluant itérativement.
* **Prochaine étape :** Passage à la **Phase 4 : ML Pipeline, Preprocessing & Feature Engineering**.

</details>

---

<details>
<summary>🇬🇧 <b>English Version (Click to expand)</b></summary>

### *This document details the data quality strategy, autonomous quality gate, and the execution of the validation and auto-correction module (`src/validation.py`).*

---

## 1. Context & Synthetic Data Strategy

Within the **PharmaSupply-ML** project framework, pharmaceutical demand data models realistic supply chain scenarios (seasonality, stockouts, order variability).

* **Dual-Mode Philosophy:** The validation pipeline acts as more than a rigid binary blocker; it combines strict structural checks with smart auto-correction to ensure operational continuity.
* **Logistics Readiness (FEFO):** Native integration of traceability and expiration parameters to feed downstream inventory management logic (First Expired, First Out).

---

## 2. Operational Guardrails & Auto-Correction

The `src/validation.py` module acts as an autonomous **Quality Gate** structured across four main stages:

1. **Critical Structural Checks (Blocking):**
   * Verifies that the `raw_pharmaceutical_demand` staging table is not empty.
   * Validates the presence of mandatory columns (`date`, `store_id`, `product_id`, `target_demand`, `sales_volume`, `stock_on_hand`, `supplier_lead_time`, `safety_threshold`). Missing mandatory columns raise a critical `ValueError`.
2. **Completeness & Smart Imputation:**
   * Detects missing values (`NaN`/`NULL`) across required fields with automated fallback imputation (0 for numeric features, `"UNKNOWN"` for categorical ones).
3. **Optional Logistical & FEFO Fields:**
   * Handles safe fallbacks for sizing and expiration attributes if absent (`batch_size = 1`, `moq = 1`, `shelf_life_days = 365`, `critical_expiry_days` defaulted to match shelf life).
4. **Business Rule Auto-Correction (Soft Fixes):**
   * Automatically corrects negative or abnormal values for demand (`target_demand`), sales volumes (`sales_volume`), and stock levels (`stock_on_hand`) by clipping them to zero.
   * Normalizes invalid supplier lead times (`supplier_lead_time` defaulted to 3 days) and safety thresholds (`safety_threshold` defaulted to 150.0).
   * Enforces cross-column logic: sales (`sales_volume`) cannot exceed true demand (`target_demand`), adjusting rows dynamically when violations occur.

---

## 3. Pipeline Behavior

* **Hard Fail:** Triggered exclusively upon critical structural corruption (empty table or missing required schema columns).
* **Resilient Soft Auto-Correction:** Minor anomalies are dynamically fixed with clear console logging, returning a pristine DataFrame ready for downstream Feature Engineering.

---

## 4. Quality Pipeline Execution

The script runs directly within the project virtual environment:

```bash
conda activate pharma-supply
python src/validation.py
```

---

## 5. Status & Next Steps

* **Status:** Synchronized with the Quality Gate source code and evolving iteratively.
* **Next Step:** Proceed to **Phase 4: ML Pipeline, Preprocessing & Feature Engineering**.

</details>