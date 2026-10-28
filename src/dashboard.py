"""
===============================================================================
PharmaSupply-ML: Interactive Streamlit Decision Dashboard (Vertical Flow)
===============================================================================

OBJECTIVE & USAGE:
------------------
This script provides a professional, interactive MLOps decision-making interface
following a top-to-bottom vertical flow architecture:
1. Top Bar (En-tête compact): Global store selector and free product search/inspection.
2. Control Center (Centre de Contrôle): Interactive alerts queue fetched from the
   backend, strictly sorted by operational criticality (Level 1: Paramétrage,
   Level 2: Rupture & Péremption FEFO, Level 3: Dérive du modèle / Drift).
3. Full-Width Detail View (Fiche détaillée): Complete operational inspection, sales history
   visualization, and pragmatic action buttons for simulation (safety stock recalculation,
   FEFO batch processing).

Author: YCS-TLSE
Project: PharmaSupply-ML
"""

import os

import pandas as pd
import requests
import streamlit as st

# Configure Streamlit page layout and metadata headers
st.set_page_config(
    page_title="PharmaSupply - Centre de Contrôle & Réapprovisionnement",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",  # Flux vertical en pleine largeur sans sidebar intrusive
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING (Harmonized vertical layout & severity styling)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 98% !important;
        }
        header[data-testid="stHeader"], footer {
            display: none !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff;
            border: 1px solid #cbd5e1 !important;
            border-left: 5px solid #2563eb !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            color: #0f172a !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: #475569 !important;
        }
        .card-small {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 8px;
        }
        .card-small-label {
            font-size: 0.8rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 600;
            margin-bottom: 2px;
        }
        .card-small-value {
            font-size: 1.1rem;
            color: #1e293b;
            font-weight: 700;
        }
    </style>
""",
    unsafe_allow_html=True,
)

API_BASE_URL = os.getenv(
    "API_BASE_URL", "http://localhost:8000"
)


# -----------------------------------------------------------------------------
# API UTILITY FUNCTIONS WITH CACHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_stores():
    """Fetches available pharmacy store identifiers from the API[cite: 13]."""
    try:
        res = requests.get(
            f"{API_BASE_URL}/stores", timeout=5
        )
        if res.status_code == 200:
            return res.json().get("stores", ["1"])
    except Exception:
        pass
    return ["1"]


@st.cache_data(ttl=30)
def fetch_products(store_id):
    """Fetches available product identifiers for a given store[cite: 13]."""
    try:
        res = requests.get(
            f"{API_BASE_URL}/stores/{store_id}/products",
            timeout=5,
        )
        if res.status_code == 200:
            return res.json().get("products", ["MED_001"])
    except Exception:
        pass
    return ["MED_001"]


@st.cache_data(ttl=15)
def fetch_store_alerts(store_id):
    """Fetches prioritized operational alerts from the backend audit endpoint."""
    try:
        res = requests.get(
            f"{API_BASE_URL}/stores/{store_id}/alerts",
            timeout=5,
        )
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {
        "store_id": store_id,
        "model_status": "STABLE",
        "alerts": [],
    }


def fetch_latest_features(store_id, product_id):
    """Auto-fetches latest operational features, lags, and constraints for inference[cite: 13]."""
    try:
        res = requests.get(
            f"{API_BASE_URL}/features/{store_id}/{product_id}",
            timeout=5,
        )
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {
        "stock_on_hand": 100.0,
        "supplier_lead_time": 3.0,
        "safety_threshold": 150.0,
        "batch_size": 1,
        "moq": 1,
        "shelf_life_days": 365,
        "critical_expiry_days": 365,
        "lag_1": 20.0,
        "lag_7": 140.0,
        "lag_14": 280.0,
        "lag_30": 600.0,
    }


# -----------------------------------------------------------------------------
# 1. TOP BAR (EN-TÊTE COMPACT - FLUX VERTICAL)
# -----------------------------------------------------------------------------
st.title(
    "🏥 PharmaSupply - Centre de Contrôle & Réapprovisionnement"
)

top_col1, top_col2, top_col3 = st.columns([1.5, 2.5, 2.0])

with top_col1:
    stores = fetch_stores()
    selected_store_id = st.selectbox(
        "🌐 Magasin Actif", options=stores, index=0
    )

with top_col2:
    products = fetch_products(selected_store_id)
    # Recherche libre / sélecteur direct de médicament pour inspection hors alertes
    search_query = st.selectbox(
        "🔍 Inspection Directe (Recherche / Produit)",
        options=[
            "-- Sélectionner depuis les alertes ou recherche libre --"
        ]
        + products,
        index=0,
    )

with top_col3:
    # Récupération des alertes globales du magasin et statut du modèle
    alerts_data = fetch_store_alerts(selected_store_id)
    model_status = alerts_data.get("model_status", "STABLE")
    alert_list = alerts_data.get("alerts", [])

    status_color = (
        "🟢"
        if model_status == "STABLE"
        else ("🟠" if model_status == "WARNING" else "🔴")
    )
    st.markdown(
        f"**État Global IA / Drift** : {status_color} `{model_status}`"
    )
    st.markdown(
        f"**Anomalies Actives** : `{len(alert_list)} signalements`"
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 2. CENTRE DE CONTRÔLE (FILE D'ATTENTE DES ANOMALIES - GESTION PAR EXCEPTION)
# -----------------------------------------------------------------------------
st.subheader(
    "🚨 Centre de Contrôle : File d'Attente des Anomalies Prioritaires"
)

selected_product_id = None

if alert_list:
    # Construction d'options lisibles pour la file d'attente triée par criticité
    alert_options = []
    alert_map = {}

    for alert in alert_list:
        lvl = alert["level"]
        icon = (
            "🔴"
            if lvl == 1
            else ("🟠" if lvl == 2 else "⚠️")
        )
        label = f"{icon} [{alert['category']}] Produit : {alert['product_id']} — {alert['message']}"
        alert_options.append(label)
        alert_map[label] = alert["product_id"]

    selected_alert_label = st.selectbox(
        "Sélectionner une anomalie prioritaire à traiter :",
        options=["-- Choisir dans la file d'attente --"]
        + alert_options,
    )

    if (
        selected_alert_label
        != "-- Choisir dans la file d'attente --"
    ):
        selected_product_id = alert_map[
            selected_alert_label
        ]
else:
    st.success(
        "✅ Aucune anomalie critique détectée dans ce magasin. Tous les voyants sont au vert !"
    )

# Si l'utilisateur utilise la recherche libre dans la Top Bar, cela prend le pas
if (
    search_query
    != "-- Sélectionner depuis les alertes ou recherche libre --"
):
    selected_product_id = search_query

# Si aucun produit n'est sélectionné, on prend par défaut le premier produit disponible
if not selected_product_id and products:
    selected_product_id = products[0]

st.markdown("---")

# -----------------------------------------------------------------------------
# 3. FICHE DÉTAILLÉE EN PLEINE LARGEUR (FULL-WIDTH DETAIL VIEW)
# -----------------------------------------------------------------------------
if selected_product_id:
    st.subheader(
        f"📋 Fiche Détaillée & Pilotage : `{selected_product_id}` (Magasin {selected_store_id})"
    )

    # Récupération des caractéristiques et état du produit sélectionné
    features = fetch_latest_features(
        selected_store_id, selected_product_id
    )

    # Gestion de session pour override du safety threshold si simulé
    override_key = f"override_safety_{selected_product_id}"
    if override_key in st.session_state:
        features["safety_threshold"] = st.session_state[
            override_key
        ]

    # Construction du payload pour l'API /predict
    payload = {
        "store_id": selected_store_id,
        "product_id": selected_product_id,
        "stock_on_hand": features["stock_on_hand"],
        "supplier_lead_time": features[
            "supplier_lead_time"
        ],
        "safety_threshold": features["safety_threshold"],
        "batch_size": features.get("batch_size", 1),
        "moq": features.get("moq", 1),
        "shelf_life_days": features.get(
            "shelf_life_days", 365
        ),
        "lag_1": features["lag_1"],
        "lag_7": features["lag_7"],
        "lag_14": features["lag_14"],
        "lag_30": features["lag_30"],
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=5,
        )
        if response.status_code == 200:
            result = response.json()
            predicted_demand = float(
                result.get("predicted_demand", 0)
            )
            projected_stock = float(
                result.get("projected_stock", 0)
            )
            reorder_needed = result.get(
                "reorder_needed", False
            )
            order_quantity = result.get("order_quantity", 0)

            daily_demand = (
                predicted_demand / 7.0
                if predicted_demand > 0
                else 1.0
            )
            lead_time = float(
                features["supplier_lead_time"]
            )
            safety_threshold = float(
                features["safety_threshold"]
            )
            ddlt = daily_demand * lead_time
            optimal_safety = round(ddlt * 2.0, 0)
            critical_expiry = int(
                features.get("critical_expiry_days", 365)
            )

            # --- METRICS CONTAINER EN PLEINE LARGEUR ---
            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        label="Stock Physique Actuel",
                        value=f"{int(features['stock_on_hand'])} units",
                    )
                with m2:
                    st.metric(
                        label="Demande Prévue (7j)",
                        value=f"{int(predicted_demand)} units",
                        delta=-int(predicted_demand),
                        delta_color="inverse",
                    )
                with m3:
                    st.metric(
                        label="Stock Projeté (J+7)",
                        value=f"{int(projected_stock)} units",
                    )
                with m4:
                    st.metric(
                        label="Péremption Critique (FEFO)",
                        value=f"{critical_expiry} jours",
                        delta="Risque lot urgent"
                        if critical_expiry < 30
                        else "Normal",
                        delta_color="inverse"
                        if critical_expiry < 30
                        else "normal",
                    )

            # --- BANNES D'ACTION ET RECOMMANDATION MÉTIER ---
            if critical_expiry < 30:
                st.error(
                    f"🟠 **ALERTE PÉREMPTION CRITIQUE (FEFO)**  \n"
                    f"Le lot le plus urgent expire dans **{critical_expiry} jours** ! Écoulement prioritaire requis."
                )

            if reorder_needed:
                st.warning(
                    f"⚠️ **RÉAPPROVISIONNEMENT REQUIS**  \n"
                    f"Quantité commandée calculée (en tenant compte du MOQ et du Batch Size) : **{order_quantity} unités**."
                )
            else:
                st.success(
                    "✅ **STOCK SÉCURISÉ** — Aucun réapprovisionnement immédiat nécessaire."
                )

            st.markdown("---")

            # --- SECTION GRAPHIQUE D'HISTORIQUE DE VENTES ET SIMULATIONS PRAGMATIQUES ---
            col_chart, col_actions = st.columns([2.5, 1.5])

            with col_chart:
                st.markdown(
                    "##### 📈 Historique des Ventes (Lags & Dynamique)"
                )
                lag_df = pd.DataFrame(
                    {
                        "Période": [
                            "J-30",
                            "J-14",
                            "J-7",
                            "J-1 (Dernier)",
                        ],
                        "Volume de Ventes": [
                            features["lag_30"] / 30.0,
                            features["lag_14"] / 14.0,
                            features["lag_7"] / 7.0,
                            features["lag_1"],
                        ],
                    }
                )
                st.bar_chart(lag_df.set_index("Période"))

            with col_actions:
                st.markdown(
                    "##### ⚙️ Actions & Simulations Métier"
                )

                # Bouton 1 : Recalcul automatique du stock de sécurité
                if st.button(
                    "🔄 Recalculer le stock de sécurité optimal",
                    use_container_width=True,
                ):
                    st.session_state[override_key] = (
                        optimal_safety
                    )
                    st.success(
                        f"Seuil mis à jour à {optimal_safety} unités !"
                    )
                    st.rerun()

                # Bouton 2 : Simulation de commande / écoulement prioritaire
                if st.button(
                    "📦 Simuler Commande / Purge FEFO",
                    use_container_width=True,
                ):
                    st.info(
                        f"Simulation validée : Commande de {order_quantity} unités enregistrée et rotation des lots simulée."
                    )

            # --- TABLEAU DES PARAMÈTRES LOGISTIQUES ET DE RÉFÉRENCE ---
            st.markdown(
                "##### 📊 Paramètres Logistiques & Contraintes"
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f"""<div class="card-small"><div class="card-small-label">Seuil de Sécurité</div><div class="card-small-value">{safety_threshold} units</div></div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="card-small"><div class="card-small-label">Délai Fournisseur</div><div class="card-small-value">{lead_time} days</div></div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""<div class="card-small"><div class="card-small-label">Batch Size (Colisage)</div><div class="card-small-value">{features.get("batch_size", 1)} units</div></div>""",
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""<div class="card-small"><div class="card-small-label">MOQ (Min. Commande)</div><div class="card-small-value">{features.get("moq", 1)} units</div></div>""",
                    unsafe_allow_html=True,
                )

        else:
            st.error(
                f"Erreur API ({response.status_code}) : {response.text}"
            )
    except Exception as e:
        st.error(
            f"Impossible de joindre l'API backend : {e}"
        )
