"""
===============================================================================
PharmaSupply-ML: Thin-Client Interactive Decision Dashboard (Compact UI)
===============================================================================
"""

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Configure Streamlit page layout and metadata headers
st.set_page_config(
    page_title="PharmaSupply - Pilotage & Prévisions IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING (Soft Slate Gray & Compact Layout)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp, .main, p, span, label, div, h1, h2, h3, h4, h5, h6 {
            color: #334155 !important;
        }
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 95% !important;
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
            font-weight: 600;
            margin-bottom: 2px;
        }
        .card-small-value {
            font-size: 1.1rem;
            color: #334155;
            font-weight: 700;
        }
        div[data-baseweb="select"] span {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

API_BASE_URL = os.getenv(
    "API_BASE_URL", "http://localhost:8000"
)


# -----------------------------------------------------------------------------
# API UTILITY FUNCTIONS WITH CACHING & MOCK FALLBACK
# -----------------------------------------------------------------------------
def _read_local_model_status() -> str:
    """Read local model_status.txt for mock/standalone testing when API is offline."""
    candidates = [
        Path("model_status.txt"),
        Path(__file__).resolve().parent.parent
        / "model_status.txt",
    ]
    for p in candidates:
        if p.exists():
            try:
                content = p.read_text(
                    encoding="utf-8"
                ).strip()
                if content in [
                    "STABLE",
                    "WARNING",
                    "CRITICAL",
                ]:
                    return content
                elif content == "OK":
                    return "STABLE"
            except Exception:
                pass
    return "STABLE"


def _build_model_health_payload(status_val: str) -> dict:
    messages = {
        "STABLE": "🟢 STABLE — Le modèle de prévision est fiable",
        "WARNING": (
            "🟠 WARNING — Performance du modèle sous surveillance : nous vous"
            " invitons à considérer les résultats ci-dessous avec une vigilance renforcée. Nos équipes optimisent le système."
        ),
        "CRITICAL": (
            "🔴 CRITICAL — Alerte critique sur le modèle de prévision, nous vous invitons à considérer les résultats ci dessous avec une extrême vigilance, le"
            " support client vous contactera sous 24h."
        ),
    }
    return {
        "status": status_val,
        "message": messages.get(
            status_val, messages["STABLE"]
        ),
    }


@st.cache_data(ttl=5)
def fetch_model_health():
    """Fetch model operational health status from backend API, with local file fallback."""
    try:
        res = requests.get(
            f"{API_BASE_URL}/model/health", timeout=3
        )
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    local_status = _read_local_model_status()
    return _build_model_health_payload(local_status)


@st.cache_data(ttl=30)
def fetch_stores():
    """Fetch available store IDs from backend API."""
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
    """Fetch available products for a specific store from backend API."""
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


@st.cache_data(ttl=5)
def fetch_store_alerts(store_id):
    """Fetch active alerts for a specific store from backend API, with mock fallback."""
    try:
        res = requests.get(
            f"{API_BASE_URL}/stores/{store_id}/alerts",
            timeout=3,
        )
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    fallback_status = _read_local_model_status()
    mock_alerts = []
    if fallback_status in ["WARNING", "CRITICAL"]:
        mock_alerts.append(
            {
                "product_id": "GLOBAL_MODEL",
                "level": 3,
                "category": "DRIFT",
                "severity": "YELLOW",
                "message": (
                    "Instabilité de l'IA détectée (Statut du modèle :"
                    f" {fallback_status}). Rapport Evidently à inspecter."
                ),
                "current_stock": 0.0,
                "critical_expiry_days": None,
                "safety_threshold": 0.0,
            }
        )
    return {
        "store_id": store_id,
        "model_status": fallback_status,
        "alerts": mock_alerts,
    }


def fetch_latest_features(store_id, product_id):
    """Fetch latest feature set for a store and product pair."""
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
# 1. APPLICATION HEADER & UNIFIED CONTROL BANNER (MODEL STATUS & NETWORK SUMMARY)
# -----------------------------------------------------------------------------
st.title(
    "PharmaSupply — Pilotage multi-sites & Prévisions IA pour la gestion des stocks"
)

# Fetch model health status
model_health = fetch_model_health()
status_message = model_health.get(
    "message",
    "🟢 STABLE — Le modèle de prévision est fiable",
)
model_status = model_health.get("status", "STABLE")

# Fetch stores and calculate network anomalies
stores = fetch_stores()
stores_with_issues_count = sum(
    1
    for s in stores
    if len(fetch_store_alerts(s).get("alerts", [])) > 0
)

# Network summary text formatting
if stores_with_issues_count > 0:
    network_summary = f"Bilan réseau : <span style='color:#dc2626; font-weight:700;'><b>{stores_with_issues_count} site(s)</span> présentent au moins une anomalie."
else:
    network_summary = f"<b>Bilan réseau :</b> Tous les <b>{len(stores)} site(s)</b> sont opérationnels."

# Background logic: Soft pastel red if network anomalies exist OR if model is WARNING/CRITICAL
is_alert_condition = (stores_with_issues_count > 0) or (
    model_status in ["WARNING", "CRITICAL"]
)

if is_alert_condition:
    banner_bg = "#fee2e2"  # Soft pastel red
    banner_border = "#fca5a5"
else:
    banner_bg = "#f8fafc"  # Neutral soft gray
    banner_border = "#e2e8f0"

# Unified centered control box combining Model Status (1st line) and Network Summary (2nd line)
st.markdown(
    f"""
    <div style='background-color: {banner_bg}; border: 1px solid {banner_border}; border-radius: 8px; padding: 14px 18px; margin-top: 4px; margin-bottom: 16px; color: #334155; text-align: center; line-height: 1.6;'>
        <div style='font-size: 1.3rem; font-weight: 500; margin-bottom: 4px;'>
            {status_message}
        </div>
        <div style='font-size: 1.5rem;'>
            {network_summary}
        </div>
        <div style='font-size: 0.9rem; font-style: italic; color: #64748b; font-weight: 400; margin-top: 2px;'>
            (afficher le détail de chaque anomalie par sélection ci-dessous)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. SELECTION PANEL (SITE & ANOMALY FILTER)
# -----------------------------------------------------------------------------
sel_col_left, sel_col_right = st.columns(2)
selected_product_id = None

with sel_col_left:
    with st.container(border=True):
        st.markdown("##### Sélection du site")

        store_options_labels = []
        store_map = {}
        for s in stores:
            s_alerts = fetch_store_alerts(s).get(
                "alerts", []
            )
            count = len(s_alerts)
            icon = "🔴" if count > 0 else "🟢"
            label = f"Site {s} — {icon} {count} anomalie(s)"
            store_options_labels.append(label)
            store_map[label] = s

        chosen_store_label = st.selectbox(
            "Choisir le site actif :",
            options=store_options_labels,
            key="store_enriched_selector",
        )
        selected_store_id = store_map[chosen_store_label]

        current_store_data = fetch_store_alerts(
            selected_store_id
        )
        alert_list = current_store_data.get("alerts", [])

with sel_col_right:
    with st.container(border=True):
        st.markdown("##### Anomalies / recherche directe")

        if alert_list:
            alert_options = [
                "-- Sélectionner une anomalie prioritaire --"
            ]
            alert_map = {}
            for alert in alert_list:
                lvl = alert["level"]
                icon = (
                    "🔴"
                    if lvl == 1
                    else ("🟠" if lvl == 2 else "⚠️")
                )
                label = f"{icon} [{alert['category']}] {alert['product_id']} — {alert['message']}"
                alert_options.append(label)
                alert_map[label] = alert["product_id"]

            chosen_alert = st.selectbox(
                "Traiter par exception :",
                options=alert_options,
                key="alert_selector",
            )
            if (
                chosen_alert
                != "-- Sélectionner une anomalie prioritaire --"
            ):
                selected_product_id = alert_map[
                    chosen_alert
                ]
        else:
            products = fetch_products(selected_store_id)
            search_query = st.selectbox(
                "Site stable. Recherche directe :",
                options=["-- Sélectionner un médicament --"]
                + products,
                key="direct_search_selector",
            )
            if (
                search_query
                != "-- Sélectionner un médicament --"
            ):
                selected_product_id = search_query

st.markdown("---")


# -----------------------------------------------------------------------------
# 3. DETAILED PRODUCT SHEET & DECISION CONTROL
# -----------------------------------------------------------------------------
if selected_product_id:
    st.subheader(
        f"Fiche d'identité & pilotage : `{selected_product_id}` (Site {selected_store_id})"
    )

    features = fetch_latest_features(
        selected_store_id, selected_product_id
    )

    override_key = f"override_safety_{selected_product_id}"
    if override_key in st.session_state:
        features["safety_threshold"] = st.session_state[
            override_key
        ]

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

            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        label="Stock physique actuel",
                        value=f"{int(features['stock_on_hand'])} units",
                    )
                with m2:
                    st.metric(
                        label="Demande prévue (7j)",
                        value=f"{int(predicted_demand)} units",
                        delta=-int(predicted_demand),
                        delta_color="inverse",
                    )
                with m3:
                    st.metric(
                        label="Stock projeté (J+7)",
                        value=f"{int(projected_stock)} units",
                    )
                with m4:
                    st.metric(
                        label="Péremption critique (FEFO)",
                        value=f"{int(critical_expiry)} jours",
                        delta="Risque lot urgent"
                        if critical_expiry < 30
                        else "Normal",
                        delta_color="inverse"
                        if critical_expiry < 30
                        else "normal",
                    )

            if critical_expiry < 30:
                st.error(
                    f"**ALERTE PÉREMPTION CRITIQUE (FEFO)** : Le lot urgent expire dans **{critical_expiry} jours** !"
                )

            if reorder_needed:
                st.warning(
                    f"**RÉAPPROVISIONNEMENT REQUIS** : Quantité commandée calculée : **{order_quantity} unités**."
                )
            else:
                st.success(
                    "**STOCK SÉCURISÉ** — Aucun réapprovisionnement immédiat nécessaire."
                )

            st.markdown("---")

            col_chart, col_actions = st.columns([2.5, 1.5])

            with col_chart:
                st.markdown(
                    "##### Historique des ventes & dynamique"
                )
                lag_df = pd.DataFrame(
                    {
                        "Période": [
                            "J-30",
                            "J-14",
                            "J-7",
                            "J-1 (Dernier)",
                        ],
                        "Volume de ventes": [
                            features["lag_30"] / 30.0,
                            features["lag_14"] / 14.0,
                            features["lag_7"] / 7.0,
                            features["lag_1"],
                        ],
                    }
                )
                st.bar_chart(lag_df.set_index("Période"))

            with col_actions:
                st.markdown("##### Actions & simulations")
                if st.button(
                    "Recalculer le stock de sécurité optimal",
                    use_container_width=True,
                ):
                    st.session_state[override_key] = (
                        optimal_safety
                    )
                    st.cache_data.clear()  # Flush cache for instant UI refresh
                    st.success(
                        f"Seuil mis à jour à {optimal_safety} unités !"
                    )
                    st.rerun()

                if st.button(
                    "Simuler commande / purge FEFO",
                    use_container_width=True,
                ):
                    st.cache_data.clear()  # Flush cache for instant UI refresh
                    st.info(
                        f"Simulation validée : Commande de {order_quantity} unités enregistrée."
                    )
                    st.rerun()

            st.markdown("##### Paramètres logistiques")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f"""<div class="card-small"><div class="card-small-label">Seuil de sécurité</div><div class="card-small-value">{safety_threshold} units</div></div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="card-small"><div class="card-small-label">Délai fournisseur</div><div class="card-small-value">{lead_time} days</div></div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""<div class="card-small"><div class="card-small-label">Batch size (colisage)</div><div class="card-small-value">{features.get("batch_size", 1)} units</div></div>""",
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""<div class="card-small"><div class="card-small-label">MOQ (min. commande)</div><div class="card-small-value">{features.get("moq", 1)} units</div></div>""",
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
