# -*- coding: utf-8 -*-
"""
app.py
======
App de Streamlit que expone 3 modelos de Machine Learning entrenados sobre
el histórico de proveedores sancionados (OSCE/Tribunal de Contrataciones):

1. Estimación del monto de la multa (S/)          -> Regresión (log1p)
2. Clasificación del tipo/código de infracción      -> Clasificación
3. Estimación de la duración de la inhabilitación   -> Regresión (días)

Cada predicción se registra en Supabase (Postgres) para auditoría/telemetría,
siguiendo la arquitectura desacoplada de la guía de despliegue.
"""

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from supabase import Client, create_client

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Predicción de Sanciones a Proveedores del Estado",
    page_icon="⚖️",
    layout="centered",
)


# ---------------------------------------------------------------------------
# 1. Conexión segura a Supabase
# ---------------------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


try:
    supabase = init_supabase()
    SUPABASE_OK = True
except Exception:
    # Permite correr la app localmente aunque aún no se hayan configurado
    # los secrets (solo se pierde el registro/auditoría en la nube).
    supabase = None
    SUPABASE_OK = False


# ---------------------------------------------------------------------------
# 2. Carga de los 3 modelos y catálogo de infracciones
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    modelo_monto = joblib.load(MODELS_DIR / "modelo_monto.pkl")
    modelo_infraccion = joblib.load(MODELS_DIR / "modelo_infraccion.pkl")
    modelo_duracion = joblib.load(MODELS_DIR / "modelo_duracion.pkl")
    with open(MODELS_DIR / "catalogo_infracciones.json", "r", encoding="utf-8") as f:
        catalogo = json.load(f)
    return modelo_monto, modelo_infraccion, modelo_duracion, catalogo


modelo_monto, modelo_infraccion, modelo_duracion, catalogo = load_models()

CODIGOS_INFRACCION = sorted(list(catalogo.keys()))
TIPOS_PROVEEDOR = ["PERSONA_NATURAL", "PERSONA_JURIDICA", "OTRO"]


def registrar_en_supabase(tipo_prediccion: str, inputs: dict, resultado):
    """Inserta el registro de auditoría en la tabla predicciones_log."""
    if not SUPABASE_OK:
        st.caption("⚠️ Supabase no configurado: la predicción no se registró en la nube.")
        return
    payload = {
        "tipo_prediccion": tipo_prediccion,
        "inputs_usuario": inputs,
        "resultado_prediccion": str(resultado),
    }
    try:
        supabase.table("predicciones_log").insert(payload).execute()
        st.caption("✓ Consulta registrada de manera segura en Supabase.")
    except Exception as e:
        st.error(f"Error al persistir datos en Supabase: {e}")


# ---------------------------------------------------------------------------
# 3. Interfaz
# ---------------------------------------------------------------------------
st.title("⚖️ Predicción de Sanciones a Proveedores del Estado 🇵🇪")
st.caption(
    "Modelos entrenados sobre el histórico de proveedores sancionados "
    "(inhabilitados) ante el OSCE / Tribunal de Contrataciones del Estado."
)

tab1, tab2, tab3 = st.tabs(
    [
        "💰 Monto de la multa",
        "🏷️ Tipo de infracción",
        "📅 Duración de la sanción",
    ]
)

# --- TAB 1: Estimación del monto -------------------------------------------
with tab1:
    st.subheader("Estimación del monto económico de la multa")
    st.write("Predice el monto (S/) según la infracción cometida y el tiempo de inhabilitación.")

    col1, col2 = st.columns(2)
    with col1:
        cod_infr_1 = st.selectbox(
            "Código de infracción:",
            CODIGOS_INFRACCION,
            format_func=lambda c: f"{c} - {catalogo.get(c, '')[:45]}...",
            key="cod_infr_1",
        )
    with col2:
        tipo_prov_1 = st.selectbox("Tipo de proveedor:", TIPOS_PROVEEDOR, key="tipo_prov_1")

    duracion_1 = st.number_input(
        "Tiempo de inhabilitación (días):", min_value=1, max_value=3650, value=120, step=1
    )

    if st.button("Estimar monto de la multa", type="primary"):
        X_new = pd.DataFrame(
            [{"COD_INFRACCION": cod_infr_1, "TIPO_PROVEEDOR": tipo_prov_1, "DURACION_DIAS": duracion_1}]
        )
        pred_log = modelo_monto.predict(X_new)[0]
        pred_monto = float(np.expm1(pred_log))
        st.success(f"💰 Monto estimado de la multa: **S/ {pred_monto:,.2f}**")

        registrar_en_supabase(
            "estimacion_monto",
            {"cod_infraccion": cod_infr_1, "tipo_proveedor": tipo_prov_1, "duracion_dias": duracion_1},
            round(pred_monto, 2),
        )

# --- TAB 2: Clasificación del tipo de infracción ----------------------------
with tab2:
    st.subheader("Clasificación del tipo / gravedad de infracción")
    st.write("Predice el código de infracción más probable según el perfil/historial del proveedor.")

    col1, col2 = st.columns(2)
    with col1:
        tipo_prov_2 = st.selectbox("Tipo de proveedor:", TIPOS_PROVEEDOR, key="tipo_prov_2")
        n_previas = st.number_input(
            "N° de sanciones previas del proveedor:", min_value=0, max_value=50, value=0, step=1
        )
    with col2:
        monto_prom_previo = st.number_input(
            "Monto promedio de sanciones previas (S/):", min_value=0.0, value=24750.0, step=100.0
        )
        fecha_ref = st.date_input("Fecha de referencia:", value=date.today())

    if st.button("Predecir tipo de infracción", type="primary"):
        X_new = pd.DataFrame(
            [
                {
                    "TIPO_PROVEEDOR": tipo_prov_2,
                    "MONTO_PROMEDIO_PREVIO": monto_prom_previo,
                    "N_SANCIONES_PREVIAS": n_previas,
                    "MES_INICIO": fecha_ref.month,
                    "ANIO_INICIO": fecha_ref.year,
                }
            ]
        )
        pred_cod = modelo_infraccion.predict(X_new)[0]
        proba = modelo_infraccion.predict_proba(X_new)[0]
        clases = modelo_infraccion.named_steps["clf"].classes_
        confianza = float(proba[list(clases).index(pred_cod)])

        descripcion = catalogo.get(pred_cod, "Categoría agrupada (infracciones poco frecuentes)")
        st.success(f"🏷️ Código de infracción más probable: **{pred_cod}**")
        st.info(f"📄 Descripción: {descripcion}")
        st.caption(f"Confianza del modelo: {confianza:.1%}")

        registrar_en_supabase(
            "clasificacion_infraccion",
            {
                "tipo_proveedor": tipo_prov_2,
                "monto_promedio_previo": monto_prom_previo,
                "n_sanciones_previas": n_previas,
                "mes": fecha_ref.month,
                "anio": fecha_ref.year,
            },
            pred_cod,
        )

# --- TAB 3: Duración de la sanción ------------------------------------------
with tab3:
    st.subheader("Estimación del tiempo de sanción (días)")
    st.write("Predice cuántos días durará la inhabilitación según la infracción y el monto de la multa.")

    col1, col2 = st.columns(2)
    with col1:
        cod_infr_3 = st.selectbox(
            "Código de infracción:",
            CODIGOS_INFRACCION,
            format_func=lambda c: f"{c} - {catalogo.get(c, '')[:45]}...",
            key="cod_infr_3",
        )
    with col2:
        tipo_prov_3 = st.selectbox("Tipo de proveedor:", TIPOS_PROVEEDOR, key="tipo_prov_3")

    monto_3 = st.number_input("Monto de la multa (S/):", min_value=0.0, value=24750.0, step=100.0)

    if st.button("Estimar duración de la sanción", type="primary"):
        X_new = pd.DataFrame(
            [{"COD_INFRACCION": cod_infr_3, "TIPO_PROVEEDOR": tipo_prov_3, "MONTO_NUM": monto_3}]
        )
        pred_dias = float(modelo_duracion.predict(X_new)[0])
        st.success(f"📅 Duración estimada de la inhabilitación: **{pred_dias:,.0f} días** (~{pred_dias/30:.1f} meses)")

        registrar_en_supabase(
            "estimacion_duracion",
            {"cod_infraccion": cod_infr_3, "tipo_proveedor": tipo_prov_3, "monto": monto_3},
            round(pred_dias, 1),
        )

st.divider()
with st.expander("ℹ️ Sobre los modelos"):
    st.write(
        """
        - **Monto de la multa**: `RandomForestRegressor` entrenado sobre `log1p(MONTO)`
          (el monto tiene una distribución muy sesgada por outliers de hasta S/20M).
        - **Tipo de infracción**: `RandomForestClassifier` balanceado por clases; los
          códigos con menos de 15 observaciones se agruparon como `OTROS`.
        - **Duración de la sanción**: `RandomForestRegressor` sobre la diferencia en
          días entre `FECHA_INICIO` y `FECHA_FIN`.

        Ver `models/metricas.json` para el detalle de desempeño de cada modelo.
        """
    )
