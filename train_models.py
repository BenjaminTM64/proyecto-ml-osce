# -*- coding: utf-8 -*-
"""
train_models.py
================
Entrena y serializa los 3 modelos de Machine Learning del proyecto:

1. modelo_monto.pkl        -> Regresión: predice el MONTO (S/) de la multa
2. modelo_infraccion.pkl   -> Clasificación: predice el ID_MOTIVO_INFRACCION
3. modelo_duracion.pkl     -> Regresión: predice la duración de la sanción (días)

Uso:
    python train_models.py

Requiere: data/sancionados_multa.csv (separador '|', encoding latin-1)
Genera los .pkl en la carpeta models/ y un reporte de métricas en
models/metricas.json
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "sancionados_multa.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA DE DATOS
# ---------------------------------------------------------------------------
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep="|", dtype=str, encoding="utf-8")

    # --- Fechas ---
    df["FECHA_INICIO_DT"] = pd.to_datetime(df["FECHA_INICIO"], format="%Y%m%d", errors="coerce")
    df["FECHA_FIN_DT"] = pd.to_datetime(df["FECHA_FIN"], format="%Y%m%d", errors="coerce")

    # --- Duración de la sanción en días (variable objetivo del modelo 3 y
    # feature de los modelos 1 y 2) ---
    df["DURACION_DIAS"] = (df["FECHA_FIN_DT"] - df["FECHA_INICIO_DT"]).dt.days

    # --- Monto como número ---
    df["MONTO_NUM"] = pd.to_numeric(df["MONTO"], errors="coerce")

    # --- Código de infracción "limpio": algunos registros traen varios
    # códigos separados por coma (ej. "237,245,"). Nos quedamos con el
    # primero, que es la infracción principal. ---
    df["COD_INFRACCION"] = (
        df["ID_MOTIVO_INFRACCION"].str.split(",").str[0].str.strip()
    )
    df = df[df["COD_INFRACCION"].notna() & (df["COD_INFRACCION"] != "")]

    # --- Tipo de proveedor a partir de los 2 primeros dígitos del RUC ---
    # 10 = persona natural | 15/17 = persona natural (regímenes especiales)
    # 20 = persona jurídica | 99/otros = no domiciliado / otros
    df["PREFIJO_RUC"] = df["RUC"].str[:2]

    def tipo_ruc(p):
        if p == "20":
            return "PERSONA_JURIDICA"
        if p in ("10", "15", "17"):
            return "PERSONA_NATURAL"
        return "OTRO"

    df["TIPO_PROVEEDOR"] = df["PREFIJO_RUC"].apply(tipo_ruc)

    # --- Features temporales de la fecha de inicio de la sanción ---
    df["ANIO_INICIO"] = df["FECHA_INICIO_DT"].dt.year
    df["MES_INICIO"] = df["FECHA_INICIO_DT"].dt.month
    df["DIA_SEMANA_INICIO"] = df["FECHA_INICIO_DT"].dt.dayofweek

    # --- Historial: número de sanciones previas del mismo RUC (a la fecha
    # de inicio de cada sanción) y monto histórico promedio previo. Esto le
    # da al modelo de clasificación una noción de "perfil / historial" del
    # proveedor, tal como pide el enunciado. ---
    df = df.sort_values("FECHA_INICIO_DT")
    df["N_SANCIONES_PREVIAS"] = df.groupby("RUC").cumcount()
    df["MONTO_PROMEDIO_PREVIO"] = (
        df.groupby("RUC")["MONTO_NUM"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["MONTO_PROMEDIO_PREVIO"] = df["MONTO_PROMEDIO_PREVIO"].fillna(
        df["MONTO_NUM"].median()
    )

    return df


# ---------------------------------------------------------------------------
# 2. MODELO 1: REGRESIÓN DEL MONTO DE LA MULTA
# ---------------------------------------------------------------------------
def entrenar_modelo_monto(df: pd.DataFrame) -> dict:
    cols_cat = ["COD_INFRACCION", "TIPO_PROVEEDOR"]
    cols_num = ["DURACION_DIAS"]

    data = df.dropna(subset=cols_cat + cols_num + ["MONTO_NUM"]).copy()
    data = data[data["DURACION_DIAS"] >= 0]

    # El monto tiene una distribucion muy sesgada (mediana ~S/24,750 pero
    # existen multas de hasta S/20,000,000). Entrenamos en escala
    # logaritmica (log1p) para que esos outliers no dominen el error, y
    # revertimos con expm1 al predecir. Esto es estandar para variables
    # monetarias muy asimetricas.
    X = data[cols_cat + cols_num]
    y = data["MONTO_NUM"]
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )
    y_test = np.expm1(y_test_log)

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
        ],
        remainder="passthrough",
    )

    modelo = Pipeline(
        steps=[
            ("prep", preprocesador),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
    modelo.fit(X_train, y_train_log)
    y_pred_log = modelo.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    metricas = {
        "mae_soles": float(mean_absolute_error(y_test, y_pred)),
        "mae_log": float(mean_absolute_error(y_test_log, y_pred_log)),
        "r2_log": float(r2_score(y_test_log, y_pred_log)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": cols_cat + cols_num,
        "nota": "Target entrenado en escala log1p(MONTO) por alta asimetria; predict_monto() revierte con expm1.",
    }

    joblib.dump(modelo, MODELS_DIR / "modelo_monto.pkl")
    print("[OK] modelo_monto.pkl entrenado ->", metricas)
    return metricas


# ---------------------------------------------------------------------------
# 3. MODELO 2: CLASIFICACIÓN DEL TIPO / GRAVEDAD DE INFRACCIÓN
# ---------------------------------------------------------------------------
def entrenar_modelo_infraccion(df: pd.DataFrame) -> dict:
    cols_cat = ["TIPO_PROVEEDOR"]
    cols_num = ["MONTO_PROMEDIO_PREVIO", "N_SANCIONES_PREVIAS", "MES_INICIO", "ANIO_INICIO"]

    data = df.dropna(subset=cols_cat + cols_num + ["COD_INFRACCION"]).copy()

    # Agrupamos códigos con muy pocas observaciones en "OTROS" para que el
    # modelo pueda aprender patrones razonables (el dataset está muy
    # desbalanceado: el código 237 concentra ~95% de los casos).
    conteo = data["COD_INFRACCION"].value_counts()
    codigos_validos = conteo[conteo >= 15].index
    data["TARGET_INFRACCION"] = data["COD_INFRACCION"].where(
        data["COD_INFRACCION"].isin(codigos_validos), "OTROS"
    )

    X = data[cols_cat + cols_num]
    y = data["TARGET_INFRACCION"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
        ],
        remainder="passthrough",
    )

    modelo = Pipeline(
        steps=[
            ("prep", preprocesador),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "clases": sorted(y.unique().tolist()),
        "features": cols_cat + cols_num,
        "nota": (
            "Dataset desbalanceado: el código 237 concentra la gran mayoría "
            "de los casos. Se agruparon códigos con <15 observaciones en 'OTROS'."
        ),
    }

    joblib.dump(modelo, MODELS_DIR / "modelo_infraccion.pkl")
    print("[OK] modelo_infraccion.pkl entrenado ->", metricas)
    return metricas


# ---------------------------------------------------------------------------
# 4. MODELO 3: REGRESIÓN DE LA DURACIÓN DE LA SANCIÓN (DÍAS)
# ---------------------------------------------------------------------------
def entrenar_modelo_duracion(df: pd.DataFrame) -> dict:
    cols_cat = ["COD_INFRACCION", "TIPO_PROVEEDOR"]
    cols_num = ["MONTO_NUM"]

    data = df.dropna(subset=cols_cat + cols_num + ["DURACION_DIAS"]).copy()
    data = data[(data["DURACION_DIAS"] >= 0) & (data["DURACION_DIAS"] <= 3650)]

    X = data[cols_cat + cols_num]
    y = data["DURACION_DIAS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
        ],
        remainder="passthrough",
    )

    modelo = Pipeline(
        steps=[
            ("prep", preprocesador),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = {
        "mae_dias": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": cols_cat + cols_num,
    }

    joblib.dump(modelo, MODELS_DIR / "modelo_duracion.pkl")
    print("[OK] modelo_duracion.pkl entrenado ->", metricas)
    return metricas


# ---------------------------------------------------------------------------
# 5. TABLA DE REFERENCIA: código -> descripción de infracción (para la UI)
# ---------------------------------------------------------------------------
def exportar_catalogo_infracciones(df: pd.DataFrame):
    catalogo = (
        df.dropna(subset=["COD_INFRACCION", "DE_MOTIVO_INFRACCION"])
        .groupby("COD_INFRACCION")["DE_MOTIVO_INFRACCION"]
        .first()
        .to_dict()
    )
    with open(MODELS_DIR / "catalogo_infracciones.json", "w", encoding="utf-8") as f:
        json.dump(catalogo, f, ensure_ascii=False, indent=2)
    print(f"[OK] catalogo_infracciones.json ({len(catalogo)} códigos)")
    return catalogo


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Cargando y limpiando datos...")
    df = cargar_datos()
    print(f"Registros utilizables: {len(df)}")

    print("\nEntrenando modelo 1/3 (monto de la multa)...")
    m1 = entrenar_modelo_monto(df)

    print("\nEntrenando modelo 2/3 (tipo de infracción)...")
    m2 = entrenar_modelo_infraccion(df)

    print("\nEntrenando modelo 3/3 (duración de la sanción)...")
    m3 = entrenar_modelo_duracion(df)

    exportar_catalogo_infracciones(df)

    reporte = {"modelo_monto": m1, "modelo_infraccion": m2, "modelo_duracion": m3}
    with open(MODELS_DIR / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print("\n✅ Listo. Modelos guardados en la carpeta models/")
