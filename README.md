# Predicción de Sanciones a Proveedores del Estado

App de Streamlit con 3 modelos de Machine Learning entrenados sobre
`sancionados_multa_.csv`, integrada con Supabase para auditoría, lista para
desplegar en Streamlit Community Cloud según la Guía de Implementación.

## ¿Qué predice cada modelo?

| Modelo | Archivo | Tipo | Predice | Basado en |
|---|---|---|---|---|
| 1 | `models/modelo_monto.pkl` | Regresión | Monto de la multa (S/) | Código de infracción + tiempo de inhabilitación |
| 2 | `models/modelo_infraccion.pkl` | Clasificación | Código de infracción (`ID_MOTIVO_INFRACCION`) | Tipo de proveedor + historial (n° de sanciones previas, monto promedio previo) + fecha |
| 3 | `models/modelo_duracion.pkl` | Regresión | Duración de la inhabilitación (días) | Código de infracción + tipo de proveedor + monto |

Métricas de desempeño de cada uno están en `models/metricas.json` (se
regeneran cada vez que corres `train_models.py`).

**Nota sobre los datos:** el dataset está muy desbalanceado — el código de
infracción `237` concentra ~95% de los casos, y el monto tiene outliers de
hasta S/20,000,000 frente a una mediana de ~S/24,750. Por eso:
- El modelo de monto se entrena en escala `log1p` y se revierte con `expm1`.
- El modelo de clasificación agrupa códigos con <15 observaciones en `OTROS`
  y usa `class_weight="balanced"`.

## Estructura del proyecto

```
proyecto-ml/
├── .streamlit/
│   └── secrets.toml.example   <- Plantilla (copiar como secrets.toml, NO subir el real)
├── data/
│   └── sancionados_multa.csv  <- Dataset fuente (UTF-8, separador '|')
├── models/
│   ├── modelo_monto.pkl
│   ├── modelo_infraccion.pkl
│   ├── modelo_duracion.pkl
│   ├── catalogo_infracciones.json
│   └── metricas.json
├── app.py                     <- App principal de Streamlit (UI + Backend)
├── train_models.py            <- Script de entrenamiento (regenera los .pkl)
├── requirements.txt
├── supabase_setup.sql         <- SQL para crear la tabla de auditoría
├── .gitignore
└── README.md
```
