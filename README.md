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

## Pasos para integrar (siguiendo la guía)

### 1. Supabase
1. Crear proyecto en [supabase.com](https://supabase.com) con tu cuenta de GitHub.
2. En **SQL Editor**, ejecutar el contenido de `supabase_setup.sql`.
3. En **Project Settings -> API**, copiar la `Project URL` y la `anon public API Key`.

### 2. Desarrollo local (opcional, para probar antes de desplegar)
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Editar .streamlit/secrets.toml con tus credenciales reales de Supabase
streamlit run app.py
```
Si quieres reentrenar los modelos con datos actualizados:
```bash
python train_models.py
```

### 3. Subir a GitHub
```bash
git init
git add .
git commit -m "App ML: predicción de sanciones a proveedores"
git branch -M main
git remote add origin <URL_DE_TU_REPO>
git push -u origin main
```
`secrets.toml` **no** se sube porque está en `.gitignore` (verifica antes del push).
Si algún `.pkl` supera 25 MB, usa Git LFS (`git lfs track "*.pkl"`); en este
proyecto el más pesado (`modelo_infraccion.pkl`) pesa ~20 MB, así que no es
obligatorio, pero puedes activarlo igual por buena práctica.

### 4. Desplegar en Streamlit Community Cloud
1. Ir a [share.streamlit.io](https://share.streamlit.io) e iniciar sesión con GitHub.
2. **Create app** -> seleccionar el repositorio, la rama `main` y `app.py` como archivo principal.
3. Antes de desplegar, abrir **Advanced settings... -> Secrets** y pegar:
   ```toml
   SUPABASE_URL = "https://tu-id-proyecto.supabase.co"
   SUPABASE_KEY = "tu-clave-anon-public-jwt"
   ```
4. Clic en **Deploy**. Al usar cualquiera de las 3 pestañas de la app y
   presionar el botón de predicción, el resultado se guarda automáticamente
   en la tabla `predicciones_log` de Supabase.

## Verificación
- Cada predicción exitosa muestra `✓ Consulta registrada de manera segura en Supabase.`
- Puedes confirmar los registros en Supabase -> Table Editor -> `predicciones_log`.
