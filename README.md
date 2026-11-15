# Predicción de precios de casas con Machine Learning

Aplicación en Python (backend y frontend) que predice el valor de una casa en California usando un modelo Random Forest.

**Autor:** Mateo Sebastián Decimavilla Añarumba
**Asignatura:** Proyectos de Machine Learning - Período 2026-2

## Dataset

California Housing, datos reales del censo de EE.UU. de 1990 (20.640 sectores, 8 variables y el valor mediano de la casa).

**Fuente:** Pace, R. K. y Barry, R. (1997). *Sparse Spatial Autoregressions*. Statistics & Probability Letters, 33(3), 291-297. Se obtiene con `sklearn.datasets.fetch_california_housing`.

## Tecnologías

- Python, pandas y scikit-learn (modelo RandomForestRegressor)
- FastAPI y Uvicorn (backend / API)
- Streamlit (interfaz gráfica)
- Git y GitHub

## Estructura

```
backend/   descargar_datos.py, entrenar.py, api.py
frontend/  app.py
data/      california_housing.csv
models/    modelo entrenado (no se sube a GitHub)
```

## Cómo ejecutarlo

1. Crear y activar el entorno virtual, e instalar las librerías:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Descargar el dataset y entrenar el modelo inicial:

```
python backend/descargar_datos.py
python backend/entrenar.py
```

3. Encender la API (terminal 1):

```
uvicorn backend.api:app --reload
```

4. Abrir la interfaz (terminal 2, con el entorno activado):

```
streamlit run frontend/app.py
```

## Funcionalidades

- Entrenar el modelo desde la interfaz.
- Configurar parámetros (número de árboles, profundidad y porcentaje de prueba).
- Mostrar métricas: MAE, RMSE y R2.
- Predecir el precio de una casa nueva.
- API REST con documentación en `http://127.0.0.1:8000/docs`.

## Resultados

Con 100 árboles y profundidad 10 el modelo obtuvo R2 = 0,77 y MAE = 0,37 (unos 36.600 USD de error promedio).  
