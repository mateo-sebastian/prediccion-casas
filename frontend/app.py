import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.title("Predicción de precios de casas")
st.caption("Dataset: California Housing (Pace & Barry, 1997)")

# 1. Parámetros y entrenamiento
st.header("1. Entrenar el modelo")
n_estimators = st.slider("Número de árboles (n_estimators)", 10, 300, 100)
max_depth = st.slider("Profundidad máxima (max_depth)", 2, 30, 10)
test_size = st.slider("Porcentaje de prueba (test_size)", 0.1, 0.5, 0.2)

if st.button("Entrenar modelo"):
    with st.spinner("Entrenando..."):
        r = requests.post(
            f"{API}/entrenar",
            json={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "test_size": test_size,
            },
        )
    m = r.json()["metricas"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", m["MAE"])
    c2.metric("RMSE", m["RMSE"])
    c3.metric("R2", m["R2"])

# 2. Nuevo caso
st.header("2. Predecir el precio de una casa")

CIUDADES = {
    "Los Ángeles": (34.05, -118.24),
    "San Francisco": (37.77, -122.42),
    "San Diego": (32.72, -117.16),
    "Sacramento": (38.58, -121.49),
    "San José": (37.34, -121.89),
}

col1, col2 = st.columns(2)
with col1:
    ingreso = st.number_input(
        "Ingreso anual de la familia (USD)", 5000, 150000, 40000, step=1000
    )
    antiguedad = st.number_input("Antigüedad de la casa (años)", 1, 52, 20, step=1)
    habitaciones = st.number_input("Número de habitaciones", 1, 15, 5, step=1)
    dormitorios = st.number_input("Número de dormitorios", 1, 10, 1, step=1)
with col2:
    personas = st.number_input("Personas que viven en la casa", 1, 10, 3, step=1)
    poblacion = st.number_input("Habitantes del sector", 100, 5000, 1000, step=100)
    ciudad = st.selectbox("Ciudad", list(CIUDADES.keys()))

if st.button("Predecir precio"):
    lat, lon = CIUDADES[ciudad]
    casa = {
        "MedInc": ingreso / 10000,
        "HouseAge": antiguedad,
        "AveRooms": habitaciones,
        "AveBedrms": dormitorios,
        "Population": poblacion,
        "AveOccup": personas,
        "Latitude": lat,
        "Longitude": lon,
    }
    r = requests.post(f"{API}/predecir", json=casa)
    st.success(f"Precio estimado: ${r.json()['precio_usd']:,} USD")   