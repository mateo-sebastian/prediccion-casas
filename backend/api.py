import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from backend.entrenar import RUTA_MODELO, entrenar

app = FastAPI(title="API Prediccion de Precios de Casas")


class Casa(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class Parametros(BaseModel):
    n_estimators: int = 100
    max_depth: int = 10
    test_size: float = 0.2


@app.get("/")
def inicio():
    return {"mensaje": "API activa"}


@app.post("/entrenar")
def entrenar_modelo(p: Parametros):
    metricas = entrenar(p.n_estimators, p.max_depth, p.test_size)
    return {"metricas": metricas}


@app.post("/predecir")
def predecir(casa: Casa):
    paquete = joblib.load(RUTA_MODELO)
    datos = pd.DataFrame([casa.model_dump()])[paquete["columnas"]]
    pred = paquete["modelo"].predict(datos)[0]
    return {"precio_usd": round(float(pred) * 100000)} 