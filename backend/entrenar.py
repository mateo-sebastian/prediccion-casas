import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RUTA_DATOS = "data/california_housing.csv"
RUTA_MODELO = "models/modelo.joblib"


def entrenar(n_estimators=100, max_depth=10, test_size=0.2):
    df = pd.read_csv(RUTA_DATOS)
    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    modelo = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)
    metricas = {
        "MAE": round(mean_absolute_error(y_test, pred), 4),
        "RMSE": round(mean_squared_error(y_test, pred) ** 0.5, 4),
        "R2": round(r2_score(y_test, pred), 4),
    }

    joblib.dump({"modelo": modelo, "columnas": list(X.columns)}, RUTA_MODELO)
    return metricas


if __name__ == "__main__":
    print(entrenar()) 