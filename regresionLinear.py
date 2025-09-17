import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid
from sklearn.linear_model import LinearRegression


data = {
    "Cantidad_Producida": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
    "Costo_Transporte": [50, 60, 65, 70, 80, 85, 90, 100, 110, 120],
    "Precio_Fruta": [2.5, 2.7, 3.0, 3.2, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5]
}

df = pd.DataFrame(data)
x = df[["Cantidad_Producida", "Costo_Transporte"]]
y = df["Precio_Fruta"]

model = LinearRegression()
model.fit(x,y)

def calcularPrecio(cantidad, costo):
    result = model.predict([[cantidad, costo]])[0]
    return result

def generarGrafico(cantidad, costo, prediccion):
    X_plot = pd.DataFrame({
        "Cantidad_Producida": range(100, 600, 50),
        "Costo_Transporte": [costo] * 10
    })

    y_pred = model.predict(X_plot)

    plt.figure(figsize=(7, 5))
    plt.scatter(df["Cantidad_Producida"], y, color="blue", label="Datos reales")
    plt.plot(X_plot["Cantidad_Producida"], y_pred, color="red", linewidth=2, label="Línea de regresión")

    plt.scatter(cantidad, prediccion, color="green", s=100, marker="X", label="Predicción")

    plt.xlabel("Cantidad producida (kg)")
    plt.ylabel("Precio de la fruta (USD)")
    plt.title("Regresión Lineal (predicción dinámica)")
    plt.legend()
    plt.tight_layout()

    filename = f"grafico_{uuid.uuid4().hex}.png"
    ruta = os.path.join("static", filename)
    plt.savefig(ruta)
    plt.close()

    return filename
