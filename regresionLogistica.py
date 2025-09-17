import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv("dataset_desempleo.csv")

X = data.drop("Desempleado", axis=1)
y = data["Desempleado"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

def evaluate():
    y_pred = log_model.predict(X_test_scaled)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.savefig("static/conf_matrix.png")
    plt.close()

    report = classification_report(y_test, y_pred, target_names=["No", "Sí"])
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, report, "conf_matrix.png"

def predict_label(features, threshold=0.5):
    niveles = ["Primario", "Secundario", "Universitario", "Posgrado"]
    conocimientos_txt = ["Ninguno", "Básico", "Intermedio", "Avanzado"]

    X_new = np.array([features])
    X_new_scaled = scaler.transform(X_new)

    probas = log_model.predict_proba(X_new_scaled)[0]
    prob_empleado = probas[0]      
    prob_desempleado = probas[1] 

    if prob_desempleado >= threshold:
        label = "No"
        estado = "Desempleado"
        probability = prob_desempleado
    else:
        label = "Sí"
        estado = "Empleado"
        probability = prob_empleado

    description = (
        f"Una persona con nivel educativo <b>{niveles[features[0]]}</b>, "
        f"{features[1]} años de experiencia, "
        f"conocimientos técnicos <b>{conocimientos_txt[features[2]]}</b> "
        f"y edad de {features[3]} años, "
        f"tiene una probabilidad de <b>{probability*100:.2f}%</b> de estar "
        f"<b>{estado}</b>."
        f"Decision finale:{label} "
    )

    return label, probability, description
