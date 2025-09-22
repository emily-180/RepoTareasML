import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool

df = pd.read_csv("reseñas_productos.csv")

X = df.drop("sentimiento", axis=1)
y = df["sentimiento"]

cat_features = ["categoria_producto", "pais_origen"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    loss_function="Logloss",   
    eval_metric="Accuracy",
    random_seed=42,
    verbose=50
)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

model.fit(train_pool, eval_set=test_pool)

def evaluate():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] 

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"], output_dict=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negativo","Positivo"], yticklabels=["Negativo","Positivo"])
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión - CatBoost")
    plt.savefig("static/conf_matrix_catboost.png")
    plt.close()

    return {
        "accuracy": round(acc, 4),
        "report": report,
        "conf_matrix_file": "static/conf_matrix_catboost.png"
    }

def predict_label(features, threshold=0.5):
    """
    features = [longitud_texto, palabras_clave, puntuacion_asignada, categoria_producto, pais_origen]
    """
    X_new = pd.DataFrame([{
        "longitud_texto": features[0],
        "palabras_clave": features[1],
        "puntuacion_asignada": features[2],
        "categoria_producto": features[3],
        "pais_origen": features[4]
    }])

    proba = model.predict_proba(X_new)[0][1] 
    label = "Positivo" if proba >= threshold else "Negativo"

    return label, round(proba, 4)
