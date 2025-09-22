from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)
import regresionLinear
import regresionLogistica

@app.route("/")  
def home():
    return redirect(url_for("index"))

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/caso1")
def caso1():
    return render_template("caso1.html")

@app.route("/caso2")
def caso2():
    return render_template("caso2.html")

@app.route("/caso3")
def caso3():
    return render_template("caso3.html")

@app.route("/caso4")
def caso4():
    return render_template("caso4.html")

@app.route("/conceptoLineal")
def conceptoLineal():
    return render_template("conceptoLineal.html")

@app.route("/praticoLineal", methods=["GET"])
def praticoLineal_form():
    return render_template("praticoLineal.html", grafico=None, result=None)

@app.route("/praticoLineal", methods=["POST"])
def praticoLineal_result():
    cantidad = float(request.form["cantidad"])
    costo = float(request.form["costo"])
    calculateResult = regresionLinear.calcularPrecio(cantidad, costo)
    grafico = regresionLinear.generarGrafico(cantidad, costo, calculateResult)

    return render_template("praticoLineal.html", result=calculateResult, grafico=grafico)

@app.route("/conceptoLogistica")
def conceptoLogistica():
    return render_template("conceptoLogistica.html")

@app.route("/praticoLogistica", methods=["GET", "POST"])
def praticoLogistica():
    result = None
    label = None
    probability = None
    description = None
    accuracy, report, conf_matrix_file = regresionLogistica.evaluate()

    if request.method == "POST":
        nivel = int(request.form["nivel"])
        experiencia = int(request.form["experiencia"])
        conocimientos = int(request.form["tecnicos"])
        edad = int(request.form["edad"])

        label, probability, description = regresionLogistica.predict_label(
            [nivel, experiencia, conocimientos, edad]
        )

        result = description

    return render_template(
        "praticoLogistica.html",
        result=result,                  
        label=label,                    
        probability=f"{(probability*100):.2f}%" if probability else None,
        accuracy=f"{accuracy*100:.2f}%",
        report=report,                  
        conf_matrix_file=conf_matrix_file
    )

@app.route("/conceptoTiposClasificacion")
def conceptoTiposClasificacion():
    return render_template("conceptoTiposClasificacion.html")


if __name__ == "__main__":
    app.run(debug=True)