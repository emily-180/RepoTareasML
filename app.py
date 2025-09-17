from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)
import regresionLinear

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

if __name__ == "__main__":
    app.run(debug=True)