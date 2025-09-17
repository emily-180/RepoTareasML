from flask import Flask, render_template, redirect, url_for
app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)