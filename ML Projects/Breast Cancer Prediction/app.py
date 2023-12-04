from flask import Flask,request, render_template
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__) 


@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/treatment')
def treatment():
    return render_template("treatment.html")

@app.route('/news')
def news():
    return render_template("news.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    return render_template("predict.html")

@app.route('/result', methods=["GET","POST"])
def result():
    if request.method == "POST":
        texture_mean = float(request.form["texture_mean"])
        smoothness_mean = float(request.form["smoothness_mean"])
        compactness_mean = float(request.form["compactness_mean"])
        symmetry_mean = float(request.form["symmetry_mean"])
        fractal_dimension_mean = float(request.form["fractal_dimension_mean"])
        texture_se = float(request.form["texture_se"])
        smoothness_se = float(request.form["smoothness_se"])
        symmetry_se = float(request.form["symmetry_se"])
        symmetry_worst = float(request.form["symmetry_worst"])

        pred = model.predict([[texture_mean, smoothness_mean, compactness_mean, symmetry_mean,
                               fractal_dimension_mean, texture_se, smoothness_se, symmetry_se, symmetry_worst]])

        output = pred[0]
        if output == 0:
            return render_template("result.html" , predicted_text = "Breast Cancer Not Predicted!🎗️❤️")
            
        
        else:
            return render_template("result.html" , predicted_text = "💔Breast Cancer Predicted.🎗️")
            
    return render_template("result.html")


if __name__ == '__main__':
    app.run(debug=True)
