import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Creat flask app
app = Flask(__name__)

#load the pickle model
model = pickle.load(open("Final_Cobb_Progress.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    prediction = np.round(prediction[0],1)

    return render_template("index.html", 
    original_input={
        "Initial Cobb angle (°)":float_features[0],
        "Flexibility (%)":float_features[1],
        "Lumbar lordosis angle (°)":float_features[2],
        "Thoracic kyphosis angle (°)":float_features[3],
        "Age at prediction (years)":float_features[4],
        "Number of levels involved":float_features[5],
        "Risser"+" stage at initial diagnosis":float_features[6],

    } ,prediction_text = " The Final Cobb angle is {} degrees".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)


