from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    GENDER = int(request.form.get('GENDER',False))
    AGE = int(request.form.get('AGE',False))
    SMOKING = int(request.form.get('SMOKING',False))
    YELLOW_FINGERS = int(request.form.get('YELLOW_FINGERS',False))
    ANXIETY = int(request.form.get('ANXIETY',False))
    PEER_PRESSURE = int(request.form.get('PEER_PRESSURE',False))
    CHRONIC_DISEASE = int(request.form.get('CHRONIC_DISEASE',False))
    FATIGUE	 = int(request.form.get('FATIGUE',False))
    ALLERGY = int(request.form.get('ALLERGY',False))
    WHEEZING = int(request.form.get('WHEEZING',False))
    COUGHING = int(request.form.get('COUGHING',False))
    ALCOHOL_CONSUMING = int(request.form.get('ALCOHOL_CONSUMING',False))
    SHORTNESS_OF_BREATH= int(request.form.get('SHORTNESS_OF_BREATH',False))
    SWALLOWING_DIFFICULTY= int(request.form.get('SWALLOWING_DIFFICULTY',False))
    CHEST_PAIN= int(request.form.get('CHEST_PAIN',False))

    x = np.array([GENDER,AGE ,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,ALLERGY,FATIGUE,WHEEZING,COUGHING,ALCOHOL_CONSUMING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN]).reshape(1, -1)

    scaler_path = os.path.join('C:/Users/sakshi/Downloads/Lung_Cancer Project','models/scaler.pkl')
    scaler = None
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    x = scaler.transform(x)

    model_path = os.path.join('C:/Users/sakshi/Downloads/Lung_Cancer Project','models/knn.sav')
    knn = joblib.load(model_path)

    y_pred = knn.predict(x)

    # for Lung Cancer Risk
    if y_pred == 0:
        return render_template('nolungcancer.html')
    else:
        return render_template('lungcancer.html')


if __name__ == "__main__":
    app.run(debug=True,port=7368)
