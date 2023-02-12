from flask import Flask, jsonify, request
import json
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data
    data = request.get_json()
    
    # Extract the mean and std values
    ht_mean = data["HT"]["Mean"]
    ht_std = data["HT"]["STD"]
    rpt_mean = data["RPT"]["Mean"]
    rpt_std = data["RPT"]["STD"]
    ppt_mean = data["PPT"]["Mean"]
    ppt_std = data["PPT"]["STD"]
    rrt_mean = data["RRT"]["Mean"]
    rrt_std = data["RRT"]["STD"]
    
    # Get the selected model
    model_name = data["Model"]

    # Load the saved models
    if model_name == "SVM":
        model = joblib.load("svm_model.joblib")
    elif model_name == "RF":
        model = joblib.load("rf_model.joblib")
    elif model_name == "XGB":
        model = joblib.load("xgb_model.joblib")
    else:
        return "Invalid Model Name"
    
    # Make the prediction
    feature = np.array([[ht_mean, ht_std, rpt_mean, rpt_std, ppt_mean, ppt_std, rrt_mean, rrt_std]])
    user_id = model.predict(feature)

    # Transform the class labels using the label encoder
    le = joblib.load("label_encoder.joblib")
    user_id = le.inverse_transform(user_id)

    # Return the result
    return jsonify({'UserID': int(user_id[0])})

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port=8080)