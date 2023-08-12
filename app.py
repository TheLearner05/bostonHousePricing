from flask import Flask, render_template, request, app,url_for
import flask 
from flask import jsonify
import pickle
import numpy as np
import pandas as pd
import json


app = Flask(__name__)

#load the model
model = pickle.load(open("regModel.pkl","rb"))
scalar = pickle.load(open("scaling.pkl",'rb'))
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict_api",methods=["POST"])
def predict_api():
    data = request.json['data']
    print(data)
    print(data)
    #standardize the input 
    print(np.array(list(data.values())).reshape(1,-1))

    nData  = scalar.transform(np.array(list(data.values())).reshape(1,-1))

    output = model.predict(nData)
    print(output[0])
    return json.dumps(output[0])

if __name__ == "__main__":
    app.run(debug=True)


