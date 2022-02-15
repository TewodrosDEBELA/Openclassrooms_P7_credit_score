# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:10:00 2022

@author: Tewod
"""


#API FLASK run (commande : python api/api.py)
# Local Adresse :  http://127.0.0.1:5000/credit/IDclient
# adresse distance : https://api-prediction-credit.herokuapp.com/credit/idclient
# Github depo : https://github.com/DeepScienceData/API-Prediction

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from zipfile import ZipFile
import json



app = Flask(__name__)
# tell Flask to use the above defined config

@app.route("/")
def hello():
    """
    Ping the API.
    """
    return jsonify({"text":"Hello, the API is up and running..." })

@app.route('/credit/<id_client>', methods=['GET'])
def credit(id_client):

        pickle_in = open('C:/Users/Tewod/OneDrive/Bureau/Openclassrooms/projets/projet7/Mamadou/RandomForestClassifier.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        
        sample = pd.read_csv('C:/Users/Tewod/OneDrive/Bureau/Openclassrooms/projets/projet7/X_sample.csv', index_col='SK_ID_CURR', encoding ='utf-8')
        X=sample.iloc[:, :126]
        score = clf.predict_proba(X[X.index == int(id_client)])[:,1]
        predict = clf.predict(X[X.index == int(id_client)])

        # round the predict proba value and set to new variable
        percent_score = score*100
        id_risk = np.round(percent_score, 3)
        # create JSON object
        output = {'prediction': int(predict), 'client risk in %': float(id_risk)}


        print('Nouvelle Prédiction : \n', output)

        return jsonify(output)
if __name__ == '__main__':
    app.run(debug=True)
