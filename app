# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 23:41:36 2022
@author: Tewodros
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from joblib import load
import json
from lightgbm import LGBMClassifier


app = Flask(__name__)

@app.route("/")
def hello():
    """
    Ping the API.
    """
    return jsonify({"text":"Hello, the API is intended for predicting credit risk" })

@app.route('/credit/', methods=['GET'])

@app.route('/credit/<int:id_client>' , methods=['GET'])


def credit(id_client):
        pickle_in = open('lgbm_classifier.pickle', 'rb') 

        clf = pickle.load(pickle_in)
        sample = pd.read_csv('X_test_sample.csv', index_col='SK_ID_CURR', encoding ='utf-8')
         # Testing with one client id
       
        id = id_client
        score = clf.predict_proba(sample.loc[[id]])[:,1]
        predict = clf.predict(sample.loc[[id]])

        # round the predict proba value and set to new variable
        percent_score = score*100
        
        id_risk = np.round(percent_score, 3)
        # create JSON object
        output = {'prediction': int(predict), 'client risk in %': float(id_risk)}

        print('Nouvelle Prédiction : \n', output)
        
        return output
    
     
    
if __name__ == '__main__':
     app.run()
