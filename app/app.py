import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from joblib import load, dump
from zipfile import ZipFile
import json
from lightgbm import LGBMClassifier


app = Flask(__name__)

#API_url = "http://127.0.0.1:5000/credit/" + id_client

# tell Flask to use the above defined config

clf = load('model/lgbm_classifier.pickle')
z = ZipFile("X_test_final.zip")
sample = pd.read_csv(z.open('X_test_final.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

X=sample.copy()

clf.predict_proba(X.loc[[362145]])[:,1]


@app.route('/home')
def home():
    return jsonify(username='eduCBA' , account='Premium' , validity='200 days')
    

@app.route('/credit/', methods=['GET'])

@app.route('/credit/<int:id_client>' , methods=['GET'])


def credit(id_client):
    
        id = id_client
        score = clf.predict_proba(X.loc[[id]])[:,1]
        predict = clf.predict(X.loc[[id]])

        # round the predict proba value and set to new variable
        percent_score = score*100
        
        id_risk = np.round(percent_score, 3)
        # create JSON object
        output = {'prediction': int(predict), 'client risk in %': float(id_risk)}

        print('Nouvelle Prédiction : \n', output)
        
        return jsonify(output)
    
#API_url = "http://127.0.0.1:5000/credit/" + '<int:id_client>'     
    
if __name__ == '__main__':
   app.run(debug=True)
