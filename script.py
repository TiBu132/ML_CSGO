#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
#creating instance of the class
app=Flask(__name__)

scaler_fitted = pickle.load(open('checkpoints/scaler_fitted.pkl', 'rb'))

pca_fitted = pickle.load(open('checkpoints/pca_fitted.pkl', 'rb'))

def transformar_datos(new_data):
    new_data_point = np.array([new_data])
    new_data_scaled = scaler_fitted.transform(new_data_point)
    new_data_pca = pca_fitted.transform(new_data_scaled)
    new_data_final = np.c_[new_data_pca]

    return new_data_final

#to tell flask what url shoud trigger the function index()
@app.route('/', methods=['GET','POST'])
def index():    
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    lista=[]
    loaded_model = pickle.load(open("checkpoints/model.pkl","rb"))
    result = loaded_model.predict(to_predict_list)
    proba = loaded_model.predict_proba(to_predict_list)
    lista.append(result)
    lista.append(proba)
    return lista

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        data = request.form
        form_values = [value for key, value in data.items()]
        data_topredict = transformar_datos(form_values)
        try:
            result = ValuePredictor(data_topredict)
        except ValueError:
            prediction='Error en el formato de los datos'

        return render_template("result.html", result=result)


if __name__=="__main__":

    app.run(port=5001)