import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('DiabetesPredictor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    Pregnancies = float(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    finalFeatures = np.array([[Pregnancies,Glucose,BloodPressure,BMI,DiabetesPedigreeFunction]])
    prediction = model.predict(finalFeatures)
    
    if prediction==1:
        prediction="Positive"
    else:
        prediction="Negative"
    

    return render_template('index.html', prediction_text='Result of Diabetes Test:{}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)