from flask import Flask
from flask import render_template, request

# modeling packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.externals import joblib
L1_logistic = joblib.load('model/model.pkl')
# create the flask object
app = Flask(__name__)

@app.route('/titanic', methods=['GET','POST'])
def titanic():
    data = {}   # data object to be passed back to the web page
    if request.form:
        # get the input data
        form_data = request.form
        data['form'] = form_data
        predict_class = float(form_data['predict_class'])
        predict_age = float(form_data['predict_age'])
        predict_sibsp = float(form_data['predict_sibsp'])
        predict_parch = float(form_data['predict_parch'])
        predict_fare = float(form_data['predict_fare'])
        predict_sex = form_data['predict_sex']

        # convert the sex from text to binary
        if predict_sex == 'M':
            sex = 0
        else:
            sex = 1
        input_data = np.array([predict_class, predict_age, predict_sibsp, predict_parch, predict_fare, sex])

        # get prediction
        prediction = L1_logistic.predict_proba(input_data.reshape(1, -1))
        prediction = prediction[0][1] # probability of survival
        data['prediction'] = '{:.1f}% Chance of Survival'.format(prediction * 100)
        
    return render_template('titanic_finished.html', data=data)


if __name__ == '__main__':

    # start the app
    app.run(debug=True)
