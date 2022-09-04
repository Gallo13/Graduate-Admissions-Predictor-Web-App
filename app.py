# Created by: Jess Gallo
# Date Created: 08/25/2022
# Last Modified: 08/28/2022
# Description: Model Deployment using Flask and Heroku

from flask import Flask, request, render_template, url_for
import numpy as np
import pickle


app = Flask(__name__)
model= pickle.load(open('admission_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]  # do i need the index??
    return render_template('index.html', prediction_text='Graduate admittance chances are: {}%'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
