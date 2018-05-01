from flask import Flask, render_template, request, flash, redirect, url_for
import os
import numpy as np
from os.path import abspath, dirname
import pandas as pd
from movie_project import apply_classification

app = Flask(__name__)
app.config.from_object(__name__)

@app.route('/', methods =['POST', 'GET'])
def home():
    return render_template('index.html')

@app.route('/introduction/', methods =['POST', 'GET'])
def introduction():
    return render_template('introduction.html')

@app.route('/visualization/', methods =['POST', 'GET'])
def visualization():
    return render_template('visualization.html')

@app.route('/predict/', methods =['POST', 'GET'])
def prediction():
    if request.method == "POST":
        return redirect(url_for('upload'))
    return render_template('predict.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        budget = request.form['budget']
        dfl = request.form['dfl']
        afl1 = request.form['afl1']
        afl2 = request.form['afl2']
        duration = request.form['duration']
        df = pd.read_csv('moviewprofit.csv')
        answer = apply_classification(df, budget, dfl, afl1, afl2, duration)
        print (answer)
        return render_template('result.html', **locals())
    else:
        return render_template('predict.html')


if __name__ == '__main__':
  port = int(os.environ.get('PORT', 9999))
  app.debug = True
  print('Running on port ' + str(port))
  app.run('0.0.0.0',port)