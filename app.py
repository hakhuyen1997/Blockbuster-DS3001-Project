from flask import Flask, render_template, request, flash, redirect, url_for
import os
import numpy as np
from os.path import abspath, dirname
import pandas as pd

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
    return render_template('predict.html')

if __name__ == '__main__':
  port = int(os.environ.get('PORT', 9999))
  app.debug = True
  print('Running on port ' + str(port))
  app.run('0.0.0.0',port)