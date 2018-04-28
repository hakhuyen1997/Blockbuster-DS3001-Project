#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:22:35 2018

@author: khuyencao1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


df = pd.read_csv('moviewprofit.csv')

# Missing columns visualizations using bar chart
print (df.isnull().sum().sort_values())
df.isnull().sum().sort_values().plot.bar()

# Missing columns visualization using missingno and null values correlation heat
missingValueColumns = df.columns[df.isnull().any()].tolist()
msno.bar(df[missingValueColumns],\
            figsize=(20,8),color="#34495e",fontsize=12,labels=True,)
msno.matrix(df[missingValueColumns],width_ratios=(10,1),\
            figsize=(20,8),color=(0,0, 0),fontsize=12,sparkline=True,labels=True)

# Pearson correlation heat map matrix among all existing attributes
import seaborn as sns
plt.subplots(figsize=(20,15))
corr = df.corr()
sns.heatmap(corr, annot=True)

#Simple EDA for visualization
# Top director
df.groupby('director_name').gross.mean().sort_values(ascending=False).nlargest(10)
df_director = df.groupby('director_name').gross.sum().sort_values(ascending=False).nlargest(10)
df.groupby('director_name').gross.sum().sort_values(ascending=False).nlargest(10)
# Top actor
df.groupby('actor_1_name').gross.mean().sort_values(ascending=False).nlargest(10)
df.groupby('actor_1_name').gross.sum().sort_values(ascending=False).nlargest(10)
df.groupby('actor_2_name').gross.mean().sort_values(ascending=False).nlargest(10)
df.groupby('actor_2_name').gross.sum().sort_values(ascending=False).nlargest(10)
df.groupby('actor_3_name').gross.mean().sort_values(ascending=False).nlargest(10)
df.groupby('actor_3_name').gross.sum().sort_values(ascending=False).nlargest(10)

#Preprocessing and handling missing values
df = df.fillna(df.median())
#print (df.isnull().all())
#print (df.corr().gross.sort_values())

#Create column decide succesfulness based on profit
def isSuccessful(row):
    if row['profit'] >= 3.0:
        val = '1'
    else:
        val = '0'
    return val
df['isSuccessful'] = df.apply(isSuccessful, axis=1)
#print (df.head())

#Apply different classification models
X = df[['budget', 'director_facebook_likes', 'actor_1_facebook_likes', 'duration', 'cast_total_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes']]
y = df.isSuccessful

X_train = X.head(3362)
y_train = y.head(3362)
X_test = X.tail(1681)
y_test = y.tail(1681)

#Decision Tree
from sklearn import tree
from sklearn.metrics import accuracy_score

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict1 = model.predict(X_test)
print (accuracy_score(y_test, y_predict1))

#Linear Classifier
from sklearn import linear_model

clf = linear_model.SGDClassifier()
clf.fit(X_train, y_train)
y_predict2 = clf.predict(X_test)
print (accuracy_score(y_test, y_predict2))

#Random Forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=, random_state=0)
rf.fit(X_train, y_train)
y_predict3 = rf.predict(X_test)
print (accuracy_score(y_test, y_predict3))


