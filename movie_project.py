#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:22:35 2018

@author: khuyencao1
"""

import pandas as pd
from xgboost import XGBClassifier

def isSuccessful(row):
    if row['profit'] >= 6.59:
        val = '1'
    else:
        val = '0'
    return val

def preprocessing_data(df):
    df = df.fillna(df.median())
    df['isSuccessful'] = df.apply(isSuccessful, axis=1)
    return df


def apply_xgboost(X_test, df):
    df = preprocessing_data(df)
    X = df[['budget', 'director_facebook_likes', 'actor_1_facebook_likes', 'duration', 'cast_total_facebook_likes','actor_2_facebook_likes', 'actor_3_facebook_likes']]
    X = X.fillna(X.median())
    y = df.isSuccessful
    X_train = X.head(3362)
    y_train = y.head(3362)
    xgboost = XGBClassifier()
    xgboost.fit(X_train, y_train)
    y_predict = xgboost.predict(X_test)
    return y_predict


def apply_classification(df, budget, dfl, afl1, afl2, duration):
    X_test = pd.read_csv('stub.csv')
    X_test['budget'] = float(budget)
    X_test['director_facebook_likes'] = float(dfl)
    X_test['actor_1_facebook_likes'] = float(afl1)
    X_test['duration'] = float(duration)
    X_test['actor_2_facebook_likes'] = float(afl2)
    X_test['cast_total_facebook_likes'] = df.min()['cast_total_facebook_likes']
    X_test['actor_3_facebook_likes'] = df.min()['actor_3_facebook_likes']
    print (X_test)
    result_predict = apply_xgboost(X_test, df)
    print (result_predict)
    if result_predict == '0':
        return "NO"
    else:
        return "YES"

if __name__ == "__main__":
    df = pd.read_csv('moviewprofit.csv')
    budget = 0
    dfl = 0
    afl1 = 0
    afl2 = 0
    afl3 = 0
    duration = 0
    preprocessing_data(df)
    apply_classification(df, budget, dfl, afl1, afl2, duration)


