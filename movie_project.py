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
            #figsize=(20,8),color="#34495e",fontsize=12,labels=True,)
msno.matrix(df[missingValueColumns],width_ratios=(10,1),\
            figsize=(20,8),color=(0,0, 0),fontsize=12,sparkline=True,labels=True)

# Pearson correlation heat map matrix among all existing attributes
import seaborn as sns
plt.subplots(figsize=(20,15))
corr = df.corr()
sns.heatmap(corr, annot=True)

#Simple EDA for visualization
df.groupby('director_name').gross.mean().sort_values(ascending=False).nlargest(10)
df_director = df.groupby('director_name').gross.sum().sort_values(ascending=False).nlargest(10)
df.groupby('director_name').gross.sum().sort_values(ascending=False).nlargest(10)

df.groupby('actor_1_name').gross.mean().sort_values(ascending=False).nlargest(10)
df.groupby('actor_1_name').gross.sum().sort_values(ascending=False).nlargest(10)
df.groupby('actor_2_name').gross.().sort_values(ascending=False).nlargest(10)
df.groupby('actor_2_name').gross.sum().sort_values(ascending=False).nlargest(10)
df.groupby('actor__name').gross.mean().sort_values(ascending=False).nlargest(10)
df.groupby('actor_3_name').gross.sum().sort_values(ascending=False).nlargest(10)

#Preprocessing and handling missing values

#Apply different classification models

