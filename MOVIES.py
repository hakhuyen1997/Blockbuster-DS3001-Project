import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import plotly.graph_objs as go

data = pd.read_csv('/Users/ylee/Desktop/DS 3001/Final Project/movie.csv', error_bad_lines=False,encoding = "ISO-8859-1")
mtypes = data.dtypes

shape = data.shape


profit = (((data['gross'].values)-(data['budget'].values))/(data['gross'].values))*100
data.loc[:,'profit'] = pd.Series(profit, index=data.index)

percentProfit = sns.jointplot(x="title_year", y="profit",kind='scatter',
                              size=10 ,ylim = [0,110],xlim= [1980,2020],data=data, color = 'g', linewidth=1)

data.describe().to_csv('dataDescribed.csv')

plt.show()