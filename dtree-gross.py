
# coding: utf-8

# In[8]:


import numpy as np
import numpy.random as rnd
import os

# to make this notebook's output stable across runs
rnd.seed(42)

# To plot pretty figures
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


import os
import pandas as pd


DATA_PATH = ""
csv_path = os.path.join(DATA_PATH, "movie_metadata.csv")
movie_data = pd.read_csv(csv_path)

movie_data.head()
movie_data.tail()
movie_data.info()


# In[28]:


from sklearn.cross_validation import train_test_split
movie_data2=movie_data[['gross','num_voted_users','num_user_for_reviews','movie_facebook_likes']]
md=movie_data2.fillna(movie_data2.median())
dataset=md.values
type(dataset)


# In[29]:


y=dataset[:,3]
x=dataset[:,1:3]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)


# In[30]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris

tree_clf=DecisionTreeClassifier(max_depth=3,random_state=42)
tree_clf.fit(xtrain,ytrain)


# In[31]:


tree_clf.score(xtest,ytest)

