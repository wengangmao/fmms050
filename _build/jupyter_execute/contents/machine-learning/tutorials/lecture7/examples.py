#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.model_selection import cross_val_score as cv
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier as RFC
from time import time
import datetime


# In[2]:


data = load_breast_cancer()
X = data.data
y = data.target


# In[3]:


y


# In[5]:


dnn = DNN(hidden_layer_sizes=(200,),random_state=420)

cv(dnn,X,y,cv=5).mean()


# In[6]:


dnn.fit(X,y).predict(X)


# In[7]:


dnn.fit(X,y).predict_proba(X)


# In[8]:


times = time()
dnn = DNN(hidden_layer_sizes=(200,),random_state=420)
print(cv(dnn,X,y,cv=5).mean())
print(time() - times)


# In[9]:


times = time()
clf_rfc = RFC(n_estimators=200,random_state=420)
print(cv(clf_rfc,X,y,cv=5).mean())
print(time() - times)


# In[10]:


times = time()
dnn = DNN(hidden_layer_sizes=(50,),random_state=420)
print(cv(dnn,X,y,cv=5).mean())
print(time() - times)


# In[11]:


times = time()
dnn = DNN(hidden_layer_sizes=(50,50),random_state=420)
print(cv(dnn,X,y,cv=5).mean())
print(time() - times)


# In[12]:


times = time()
dnn = DNN(hidden_layer_sizes=(50,100),random_state=420)
print(cv(dnn,X,y,cv=5).mean())
print(time() - times)


# In[13]:


times = time()
dnn = DNN(hidden_layer_sizes=(100,100,100),random_state=420)
print(cv(dnn,X,y,cv=5).mean())
print(time() - times)


# In[18]:


from sklearn.preprocessing import StandardScaler as SS
X_ = SS().fit_transform(X)
times = time()
dnn = DNN(hidden_layer_sizes=(200,50),random_state=420)
print(cv(dnn,X_,y,cv=5).mean())
print(time() - times)


# In[24]:


for activef in ["identity","logistic","tanh","relu"]:
    times = time()
    dnn = DNN(hidden_layer_sizes=(200,50),
                activation = activef,
                max_iter = 2000,
                random_state=420)
    print(activef, cv(dnn,X_,y,cv=5).mean())
    print(time() - times)


# In[25]:


dnn.fit(X,y).out_activation_


# In[ ]:




