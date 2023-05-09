#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.model_selection import cross_val_score as cv
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier as RFC
from time import time
import datetime


# In[5]:


data = load_breast_cancer()
X = data.data
y = data.target


# In[6]:


from sklearn.preprocessing import StandardScaler as SS
X_ = SS().fit_transform(X)
times = time()
dnn = DNN(hidden_layer_sizes=(200,50),random_state=420)
print(cv(dnn,X_,y,cv=5).mean())
print(time() - times)


# In[7]:


dnn = DNN(hidden_layer_sizes=(20,),
        activation="relu",
        solver="sgd",
        learning_rate_init = 0.5,
        learning_rate = "invscaling",
        power_t = 0.1,
        batch_size=200,
        max_iter=3000,
        random_state=420).fit(X_,y)


# In[8]:


dnn.coefs_


# In[9]:


type(dnn.coefs_)


# In[11]:


for item in dnn.coefs_:
    print(item.shape)


# In[12]:


X_.shape


# In[ ]:


# how many parameters?


# In[17]:


dnn.coefs_[0][0].shape # w1^t


# In[14]:


dnn.intercepts_


# In[15]:


for item in dnn.intercepts_:
    print(item.shape)


# In[16]:


dnn.loss_


# In[ ]:




