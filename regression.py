#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[3]:


boston = load_boston()
print(boston.DESCR)


# In[4]:


dataset = boston.data
for name, index in enumerate(boston.feature_names):
    print(index, name)


# In[13]:


data = dataset[:,12].reshape(-1,1)


# In[14]:


np.shape(dataset)


# In[15]:


target = boston.target.reshape(-1-1)


# In[16]:


np.shape(target)


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='yellow')
plt.xlabel('lower income population ')
plt.ylabel('cost of house')
plt.show()


# In[20]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(data, target)


# In[21]:


pred = reg.predict(data)


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='green')
plt.scatter(data,pred,color='red')
plt.xlabel('lower income population ')
plt.ylabel('cost of house')
plt.show()


# In[28]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# In[33]:


model = make_pipeline(PolynomialFeatures(5), reg)


# In[34]:


model.fit(data, target)


# In[35]:


pred = model.predict(data)


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data,target,color='green')
plt.scatter(data,pred,color='red')
plt.xlabel('lower income population ')
plt.ylabel('cost of house')
plt.show()


# In[ ]:




