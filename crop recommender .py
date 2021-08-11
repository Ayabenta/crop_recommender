#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd 
from sklearn.model_selection import train_test_split


# In[45]:


df =pd.read_csv('crop.csv')
df


# In[17]:


X = df[df.columns.tolist()[:-1]]
Y = df['label']


# In[48]:


type(Y)


# In[28]:


x_train ,x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=2)


# In[30]:


from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier(n_estimators = 100, random_state = 0)
RF.fit(x_train,y_train)


# In[32]:


from sklearn.metrics import accuracy_score 


# In[38]:


y_pred= RF.predict(x_test)


# In[41]:


accuracy_score(y_test, y_pred)


# In[51]:


import numpy as np
X = np.array([[90,40,45,24,78,6.5,2]])


# In[62]:


print("recommended crop is "+RF.predict(X)[0])


# In[63]:


import pickle


# In[64]:


RF_Model_pkl = pickle.dump(RF,open('croprecommender.pkl', 'wb')) 


# In[ ]:




