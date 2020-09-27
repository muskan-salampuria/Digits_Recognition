#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()


# In[3]:


dir(digits)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])


# In[6]:


df=pd.DataFrame(digits.data)


# In[7]:


df['target']=digits.target


# In[8]:


df.head()


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)


# In[16]:


len(X_test)


# In[40]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=40)
model.fit(X_train,y_train)


# In[41]:


model.score(X_test,y_test)


# In[44]:


y_predicted=model.predict(X_test)


# In[45]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predicted)


# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel('Truth')


# In[ ]:




