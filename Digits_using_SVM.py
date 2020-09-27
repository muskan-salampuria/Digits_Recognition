#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()


# In[6]:


dir(digits)


# In[9]:


df=pd.DataFrame(digits.data)
df.head()


# In[10]:


df['target']=digits.target
df.head()


# In[12]:


X=df.drop(['target'],axis='columns')
y=df.target


# In[29]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[80]:


from sklearn.svm import SVC
model=SVC(kernel='linear',gamma=1,C=1)
model.fit(X_train,y_train)


# In[83]:


print(model.score(X_test,y_test))


# In[ ]:




