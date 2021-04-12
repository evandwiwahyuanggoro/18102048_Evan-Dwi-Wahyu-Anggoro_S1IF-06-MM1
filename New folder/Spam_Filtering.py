#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd //18102048_Evan Dwi Wahyu Anggoro_S1IF-06-MM1


# In[52]:


data=pd.read_csv("spam.csv", encoding="latin-1")


# In[53]:


data.head(5)


# In[6]:


data.columns


# In[7]:


data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)


# In[8]:


data.head()


# In[9]:


data['class']=data['class'].map({'ham':0, 'spam':1})


# In[10]:


data.head()


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer


# In[20]:


cv=CountVectorizer()


# In[21]:


x=data['message']
y=data['class']


# In[22]:


x.shape


# In[23]:


y.shape


# In[24]:


x=cv.fit_transform(x)


# In[25]:


x


# 1. The Cat
# 2. The Dog
# 3. The Bird
# 
#       The Cat Dog Bird
# 1.     1   1   0   0
# 2.      1  0    1   0
# 3.      1  0    0   1

# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


# In[28]:


x_train.shape


# In[29]:


from sklearn.naive_bayes import MultinomialNB


# In[30]:


model=MultinomialNB()


# In[31]:


model.fit(x_train, y_train)


# In[33]:


result=model.score(x_test, y_test)


# In[34]:


result=result*100


# In[35]:


result


# In[38]:


import pickle


# In[39]:


pickle.dump(model, open("spam.pkl","wb"))


# In[40]:


pickle.dump(cv, open("vectorizer.pkl","wb"))


# In[41]:


clf=pickle.load(open("spam.pkl","rb"))


# In[42]:


clf


# In[46]:


msg="I am sleep"
data=[msg]
vect=cv.transform(data).toarray()
result=model.predict(vect)
print(result)


# In[ ]:




