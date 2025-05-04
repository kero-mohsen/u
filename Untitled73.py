#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import matplotlib.pyplot as plt
import joblib


# In[ ]:


data = pd.read_csv(r)


# In[ ]:


data


# In[ ]:


data = data.drop_duplicates()


# In[ ]:


data.hist(bins=50, figsize=(20, 15))


# In[ ]:


label_encoders = {}
for column in ['','','']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])
data = pd.get_dummies(data, columns=['furnishingstatus'])


# In[ ]:


X = data.drop('price', axis=1)
Y = data['price']


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=20, random_state=42)


# In[3]:


model = LinearRegression()
model.fit(X_train, Y_train)


# In[ ]:


Y_predict = model.predict(X_test)


# In[ ]:


mse = mean_squared_error(Y_test, Y_predict)


# In[ ]:


print("mean squard error = ", mse)


# In[ ]:


joblib.dump(model, r"C:\Users\kero mohsen\Downloads\Housing.pk1")


# In[4]:


def predict_hosue_price(model, features):
    input_data = pd.DataFrame([features], columns=X.columns)
    predict_price = model.predict(input_data)
    return predict_price


# In[ ]:


user_input = {}
for column in X.columns:
    user_input[column] = int(input(f"Enter {column}: "))
predict_price = predict_hosue_price(model, user_input)
print('Predict Price for house : ', predict_price)


# In[ ]:




