#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# # Task-2
# ## To Explore Supervised Machine Learning
# 
# In this task we will predict the percentage of marks that astudent is expected to score based upon the number of hours they studied.

# # Import all libraries

# In[1]:


#import libraries 
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read the file 

# In[2]:


#read data which is stored 
url=("http://bit.ly/w-data")
df=pd.read_csv(url)


# # Print file

# In[3]:


#print data
print("Data is succesfully imported:-")
df


# In[4]:


#print first five records
df.head()


# In[5]:


#print the last five records
df.tail()


# In[6]:


#describe  the dataset 
df.describe()


# In[7]:


df.dtypes


# In[8]:


df['Scores']=df['Scores'].astype('float64')
df.dtypes


# In[9]:


df.corr()


# In[10]:


#plotting the distribution of score
plt.plot(df['Hours'],df['Scores'],'g*')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel("Percentag Scored")
plt.show()


# # Prepare Data For Training And Test

# In[11]:


x=df.iloc[:,0:1].values
y=df.iloc[:,1:].values
y


# # Splitting Into Train And Test

# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# # Training The Model

# In[13]:


#Traning The Dataset
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print("Training complete.")


# In[14]:


#predication of marks
print(x_test)
y_predict=model.predict(x_test)
print(y_predict)


# # Visualising The Training Set

# In[15]:


#plotting the regression line
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,model.predict(x_train),color="red")
plt.title("Hours Vs Study")
plt.xlabel("Hours study")
plt.ylabel("Scores(%)")
plt.show()


# # Visualizing The Test Set

# In[16]:


#plotting test set
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,model.predict(x_test),color="green")
plt.title("Hours Vs Study")
plt.xlabel("Hours study")
plt.ylabel("Scores(%)")
plt.show()


# # Prediction Scored For 9.25 Hours Study

# In[17]:


#making requried prediction
print("No. of Hours:-",9.25)
print("Predicted Score in percentage:-",model.predict([[9.25]]))


# # R-Square

# In[18]:


model.score(x_train,y_train)


# # Adjust R-square

# In[19]:


x_train.shape


# # Model Evaluation

# In[20]:


#Evaluation of the model
from sklearn import metrics
print("Mean Absolute Error:-",
metrics.mean_absolute_error(y_test,y_predict))
print("Mean square Error:-",
metrics.mean_squared_error(y_test,y_predict))


#  ## *************************************Complete task-2****************************************************
