#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


df_train=pd.read_csv('C:/Users/ravin/OneDrive/Desktop/archive (1)/train.csv')


# In[3]:


df_train.head()


# In[4]:


df_test=pd.read_csv('C:/Users/ravin/OneDrive/Desktop/archive (1)/test.csv')


# In[5]:


df_test.head()


# In[6]:


df=df_train.append(df_test)


# In[7]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.drop(['User_ID'],axis=1,inplace=True)


# In[10]:


df.head()


# In[11]:


df['Gender']=df['Gender'].map({'F':0,'M':1})


# In[12]:


df.head()


# In[13]:


df['Age'].unique()


# In[14]:


#pd.get_dummies(df['Age'],drop_first=True)
df['Age'] = df['Age'].map({'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7})


# In[15]:


df.head()


# In[16]:


df_city=pd.get_dummies(df['City_Category'],drop_first=True)


# In[17]:


df_city


# In[18]:


df=pd.concat([df,df_city],axis=1)


# In[19]:


df


# In[20]:


df.drop('City_Category',axis=1,inplace=True)


# In[21]:


df.drop('C',axis=1,inplace=True)


# In[22]:


df.head()


# In[23]:


df=pd.concat([df,df_city],axis=1)


# In[24]:


df.head()


# In[25]:


df.isnull().sum()


# In[26]:


df['Product_Category_2'].value_counts()


# In[27]:


df['Product_Category_2'].mode()[0]


# In[28]:


df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[29]:


df['Product_Category_2'].isnull().sum()


# In[81]:


df['Product_Category_3'].mode()[0]


# In[30]:


df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[31]:


df['Product_Category_3'].isnull().sum()


# In[32]:


df.head()


# In[33]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')


# In[43]:


df.head()


# In[35]:


df.info()


# In[36]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)


# In[96]:


df.info()


# In[44]:


df['B']=df['B'].astype(int)


# In[45]:


df['C']=df['C'].astype(int)


# In[40]:


df.info()


# In[49]:


df.head()


# In[54]:


sns.barplot('Age','Purchase',hue='Gender',data=df)


# In[51]:


df1=df


# sns.barplot(x='Age',y='Purchase',hue='Gender',data=df1)

# sns.barplot('Age','Purchase',hue='Gender',data=df)

# In[59]:


sns.barplot(x='Age',y='Purchase',hue='Gender',data=df)


# # men do purchase more than women
# 

# In[60]:


sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=df)


# In[61]:


sns.barplot(x='Product_Category_1',y='Purchase',hue='Gender',data=df)


# In[62]:


sns.barplot(x='Product_Category_2',y='Purchase',hue='Gender',data=df)


# In[63]:


sns.barplot(x='Product_Category_3',y='Purchase',hue='Gender',data=df)


# In[67]:


df_test=df[df['Purchase'].isnull()]


# In[68]:


df_train=df[~df['Purchase'].isnull()]


# In[70]:


df_train.head()


# In[71]:


df_test.head()


# In[81]:


x=df_train


# In[83]:


x.head()


# In[85]:


x.shape


# In[86]:


x.head()


# In[90]:


y=df_test['Purchase']


# In[91]:


y.head()


# In[ ]:





# In[98]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   df_train, df_test, test_size=0.33, random_state=42)


# In[ ]:





# In[96]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[ ]:


x_train=sc.fit_transform(x_train)
y_train=sc.fit_transform(x_test)


# In[99]:


from sklearn.datasets import load_iris


# In[100]:


iris=load_iris()


# In[101]:


iris


# In[102]:


iris.data


# In[103]:


from sklearn.tree import DecisionTreeClassifier


# In[110]:


classiffier=DecisionTreeClassifier()
classiffier.fit(iris.data,iris.target)


# In[112]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(classiffier, filled=True)


# In[ ]:




