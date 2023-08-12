#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df_train=pd.read_excel("C:/Users/ravin/Downloads/5-Days-Live-EDA-and-Feature-Engineering-main/5-Days-Live-EDA-and-Feature-Engineering-main/Flight Prediction/Data_Train.xlsx")


# In[49]:


df_train.head()


# In[50]:


df_test=pd.read_excel('C:/Users/ravin/Downloads/5-Days-Live-EDA-and-Feature-Engineering-main/5-Days-Live-EDA-and-Feature-Engineering-main/Flight Prediction/Test_set.xlsx')


# In[51]:


df_test.head()


# In[52]:


final_df=df_train.append(df_test)


# In[14]:


final_df.head()


# In[53]:


final_df.tail()


# In[54]:


final_df.info()


# In[55]:


final_df['Date']=final_df['Date_of_Journey'].str.split('/').str[0]
final_df['month']=final_df['Date_of_Journey'].str.split('/').str[1]
final_df['year']=final_df['Date_of_Journey'].str.split('/').str[2]


# In[ ]:


#final_df['Date']=final_df['Date_of_Journey'].(apply lambda x:x.split('/')[0])
#final_df['month']=final_df['Date_of_Journey'].(apply lambda x:x.split('/')[1])
#final_df['year']=final_df['Date_of_Journey'].(apply lambda x:x.split('/')[2])


# In[56]:


final_df['Date']=final_df['Date'].astype(int)


# In[87]:


final_df['month']=final_df['month'].astype(int)
final_df['year']=final_df['year'].astype(int)


# In[58]:


final_df.info()


# In[61]:


final_df.drop('Date_of_Journey',axis=1,inplace=True)


# In[63]:


final_df.head()


# In[65]:


final_df['Arrival_Time'].str.split(' ').str[0]


# In[68]:


final_df['Arrival_Time']=final_df['Arrival_Time'].apply (lambda x:x.split(' ')[0])


# In[69]:


final_df.head(1)


# In[72]:


final_df['Arrival_hour']=final_df['Arrival_Time'].str.split(':').str[0]
final_df['Arrival_min']=final_df['Arrival_Time'].str.split(':').str[1]


# In[73]:


final_df.head(1)


# In[74]:


final_df['Arrival_hour']=final_df['Arrival_hour'].astype(int)
final_df['Arrival_min']=final_df['Arrival_min'].astype(int)


# In[77]:


final_df.drop('Arrival_Time',axis=1,inplace=True)


# In[79]:


final_df.head(1)


# In[80]:


final_df['Dep_hour']=final_df['Dep_Time'].str.split(':').str[0]
final_df['Dep_min']=final_df['Dep_Time'].str.split(':').str[1]


# In[83]:


final_df.drop('Dep_Time',axis=1,inplace=True)


# In[84]:


final_df.info()


# In[85]:


final_df['Dep_hour']=final_df['Dep_hour'].astype(int)
final_df['Dep_min']=final_df['Dep_min'].astype(int)


# In[88]:


final_df.info()


# In[115]:


final_df.drop('Total_Stops',axis=1,inplace=True)


# In[116]:


final_df.info()


# In[130]:


final_df['duration_hour']=final_df['Duration'].str.split('h').str[0]


# In[131]:


final_df.info()


# In[137]:


final_df.drop(6474,axis=0,inplace=True)


# In[138]:


final_df.drop(2660,axis=0,inplace=True)


# In[139]:


final_df['duration_hour']=final_df['duration_hour'].astype('int')


# In[143]:


final_df.drop('Duration',axis=1,inplace=True)


# In[148]:


final_df.head(1)


# In[145]:


df['Airline'].unique()


# In[146]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()


# In[147]:


final_df['Airline']=labelencoder.fit_transform(final_df['Airline'])


# In[149]:


final_df['Source']=labelencoder.fit_transform(final_df['Source'])
final_df['Destination']=labelencoder.fit_transform(final_df['Destination'])


# In[150]:


final_df.info()


# In[151]:


final_df.shape


# In[152]:


final_df.head(2)


# In[153]:


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()


# In[155]:


pd.get_dummies(final_df,columns=['Airline','Source','Destination','Additional_Info'])


# In[ ]:




