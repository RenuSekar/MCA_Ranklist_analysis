#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[19]:


#Saravana
rank_list = pd.read_csv('/kaggle/input/mca-ranklist-analysis/cleaned_rank_list.csv')
provisional_list = pd.read_csv('/kaggle/input/mca-ranklist-analysis/cleaned_prov_allot.csv')


# In[3]:


rank_list_df = pd.DataFrame(rank_list)
provisonal_df = pd.DataFrame(provisional_list)


# In[4]:


rank_list_df.head


# In[5]:


provisonal_df.head


# In[6]:


#I dont know what we can derive from this datasets, I just wanted to try how linear regressions and some other regressions works. 
#If you have any good idea with this, go on I'm in.


# In[7]:


condition_value = 'OC'
new_value = 0
rank_list_df.loc[rank_list_df['COMMUNITY']==condition_value, 'COMMUNITY_RANK'] = new_value


# In[8]:


X = rank_list_df[['TANCET_MARK', 'COMMUNITY']]
y = rank_list_df['COMMUNITY_RANK']


# In[9]:


X = pd.get_dummies(X, columns=['COMMUNITY'])


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[12]:


model = RandomForestClassifier()
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


# In[13]:


best_model = grid_search.best_estimator_


# In[14]:


y_pred = best_model.predict(X_test)


# In[15]:


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))


# In[16]:


def predict_new_data(new_data):
    """
    Predict the community rank for new input data.
    
    Parameters:
    new_data (pd.DataFrame): DataFrame containing new input data with the same structure as the original data.
    
    Returns:
    np.array: Predictions for the new input data.
    """
    # One-hot encode the new data
    new_data = pd.get_dummies(new_data, columns=['COMMUNITY'])
    
    # Align the new data with the columns of the training data
    new_data = new_data.reindex(columns=X.columns, fill_value=0)
    
    # Scale the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Predict using the best model
    predictions = best_model.predict(new_data_scaled)
    
    return predictions


# In[17]:


new_input_data = pd.DataFrame({
    'TANCET_MARK': [78.5, 85.0],
    'COMMUNITY': ['BC', 'MBC']
})


# In[18]:


predictions = predict_new_data(new_input_data)
print(predictions)


# In[ ]:




