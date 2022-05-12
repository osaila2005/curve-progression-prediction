#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
 
features = pd.read_excel("Features_AR1.xlsx")
#features=features.round(1)
features.head(10)


# In[2]:


# Use numpy to convert to arrays
import numpy as np
import pickle
# Labels are the values we want to predict
labels = np.array(features['Cobb_Final'])
# Remove the labels from the features
#features= features.drop


# In[3]:


# Use numpy to convert to arrays
import numpy as np
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
# Labels are the values we want to predict
labels = np.array(features['Cobb_Final'])
#cat_Features = np.array(features['Num_Levels_Involved'])
#Lenke_mapping = {4:1,5:2, 6:3,7:4, 8:5,9:6, 10:7}
#features['Lenke'] = features['Lenke'].map(Lenke_mapping)
num_levels_mapping = {4:1,5:2, 6:3,7:4, 8:5,9:6, 10:7}
features['Number of levels involved'] = features['Number of levels involved'].map(num_levels_mapping)
Apex_First_TP_mapping = {6:1,7:2, 8:3,9:4, 10:5,11:6, 12:7, 13:8, 14:9, 15:10}
features['Apex_First_TP'] = features['Apex_First_TP'].map(Apex_First_TP_mapping)
Risser_First_TP_mapping = {-2:1,-1:2, 0:3,1:4, 2:5,3:6, 0.75:7, 4:8, 5:9}
features['Risser "+" stage at first visit'] = features['Risser "+" stage at first visit'].map(Risser_First_TP_mapping)

features['Lenke'] = class_le.fit_transform(features['Lenke'].values)
features['Brace'] = class_le.fit_transform(features['Brace'].values)
features['Gender'] = class_le.fit_transform(features['Gender'].values)
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('Cobb_Final', axis = 1)
#features= features.drop('Gender', axis = 1)
features= features.drop('Brace', axis = 1)
#features= features.drop('Risser_First_TP', axis = 1)
#features= features.drop('Num_Levels_Involved', axis = 1)
features= features.drop('Apex_First_TP', axis = 1)
#features= features.drop('Age_Last', axis = 1)
features= features.drop('Vertebral Wedging', axis = 1)
features= features.drop('Lenke', axis = 1)
features= features.drop('Axial Rotation', axis = 1)
#features= features.drop('Cobb_Initial', axis = 1)
#features= features.drop('Age_First', axis = 1)
#features= features.drop('K', axis = 1)
#features= features.drop('Flexibility', axis = 1)
features= features.drop('Time_Span', axis = 1)
#features= features.drop('L', axis = 1)
features= features.drop('Delta_Cobb', axis = 1)
features= features.drop('Delta_Cobb>=10', axis = 1)
features= features.drop('Final_Cobb>=45', axis = 1)
features= features.drop('ID', axis = 1)


#features=features.drop(labels = ["Time_span","K_first","Flexibility","Gender","L_First","Lenke","Apex_First_TP","num_Levels_involved_First_TP","Risser_first_visit","age_first_visit","Apex_wedge_First_TP","age_last_visit","Brace"], axis=1)
#df=df.drop(labels = ["Time_span","K_first","Flexibility","Gender","L_First","Lenke","Apex_First_TP","num_Levels_involved_First_TP","Risser_first_visit","age_first_visit","Apex_wedge_First_TP","age_last_visit","Brace"], axis=1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
print(feature_list)
print(type(features))

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#train_features = sc.fit_transform(train_features)
#test_features = sc.transform(test_features)


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

rf=RandomForestRegressor(bootstrap= True, max_depth= 30, max_features= 5, min_samples_leaf= 2, min_samples_split= 7,
 n_estimators=291 , random_state = 42, oob_score=True)

# Train the model on training data
scores_training=cross_val_score(rf,train_features, train_labels,cv=10, scoring='neg_mean_absolute_error');
print (scores_training)
print (scores_training.mean())

rf.fit(train_features, train_labels)

# Make pickle file of  my model
pickle.dump(rf,open("Final_Cobb_Progress.pkl","wb"))





