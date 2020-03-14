#!/usr/bin/env python
# coding: utf-8

# # Pratical Machine Learning 
# ## Fátima Vilela Candal

# # Data processing
# The original training and test data has 160 variables. The columns with NA entries have been removed. Five (5) variables were removed.

# In[498]:


# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math

from sklearn.model_selection import train_test_split 

from sklearn.neighbors import KNeighborsClassifier  

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# In[499]:


# Importing the Dataset 
df = pd.read_csv('pml-training.csv')


# In[500]:


# Clear all null data
df.dropna(inplace=True)


# In[501]:


# Total rows and columns 
print("Train data line and colum: {}".format(df.shape))


# In[502]:


# Columns present in the dataset 
print(df.columns) 


# In[503]:


# Data types
df.dtypes


# In[504]:


# Presentation of null data 
df.isnull().sum()


# In[528]:


# Data presented 20 columns and 05 rows
df.head(5)


# In[506]:


# Descriptive statistics for each column
df.describe()


# # Train Test Split
# We will divide our dataset into training and test splits, which gives us a better idea as to how our algorithm performed during the testing phase. This way our algorithm is tested on un-seen data, as it would be in a production application.

# In[551]:


# Preprocessing
# The next step is to split our dataset into its attributes and labels

cols = ['raw_timestamp_part_1',
'raw_timestamp_part_2',
'num_window',
'roll_belt',
'pitch_belt',
'yaw_belt',
'gyros_forearm_x',
'gyros_forearm_y',
'gyros_forearm_z',
'accel_forearm_x',
'accel_forearm_y',
'accel_forearm_z',
'magnet_forearm_x',
'magnet_forearm_y',
'magnet_forearm_z']


X = df[cols]
y = df.classe

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  


# In[542]:


print (X_train.shape)


# In[543]:


print (X_test.shape)


# #  Model
# The first step is to import the KNeighborsClassifier class from the sklearn.neighbors library. In the second line, this class is initialized with one parameter, i.e. n_neigbours. This is basically the value for the K. There is no ideal value for K and it is selected after testing and evaluation, however to start out, 5 seems to be the most commonly used value for KNN algorithm.
# 
# After all the work of data preparation, creating and training the model KNN regression model from skicit-learn, instantiate the model, and fit  the model on the training data. 

# # Predictions
# It is extremely straight forward to train the KNN algorithm and make predictions with it, especially when using Scikit-Learn.

# In[544]:


# from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=20)  
classifier.fit(X_train, y_train)  


# In[545]:


# Prediction
y_pred = classifier.predict(X_test) 
y_pred 


# # Evaluating the Algorithm
# For evaluating an algorithm, confusion matrix, precision, recall and f1 score are the most commonly used metrics. The confusion_matrix and classification_report methods of the sklearn.metrics can be used to calculate these metrics. 

# In[547]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # Comparing Error Rate with the K Value
# In the training and prediction section we said that there is no way to know beforehand which value of K that yields the best results in the first go. 
# 
# We randomly chose 5 as the K value and it just happen to result in 100% accuracy.
# 
# One way find the best value of K is to plot the graph of K value and the corresponding error rate for the dataset.
# 
# We will plot the mean error for the predicted values of test set for all the K values between 1 and 40.

# In[548]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[549]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# # Conclusion
# KNN is a simple yet powerful classification algorithm. It requires no training for making predictions, which is typically one of the most difficult parts of a machine learning algorithm. The KNN algorithm have been widely used to find document similarity and pattern recognition. It has also been employed for developing recommender systems and for dimensionality reduction and pre-processing steps for computer vision, particularly face recognition tasks.
# 
# 
