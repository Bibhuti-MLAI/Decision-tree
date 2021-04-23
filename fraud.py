# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 21:40:58 2021

@author: Bibhuti
"""

import pandas as pd
import numpy as np
#loading the dataset
df = pd.read_csv("C:/Users/Bibhuti/OneDrive/Desktop/360digiTMG assignment/Decision Tree/Datasets_DTRF/Fraud_check.csv")

d_types =["binary","nominl","ratio","ratio","ratio","binary"]

data_details =pd.DataFrame({"column name":df.columns,
                            "data types ":d_types,
                            "data types-p":df.dtypes})
#details of df 
df.info()
df.describe()

#data types        
df.dtypes
#checking for na value
df.isna().sum()
df.isnull().sum()
#checking unique value for each columns
df.nunique()
df.Undergrad.value_counts()

#variance of df
df.var()

from sklearn.preprocessing import LabelEncoder
#converting into binary
lb = LabelEncoder()

df["Undergrad"] = lb.fit_transform(df["Undergrad"])
df["Marital.Status"] = lb.fit_transform(df["Marital.Status"])
df["Urban"] = lb.fit_transform(df["Urban"])

df.dtypes

EDA ={"column ": df.columns,
      "mean": df.mean(),
      "median":df.median(),
      "mode":df.mode(),
      "standard deviation": df.std(),
      "variance":df.var(),
      "skewness":df.skew(),
      "kurtosis":df.kurt()}

EDA

# covariance for data set 
covariance = df.cov()
covariance


####### graphical repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(df.iloc[:, :])

#treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
df['income'] = df['Taxable.Income'].apply(lambda x: 'Good' if x <= 30000 else 'Risky')
df.nunique()
df.income.value_counts()

colnames = list(df.columns)

predictors = colnames[0:2]+colnames[3:6]
target = colnames[6]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2,random_state=7)

"""
5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Decision Tree and Random Forest on the given datasets.
5.3	Train and Test the data and perform cross validation techniques, compare accuracies, precision and recall and explain about them.
5.4	Briefly explain the model output in the documentation. """


from sklearn.tree import DecisionTreeClassifier as DT


model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

#model is over fitting so we are building random forest

###########  Random forest 

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(train[predictors], train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))


######
# GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

param_grid = {"max_features": [2,3,4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10,15,17,21,22]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(train[predictors], train[target])

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(test[target], cv_rf_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_rf_clf_grid.predict(test[predictors]))
