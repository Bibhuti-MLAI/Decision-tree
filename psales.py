# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 21:27:08 2021

@author: Bibhuti
"""

import pandas as pd
import numpy as np


#loading the dataset
df = pd.read_csv("C:/Users/Bibhuti/OneDrive/Desktop/360digiTMG assignment/Decision Tree/Datasets_DTRF/Company_Data.csv")


d_types =["ratio","nominal","ratio","ratio","ratio","ratio","oridanal","ratio","oridanal","binary","binary"]

data_details =pd.DataFrame({"column name":df.columns,
                            "data types ":d_types,
                            "data types-p":df.dtypes})
      
df.dtypes


#checking for na value
df.isna().sum()
df.isnull().sum()

df.nunique()

#variance of df
df.var()

#checking unique value in Advertising, column with count
df['Advertising'].unique()
capitalgain_uni_count = pd.value_counts(df['Advertising'])
capitalgain_uni_count


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


#boxplot for every columns
df.columns
df.nunique()

boxplot = df.boxplot(column=["Sales","CompPrice","Income","Advertising","Population","Price","Age"])   



from sklearn.preprocessing import LabelEncoder
#converting into binary
lb = LabelEncoder()

df["ShelveLoc"] = lb.fit_transform(df["ShelveLoc"])
df["Urban"] = lb.fit_transform(df["Urban"])
df["US"] = lb.fit_transform(df["US"])
df.dtypes

colnames = list(df.columns)

predictors = colnames[1:11]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2,random_state=7)

# import the regressor 
from sklearn.tree import DecisionTreeRegressor  
  
# create a regressor object 
model = DecisionTreeRegressor(random_state = 0)  
  
# fit the regressor with X and Y data 
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])

 # Test Data Accuracy 
np.sqrt(((preds - test[target]) ** 2).mean())

# Prediction on Train Data
preds_ = model.predict(train[predictors])

 # Train Data Accuracy
np.sqrt(((preds_ - train[target]) ** 2).mean())

#model is over fitting so we are building random forest

###########  Random forest 



from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=500, n_jobs=1, random_state=42)

rf_reg.fit(train[predictors], train[target])


# Prediction on Test Data
preds = rf_reg.predict(test[predictors])

 # Test Data Accuracy 
np.sqrt(((preds - test[target]) ** 2).mean())
