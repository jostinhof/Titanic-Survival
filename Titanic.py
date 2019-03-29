# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:56:38 2019

@author: Jostin Joseph
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset= pd.read_csv('https://s3.eu-geo.objectstorage.softlayer.net/ml-datasetstore/titanic_ship.csv')
 
titanic=dataset[['Age','Sex','Pclass','Parch','SibSp','Fare','Survived']]

X=titanic[['Age','Sex','Pclass','Parch','SibSp','Fare']].values

y=titanic[['Survived']].values

#Cleaning the data

from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

imputer.fit(X[:, 0:1])

X[:,0:1]=imputer.fit_transform(X[:,0:1])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder=LabelEncoder()

labelencoder.fit(X[:,1])
X[:,1]=labelencoder.transform(X[:,1])



from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

#Splitting the data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=42)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
classi=LogisticRegression(random_state=0)
classi.fit(X_train,y_train)


y_pred=classi.predict(X_test)

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test,y_pred)