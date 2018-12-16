# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:01:44 2018

@author: Yadav
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
location='C:\\Users\\Yadav\\Desktop\\energydata_complete.csv'
data=pd.read_csv(location)
#X=data[['T1','RH_1','T2','RH_2','T3','RH_3']]
X=data[['T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9']]
Y=data[['Appliances','lights']]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.34,random_state=20)

'''
X_train=preprocessing.normalize(X_train)
X_test=preprocessing.normalize(X_test)
Y_train=preprocessing.normalize(Y_train)
Y_test=preprocessing.normalize(Y_test)
'''
print("done")
model_1=MultiOutputRegressor(RandomForestRegressor())
model_1.fit(X_train,Y_train)
Y_predict=model_1.predict(X_test)
print(Y_predict)
err_1=metrics.mean_squared_error(Y_test,Y_predict)
print("the rmse value is")
print(np.sqrt(err_1))
print("the r2 value is")
res=r2_score(Y_test,Y_predict)
print(res)
plt.plot(X_test,Y_test)
plt.xlabel('input')
plt.ylabel('output')
plt.show()
plt.plot(X_test,Y_predict)
plt.xlabel('input')
plt.ylabel('output')
plt.show()
