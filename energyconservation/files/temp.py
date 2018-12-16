# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
location='C:\\Users\\Yadav\\Desktop\\energydata_complete.csv'

#print("please enter url of file")
#string=input("please enter full path of file and put double backslashes:")
X1=pd.read_csv(location)
#print(X1)
#splitting test and train data
Y=X1[['Appliances','lights']]
#print(X)
#X=X1[['T1','RH_1','T2','RH_2','T3','RH_3']]
X=X1[['T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9']]
#print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.34,random_state=20)
'''
X_train=preprocessing.normalize(X_train)
X_test=preprocessing.normalize(X_test)
Y_train=preprocessing.normalize(Y_train)
Y_test=preprocessing.normalize(Y_test)
'''
#creating and fitting model
model=LinearRegression(fit_intercept=True)
model.fit(X_train,Y_train)
#predicting output
    #model.coef_ this is for coefficient of attributes in linear regression
    #model.intercept_ this is for intercept value
Y_predict=model.predict(X_test)
# calculate root absolute percentage error 
'''
result1=np.mean(np.abs((Y_predict-Y_test)/Y_test))*100
print("the root absolute percent error is") 
print(result1)
'''
result=metrics.mean_squared_error(Y_test,Y_predict)
print("the rmse value is")
print(np.sqrt(result))
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
