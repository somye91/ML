# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:27:33 2018

@author: Somye
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LinReg:
    
    def fit(self,df):
        num=0
        den=0
        X_mean=np.average(df.iloc[:,0].values)
        y_mean=np.average(df.iloc[:,1].values)
       
        #X_train=df.iloc[:,0].values
        #y_train=df.iloc[:,1].values
        #for x in range(0,1):
        for x in range(0,700):
                num+=(df.values[x,0]- X_mean)*(df.values[x,1]- y_mean)
                den+=(df.values[x,0]-X_mean)*(df.values[x,0]-X_mean)
                #print(df.values[x,0],df.values[x,1])
        
        self.m=num/den
        self.b=y_mean -self.m*X_mean
        
        
    def predict(self,x):
        return self.m*x +self.b
    
    
dataset_train=pd.read_csv("C:/Users/Somye/Downloads/random-linear-regression/train.csv")
dataset_test=pd.read_csv("C:/Users/Somye/Downloads/random-linear-regression/test.csv")
#X_train=dataset_train.iloc[:,0].values
#y_train=dataset_train.iloc[:,1].values
lr=LinReg()
lr.fit(dataset_train)
vfunc=np.vectorize(lr.predict)
y_pred=vfunc(dataset_train.iloc[:,0].values)
plt.scatter(dataset_train.iloc[:,0],dataset_train.iloc[:,1],color='blue')
plt.plot(dataset_train.iloc[:,0],y_pred,color='red')
