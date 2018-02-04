# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 19:40:35 2018

@author: Somye
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LinReg:
    learning_rate=.0001
    m=0
    b=0
    err=0
    def fit(self,df):
        m_grad=0
        b_grad=0
        
        for x in range(0,df.values.shape[0]):
                y_pred=self.m*df.values[x,0]+self.b
                error= y_pred - df.values[x,1]
                
                """if error>self.err:
                    self.learning_rate=self.learning_rate/10
                else:
                    self.learning_rate+=self.learning_rate*.05
                """
                
                m_grad+=error*df.values[x,0]/df.values.shape[0]
                b_grad+=error/df.values.shape[0]
                
        
        
        self.m=self.m - m_grad * self.learning_rate
        self.b=self.b - b_grad * self.learning_rate
        print(self.m,self.b)           
        
        
    def predict(self,x):
        
        return self.m*x +self.b
    
dataset_train=pd.read_csv("C:/Users/Somye/Downloads/random-linear-regression/train.csv")
dataset_test=pd.read_csv("C:/Users/Somye/Downloads/random-linear-regression/test.csv")

lr=LinReg()


##Call this 100 times for fradient adjustment
for i in range(500):
    lr.fit(dataset_test)

#Vevtorize numpy array to apply predict function to every element. Returns a numpy array    
vfunc=np.vectorize(lr.predict)
pred_y=vfunc(dataset_test.iloc[:,0].values)
plt.scatter(dataset_test.iloc[:,0],dataset_test.iloc[:,1],color='blue')
plt.plot(dataset_test.iloc[:,0],pred_y,color='red')
plt.show()
     
#print(a)