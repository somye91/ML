# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:30:36 2018

@author: Somye
"""



##Iris dataset
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
class LogReg:
    
    def predict(self,X,weights):
        
        z=np.dot(X,weights)
        return self.sigmoid(z)
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def fit(self,X,y,weights,iteration,lr):
        self.weights=weights
        
        self.costhist=[]
        for i in range(iteration):
            pred=self.predict(X,weights)
            error=pred -y
            self.weights-= (lr/X.shape[0] ) * np.dot(X.T,error)
            cost=self.costfunc(X,y,weights)
            self.costhist.append([i,cost])      
            
    def costfunc(self,X,y,weights):
        pred=self.predict(X,weights)
        c1=y*np.log(pred)
        c2=(1-y)*np.log(1-pred)
        c=-(c1+c2)
        cost=c.sum()/y.shape[0]
        return cost
        
    def decision_boundary(self,prob):
        return 1 if prob >= .5 else 0

    def classify(self,preds):
        '''
        input  - N element array of predictions between 0 and 1
        output - N element array of 0s (False) and 1s (True)
        '''
        decision_boundar = np.vectorize(self.decision_boundary)
        return decision_boundar(preds).flatten()


        
        
        
        
data = datasets.load_iris()
X = data.data[:100, :2]
y = data.target[:100]
X_full = data.data[:100, :]
# 1 to 50 are 0   Setosa
# 51 to 100 are 1   Versicolor

##ou can easily incorporate intercept by adding a colum of ones into your X:
##Bias
X = np.hstack([np.ones([X.shape[0],1]), X])
weights=np.zeros(X.shape[1]) 
logreg=LogReg()
iteration=100000
lr=.001
logreg.fit(X,y,weights,iteration,lr)

print(logreg.classify(logreg.predict(X,logreg.weights)))
a=np.array(logreg.costhist)
plt.plot(a[:,0],a[:,1])  #Cost function plot

    
