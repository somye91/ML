# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:45:34 2018

@author: Somye
"""

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
class NeuralNet:
    
    def __init__(self,I,H,O):
        self.w_ih=np.random.uniform(size=(I,H))
        self.w_ho=np.random.uniform(size=(H,O))
        
    def sigmoid(self,Z,deriv="false"):
        if deriv=="false":
            return 1/(1+np.exp(-Z))
        else:
            return Z*(1-Z)

    def predict(self,X):
        A_H=X.dot(self.w_ih)
        Z_H=self.sigmoid(A_H)
        A_O=Z_H.dot(self.w_ho)
        Z_O=self.sigmoid(A_O)  #this is y, our prediction
        return Z_O,Z_H
    
    def train(self,X,y,iteration,lr):
        samples=X.shape[0]
        for i in range(iteration):
            
            Z_O,Z_H=self.predict(X)
            error=Z_O - y
            dw_ho = error * self.sigmoid(Z_O,deriv="true")                        # delta Z
            dw_ih = dw_ho.dot(self.w_ho.T) * self.sigmoid(Z_H,deriv="true")             # delta H
            self.w_ho -=  Z_H.T.dot(dw_ho)  *lr/samples                        # update output layer weights
            self.w_ih -=  X.T.dot(dw_ih) *lr/samples
            print("Iteration: ",i,self.w_ho,self.w_ih)
            
        
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([ [0],   [1],   [1],   [1]])
nn=NeuralNet(2,3,1)   #Input, Hidden, Output
iteration=40000
lr=.1
nn.train(X,y,iteration,lr)
print(nn.predict(X)[0])