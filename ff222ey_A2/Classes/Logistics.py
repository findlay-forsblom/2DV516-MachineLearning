#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:14:57 2020

@author: findlayforsblom
"""

import numpy as np

class LogisticRegression:
    def __init__(self, alpha, iterations, normalize = True, extend = True):
        self.alpha = alpha
        self.iterations = iterations
        self.normalize = normalize
        self.extend = extend
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        if (self.normalize):
             self.u = np.mean(X, axis=0)
             self.std = np.std(X, axis=0)
             self.Xe = self.__featureScaling__(X)
    
             if self.extend:
                 self.Xne = self.__extendArray__(self.Xe)
             else:
                 self.Xne = self.Xe
                 
        else:
             if self.extend:
                 self.Xne = self.__extendArray__(self.X)
             else:
                 self.Xne = self.X
                 #print(self.Xne)
        
        self.__computeBeta__()
    
    def __extendArray__(self,X):
        N = X.shape[0]
        #X = self.__featureScaling__(X)
        return np.c_[np.ones(N), X]
    
    def __featureScaling__(self, X):
        return (X - self.u) / self.std
    
    def predict(self,X):
        if self.normalize:
             Xe = self.__featureScaling__(X)
             if self.extend:
                 Xne = self.__extendArray__(Xe)
             else:
                 Xne = Xe
        else:
            if self.extend:
                 Xne = self.__extendArray__(X)
            else:
                 Xne = X
            
        theta = self.theta
        return self.__sigmoidFunction__(np.dot(Xne, theta))
        
    
    def __sigmoidFunction__(self, X):
        return 1/(1 + np.exp(-X))
    
    def getTrainingErrors(self):
        y = self.y
        trainingerrors = self.__sigmoidFunction__(np.dot(self.Xne, self.theta))
        trainingerrors = trainingerrors > 0.5
        y_test = y == 1
        errors = y_test[y_test !=trainingerrors]
        errors = len(errors)
        return errors
        
    
    def __costFunction__(self, Xne, theta):
        y = self.y
        #ypred = predict(Xne, X)
        N = Xne.shape[0]
        j = y.T.dot(np.log((self.__sigmoidFunction__(np.dot(Xne, theta)))))
        return -1/N * (j + (1-y).T.dot(np.log(1-(self.__sigmoidFunction__(np.dot(Xne, theta))))))
    
    def getCost(self):
        return self.cost
    
    def getBetas(self):
        return self.theta
    
    def __computeBeta__(self):
        eta = self.alpha
        iterations = self.iterations
        y = self.y
        
        Xne = self.Xne
        
        #print(Xne)
    
        N = Xne.shape[0]
    
        theta = np.random.randn(Xne.shape[1])
        #theta = [0,0,0,0,0,0,0,0,0,0]
        MSE = []
        for iteration in range(iterations):
            gradients = Xne.T.dot((self.__sigmoidFunction__(np.dot(Xne, theta)) - y))
            theta = theta - ((eta * gradients )/N)
            #print(theta)
            J = self.__costFunction__(Xne, theta)
            MSE.append(J)
        self.theta = theta
        self.cost = MSE