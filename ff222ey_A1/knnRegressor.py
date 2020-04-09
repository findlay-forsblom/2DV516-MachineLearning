#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:09:55 2020

@author: findlayforsblom
"""

import numpy as np

class KNeighborsRegressor:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, Xtest):
        ypred = []
        lenght = len(Xtest)
        X = self.X
        y = self.y
        for value in range(lenght):
            idx = np.argpartition((np.abs(X- Xtest[value])), self.n_neighbors)[:self.n_neighbors] # finds the indices k nearest neighbours
            ypred.append(np.sum(y[idx]) / len(idx)) #sums the values of the k nearest nieggbors
        ypred = np.array(ypred)
        return ypred
        
    def getTraningerrors(self):
        X = np.copy(self.X)
        y = self.y
        ypred = self.predict(X)
        sums = (ypred - y) ** 2
        return round((np.sum(sums)) / len(ypred), 2)   
        
    