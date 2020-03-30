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
        for value in range(lenght):
            sum = 0
            temp = np.copy(self.X)
            yTemp = np.copy(self.y)
            for num in range(self.n_neighbors):
                idx = (np.abs(temp - Xtest[value])).argmin()
                yvalue = yTemp[idx]
                sum += yvalue
                temp = np.delete(temp, idx)
                yTemp = np.delete(yTemp, idx)
            ypred.append(sum / self.n_neighbors)
        ypred = np.array(ypred)
        return ypred
    
    def getTraningerrors(self):
        X = np.copy(self.X)
        y = self.y
        ypred = self.predict(X)
        sums = (ypred - y) ** 2
        return round((np.sum(sums)) / len(ypred), 2)   
        
    