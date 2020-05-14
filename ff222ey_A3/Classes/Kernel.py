#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:05:04 2020

@author: findlayforsblom
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class AnovaKernel(BaseEstimator,TransformerMixin):
    def __init__(self, sigma = 1.0, d = 1):
        super(AnovaKernel,self).__init__()
        self.sigma = sigma
        self.d = d

    def transform(self, X):
        return self.computeGram(X, self.X_train_, sig = self.sigma, d = self.d)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self

    def anova(self, xi, xj, sig, d):
        return np.sum(((xi-xj) ** 2) * (-sig)) ** d
    
    def computeGram(self, X,Y,sig, d):
        n = X.shape[0]
        m = Y.shape[0]
        
        gram = np.zeros([n, m])
        for i in range(n):
            for j in range(m):
                gram[i,j] = self.anova(X[i], Y[j], sig, d)
        return gram