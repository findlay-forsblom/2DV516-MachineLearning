#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:29:24 2020

@author: findlayforsblom
"""
import pandas as pd
import numpy as np

def computeMesh(X,Y,h):
    x_min, x_max = X.min()-0.2, X.max()+0.2
    y_min, y_max = Y.min()-0.2, Y.max()+0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h)) # Mesh Grid
    x1,x2 = xx.ravel(), yy.ravel()
    
    return (x1 ,x2, xx, yy)

def anova(xi, xj, sig, d):
    return np.sum(((xi-xj) ** 2) * (-sig)) ** d
    
def computeGram(X,Y,sig, d):
    n = X.shape[0]
    m = Y.shape[0]
    
    gram = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            gram[i,j] = anova(X[i], Y[j], sig, d)
    return gram