#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:47:05 2020

@author: findlayforsblom
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def mapFeatures(X1,X2,D, Ones = True):
    one = np.ones([len(X1),1])
    if Ones:
        Xe = np.c_[one,X1,X2] # Start with [1,X1,X2]
    else:
        Xe = np.c_[X1, X2]
    for i in range(2,D+1):
        for j in range(0,i+1):
            Xnew = X1**(i-j)*X2**j # type (N)
            Xnew = Xnew.reshape(-1,1) # type (N,1) required by append
            Xe = np.append(Xe,Xnew,1) # axis = 1 ==> append column
    return Xe

def computeMesh(X,Y,h):
    x_min, x_max = X.min()-0.2, X.max()+0.2
    y_min, y_max = Y.min()-0.2, Y.max()+0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h)) # Mesh Grid
    x1,x2 = xx.ravel(), yy.ravel()
    
    return (x1 ,x2, xx, yy)

def logisticRegSklearn(Xe, y, C=1000.0, tol=1e-6, max_iter=1000):
    logreg = LogisticRegression(solver='lbfgs', C=C, tol=tol, max_iter= max_iter)
    logreg.fit(Xe, y)  # fit the model with data
    y_pred = logreg.predict(Xe)  # predict
    errors = np.sum(y_pred != y)  # compare y with y_pred
    return logreg, errors

def degree(x,n):
    arr = np.c_[np.ones((x.shape[0],n))]
    for i in range(n):
        arr[:,i] = np.power(x.flatten(),(i+1))
    return arr