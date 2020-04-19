#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 07:28:40 2020

@author: findlayforsblom

Exercise 6
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

dataset = pd.read_csv('./Datasets/GPUbenchmark.csv', header=None) # Reading from the csv file

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Task 1
p = X.shape[1]
models = np.ones((X.shape[0], 1))
features = np.copy(X)
for k in range(p):
    mse = np.array([])
    X_test = np.copy(models)
    for t in range(p -k):
        #print(' ')
        X_test = np.c_[X_test, features[:,t]]
        #print(X_test)
        reg = LinearRegression().fit(X_test, y)
        y_pred = reg.predict(X_test)
        X_test = np.delete(X_test, [X_test.shape[1]-1], 1)
        sums = (y_pred - y) ** 2
        sums = (np.sum(sums)) / len(y_pred)
        mse = np.append(mse, sums)
    print(mse)
    argmin = mse.argmin()
    models = np.c_[models, features[:, (argmin)]]
    features = np.delete(features, [argmin], 1)

# Task 2
validationErrors = []
for k in range(p):
    X_test = models[:,:k+2]
    reg = LinearRegression().fit(X_test, y)
    prediction = cross_val_predict(reg, X_test, y, cv=3)
    sums = (prediction - y) ** 2
    sums = (np.sum(sums)) / len(prediction)
    validationErrors.append(sums)

validationErrors = np.array(validationErrors)
min = validationErrors.argmin()
print(min)

bestModel = models[:,: min+2]