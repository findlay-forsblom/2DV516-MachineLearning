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
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

dataset = pd.read_csv('./Datasets/GPUbenchmark.csv', header=None) # Reading from the csv file

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Task 1
p = X.shape[1]
models = np.ones((X.shape[0], 1))
order = []
features = [0, 1, 2, 3, 4, 5]
traningErrors = []
for k in range(p):
    mse = np.array([])
    X_test = np.copy(models)
    for t in features:
        newFeature = np.copy(X[:,t])
        X_test = np.c_[X_test, newFeature]
        reg = LinearRegression(fit_intercept = False, normalize = True).fit(X_test, y)
        y_pred = reg.predict(X_test)
        X_test = np.delete(X_test, [X_test.shape[1]-1], 1)
        sums = (y_pred - y) ** 2
        sums = (np.sum(sums)) / len(y_pred)
        mse = np.append(mse, sums)
    argmin = mse.argmin()
    order.append(features[argmin])
    traningErrors.append(mse[argmin])
    models = np.c_[models, X[:, features[argmin]]]
    features = np.delete(features, [argmin], 0)

# Task 2
validationErrors = []
for k in range(p):
    X_test = models[:,:k+2]
    reg = LinearRegression(fit_intercept = False, normalize = True)
    prediction = cross_val_predict(reg, X_test, y, cv=3)
    sums = (prediction - y) ** 2
    sums = (np.sum(sums)) / len(prediction)
    validationErrors.append(sums)

validationErrors = np.array(validationErrors)
min = validationErrors.argmin()

validationErrors = np.array(validationErrors)
pos = validationErrors.argmin()

fig, ax = plt.subplots()
ax.plot(list(range(1,7)), traningErrors, '-', label='training sample')
ax.plot(list(range(1,7)), validationErrors, '-', label='validation sample')
ax.axvline(x=list(range(1,7))[pos], linestyle = '--', label = 'best fit')
ax.set_xlabel('Model Complexity (Number of featuress)')
ax.set_ylabel('Mean Squared Error')
ax.legend()
plt.show()

string = 'Task 2\n Best model Y = B0 + '
for k in range(min):
    string += f'B{k+1}*X{order[k]+1} + '

string = string[:-2] +f' \n The most Important feature is X{order[0]+1}'
print(string)

