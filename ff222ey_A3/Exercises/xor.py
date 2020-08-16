# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:26:31 2020

@author: findl
"""


import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([
    0,0,0,
    0,0,1,
    0,1,0,
    0,1,1,
    1,0,0,
    1,0,1,
    1,1,0,
    1,1,1,
]).reshape(8,3)


x = np.array([1, 1])

y = np.array([0,0,0,1,0,1,1,0])

clf = MLPClassifier(activation = 'logistic', solver = 'lbfgs', hidden_layer_sizes = [2], random_state = 10)
clf.fit(X,y)
clf.predict(X)
weights = clf.coefs_
clf.intercepts_

w1 = weights[0]
w2 = weights[1]
b1 = clf.intercepts_[0]
b2 = clf.intercepts_[1]


np.dot(w1.T, x) + b1

x2 = np.array([0,1])

np.dot(w2.T, x2) + b2



clf.predict(np.array([[1,0]]))




x = np.array([1, 1, 1])


w1_test = np.round(w1, decimals = 2).T
w2_test = np.round(w2, decimals = 2).T
b1_test = np.round(b1, decimals = 2)
b2_test = np.round(b2, decimals = 2)

np.dot(w1_test, x) + b1_test

x2 = np.array([0,0])

np.dot(w2_test, x2) + b2_test

