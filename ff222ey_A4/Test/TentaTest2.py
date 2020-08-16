# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:32:49 2020

@author: findl
"""

import numpy as np
from sklearn import preprocessing

X = np.array([
    [2.1, 5, 150],
    [1.2, 6, 170],
    [4.4, 4, 130],
    [5.4, 4, 100],
    [12.1, 7, 208],
    [5, 5, 140],
    [6.3, 5, 145],
    [0.3, 5, 160],
    [0.7, 4, 125],
    [5.5, 4, 135],
    [3.2, 4, 141],
    [11.1, 5, 98],
    ])

y = np.array([4.3, 7.3, 2.2, 1.9, 10.1, 3, 3, 6.6, 8, 4.2, 4.1, 8.3])

x_test = np.array([4,5, 150])

final = np.abs(x_test - X)
distance = np.sum(final, axis = 1)

sort_index = np.argsort(distance) [0:3]

closest = y[sort_index]

np.mean(closest)


def __featureScaling__(X, u, std):
        return (X - u) / std


u = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_normalized = __featureScaling__(X, u, std)
X_normalized[:,[1,2]]= X[:,[1,2]]

x_test_normalized = __featureScaling__(np.array([x_test]), u, std)
x_test_normalized[:,[1,2]]= np.array([x_test])[:,[1,2]].flatten()

x_test_normalized = x_test_normalized.flatten()




# X_normalized[:,0] = preprocessing.normalize(np.array([X[:,0]]), norm='l2')
final = np.abs(x_test_normalized - X_normalized)
distance_normalized = np.sum(final, axis = 1)

sort_index_normalized = np.argsort(distance_normalized) [0:3]

closest = y[sort_index_normalized]

np.mean(closest)

u = np.mean(X, axis=0)
std = np.std(X, axis=0)
Xe = __featureScaling__(X, u, std)
