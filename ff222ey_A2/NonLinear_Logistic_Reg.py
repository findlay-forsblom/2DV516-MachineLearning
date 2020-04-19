#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:45:09 2020

@author: findlayforsblom
"""

from Classes.Logistics import LogisticRegression
from Classes.RandomFunctions import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

#Task 1

dataset = pd.read_csv('./Datasets/microchips.csv', header=None) # Reading from the csv file
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

dataset = dataset.to_numpy()
np.random.shuffle(dataset)

X = dataset[:, :-1]
y = dataset[:, -1]

fig, ax = plt.subplots()
cdict = {0: 'red', 1: 'green'}
type = {0: 'Chip not Ok', 1:'Chip Ok'}
for g in np.unique(y):
    ix = np.where(y == g)
    ax.scatter(X[ix,0], X[ix,1], c = cdict[g], label = type[g], s = 10)
ax.legend()

# Task 2
degree = 2
X_Quad = mapFeatures(X[:,0], X[:,1], degree)
allocation = 0.8
split = int(allocation * dataset.shape[0])

X_train, y_train = X_Quad[:split], y[:split]
X_test, y_test = X_Quad[split:], y[split:]

itera = 10000
alf = 0.5
 
clf = LogisticRegression(alpha = alf, iterations = itera, normalize = False, extend = False)
clf.fit(X_train, y_train)
mse = clf.getCost()

x1, x2, xx, yy = computeMesh(X[:, 0], X[:, 1], 0.01 )
XXe = mapFeatures(x1,x2,degree)

p = clf.predict(XXe)

classes = p > 0.5
clzmesh = classes.reshape(xx.shape)


fig1, ax1 = plt.subplots(1, 2)
ax1[0].plot(list(range(itera)), mse, '-')
ax1[0].set_xlabel('iterations')
ax1[0].set_ylabel('J(B)')
ax1[0].set_title(f'N = {itera} and Alpha = {alf} ')

errors = clf.getTrainingErrors()


ax1[1].pcolormesh(xx, yy, clzmesh, cmap=cmap_light)
ax1[1].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
ax1[1].set_title(f'Training errors {errors}')


# Task 4
degree = 5
X_Quad = mapFeatures(X[:,0], X[:,1], degree)
allocation = 0.8
split = int(allocation * dataset.shape[0])

X_train, y_train = X_Quad[:split], y[:split]


clf = LogisticRegression(alpha = alf, iterations = itera, normalize = False, extend = False)
clf.fit(X_train, y_train)
mse = clf.getCost()

x1, x2, xx, yy = computeMesh(X[:, 0], X[:, 1], 0.01 )
XXe = mapFeatures(x1,x2,degree)

p = clf.predict(XXe)

classes = p > 0.5
clzmesh = classes.reshape(xx.shape)

fig2, ax2 = plt.subplots(1, 2)
ax2[0].plot(list(range(itera)), mse, '-')
ax2[0].set_xlabel('iterations')
ax2[0].set_ylabel('J(B)')
ax2[0].set_title(f'N = {itera} and Alpha = {alf} ')

errors = clf.getTrainingErrors()


ax2[1].pcolormesh(xx, yy, clzmesh, cmap=cmap_light)
ax2[1].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
ax2[1].set_title(f'Training errors {errors}')
plt.show()