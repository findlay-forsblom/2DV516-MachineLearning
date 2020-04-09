#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:30:42 2020

@author: findlayforsblom

Exercise 4
"""

# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Importing the dataset
dataset = pd.read_csv('microchips.csv')

thisdict = {
  1: "OK",
  0: "Fail"
}

X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 2].values

x1 = dataset.iloc[:56, [0]].values #ok
x2 = dataset.iloc[:56, [1]].values #ok
x3 = dataset.iloc[56:, [0]].values #not ok
x4 = dataset.iloc[56:, [1]].values #not ok

X_test = [[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]]
X_test = np.array(X_test).reshape(3,2)

fig, ax = plt.subplots()
ok = ax.scatter(x1,x2,color='green')
notOk = ax.scatter(x3,x4,color='red')
ax.legend((ok, notOk), ('Chip OK', 'Chip Failed') )
ax.set_title('Plot of orignal data')

numOfRows = X_test.shape[0]
ks = [1,3,5,7]
for k in ks:
    classifier = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
    classifier.fit(X,y)
    print(f'k = {k}')
    counter = 1 
    ypred = classifier.predict(X_test)
     
    for row in range(numOfRows):
        Xtest= X_test[row]
        print(f'\t chip{counter}: {Xtest} ==> {thisdict[ypred[row]]}')
        counter +=1

h = 0.05
x_min, x_max = X[:, 0].min()-0.2, X[:, 0].max()+0.2
y_min, y_max = X[:, 1].min()-0.2, X[:, 1].max()+0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h)) # Mesh Grid
xy_mesh = np.c_[xx.ravel(), yy.ravel()] # Turn to Nx2 matrix

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])


count = 0
fig1, ax1 = plt.subplots(nrows=2, ncols=2)
for row in ax1:
    for col in row:
        classifier = KNeighborsClassifier(ks[count])
        classifier.fit(X,y)
        clzmesh = classifier.predict(xy_mesh)
        ypred = classifier.predict(X)
        clzmesh = clzmesh.reshape(xx.shape)
        
        errors = 0
        
        for value in range(len(ypred)):
            if ypred[value] != y[value]:
                errors +=1

        col.pcolormesh(xx, yy, clzmesh, cmap=cmap_light)
        col.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        col.set_title(f'k = {ks[count]}, traning errors = {errors} ')
        count +=1

plt.tight_layout()

plt.show()
