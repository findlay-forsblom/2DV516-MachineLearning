#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:21:15 2020

@author: findlayforsblom
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('../Datasets/artificial.csv', delimiter=',', header = None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

clf = DecisionTreeClassifier(random_state=0, max_depth = 3)

clf.fit(X_train,y_train)
clf.score(X_train, y_train)
y_pred = clf.predict(X_train)
print ('accuracy score: %0.3f' % accuracy_score(y_train, y_pred))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 5)
accuracies.mean()
accuracies.std()

h = 0.05
x_min, x_max = X[:, 0].min()-0.2, X[:, 0].max()+0.2
y_min, y_max = X[:, 1].min()-0.2, X[:, 1].max()+0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h)) # Mesh Grid
xy_mesh = np.c_[xx.ravel(), yy.ravel()] # Turn to Nx2 matrix

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
#AAFFAA
clzmesh = clf.predict(xy_mesh)
clzmesh = clzmesh.reshape(xx.shape)

fig, ax = plt.subplots()
ax.pcolormesh(xx, yy, clzmesh, cmap=cmap_light)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s= 3)
plt.show()