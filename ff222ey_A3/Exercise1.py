#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 07:33:27 2020

@author: findlayforsblom
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from Classes.functions import  *
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#ffffb3'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#e6e600'])

dataset = pd.read_csv('./Datasets/mnistsub.csv', delimiter=',', header = None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 101)

# Using gridsearch to find the best regularizattion parameter for the linear svc
clf = svm.SVC()
parameters = [{'kernel': ['poly'], 'C': [0.5, 1, 10, 5, 7, 8, 9], 
               'gamma': [ 0.5, 1, 10,'auto', 'scale'],
               'degree': [1, 2, 3]}]

grid_search = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_

"""
clf = svm.SVC(kernel = 'linear', C = 0.8)
clf.fit(X_train, y_train)

#computing the decision boundary
x1, x2, xx, yy = computeMesh(X_train[:,0], X_train[:,1], 0.02)
xy_mesh = np.c_[x1, x2] # Turn to Nx2 matrix
clzmesh = clf.predict(xy_mesh)
clzmesh = clzmesh.reshape(xx.shape)

fig, ax = plt.subplots()
ax.pcolormesh(xx, yy, clzmesh, cmap=cmap_light,linewidth=10)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, s =15)
"""

"""
clf = svm.SVC(kernel = 'rbf', C = 9, gamma = 'auto')
clf.fit(X_train, y_train)

#computing the decision boundary
x1, x2, xx, yy = computeMesh(X_train[:,0], X_train[:,1], 0.02)
xy_mesh = np.c_[x1, x2] # Turn to Nx2 matrix
clzmesh = clf.predict(xy_mesh)
clzmesh = clzmesh.reshape(xx.shape)

fig, ax = plt.subplots()
ax.pcolormesh(xx, yy, clzmesh, cmap=cmap_light,linewidth=10)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, s =15)
"""

clf = svm.SVC(kernel = 'poly', C = 0.5, degree = 1, gamma = 0.5)
clf.fit(X_train, y_train)

#computing the decision boundary
x1, x2, xx, yy = computeMesh(X_train[:,0], X_train[:,1], 0.02)
xy_mesh = np.c_[x1, x2] # Turn to Nx2 matrix
clzmesh = clf.predict(xy_mesh)
clzmesh = clzmesh.reshape(xx.shape)

fig, ax = plt.subplots()
ax.pcolormesh(xx, yy, clzmesh, cmap=cmap_light,linewidth=10)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, s =15)
