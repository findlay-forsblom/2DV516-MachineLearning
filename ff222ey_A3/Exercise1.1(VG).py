#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:18:43 2020

@author: findlayforsblom
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from Classes.functions import  *
from sklearn.pipeline import Pipeline

from Classes.Kernel import AnovaKernel

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#ffffb3'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#e6e600'])

dataset = pd.read_csv('./Datasets/mnistsub.csv', delimiter=',', header = None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 101)

pipe = Pipeline([
        ('anova', AnovaKernel()),
        ('svm', svm.SVC()),
    ])

# Set the parameter 'gamma' of our custom kernel by
# using the 'estimator__param' syntax.
cv_params = dict([
    ('anova__sigma', [0.5,1.5,3]),
    ('svm__kernel', ['precomputed']),
    ('anova__d', [2,1,3]),
    ('svm__C', [0.5, 1,2.5]),
])

kernel_X_train = computeGram(X_train, X_train, 1,1)
kernel_X_test = computeGram(X_test, X_train, 1, 1)

# Do grid search to get the best parameters.
model = GridSearchCV(pipe, cv_params, cv=5, verbose=1, n_jobs=-1)
model.fit(X_train, y_train)
best_score = model.best_score_
best_parameters = model.best_params_

#use best paramters on test ser
svc = svm.SVC(kernel='precomputed', C= 1)
svc.fit(kernel_X_train, y_train)
y_pred = svc.predict(kernel_X_test)
print ('accuracy score: %0.3f' % accuracy_score(y_test, y_pred))

#computing the decision boundary
x1, x2, xx, yy = computeMesh(X_train[:,0], X_train[:,1], 0.05)
xy_mesh = np.c_[x1, x2] # Turn to Nx2 matrix
clzmesh = svc.predict(computeGram(xy_mesh, X_train, 1, 1))
clzmesh = clzmesh.reshape(xx.shape)

fig, ax = plt.subplots()
ax.pcolormesh(xx, yy, clzmesh, cmap=cmap_light,linewidth=10)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, s =15)
plt.show()

