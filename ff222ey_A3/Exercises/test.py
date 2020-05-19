#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:09:41 2020

@author: findlayforsblom
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from random import choice
from sklearn.metrics import accuracy_score

dataset_train = pd.read_csv('./Datasets/fashion-mnist_train.csv', delimiter=',', header = None)
dataset_test = pd.read_csv('./Datasets/fashion-mnist_test.csv', delimiter=',', header = None)

dataset_train = dataset_train.to_numpy()
dataset_test = dataset_test.to_numpy()

np.random.shuffle(dataset_train)
np.random.shuffle(dataset_test)

X_train = dataset_train[:, 1:]
y_train = dataset_train[:, 0]

X_test = dataset_test[:, 1:]
y_test = dataset_test[:, 0]

from sklearn.neural_network import MLPClassifier
X_train, y_train = X_train[:10000, :], y_train[:10000]
X_test, y_test = X_test[:1000, :], y_test[:1000]

X_train = np.array(X_train, dtype=np.float32) / 255
y_train = np.array(y_train, dtype=np.float32)

X_test = np.array(X_test, dtype=np.float32) / 255
y_test = np.array(y_test, dtype=np.float32)

clf = MLPClassifier(random_state = 0, warm_start = True)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 5)

