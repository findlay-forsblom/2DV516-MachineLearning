#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 07:08:23 2020

@author: findlayforsblom
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from random import choice

dataset_train = pd.read_csv('./Datasets/fashion-mnist_train.csv', delimiter=',', header = None)
dataset_test = pd.read_csv('./Datasets/fashion-mnist_test.csv', delimiter=',', header = None)

dataset_train = dataset_train.to_numpy()
dataset_test = dataset_test.to_numpy()

np.random.shuffle(dataset_train)
np.random.shuffle(dataset_test)

dataset_train = dataset_train.astype(np.float)
arr = np.array(dataset_test, dtype=np.float32)

X_train = dataset_train[:, 1:]
y_train = dataset_train[:, 0]

X_test = dataset_test[:, 1:]
y_test = dataset_test[:, 0]

class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#Task 1
sequence = [i for i in range(len(y_train))]
fig1, ax1 = plt.subplots(nrows=4, ncols=4)
for row in ax1:
    for col in row:
       selection = choice(sequence)
       # print(selection)
       label = class_labels[y_train[selection]]
       # The rest of columns are pixels
       pixels = X_train[selection]
       pixels = np.array(pixels, dtype='uint8')
        
       # Reshape the array into 28 x 28 array (2-dimensional array)
       pixels = pixels.reshape((28, 28))
        
        # Plot
       col.title.set_text('{label}'.format(label=label))
       col.imshow(pixels, cmap='gray')
       col.set_axis_off()
plt.tight_layout()


# Task 2
from sklearn.neural_network import MLPClassifier
X_train, y_train = X_train[:100, :], y_train[:100]
X_test, y_test = X_test[:10, :], y_test[:10]
clf = MLPClassifier(random_state = 0, warm_start = True)

X_train = np.array(X_train, dtype=np.float32) / 255
y_train = np.array(y_train, dtype=np.float32)

parameters = [{'activation': ['identity', 'logistic', 'tanh', 'relu'],
               'solver': ['lbfgs', 'sgd', 'adam'],
               'alpha': [0.001, 0.01, 0.1, 1],
               'max_iter':[1000, 5000],
               'hidden_layer_sizes':[(9,), (10,), (None,10)]}]
grid_search = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_
