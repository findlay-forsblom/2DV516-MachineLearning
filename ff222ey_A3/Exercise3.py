#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:34:20 2020

@author: findlayforsblom
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

dataset_train = pd.read_csv('./Datasets/fbtrain.csv', delimiter=',', header = None)
dataset_test = pd.read_csv('./Datasets/fbtest.csv', delimiter=',', header = None)

X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, -1].values

X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, -1].values




#Task 1
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_train)
sums = (y_pred - y_train) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'Task 1 \n \tTraining mse {sums }')

y_pred = regressor.predict(X_test)
sums = (y_pred - y_test) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTest mse {sums } \n')

#Using gridsearch to find the best depth param
parameters = [{'max_depth': [4, 5, 6, 7, 8, None]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_

regressor = DecisionTreeRegressor(random_state=0, max_depth = 5)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_train)
sums = (y_pred - y_train) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTraining mse for max_depth = 5: {sums }')

y_pred = regressor.predict(X_test)
sums = (y_pred - y_test) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTest mse for max_depth = 5: {sums } \n')




#Task 2
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_train)
sums = (y_pred - y_train) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'Task 2 \n \tTraining mse {sums }')

y_pred = rf.predict(X_test)
sums = (y_pred - y_test) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTest mse {sums } \n')

#Using gridsearch to find the best depth param for random forrest
parameters = [{'max_depth': [3, 30, 10, 20, 5]}]
grid_search = GridSearchCV(estimator = rf,
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1, 
                           verbose = 1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_

#Testing on the best found params
regressor = RandomForestRegressor(random_state=0, max_depth = 30)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_train)
sums = (y_pred - y_train) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTraining mse for max_depth = 30: {sums }')

y_pred = regressor.predict(X_test)
sums = (y_pred - y_test) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTest mse for max_depth = 30: {sums } \n')




# Task 3
dataset_train = dataset_train[dataset_train[38] == 24]
X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, -1].values

dataset_test = dataset_test[dataset_test[38] == 24]
X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, -1].values

#refiting decsion tree
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_train)
sums = (y_pred - y_train) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'Task 3 \n \tTraining mse with DT: {sums }')

y_pred = regressor.predict(X_test)
sums = (y_pred - y_test) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTest mse with DT: {sums } \n')

#Using gridsearch to find the best depth param
parameters = [{'max_depth': [2, 3, 6, 8,  None]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_

# Fiting best params
regressor = DecisionTreeRegressor(random_state=0, max_depth = 3)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_train)
sums = (y_pred - y_train) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTraining mse for max_depth = 3 with DT: {sums }')

y_pred = regressor.predict(X_test)
sums = (y_pred - y_test) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTest mse for max_depth = 3 with DT: {sums } \n')

# testing randomforrest
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_train)
sums = (y_pred - y_train) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTraining mse with RF: {sums }')

y_pred = rf.predict(X_test)
sums = (y_pred - y_test) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTest mse with RF: {sums } \n')

#Using gridsearch to find the best depth param for random forrest
parameters = [{'max_depth': [15, 10, 17, 12]}]
grid_search = GridSearchCV(estimator = rf,
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1, 
                           verbose = 1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_

#Testing on the best found params
regressor = RandomForestRegressor(random_state=0, max_depth = 15)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_train)
sums = (y_pred - y_train) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTraining mse for max_depth = 15 with RF: {sums }')

y_pred = regressor.predict(X_test)
sums = (y_pred - y_test) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'\tTest mse for max_depth = 15 with RF: {sums } \n')



