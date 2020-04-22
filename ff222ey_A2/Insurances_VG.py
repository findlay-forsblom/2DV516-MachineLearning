#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:04:07 2020

@author: findlayforsblom

Exercise 7
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plt


dataset = pd.read_csv('./Datasets/insurance.csv', delimiter=';')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
X[:,4] = le.fit_transform(X[:,4])

# Creating dummy variables for the region column
df = pd.DataFrame(data=X, columns=['age', 'sex', 'bmi', 'children','smoker', 'dummy', 'shoesize'])
just_dummies = pd.get_dummies(df['dummy'])

df = pd.concat([df, just_dummies], axis=1)
df.drop(['dummy'], inplace=True, axis=1)

# Updating X with the new dummy variables
X = df.iloc[:,].values

# Avoiding the Dummy Variable Trap
X = X[:, :-1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 101)

# Fitting Standard Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
standardRegressor = LinearRegression()
standardRegressor.fit(X_train, y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
traningError = []
validationErrors = []
for num in range(1, 4):
    print(f'Degree {num} ')
    poly_reg = PolynomialFeatures(degree=num)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg.fit(X_poly, y_train)
    lin_reg_2 = LinearRegression(n_jobs = -1)
    lin_reg_2.fit(X_poly, y_train)
    y_pred = lin_reg_2.predict(X_poly)

    traniningMSE = (y_pred - y_train) ** 2
    traniningMSE = (np.sum(traniningMSE)) / len(y_pred)
    print(f'traning error {traniningMSE}')
    traningError.append(traniningMSE)

    prediction = cross_val_predict(lin_reg_2, X_poly, y_train, cv=5)
    validationError = (prediction - y_train) ** 2
    validationError = (np.sum(validationError)) / len(prediction)
    print(f'valiadtion error {validationError} \n')
    validationErrors.append(validationError)

validationErrors = np.array(validationErrors)
pos = validationErrors.argmin()

#printing results for tranning error and validation erro
fig, ax = plt.subplots()
ax.plot(list(range(1,4)), traningError, '-', label='training data')
ax.plot(list(range(1,4)), validationErrors, '-', label='validation data')
ax.axvline(x=list(range(1,4))[pos], linestyle = '--', label = 'best fit')
ax.set_xlabel('Model Complexity (Degree)')
ax.set_ylabel('Mean Squared Error')
ax.legend()
ax.set_title('The bias variance trade off')
plt.show()

#Using ridge regression in comibination with polynomial regression at degree 2
from sklearn.linear_model import Ridge
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
ridgeReg = Ridge(solver = 'sag')

from sklearn.model_selection import GridSearchCV
parameters = [{'alpha': [50, 75, 100], 
               'max_iter': [15000, 5000, 10000],
              'tol': [1e-5, 1e-6, 1e-7, 1e-4, 1e-3, 1e-2, 1e-1, 1e-8]}]

grid_search = GridSearchCV(estimator = ridgeReg,
                           param_grid = parameters,
                           cv = 5,
                           scoring = 'neg_mean_squared_error',
                           n_jobs = -1)
grid_search = grid_search.fit(X_poly[:,1:], y_train)
best_mse = grid_search.best_score_
best_parameters = grid_search.best_params_

#Using Lasso regression in combination with polynomial regression at degree 2
from sklearn import linear_model
lasso = linear_model.Lasso(normalize = True, random_state = True, selection = 'cyclic', warm_start=True)

parameters = [{'alpha': [1, 0.99, 1.01], 
               'max_iter': [3000,6000, 10000, 15000], 
               'tol': [1e-5, 1e-6, 1e-7, 1e-4, 1e-3, 1e-2, 1e-1, 1e-8]}]

grid_search = GridSearchCV(estimator = lasso,
                           param_grid = parameters,
                           cv = 5,
                           scoring = 'neg_mean_squared_error',
                           n_jobs = -1)

grid_search = grid_search.fit(X_poly[:,1:], y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Using elastic net regression in combination with polynomial regression att degree 2
from sklearn.linear_model import ElasticNet
elastReg = ElasticNet(normalize = True, warm_start = True, random_state = True, precompute = False, selection = 'cyclic')

parameters = [{'alpha': [1, 0.99, 0.98], 
               'tol': [1e+2, 1e-6, 1e-7, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0],
               'max_iter': [3000,6000, 10000, 15000],
               'l1_ratio': [0.99, 0.98, 0.95, 1]}]
grid_search = GridSearchCV(estimator = elastReg,
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1,
                           scoring = 'neg_mean_squared_error')
grid_search = grid_search.fit(X_poly[:,1:], y_train)
best_mse = grid_search.best_score_
best_parameters = grid_search.best_params_


#Model assesment
lasso = linear_model.Lasso(normalize = True, random_state = True, selection = 'cyclic', warm_start=True, max_iter = 3000, tol = 0.01)
lasso.fit(X_poly[:,1:], y_train)
y_pred = lasso.predict(poly_reg.fit_transform(X_test)[:,1:])
score = lasso.score(poly_reg.fit_transform(X_test)[:,1:], y_test)
sums = (y_pred - y_test) ** 2
sums = (np.sum(sums)) / len(y_pred)
print(f'test score = {round(score, 2)}')
print(f'test error = {sums}')
