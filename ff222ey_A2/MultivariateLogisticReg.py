#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:13:25 2020

@author: findlayforsblom

Exercise 3
"""

from Classes.Logistics import LogisticRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Task 1

dataset = pd.read_csv('./Datasets/breast_cancer.csv', header=None) # Reading from the csv file

dataset = dataset.to_numpy()
np.random.shuffle(dataset)

X = dataset[:, :-1]
y = dataset[:, -1]

# Task 2
y = y//4

allocation = 0.8 #test Ratio

split = int(allocation * dataset.shape[0])

X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print(f'Task 2 \n I allocated {100 - (allocation * 100)}% as my test data and the rest as the training data. '+
      'It is a good pratice to have around 20 - 30% as test and the rest as traning, beacuse ur model'
      +' will be more accurate when it has more training data to train on \n')

#task3
itera = 1000
alf = 0.5
 
clf = LogisticRegression(alpha = alf, iterations = itera)
clf.fit(X_train, y_train)
mse = clf.getCost()

fig, ax = plt.subplots()
ax.plot(list(range(itera)), mse, '-')
ax.set_xlabel('iterations')
ax.set_ylabel('J(B)')
ax.set_title(f'N = {itera} and Alpha = {alf} ')
plt.show()

#Task 4
errors = clf.getTrainingErrors()
print(f'Task 4 \n Traning error = {errors}, Traning accuracy = {100 - (errors/ X_train.shape[0]) *100} \n')

#Task 5
y_pred = clf.predict(X_test)
testerrors = y_pred > 0.5
y_testing = y_test == 1
testerrors = y_test[y_testing !=testerrors]
testerrors = len(testerrors)
print(f'Task 5 \n Test errors = {testerrors}, Test accuracy = {100 - (testerrors/ X_test.shape[0]) *100}\n')
