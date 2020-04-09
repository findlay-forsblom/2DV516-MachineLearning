# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:13:41 2020

@author: findlay Forsblom
"""


from Classes.Regression import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dataset = pd.read_csv('./Datasets/GPUbenchmark.csv', header=None) # Reading from the csv file

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y = np.reshape(y, (y.shape[0], 1))

regressor = LinearRegression()
Xe = regressor.fit(X,y) #Returns the normalized X Array

#Task 2
headers = ['CudaCores', 'Baseclock', 'BoostClock', 'MemorySpeed', 'MemoryConfig', 'MemoryBandwidth']
i = 0
fig1, ax1 = plt.subplots(nrows=2, ncols=3)
for row in ax1:
    for col in row:
       col.plot(Xe[:,i],y, 'o')
       col.set_xlabel(headers[i])
       col.set_ylabel('Benchmark Speed')
       i +=1
plt.tight_layout()


#Task 3
theta = regressor.computeBeta(X,y)
X_test = np.array([[2432, 1607, 1683, 8,8,256]])
ypred = regressor.predictWithNormalEqua(X_test)
print(f'Task 3 \n {ypred}\n')

#Task 4
cost = regressor.getCost()
print(f'Task 4 \n cost {cost}\n')

#Task5 
(beta, mse, itera) = regressor.computeGradient()
ypred_gradient = regressor.predictWithGradient(X_test)
print(f'Task 5 \n alpha = 0.01 and N = {itera} to get to '+
     f' 1% of the prevoius cost \n The cost function for was {mse[-1]} \n'+
     f'The predicted benchmark with gradient descent was {ypred_gradient}')

fig, ax = plt.subplots()
ax.plot(list(range(itera)), mse, '-')
ax.set_xlabel('iterations')
ax.set_ylabel('mse')
plt.show()