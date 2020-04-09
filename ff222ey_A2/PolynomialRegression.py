# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:15:37 2020

@author: findlay Forsblom ff22ey
"""


from Classes.Regression import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dataset = pd.read_csv('./Datasets/housing_price_index.csv', header=None) # Reading from the csv file
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

#Task 1
fig, ax = plt.subplots()
ax.plot(X, y, 'o')
ax.set_xlabel('year')
ax.set_ylabel('price')
plt.show()