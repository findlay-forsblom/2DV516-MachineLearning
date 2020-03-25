# -*- coding: utf-8 -*-
from knnClassifierForHand import KNeighborsClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dataset = pd.read_csv('microchips.csv')
classifer = KNeighborsClassifier(5)

X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 2].values

x1 = dataset.iloc[:56, [0]].values #ok
x2 = dataset.iloc[:56, [1]].values #ok
x3 = dataset.iloc[56:, [0]].values #not ok
x4 = dataset.iloc[56:, [1]].values #not ok

"""
plt.scatter(x1,x2,color='green')
plt.scatter(x3,x4,color='red')
plt.show()
"""


classifer.fit(X,y)
classifer.predict([-0.3, 1.0])
