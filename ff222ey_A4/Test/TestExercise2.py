# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:30:44 2020

@author: findl
"""


import numpy as np
from sklearn import preprocessing
from sklearn import datasets
from sklearn.preprocessing import normalize
iris = datasets.load_iris()
X = preprocessing.normalize(iris.data)
y = iris.target
div_threshold = 1e-9

def sammon(X, itera, thresh, alpha):
    Y = np.random.rand(np.shape(X)[0], 2)
    X = normalize(X, axis = 0)
    
    for i in range(itera):
        
    pass