#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:39:56 2020

@author: findlayforsblom
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([
    0,0,
    0,1,
    1,0,
    1,1
]).reshape(4,2)

y = np.array([0,1,1,0])

clf = MLPClassifier(activation = 'logistic', solver = 'lbfgs', hidden_layer_sizes = [2], random_state = 0)
clf.fit(X,y)
clf.predict(X)
weights = clf.coefs_
clf.intercepts_
