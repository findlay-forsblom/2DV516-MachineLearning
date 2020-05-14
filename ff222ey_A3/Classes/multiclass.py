#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:21:12 2020

@author: findlayforsblom
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class OneVsRestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def fit(self, X, y_train):
        i = 0
        classifiers = []
        for num in np.unique(y_train):
            print(num)
            y = np.isin(y_train, num)
            y = y.astype(int)
            classifier = self.classifiers[i]
            classifier.fit(X, y)
            i += 1
            classifiers.append(classifier)
        self.classifiers =classifiers

    def predict(self, X):
        rows = X.shape[0]
        y_pred = []
        
        for row in range(rows):
            self.predictions_ = list()
            for classifier in self.classifiers:
                self.predictions_.append(classifier.predict_proba(np.array([X[row]]))[0][1])
            self.predictions_ = np.array(self.predictions_)
            #print(self.predictions_)
            pred = np.argmax(self.predictions_)
            y_pred.append(pred)
        return y_pred
        
