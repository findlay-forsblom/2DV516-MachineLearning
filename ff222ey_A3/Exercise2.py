#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:55:41 2020

@author: findlayforsblom
"""

from mlxtend.data import loadlocal_mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from Classes.multiclass import OneVsRestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix

Xtrain, ytrain = loadlocal_mnist(
        images_path='./Datasets/MNIST/train-images-idx3-ubyte', 
        labels_path='./datasets/MNIST/train-labels-idx1-ubyte')

Xtest, ytest = loadlocal_mnist(
        images_path='./datasets/MNIST/t10k-images-idx3-ubyte', 
        labels_path='./datasets/MNIST/t10k-labels-idx1-ubyte')



#np.random.seed(7)
r = np.random.permutation(len(ytrain))
X, y = Xtrain[r, :], ytrain[r]
X_train, y_train = X[:10000, :], y[:10000]

#np.random.seed(7)
r = np.random.permutation(len(ytest))
X, y = Xtest[r, :], ytest[r]
X_test, y_test = X[:1000, :], y[:1000]

"""
X_train = Xtrain[:10000]
y_train = ytrain[:10000]

X_train = X_train / 255

X_test = Xtest[:1000]
y_test = ytest[:1000]

X_test = X_test / 255
"""


categories = []
number = []

for num in np.unique(y_train):
    indexes = np.where(y_train==num)[0]
    categories.append(num)
    number.append(len(indexes))

ax = sns.barplot(categories, number)
plt.title("Digits distribution (Training set)", fontsize=16)
plt.ylabel('Number of digits', fontsize=12)
plt.xlabel('Digit', fontsize=12)
plt.show()

categories = []
number = []

for num in np.unique(y_test):
    indexes = np.where(y_test==num)[0]
    categories.append(num)
    number.append(len(indexes))

ax = sns.barplot(categories, number)
plt.title("Digits distribution (Test set)", fontsize=16)
plt.ylabel('Number of digits', fontsize=12)
plt.xlabel('Digit ', fontsize=12)
plt.show()


clf = svm.SVC(kernel = 'rbf', gamma = 'scale')
parameters = [{'C': [8, 10, 7, 12]}]

grid_search = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_


#Fitting best parameters on test set and computing confusion matrix
clf = svm.SVC(kernel = 'rbf', C = 8, gamma= 'scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('accuracy score: %0.3f' % accuracy_score(y_test, y_pred))

categories = np.array(categories)
labels = np.char.mod('%d', categories)
cm = confusion_matrix(y_test, y_pred, categories)



fig, ax = plt.subplots(figsize=(10,7))  
sns.heatmap(cm, annot=True, ax = ax, cmap = 'Blues', fmt="d", linewidths=.5); #annot=True to annotate cells
sns.set(font_scale=1.4) # for label size

# labels, title and ticks
ax.set_xlabel('Predicted labels', );ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix (One Vs One)'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);

#Second part of the assignment

classifiers = []
for num in np.unique(y_train):
    svc = svm.SVC(kernel = 'rbf', C = 8, gamma= 'scale', probability = True)
    classifiers.append(svc)

classifier = OneVsRestClassifier(classifiers)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print ('accuracy score: %0.3f' % accuracy_score(y_test, y_pred))

categories = np.array(categories)
labels = np.char.mod('%d', categories)
cm = confusion_matrix(y_test, y_pred, categories)



fig, ax = plt.subplots(figsize=(10,7))  
sns.heatmap(cm, annot=True, ax = ax, cmap = 'Blues', fmt="d", linewidths=.5); #annot=True to annotate cells
sns.set(font_scale=1.4) # for label size

# labels, title and ticks
ax.set_xlabel('Predicted labels', );ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix (One Vs All)'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);


X_test[0].shape


