#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:30:33 2020

@author: findlayforsblom
"""

from mlxtend.data import loadlocal_mnist
import numpy as np
from matplotlib import pyplot as plt
from knnClassifierForHand import KNeighborsClassifier
import time

Xtrain, ytrain = loadlocal_mnist(
        images_path='./dataset/MNIST/train-images-idx3-ubyte', 
        labels_path='./dataset/MNIST/train-labels-idx1-ubyte')

Xtest, ytest = loadlocal_mnist(
        images_path='./dataset/MNIST/t10k-images-idx3-ubyte', 
        labels_path='./dataset/MNIST/t10k-labels-idx1-ubyte')
start = time.time()

Xtrain = Xtrain[:10000]
ytrain = ytrain[:10000]

Xtrain = Xtrain / 255

Xtest = Xtest[:1000]
ytest = ytest[:1000]

Xtest = Xtest / 255


"""
for num in np.unique(ytrain):
    indexes = np.where(ytrain==num)[0]
    print(f'number of {num} in the training set {len(indexes)}')

print('----------------------------------')
"""    
for num in np.unique(ytest):
    indexes = np.where(ytest==num)[0]
    print(f'number of {num} in the test set {len(indexes)}')


ks = []
scores = []

for k in range(1,15,2):
    print(k)
    classifier = KNeighborsClassifier(k)
    classifier.fit(Xtrain,ytrain)
    ypred = classifier.predict(Xtest)
    
    
    errors = ytest[ypred != ytest]
    errors = len(errors)
    score = 100 - ((errors / len(ypred)) * 100)
    
    ks.append(k)
    scores.append(score)

scores = np.array(scores)
posmax = scores.argmax()
fig, ax = plt.subplots()
ax.plot(ks, scores)
ax.plot(ks[posmax], scores[posmax], '-o', label= 'best k')
ax.set_title(f'{len(ytest)} test images on {len(ytrain)} traning images')
ax.set_xlabel('k')
ax.set_ylabel('Accuracy')
ax.legend()
print(f'best k is {ks[posmax]} with {scores[posmax]}% accuracy')
print("--- %s seconds ---" % (time.time() - start))


#print(classifier.score(Xtest, ytest))