# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:48:28 2020

@author: findl
"""


import numpy as np
import sys
import matplotlib.pyplot as plt

W2 = np.array([
    [5, -30],
    [20, -20]
    ])

b2 = np.array([10,-10])

x = np.array([0, 0])

np.dot(W2.T,x) + b2



#För clustering

cluster = np.array([
    [6,2],
    [7,2],
    [9,2],
    [5,4],
    [6,4],
    [6,5],
    [3,8] 
])

centroid = np.sum(cluster, axis = 0) / cluster.shape[0]
dist = np.sum(np.abs(cluster - centroid), axis = 1)
sse = np.sum(dist ** 2, axis = 0)

centroids = np.array([
    [2, 4.8],
    [6, 3.86],
    [5.6, 6.2]
    ])

red = np.array([2, 4.8])
blue = np.array([6, 3.86])
green = np.array([5.6, 6.2])
X = np.array([
    [2,2],
    [2,3],
    [1,5],
    [1,6],
    [4,8],
    [6,2],
    [7,2],
    [9,2],
    [5,4],
    [6,4],
    [6,5],
    [3,8],
    [5,1],
    [6,7],
    [7,7],
    [8,7],
    [2,9]  
    ])
X.shape[0]

y = np.array([0,0,0,0,0, 1,1,1,1,1,1,1, 2,2,2,2,2])

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k',c = y)

y.shape[0]

for row in range(X.shape[0]):
    val = X[row]
    mini = sys.maxsize
    for i in range(centroids.shape[0]):
        centroid = centroids[i]
        dist = np.sum(np.abs(val - centroid), axis = 0)
        if dist < mini:
            mini = dist
            y[row] = i
            print(dist)
            
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k',c = y)

    
        
