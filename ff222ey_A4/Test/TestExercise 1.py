# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:48:08 2020

@author: findl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def bkmeans (X, k, itera = 10):
    all_labels = np.zeros(X.shape[0], dtype = 'int64')
    largest_cluster = X
    ind = np.where(all_labels == 0)
    cluster_num = 0
    
    for clusters in range(k):
        #Fitting the largest cluster
        kmeans = KMeans(n_clusters=2, n_init = itera).fit(largest_cluster)
        labels = kmeans.labels_
        
        #find the positive and the negative indices and assigning new values
        pos_ind = np.where(labels == 1)
        neg_ind = np.where(labels == 0)
        labels[neg_ind] = cluster_num
        labels[pos_ind] = clusters
        all_labels[ind] = labels
        
        # Finds the largest cluster
        counts = np.bincount(all_labels)
        most_common = np.argmax(counts)
        ind = np.where(most_common == all_labels)
        largest_cluster = X[ind]
        
        #getting the number of the largest cluster
        cluster_num = all_labels[ind][0]
   
    return all_labels



X = np.array([[1, 2], [1, 4], [1, 0],
          [10, 2], [10, 4], [10, 0]])

clusters = 6

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, n_features=2, centers=clusters)

colors = ("red", "green", "blue")

y = bkmeans(X, k=clusters, itera = 10)
fig, ax = plt.subplots()
plt.scatter(X[:,0], X[:,1], alpha=0.8, c=y, s=30, cmap='viridis')


kmeans = KMeans(n_clusters=2, n_init = 1).fit(X)
kmeans.labels_



kmeans.predict([[0, 0], [12, 3]])

y = np.array([0,1,1,0])

index = {x:i for i, x in enumerate(y.ravel())}
