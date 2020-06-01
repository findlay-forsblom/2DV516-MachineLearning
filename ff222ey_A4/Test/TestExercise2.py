# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:30:44 2020

@author: findl
"""


import numpy as np
from sklearn import preprocessing
from sklearn import datasets
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
iris = datasets.load_iris()
X = preprocessing.normalize(iris.data)
y = iris.target
div_threshold = 1e-9

def sammon(X, itera, thresh, alpha):
    Y = np.random.rand(np.shape(X)[0], 2)
    
    delta_ij = euclidean_distances(X, X) #distance output space
    
    for i in range(itera):
        d_ij = euclidean_distances(Y, Y) #distance input space
        c = np.sum(delta_ij) / 2 
        E = sammonStress(d_ij, delta_ij, c)
        print (E)
        if E < thresh:
            break
        pass



def sammonStress(d_ij, delta_ij, c):
    # d_ij = euclidean_distances(dist_input, dist_input) #distance input space
    # delta_ij = euclidean_distances(dist_output, dist_output) #distance output space
    
    delta_ij[delta_ij < 1e-5] = 1e-5 #to avoid division with 0
   # scale = np.sum(delta_ij) / 2 
    dist = np.sum(((d_ij - delta_ij) ** 2 ) / delta_ij) / 2
    return dist / c


def gradient(Y, d_ij, delta_ij, alpha, c):
    rows = Y.shape[0]
    
    for i in range(rows):
        pass



itera = 10
thresh = div_threshold
alpha = 0.01


Y = np.random.rand(np.shape(X)[0], 2)
   
for i in range(itera):
    pass





#sammon stress
d_ij = euclidean_distances(Y, Y) #distance input space
delta_ij = euclidean_distances(X, X) #distance output space
delta_ij[delta_ij < 1e-5] = 1e-5
scale = np.sum(delta_ij) / 2
dist = np.sum(((d_ij - delta_ij) ** 2 ) / delta_ij) / 2
dist = dist / scale




#gradient
rows = Y.shape[0]
alpha = 0.3
c = np.sum(delta_ij) / 2 
    
for i in range(rows):
   for j in range(rows):
       if (j != i):
           pass  





sammon(X, itera, div_threshold, alpha)






"""
from sklearn.manifold import MDS
embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(X)
X_transformed.shape

fig, ax = plt.subplots()
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k',c = y)
"""