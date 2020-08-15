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
X = preprocessing.normalize(iris.data, axis = 0)
y = iris.target
div_threshold = 1e-9

# NOTE It assumess that the data (X) is already preprocessed / Normalized
def sammon(X, itera, thresh, alpha):
    Y = np.random.rand(np.shape(X)[0], 2) # random output space
    
    delta_ij = euclidean_distances(X, X) #distance input space
    
    for i in range(itera):
        d_ij = euclidean_distances(Y, Y) #distance output space
        c = np.sum(delta_ij) / 2 
        Y = gradient(Y, d_ij, delta_ij, alpha, c)
        E = sammonStress(d_ij, delta_ij, c)
        # print (E)
        if E < thresh:
            break
    return Y



def sammonStress(d_ij, delta_ij, c):
    # d_ij = euclidean_distances(dist_input, dist_input) #distance input space
    # delta_ij = euclidean_distances(dist_output, dist_output) #distance output space
    
    #delta_ij[delta_ij < 1e-5] = 1e-5 #to avoid division with 0
   # scale = np.sum(delta_ij) / 2 
    #dist = np.sum(((d_ij - delta_ij) ** 2 ) / delta_ij) / 2
    
    numerator = (d_ij - delta_ij ** 2 )
    denominator = delta_ij.copy()
    denominator[denominator < 1e-6] = 1e-6
     
    dist = np.sum(numerator / denominator) /2
    dist = dist / c

    
    
    return dist


def gradient(Y, d_ij, delta_ij, alpha, c):
    rows = Y.shape[0]
    
    for i in range(rows):
        yi = np.reshape(Y[i], (1,2))
        yj = np.delete(Y.copy(), i, axis= 0)
        
        xi = np.reshape(X[i], (1,X.shape[1]))
        xj = np.delete(X.copy(), i, axis= 0)
        
        outputSpace = np.reshape(euclidean_distances(yi, yj), (yj.shape[0], 1))
        inputSpace =  np.reshape(euclidean_distances(xi, xj), (yj.shape[0], 1))
        
        numerator = inputSpace - outputSpace
        denominator = outputSpace * inputSpace
        denominator[denominator < 1e-6] = 1e-6
        
        #TRY CHANGING C and check for results
        
        yi_ij = yi - yj
        
        first_derivitive = (-2 /c ) * np.sum((numerator / denominator) * yi_ij, axis = 0) 
        square = (yi_ij ** 2) / outputSpace
        last_part = (1 + (yi_ij/ outputSpace))
        
        second_derivitive = (-2 / c) * np.sum(1/denominator * (yi_ij - square * last_part), axis= 0)
        
        gradient = first_derivitive / abs(second_derivitive)
        
        Y[i] = Y[i] - alpha * gradient
    return Y
    
    # c2 = np.sum(inputSpace) / 2
    



itera = 1000
thresh = div_threshold
alpha = 0.01


sammon(X,itera, thresh, alpha)


Xe = sammon(X,itera, thresh, alpha)
fig, ax = plt.subplots()
plt.scatter(Xe[:,0], Xe[:,1], alpha=0.8, c=y, s=30, cmap='viridis')




#sammon stress
Y = np.random.rand(np.shape(X)[0], 2)
d_ij = euclidean_distances(Y, Y) #distance output space
delta_ij = euclidean_distances(X, X) #distance input space


numerator = (d_ij - delta_ij ** 2 )
denominator = delta_ij.copy()
denominator[denominator < 1e-5] = 1e-5
 
c = np.sum(delta_ij) / 2 
dist = np.sum(numerator / denominator) /2
dist = dist / c

#gradient
rows = Y.shape[0]
alpha = 0.3
c = np.sum(delta_ij) / 2 


lol_in = delta_ij.copy()
lol_out = d_ij.copy()

lol_in = lol_in[~np.eye(lol_in.shape[0],dtype=bool)].reshape(lol_in.shape[0],-1)
lol_out = lol_out[~np.eye(lol_out.shape[0],dtype=bool)].reshape(lol_out.shape[0],-1)

numerator = lol_in - lol_out
denominator = lol_out * lol_in
denominator[denominator < 1e-5] = 1e-5
    
for i in range(rows):
    yi = np.reshape(Y[i], (1,2))
    yj = np.delete(Y.copy(), i, axis= 0)
    
    xi = np.reshape(X[i], (1,X.shape[1]))
    xj = np.delete(X.copy(), i, axis= 0)
    
    outputSpace = np.reshape(euclidean_distances(yi, yj), (yj.shape[0], 1))
    inputSpace =  np.reshape(euclidean_distances(xi, xj), (yj.shape[0], 1))
    
    numerator = inputSpace - outputSpace
    denominator = outputSpace * inputSpace
    denominator[denominator < 1e-5] = 1e-5
    
    #TRY CHANGING C and check for results
    
    c = np.sum(euclidean_distances(xi, int(xj[xj.shape[0]/2])))
    
    yi_ij = yi - yj
    
    first_derivitive = (-2 /c ) * np.sum((numerator / denominator) * yi_ij, axis = 0) 
    square = (yi_ij ** 2) / outputSpace
    last_part = (1 + (yi_ij/ outputSpace))
    
    second_derivitive = (-2 / c) * np.sum(1/denominator * (yi_ij - square * last_part), axis= 0)
    
    gradient = first_derivitive / abs(second_derivitive)
    
    Y[i] = Y[i] - alpha * gradient
    
    # c2 = np.sum(inputSpace) / 2
    
    
    
    
    k = numerator / denominator
    
    lol =  np.sum((numerator / denominator))
    
    lol2 = np.sum(yi_ij)
    
    lol * lol2
    
    np.sum((numerator / denominator)) * yi_ij
 

party = pairwise_distances(X)


hej = np.array([
    [3,4],
    [2,3]
    ])


hej2 = np.array([
    [3,4],
    [2,3]
    ])



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