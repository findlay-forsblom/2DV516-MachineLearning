import pandas as pd
import numpy as np
from sklearn import svm
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

dataset = pd.read_csv('../Datasets/bm.csv', delimiter=',', header = None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

np.random.seed(7)
r = np.random.permutation(len(y))
X, y = X[r, :], y[r]
X_s, y_s = X[:5000, :], y[:5000]

clf = svm.SVC(kernel = 'rbf', C = 20, gamma = 0.5)
clf.fit(X_s, y_s)
y_ps = clf.predict(X_s)
supportVectors = clf.support_vectors_

h = 0.04
x_min, x_max = X_s[:, 0].min()-0.2, X_s[:, 0].max()+0.2
y_min, y_max = X_s[:, 1].min()-0.2, X_s[:, 1].max()+0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h)) # Mesh Grid
xy_mesh = np.c_[xx.ravel(), yy.ravel()] # Turn to Nx2 matrix

cmap_light = ListedColormap(['white', '#F7F9F9'])
cmap_bold = ListedColormap(['#FFFF0C', '#696969'])
#AAFFAA
clzmesh = clf.predict(xy_mesh)
clzmesh = clzmesh.reshape(xx.shape)

fig, ax = plt.subplots()
ax.pcolormesh(xx, yy, clzmesh, cmap=cmap_light, edgecolor='k', linewidth=10)
ax.scatter(X_s[:, 0], X_s[:, 1], c=y_s, cmap=cmap_bold, s= 3)

fig, ax = plt.subplots()
ax.plot(supportVectors[:,0], supportVectors[:,1], 'o')
