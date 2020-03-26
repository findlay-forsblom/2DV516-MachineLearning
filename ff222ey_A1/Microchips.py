# -*- coding: utf-8 -*-
from knnClassifierForHand import KNeighborsClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

dataset = pd.read_csv('microchips.csv')
ks = [1,3,5,7]

thisdict = {
  1: "OK",
  0: "Fail"
}

X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 2].values

x1 = dataset.iloc[:56, [0]].values #ok
x2 = dataset.iloc[:56, [1]].values #ok
x3 = dataset.iloc[56:, [0]].values #not ok
x4 = dataset.iloc[56:, [1]].values #not ok

Xtests = [[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]]
Xtests = np.array(Xtests).reshape(3,2)

plt.title('Plot of orignal data')
ok = plt.scatter(x1,x2,color='green')
notOk = plt.scatter(x3,x4,color='red')
plt.legend((ok, notOk), ('Chip OK', 'Chip Failed') )
plt.show()

numOfRows = Xtests.shape[0]

for k in ks:
    classifier = KNeighborsClassifier(k)
    classifier.fit(X,y)
    print(f'k = {k}')
    counter = 1
    ypred = classifier.predict(Xtests)
     
    for row in range(numOfRows):
        Xtest = Xtests[row]
        print(f'\t chip{counter}: {Xtest} ==> {thisdict[ypred[row]]}')
        counter +=1

fig, ax = plt.subplots(nrows=2, ncols=2)
count = 0

h = 0.02
x_min, x_max = X[:, 0].min()-0.2, X[:, 0].max()+0.2
y_min, y_max = X[:, 1].min()-0.2, X[:, 1].max()+0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h)) # Mesh Grid
xy_mesh = np.c_[xx.ravel(), yy.ravel()] # Turn to Nx2 matrix

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

for row in ax:
    for col in row:
        classifier = KNeighborsClassifier(ks[count])
        classifier.fit(X,y)
        clzmesh = classifier.predict(xy_mesh)
        clzmesh = clzmesh.reshape(xx.shape)
        col.pcolormesh(xx, yy, clzmesh, cmap=cmap_light)
        col.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        col.title.set_text(f'Decision boundary k = {ks[count]} ')
        count +=1

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.tight_layout()
plt.show()
