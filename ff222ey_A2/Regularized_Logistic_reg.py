from sklearn.linear_model import LogisticRegression
from Classes.RandomFunctions import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('./Datasets/microchips.csv', header=None) # Reading from the csv file
dataset = dataset.to_numpy()
X = dataset[:, :-1]
y = dataset[:, -1]

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Tasks 1
validationErrorsNonReg = []
k = 10
i = 1
fig1, ax1 = plt.subplots(nrows=3, ncols=3)
for row in ax1:
    for col in row:
        Xe = mapFeatures(X[:, 0], X[:, 1], i, Ones=False)  # No 1-column!
        (clf, error) = logisticRegSklearn(Xe, y)

        prediction = cross_val_predict(clf, Xe, y, cv=k)
        errors = np.sum(prediction != y)
        validationErrorsNonReg.append(errors)

        x1, x2, xx, yy = computeMesh(X[:, 0], X[:, 1], 0.01)
        XXe = mapFeatures(x1, x2, i, Ones = False)

        p = clf.predict(XXe)

        classes = p > 0.5
        clzmesh = classes.reshape(xx.shape)

        col.pcolormesh(xx, yy, clzmesh, cmap=cmap_light)
        col.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        col.set_title(f'Tr-errors = {error}, Deg ={i}')
        i += 1
fig1.suptitle('Non Regularized logstic regression')
plt.tight_layout()

validationErrorsReg = []
#Task 2
i = 1
fig2, ax2 = plt.subplots(nrows=3, ncols=3)
for row in ax2:
    for col in row:
        Xe = mapFeatures(X[:, 0], X[:, 1], i, Ones=False)  # No 1-column!
        (clf, error) = logisticRegSklearn(Xe, y, C= 1)

        prediction = cross_val_predict(clf, Xe, y, cv=k)
        errors = np.sum(prediction != y)
        validationErrorsReg.append(errors)

        x1, x2, xx, yy = computeMesh(X[:, 0], X[:, 1], 0.01)
        XXe = mapFeatures(x1, x2, i, Ones = False)

        p = clf.predict(XXe)

        classes = p > 0.5
        clzmesh = classes.reshape(xx.shape)

        col.pcolormesh(xx, yy, clzmesh, cmap=cmap_light)
        col.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        col.set_title(f'Tr-errors = {error}, Deg ={i}')
        i += 1
fig2.suptitle('Regularized logstic regression')
plt.tight_layout()

# Task 3
fig, ax = plt.subplots()
ax.plot(list(range(1,10)), validationErrorsReg, '-', label='Regularized')
ax.plot(list(range(1,10)), validationErrorsNonReg, '-', label='Non Regularized')
ax.set_xlabel('degrees')
ax.set_ylabel('Validation errors')
ax.legend()
fig.suptitle(f'CV k = {k}')
plt.show()