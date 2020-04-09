import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dataset = pd.read_csv('./Datasets/stat_females.csv', delimiter='\s+') # Reading from the csv file
X = dataset.iloc[:, [1, 2]].values
y = dataset.iloc[:, 0:1].values
u = np.mean(X, axis=0)
std = np.std(X, axis=0)

def featureScaling(X, u, std):
    print(X)
    lol = (X - u) / std
    return lol


#Task 1
fig, ax = plt.subplots(1,2)
ax[0].plot(X[:,0], y, 'o')
ax[1].plot(X[:,1], y, 'o')
ax[0].set_title('Mom and Girl')
ax[1].set_title('Dad and Girl')


#task 2
N = X.shape[0]
Xe = np.c_[np.ones(N), X ]

#Task 3
Beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)
ypred = np.sum(Beta * Xe, axis = 1)
np.sum(Beta * np.array([[1,65,70]]), axis = 1)

#Task 4
Xn = featureScaling(X)
yn = featureScaling(y)
Xne = np.c_[np.ones(N), Xn ]
fig1, ax1 = plt.subplots(1,2)
ax1[0].plot(Xne[:,1], y, 'o')
ax1[1].plot(Xne[:,2], y, 'o')
ax1[0].set_title('Mom and Girl')
ax1[1].set_title('Dad and Girl')

#Task5
Beta = np.linalg.inv(Xne.T.dot(Xne)).dot(Xne.T).dot(y)
lol = featureScaling(np.array([65,70]).reshape(1,2),u, std)
lol = np.c_[np.ones(1), lol ]
np.sum(Beta * lol, axis = 1)

#Yask 6
j = np.dot(Xe, Beta) - y
J = (j.T.dot(j)) / N

#Task 7
eta = .01
iterations = 1000

theta = np.random.randn(3,1)
MSE = []
""""
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100,1)
X_b = np.c_[np.ones((100,1)), X]
"""

for iteration in range(iterations):
    gradients = 2/N * Xne.T.dot((Xne.dot(theta) -y) )
    #print(gradients)
    theta = theta - eta * gradients
    j = np.dot(Xne, theta) - y
    J = (j.T.dot(j)) / N
    MSE.append(J[0])

MSE = np.array(MSE)
print(MSE.min())

lol = featureScaling(np.array([65,70]).reshape(1,2),u, std)
lol = np.c_[np.ones(1), lol ]
lol = np.reshape(lol, (3,1))
theta * lol
np.sum(theta * lol, axis = 0)

fig2, ax2 = plt.subplots()
ax2.plot(list(range(iterations)), MSE)

plt.show()