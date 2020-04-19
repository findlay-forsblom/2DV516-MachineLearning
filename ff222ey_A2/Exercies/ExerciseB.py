import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

dataset = pd.read_csv('./Datasets/admissiontests.csv', delimiter=',', header = None) # Reading from the csv file

#make private use before predict
def featureScaling(X, u, std):
    lol = (X - u) / std
    return lol

def sigmoidFunction(X):
    return 1/(1 + np.exp(-X))

#make private
def predict(X, beta):
    return X.dot(beta)

def computeBeta(Xne):
    eta = .5
    iterations = 1000

    N = Xne.shape[0]

    theta = [0,0,0]
    MSE = []

    for iteration in range(iterations):
        ypred = predict(Xne, theta)
        gradients = Xne.T.dot((sigmoidFunction(ypred) - y))
        theta = theta - ((eta * gradients )/N)
        print(theta)
        j = np.dot(Xne, theta) - y
        J = (j.T.dot(j)) / N
        #MSE.append(J[0])
    return theta

def logCostFunction(X,y, Xne):
    ypred = predict(Xne, X)
    N = Xne.shape[0]
    j = y.T.dot(np.log((sigmoidFunction(ypred))))
    return -1/N * (j + (1-y).T.dot(np.log(1-(sigmoidFunction(ypred)))))

X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, -1].values
u = np.mean(X, axis=0)
std = np.std(X, axis=0)

Xn = featureScaling(X, u, std)


#Task 1
fig, ax = plt.subplots()
cdict = {0: 'red', 1: 'green'}
type = {0: 'fail', 1:'pass'}
for g in np.unique(y):
    ix = np.where(y == g)
    ax.scatter(Xn[ix,0], Xn[ix,1], c = cdict[g], label = type[g], s = 10)
ax.legend()

ax.set_title('Admitted and not admitted students')

#Task 2
lol = sigmoidFunction(np.array([[0,1], [2,3]]))

#Task 3
N = Xn.shape[0]
Xne = np.c_[np.ones(N), Xn ]

#Task4
chaois = logCostFunction(np.array([0,0,0]), y, Xne)

#Task 5
beta = computeBeta(Xne)
logCostFunction(beta,y,Xne)

#Task 6
h = 0.01
x_min, x_max = Xne[:, 1].min()-0.2, Xne[:, 1].max()+0.2
y_min, y_max = Xne[:, 2].min()-0.2, Xne[:, 2].max()+0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h)) # Mesh Grid
x1,x2 = xx.ravel(), yy.ravel()
N = x1.shape[0]
xy_mesh = np.c_[np.ones(N), x1, x2] # Turn to Nx2 matrix
#N = xy_mesh.shape[0]
#xy_mesh = np.c_[np.ones(N), xy_mesh]

p = sigmoidFunction(predict(xy_mesh, beta))

classes = p > 0.5
clzmesh = classes.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

fig1, ax1 = plt.subplots()

trainingerrors = sigmoidFunction(predict(Xne, beta))
trainingerrors = trainingerrors > 0.5
y_test = y == 1
errors = y_test[y_test !=trainingerrors]
errors = len(errors)
ax1.pcolormesh(xx, yy, clzmesh, cmap=cmap_light)
ax1.scatter(Xne[:, 1], Xne[:, 2], c=y, cmap=cmap_bold)
ax1.set_title(f'Training errors {errors}')


#Task 7
lol = featureScaling(np.array([[45,85]]), u, std)
lol = np.c_[np.ones(1), lol ]
lol = sigmoidFunction(predict(lol, beta))


plt.show()

