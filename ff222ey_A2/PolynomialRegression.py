# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:15:37 2020

@author: findlay Forsblom ff22ey
"""


from Classes.Regression import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('./Datasets/housing_price_index.csv', header=None) # Reading from the csv file
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

y = np.reshape(y, (y.shape[0], 1))

#Task 1
fig, ax = plt.subplots()
ax.plot(X, y, 'o')
ax.set_xlabel('year')
ax.set_ylabel('price')

x = X + 1975

#Task 2
i = 1

fig1, ax1 = plt.subplots(nrows=2, ncols=2, dpi=300)
for row in ax1:
    for col in row:
        poly = PolynomialFeatures(degree = i)
        X_poly = poly.fit_transform(X)
        X_poly = np.delete(X_poly, 0, axis = 1)
        regressor = LinearRegression()
        regressor.fit(X_poly,y)
        
        regressor.computeBeta()
        cost = regressor.getCost()

        ypred = regressor.predictWithNormalEqua(X_poly)
        col.plot(x, y, 'o')
        col.plot(x,ypred, 'r-')
        col.set_xlabel('year')
        col.set_title(f'degree {i}, Cost = {round(cost,2)}')
        col.set_ylabel('House prices')
        i +=1
    
plt.tight_layout()
plt.show()
print('Task 2 \n degree 4 gives thw best fit as can be seen in the picture above since it has the least cost\n')


#Task 3
poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)
X_poly = np.delete(X_poly, 0, axis = 1)
arg = np.where(x.flatten() == 2015)[0][0]


price2015 = y[arg][0]

regressor2 = LinearRegression()
regressor2.fit(X_poly,y)
regressor2.computeBeta()
test = np.array (poly.fit_transform([[47]]))
test = np.delete(test, 0, axis = 1)
pricepredict2022 = regressor2.predictWithNormalEqua(test)[0]

increase = pricepredict2022 / price2015
percent = str(round(increase *100 ,2))[1:]


JonasPricePredict2022 = 2300000 * increase
print(f'Task 3 \n From year 2015 to 2022 the predicted price increase is {percent}%, meaning ' +
      f'that Jonas predicted house would be worth {JonasPricePredict2022}')

