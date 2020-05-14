# -*- coding: utf-8 -*-
"""
Created on Wed May 13 07:54:45 2020

@author: findlay
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from random import choice
import seaborn as sns

# Most of the code used was pretty much gotten from the book

fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "boot"]



#Task 1
sequence = [i for i in range(len(y_train_full))]
fig1, ax1 = plt.subplots(nrows=4, ncols=4)
for row in ax1:
    for col in row:
       selection = choice(sequence)
       # print(selection)
       label = class_names[y_train_full[selection]]
       # The rest of columns are pixels
       pixels = X_train_full[selection]
       pixels = np.array(pixels, dtype='uint8')
       
       # Plot
       col.title.set_text('{label}'.format(label=label))
       col.imshow(pixels, cmap='gray')
       col.set_axis_off()
plt.tight_layout()



#Task 2
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

val_loss, val_acc = model.evaluate(X_test, y_test)


# Task 3
from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(X_test, verbose = 1)
cm = confusion_matrix(y_test, y_pred)



fig, ax = plt.subplots(figsize=(12,9))  
sns.heatmap(cm, annot=True, ax = ax, cmap = 'Blues', fmt="d", linewidths=.5); #annot=True to annotate cells
sns.set(font_scale=1.4) # for label size

# labels, title and ticks
ax.set_xlabel('Predicted labels', );ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix Fashion MNIST'); 
ax.xaxis.set_ticklabels(class_names); ax.yaxis.set_ticklabels(class_names);
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=11)
plt.tight_layout()