import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
from tensorflow import keras
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial

#Set the random seed to get same results each run
np.random.seed(42)

#Load in the dataset
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()

Xtrain = np.reshape(Xtrain, (-1, 28, 28, 1))
Xtest = np.reshape(Xtest, (-1, 28, 28, 1))

#Training params info
height = 28
width = 28
channels = 1 #only 1 channel since mnist dataset is in grayscale
nFilters1 = 32
nFilters2 = 64
nOutputs = 10
learningRate = 0.001
nEpochs = 100
batchSize = 50
numBatches = len(Xtrain) // batchSize


conv = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding='SAME')
model = keras.models.Sequential()

model.add(conv(filters=32, kernel_size=7, input_shape=[28, 28, 1]))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(conv(filters=64, kernel_size=7))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(conv(filters=128, kernel_size=7))
model.add(conv(filters=128, kernel_size=7))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(conv(filters=256, kernel_size=7))
model.add(conv(filters=256, kernel_size=7))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(Xtrain, ytrain, epochs=nEpochs)

print(model.evaluate(Xtest, ytest))
