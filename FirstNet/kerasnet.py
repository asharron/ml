import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import numpy as np
import pandas as pd
import keras

#Load in the dataset
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
#Reshape into 1D arrays
Xtrain = Xtrain.reshape(len(Xtrain), -1)
Xtest = Xtest.reshape(len(Xtest), -1)

nInput = 28 * 28
nHidden1 = 300
nHidden2 = 100
nOutputs = 10
learningRate = 0.01
nEpochs = 50
batchSize = 50

#Create the model
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=nHidden1, activation='elu'))
model.add(keras.layers.Dense(units=nHidden2, activation='elu'))
model.add(keras.layers.Dense(units=nOutputs, activation='softmax'))

#Set the loss
model.compile(loss='sparse_categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'])

model.fit(Xtrain, ytrain, epochs=nEpochs, batch_size=batchSize)

print(model.evaluate(Xtest, ytest, batch_size=batchSize))
