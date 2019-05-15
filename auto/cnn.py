import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
import pandas as pd
import os
from datetime import datetime
from functools import partial
from matplotlib.image import imread

#Set the random seed
np.random.seed(42)

#Helper function to create minibatches
def createMinibatch(inputs, batchsize):
    inputLength = inputs.shape[0]
    indices = np.arange(inputLength)
    np.random.shuffle(indices)
    for start_idx in range(0, inputLength, batchsize):
        end_idx = min(start_idx + batchsize, inputLength)
        excerpt = indices[start_idx:end_idx]
        yield inputs[excerpt]

#Helper function to load images into memory
def loadImageBatch(imageNames, height, width, channels):
    data = []
    for imageNameIdx in range(imageNames.shape[0]):
        data.append(imread(imageNames[imageNameIdx]))
    return data

#Params for image sizes
height = 800
width = 800
channels = 3
currentDir = os.path.dirname(os.path.realpath(__file__))

#Params for NN
nEpochs = 50
batchsize = 30
codings_size = 10
nFilters1 = 3
nFilters2 = 3
nFilters3 = 3
nFilters4 = 3
nFilters5 = 3
nFilters6 = 3
nKernel1 = 1
nKernel2 = 1
nKernel3 = 1
nKernel4 = 1
nKernel5 = 1
nKernel6 = 1
nPool1 = 7
nPool2 = 7
nPool3 = 7
nDense1 = 50
nDense2 = 50

#Import the images into a numpy array
imageFiles = np.array([os.path.join(currentDir, "images", f) for f in os.listdir(os.path.join(currentDir, "images")) if os.path.isfile(os.path.join(currentDir, "images", f))])

imageHolder = tf.placeholder(tf.int32, shape=(None, None, 3))
cropImages = tf.image.resize_image_with_crop_or_pad(imageHolder, height, width)

X = tf.placeholder(tf.float32, shape=(None, height, width, channels), name="X")

#hands on 
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean

class Encoding():
    def __init__(self):
        self.codings_mean = keras.layers.Dense(codings_size)
        self.codings_log_var = keras.layers.Dense(codings_size)

    def update(self, x):
        mean = self.codings_mean(x)
        log_var = self.codings_log_var(x)
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


class EncodingLayer(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(EncodingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                        shape=(input_shape[1], self.output_dim),
                        initializer="uniform",
                        trainable=true)
        super(EncodingLayer, self).build(input_shape)
    
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

#Create the model & add layers to it
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=nFilters1, kernel_size=1, input_shape=[height, width, channels], strides=(1,1), activation="selu", padding="SAME"))
model.add(keras.layers.Conv2D(filters=nFilters2, kernel_size=nKernel2, padding="SAME", strides=(1,1), activation="selu"))
model.add(keras.layers.MaxPooling2D(pool_size=2, padding="SAME"))
model.add(keras.layers.Conv2D(filters=nFilters2, kernel_size=nKernel2, padding="SAME", strides=(1,1), activation="selu"))
model.add(keras.layers.MaxPooling2D(pool_size=2, padding="SAME"))
model.add(keras.layers.Conv2D(filters=nFilters2, kernel_size=nKernel2, padding="SAME", strides=(1,1), activation="selu"))
model.add(keras.layers.MaxPooling2D(pool_size=2, padding="SAME"))
model.add(keras.layers.Conv2D(filters=nFilters2, kernel_size=nKernel2, padding="SAME", strides=(1,1), activation="selu"))
model.add(keras.layers.MaxPooling2D(pool_size=2, padding="SAME"))
model.add(keras.layers.Conv2D(filters=nFilters2, kernel_size=nKernel2, padding="SAME", strides=(1,1), activation="selu"))
model.add(keras.layers.MaxPooling2D(pool_size=2, padding="SAME"))
model.add(keras.layers.Flatten())
encodingLayer = Encoding()
model.add(keras.layers.Lambda(lambda x: encodingLayer.update(x), output_shape=(-1, codings_size)))
model.add(keras.layers.Reshape((5, 5, 3)))
model.add(keras.layers.Conv2D(filters=nFilters2, padding="SAME", kernel_size=nKernel2, strides=(1,1), activation="selu"))
model.add(keras.layers.UpSampling2D(size=(5, 5)))
model.add(keras.layers.Conv2D(filters=nFilters2, padding="SAME", kernel_size=nKernel2, strides=(1,1), activation="selu"))
model.add(keras.layers.UpSampling2D(size=(4, 4)))
model.add(keras.layers.Conv2D(filters=3, padding="SAME", strides=(1,1), kernel_size=nKernel1, activation="sigmoid"))
model.add(keras.layers.UpSampling2D(size=(2, 2)))
model.add(keras.layers.Conv2D(filters=3, padding="SAME", strides=(1,1), kernel_size=nKernel1, activation="sigmoid"))
model.add(keras.layers.UpSampling2D(size=(2, 2)))
model.add(keras.layers.Conv2D(filters=3, padding="SAME", strides=(1,1), kernel_size=nKernel1, activation="sigmoid"))
model.add(keras.layers.UpSampling2D(size=(2, 2)))
model.add(keras.layers.Conv2D(filters=3, padding="SAME", strides=(1,1), kernel_size=nKernel1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
##Encoding Layer
#z = keras.layers.Dense(100, activation="selu")(z)
#codings_mean = keras.layers.Dense(codings_size)(z)
#codings_log_var = keras.layers.Dense(codings_size)(z)
#codings = Sampling()([codings_mean, codings_log_var])
#variational_encoder = keras.models.Model(
#        inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])
#
#decoder_inputs = keras.layers.Input(shape=[codings_size])
#x = keras.layers.Dense(100, activation="selu")(decoder_inputs)
#x = keras.layers.Dense(150, activation="selu")(x)
#x = keras.layers.Dense(28 * 28, activation="sigmoid")(x)
#outputs = keras.layers.Reshape([28, 28])(x)
#variational_decoder = keras.models.Model(inputs=[decoder_inputs], outputs=[outputs])
#
#_, _, codings = variational_encoder(inputs)
#reconstructions = variational_decoder(codings)
#variational_ae = keras.models.Model(inputs=[inputs], outputs=[reconstructions])
#
#latent_loss = -0.5 * K.sum(1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean), axis=-1)
#variational_ae.add_loss(K.mean(latent_loss) / 784.)
#variational_ae.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=[rounded_accuracy])
#history = variational_ae.fit(X_train, X_train, epochs=25, batch_size=128, validation_data=[X_valid, X_valid])
#

count = 0
with tf.Session() as sess:
    for epoch in range(nEpochs):
        batcher = createMinibatch(imageFiles, batchsize)
        Xbatch = None
        for batch in batcher:
            imageData = loadImageBatch(batch, height, width, channels)
            Xbatch = np.zeros((len(imageData), height, width, channels))
            for index, image in enumerate(imageData):
                Xbatch[index] = sess.run(cropImages, feed_dict={imageHolder: image})
            count = count + 1
            model.train_on_batch(Xbatch, Xbatch)
        print(model.evaluate(Xbatch, Xbatch))
