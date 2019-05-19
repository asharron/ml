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
codings_size = 100
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

#Create the model & add layers to it
generator = keras.models.Sequential([
    keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
    keras.layers.Reshape([7, 7, 128]),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME", activation="relu"),
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="SAME", activation="tanh")
])

descriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, padding="SAME", activation=keras.layers.LeakyReLU(0.2), input_shape=[height, width, channels]),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME", activation=keras.layers.LeakyReLU),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])

gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

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
        noise = tf.random.normal(shape=[batchsize, codings_size])
        generatedImages = generator(noise)
        XFakeOrReal = tf.concat([generatedImages, Xbatch], axis=0)
        y1 = tf.constant([[0.]] * batchsize + [[1.]] * batchsize)
        descriminator.trainable = True
        descriminator.train_on_batch(XFakeOrReal, y1)
        
        noise = tf.random.normal(shape=[batch_size, codings_size])
        y2 = tf.constant([[1.]] * batchsize)
        discriminator.trainable = False
        gan.train_on_batch(noise, y2)
