import tensorflow as tf
from tensorflow import keras
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
batchsize = 50
currentDir = os.path.dirname(os.path.realpath(__file__))

#Import the images into a numpy array
imageFiles = np.array([os.path.join(currentDir, "images", f) for f in os.listdir(os.path.join(currentDir, "images")) if os.path.isfile(os.path.join(currentDir, "images", f))])

imageHolder = tf.placeholder(tf.int32, shape=(None, None, 3))
cropImages = tf.image.resize_image_with_crop_or_pad(imageHolder, height, width)

X = tf.placeholder(tf.float32, shape=(None, height, width, channels), name="X")

############################## CNN Code
#Define the graph for the DNN Layers
with tf.name_scope("cnn"):
    #Batch norm and He initalization
    he = tf.contrib.layers.variance_scaling_initializer()
    
    conv1 = tf.layers.conv2d(X_reshape, filters=nFilters1, kernel_size=3, strides=1, padding="SAME", name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=nFilters2, kernel_size=3, strides=2, padding="SAME", name="conv2")
    pool3 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    pool3_flat = tf.reshape(pool3, shape=[-1, 32 * 14 * 14])
    fc1 = tf.layers.dense(pool3_flat, 64, activation=tf.nn.relu, name="fc1")
    logits = tf.layers.dense(fc1, nOutputs, name="outputs")

#Graph for our loss an entropy
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

#Graph for performing the actual training
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    trainingOp = optimizer.minimize(loss, name="trainingOp")

#Graph for evaluating how our model is performing
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

##############################



############################## Auto encoder code
with tf.name_scope("autonn"):
	he = tf.contrib.layers.variance_scaling_initializer()

	hidden1 = tf.layers.dense(X, nHidden1, activation=tf.nn.relu, kernel_initializer=he, name="hidden1")
	hidden2 = tf.layers.dense(hidden1, nHidden2, activation=tf.nn.relu, kernel_initializer=he, name="hidden2")
	hidden3Mean = tf.layers.dense(hidden2, nHidden3, activation=None)
	hidden3Gamma = tf.layers.dense(hidden2, nHidden3, activation=None)
	noise = tf.random_normal(tf.shape(hidden3Gamma), dtype=tf.float32)
	hidden3 = hidden3Mean + tf.exp(0.5 * hidden3Gamma) * noise
	hidden4 = tf.layers.dense(hidden3, nHidden4, activation=tf.nn.relu, kernel_initializer=he, name="hidden4")
	hidden5 = tf.layers.dense(hidden4, nHidden5, activation=tf.nn.relu, kernel_initializer=he, name="hidden5")
	logits = tf.layers.dense(hidden5, nOutputs, activation=None)
	outputs = tf.sigmoid(logits)

with tf.name_scope("loss"):
	xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
	reconstructionLoss = tf.reduce_sum(xentropy)
	latent_loss = 0.5 * tf.reduce_sum(tf.exp(hidden3Gamma) + tf.square(hidden3Mean) - 1 - hidden3Gamma)
	loss = reconstructionLoss + latent_loss

with tf.name_scope("train"):
	optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
	trainingOp = optimizer.minimize(loss)

#################################

#CNN Auto encoder graph
with tf.name_scope("cnn"):
    he = tf.contrib.layers.variance_scaling_initializer()

    conv1 = tf.layers.conv2d(X, filters=nFilters1, kernel_size=3, strides=1, padding="SAME", name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=nFilters2, kernel_size=3, strides=2, padding="SAME", name="conv2")
    pool3 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    pool3_flat = tf.reshape(pool3, shape=[-1, 64 * 14 * 14])
    fc1 = tf.layers.dense(pool3_flat, 64, activation=tf.nn.relu, name="fc1")
    encodingMean = tf.layers.dense(hidden2, nHidden3, activation=None)
    encodingGamma = tf.layers.dense(hidden2, nHidden3, activation=None)
    noise = tf.random_normal(tf.shape(encodingGamma), dtype=tf.float32)
    encodingLayer = encodingMean + tf.exp(0.5 * encodingGamma) * noise
    fc2 = tf.layers.dense(encodingLayer, 64, activation=tf.nn.relu, name="fc2")

    hidden1 = tf.layers.dense(X, nHidden1, activation=tf.nn.relu, kernel_initializer=he, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, nHidden2, activation=tf.nn.relu, kernel_initializer=he, name="hidden2")
    hidden3Mean = tf.layers.dense(hidden2, nHidden3, activation=None)
    hidden3Gamma = tf.layers.dense(hidden2, nHidden3, activation=None)
    noise = tf.random_normal(tf.shape(hidden3Gamma), dtype=tf.float32)
    hidden3 = hidden3Mean + tf.exp(0.5 * hidden3Gamma) * noise
    hidden4 = tf.layers.dense(hidden3, nHidden4, activation=tf.nn.relu, kernel_initializer=he, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, nHidden5, activation=tf.nn.relu, kernel_initializer=he, name="hidden5")
    logits = tf.layers.dense(hidden5, nOutputs, activation=None)

################### End Graph Construction

with tf.Session() as sess:
    batcher = createMinibatch(imageFiles, batchsize)
    cycle = 0
    for batch in batcher:
        imageData = loadImageBatch(batch, height, width, channels)
        Xbatch = np.zeros((len(imageData), height, width, channels))
        for index, image in enumerate(imageData):
            Xbatch[index] = sess.run(cropImages, feed_dict={imageHolder: image})
            model.train_on_batch(Xbatch, Xbatch)

print("success")
