import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial

#Set the random seed to get same results each run
np.random.seed(42)

#logging directory for tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

#Load in the dataset
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()

#Shuffle the dataset
idx = np.random.permutation(len(Xtrain))
Xtrain = Xtrain[idx]
ytrain = ytrain[idx]

#Format the training set
Xtrain = Xtrain.reshape((len(Xtrain), -1)) #Flatten to 1D array
Xtest = Xtest.reshape(len(Xtest), -1)

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

###############Graph construction phase

#First define the shape of our inputs and labels
X = tf.placeholder(tf.float32, shape=(None, 784), name="X")
X_reshape = tf.reshape(X, shape=[-1, 28, 28, 1])
y = tf.placeholder(tf.int64, shape=(None), name="y")

#Define the graph for the DNN Layers
with tf.name_scope("cnn"):
    #Batch norm and He initalization
    he = tf.contrib.layers.variance_scaling_initializer()
    
    conv1 = tf.layers.conv2d(X_reshape, filters=nFilters1, kernel_size=3, strides=1, padding="SAME", name="conv1")
    #conv2 = tf.layers.conv2d(conv1, filters=nFilters2, kernel_size=3, strides=2, padding="SAME", name="conv2")
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

mse_summary = tf.summary.scalar('MSE', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
################Execution Phase
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    for epoch in range(nEpochs):
        for iteration in range(numBatches):
            Xbatch = Xtrain[iteration * batchSize:(iteration+1) * batchSize]
            ybatch = ytrain[iteration * batchSize:(iteration+1) * batchSize]
            sess.run(trainingOp, feed_dict={X: Xbatch, y: ybatch})
            if iteration % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: Xbatch, y:ybatch})
                step = epoch * len(Xtrain) + iteration
                file_writer.add_summary(summary_str, step)
        accTrain = accuracy.eval(feed_dict={X: Xbatch, y:ybatch})
        accVal = accuracy.eval(feed_dict={X: Xtest, y: ytest})
        print("Epoch: ", epoch, "Train Acc: ", accTrain, "Val Acc: ", accVal)
    save = saver.save(sess, "./models/cnn.ckpt")

file_writer.close()
