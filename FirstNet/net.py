import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd

mnist = input_data.read_data_sets("/tmp/data/")

#Training params info
nInputs = 28 * 28
nHidden1 = 300
nHidden2 = 100
nOutputs = 10
learningRate = 0.01
nEpochs = 50
batchSize = 50

###############Graph construction phase

#First define the shape of our inputs and labels
X = tf.placeholder(tf.float32, shape=(None, nInputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

#Define the graph for the DNN Layers
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, nHidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, nHidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, nOutputs, name="outputs")


#Graph for our loss an entropy
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

#Graph for performing the actual training
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    trainingOp = optimizer.minimize(loss)

#Graph for evaluating how our model is performing
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


################Execution Phase
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(nEpochs):
        for iteration in range(mnist.train.num_examples):
            Xbatch, ybatch, = mnist.train.next_batch(batchSize)
            sess.run(trainingOp, feed_dict={X: Xbatch, y: ybatch})
        accTrain = accuracy.eval(feed_dict={X: Xbatch, y:ybatch})
        accVal = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})

        print("Epoch: ", epoch, "Train Acc: ", accTrain, "Val Acc: ", accVal)
