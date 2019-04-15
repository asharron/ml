import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial

#Load in the dataset
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
#Reshape into 1D arrays
Xtrain = Xtrain.reshape(len(Xtrain), -1)
Xtest = Xtest.reshape(len(Xtest), -1)
#Filter out for digits 0-4
trainFilter = np.where(ytrain <= 4)
testFilter = np.where(ytest <= 4)
Xtrain = Xtrain[trainFilter]
ytrain = ytrain[trainFilter]
Xtest = Xtest[testFilter]
ytest = ytest[testFilter]

#logging directory for tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

#Training params info
nInputs = 28 * 28
nHidden1 = 100
nHidden2 = 100
nHidden3 = 100
nHidden4 = 100
nHidden5 = 100
nOutputs = 5
learningRate = 0.001
nEpochs = 100
batchSize = 50
numBatches = len(Xtrain) // batchSize
dropoutRate = 0.5

###############Graph construction phase

#First define the shape of our inputs and labels
X = tf.placeholder(tf.float32, shape=(None, nInputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name="training")
XDrop = tf.layers.dropout(X, dropoutRate, training=training)


#Define the graph for the DNN Layers
with tf.name_scope("dnn"):
    #Batch norm and He initalization
    batchNorm = partial(tf.layers.batch_normalization, training=training, momentum=0.9)
    he = tf.contrib.layers.variance_scaling_initializer()

    
    hidden1 = tf.layers.dense(XDrop, nHidden1, name="hidden1", kernel_initializer=he)
    hidden1Drop = tf.layers.dropout(hidden1, dropoutRate, training=training, name="dropout1")
    bn1 = batchNorm(hidden1Drop, name="batchNorm1")
    bn1Act = tf.nn.elu(bn1, name="batchAct1")
    hidden2 = tf.layers.dense(bn1Act, nHidden2, name="hidden2", kernel_initializer=he)
    hidden2Drop = tf.layers.dropout(hidden2, dropoutRate, training=training, name="dropout2")
    bn2 = batchNorm(hidden2Drop, name="batchNorm2")
    bn2Act = tf.nn.elu(bn2, name="batchAct2")
    hidden3 = tf.layers.dense(bn2Act, nHidden3, name="hidden3", kernel_initializer=he)
    hidden3Drop = tf.layers.dropout(hidden3, dropoutRate, training=training, name="dropout3")
    bn3 = batchNorm(hidden3Drop, name="batchNorm3")
    bn3Act = tf.nn.elu(bn3, name="batchAct3")
    hidden4 = tf.layers.dense(bn3Act, nHidden4, name="hidden4", kernel_initializer=he)
    hidden4Drop = tf.layers.dropout(hidden3, dropoutRate, training=training, name="dropout4")
    bn4 = batchNorm(hidden4Drop, name="batchNorm4")
    bn4Act = tf.nn.elu(bn4, name="batchAct4")
    hidden5 = tf.layers.dense(bn4Act, nHidden5, name="hidden5", kernel_initializer=he)
    hidden5Drop = tf.layers.dropout(hidden5, dropoutRate, training=training, name="dropout5")
    bn5 = batchNorm(hidden5Drop, name="batchNorm5")
    bn5Act = tf.nn.elu(bn5, name="batchAct5")
    logits = tf.layers.dense(bn5, nOutputs, name="outputs")


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
        accTrain = accuracy.eval(feed_dict={X: Xtrain, y:ytrain})
        accVal = accuracy.eval(feed_dict={X: Xtest, y: ytest})
        print("Epoch: ", epoch, "Train Acc: ", accTrain, "Val Acc: ", accVal)
    save = saver.save(sess, "./models/classifier04.ckpt")

file_writer.close()
