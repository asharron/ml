import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import numpy as np
from datetime import datetime
from functools import partial

#Load in the dataset
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
#Rehsape into 1D arrays
Xtrain = Xtrain.reshape(len(Xtrain), -1)
Xtest = Xtest.reshape(len(Xtest), -1)
#Filter out for digits 5-9
trainFilter = np.where(ytrain > 4)
testFilter = np.where(ytest > 4)
Xtrain = Xtrain[trainFilter]
ytrain = ytrain[trainFilter]
ytrain = np.subtract(ytrain, 5)
Xtest = Xtest[testFilter]
ytest = ytest[testFilter]
ytest = np.subtract(ytest, 5)

#Setup Logging directory
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

#Training Params
nInputs = 28 * 28
nHidden1 = 100
nHidden2 = 100
nHidden3 = 100
nHidden4 = 100
nHidden5 = 100
nHidden6 = 100
nHidden7 = 100
nOutputs = 5
learningRate = 0.001
nEpochs = 100
batchSize = 50
numBatches = len(Xtrain) // batchSize
dropoutRate = 0.5


###############Graph Construction Phase

metaGraph = tf.train.import_meta_graph("./models/classifier04.ckpt.meta")

#Tell the saver which layers we want to use, in this case, hidden1-5
reuseVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden[12345]")
reuseVarsDict = dict([(var.op.name, var) for var in reuseVars])
restoredSaver = tf.train.Saver(reuseVarsDict)

#Grab X and y from old graph
X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

#Graph for the DNN
with tf.name_scope("dnn"):
    he = tf.contrib.layers.variance_scaling_initializer()

    hidden1 = tf.layers.dense(X, nHidden1, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, nHidden2, name="hidden2")
    hidden3 = tf.layers.dense(hidden2, nHidden3, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, nHidden4, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, nHidden5, name="hidden5")
    hidden5Stop = tf.stop_gradient(hidden5)
    logits = tf.layers.dense(hidden5Stop, nOutputs, name="outputs")

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

#Grab training operations and evalution
#loss = tf.get_default_graph().get_tensor_by_name("loss/loss:0")
#trainingOp = tf.get_default_graph().get_operation_by_name("train/trainingOp")
#
#correct = tf.nn.in_top_k(logits, y, 1)
#accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

mse_summary = tf.summary.scalar('MSE', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restoredSaver.restore(sess, "./models/classifier04.ckpt")
    #Train here
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
    saved = saver.save(sess, "./models/classifier59.ckpt")
