import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import numpy as np
from datetime import datetime

#Set the random seed to get the same results each time
np.random.seed(42)

#Load in the dataset
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
#Reshape into 1D arrays
Xtrain = Xtrain.reshape(len(Xtrain), -1)
Xtest = Xtest.reshape(len(Xtest), -1)

#Shuffle the dataset
idx = np.random.permutation(len(Xtrain))
Xtrain = Xtrain[idx]
ytrain = ytrain[idx]

#Split it up into two groups
split1X = Xtrain[:55000]
split1y = ytrain[:55000]
split2X = Xtrain[55000:]
split2y = ytrain[55000:]

sameCount = 0
diffCount = 0
for i in range(0, len(split1y), 2):
    if split1y[i] == split1y[i+1]:
        sameCount += 1
    else:
        diffCount += 1
print(sameCount, "::::::", diffCount)
exit()


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

#Params for DNN
learningRate = 0.01
nInputs = 28 * 28
nHidden1 = 100
nHidden2 = 100
nHidden3 = 100
nHidden4 = 100
nHidden5 = 100
nHidden6 = 10
outputs = 1
nEpochs = 100
batchSize = 50

#Graph Construction phase 
leftX = tf.placeholder(tf.float32, shape=(None, nInputs), name="leftX")
rightX = tf.placeholder(tf.float32, shape=(None, nInputs), name="rightX")

he = tf.contrib.layers.variance_scaling_initializer()

with tf.name_scope("leftDnn"):
    leftHidden1 = tf.layers.dense(leftX, nHidden1, kernel_initializer=he, activation=tf.nn.elu, name="leftHidden1")
    leftHidden2 = tf.layers.dense(leftHidden1, nHidden2, kernel_initializer=he, activation=tf.nn.elu, name="leftHidden2")
    leftHidden3 = tf.layers.dense(leftHidden2, nHidden3, kernel_initializer=he, activation=tf.nn.elu, name="leftHidden3")
    leftHidden4 = tf.layers.dense(leftHidden3, nHidden4, kernel_initializer=he, activation=tf.nn.elu, name="leftHidden4")
    leftHidden5 = tf.layers.dense(leftHidden4, nHidden5, kernel_initializer=he, activation=tf.nn.elu, name="leftHidden5")

with tf.name_scope("rightDnn"):
    rightHidden1 = tf.layers.dense(rightX, nHidden1, kernel_initializer=he, activation=tf.nn.elu, name="rightHidden1")
    rightHidden2 = tf.layers.dense(rightHidden1, nHidden2, kernel_initializer=he, activation=tf.nn.elu, name="rightHidden2")
    rightHidden3 = tf.layers.dense(rightHidden2, nHidden3, kernel_initializer=he, activation=tf.nn.elu, name="rightHidden3")
    rightHidden4 = tf.layers.dense(rightHidden3, nHidden4, kernel_initializer=he, activation=tf.nn.elu, name="rightHidden4")
    rightHidden5 = tf.layers.dense(rightHidden4, nHidden5, kernel_initializer=he, activation=tf.nn.elu, name="rightHidden5")

with tf.name_scope("dnnConcat"):
    hidden6 = tf.concat([leftHidden5, rightHidden5], axis=1)
    logits = tf.layers.dense(hidden6, nOutputs, name="outputs" )

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
