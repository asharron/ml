import tensorflow as tf 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housingData = fetch_california_housing()
m, n = housingData.data.shape

datasetBias = np.c_[np.ones((m, 1)), housingData.data]
dataset = StandardScaler().fit_transform(datasetBias)

numEpochs = 1000
learningRate = 0.1

#Declaring our input & output
X = tf.constant(dataset, dtype=tf.float32, name="X")
y = tf.constant(housingData.target.reshape(-1, 1), dtype=tf.float32, name="y")
#Creating our weights
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
#Calcuation for our feed forward, or prediction
yPred = tf.matmul(X, theta, name="predictions")
#Metric for error
error = yPred - y
#Calcualte the mean squared error
mse = tf.reduce_mean(tf.square(error), name="mse")
#Create our gradient 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
#Create our training operation
trainingOp = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(numEpochs):
        if epoch % 100 == 0:
            print("Epoch: ", epoch, " MSE = ", mse.eval())
        sess.run(trainingOp)
    best_theta = theta.eval()
    print(best_theta)


