import tensorflow as tf
from functools import partial
import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

def plot_image(image, shape=[28, 28]):
	plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
	plt.axis("off")

def createMinibatch(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt]

#Load in the dataset
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()

#Format the training set
Xtrain = Xtrain.reshape((len(Xtrain), -1)) #Flatten to 1D array
print(Xtrain.shape)
Xtrain = MinMaxScaler(feature_range=(0,1)).fit_transform(Xtrain)

nInputs = 28 * 28
nHidden1 = 500
nHidden2 = 300
nHidden3 = 30
nHidden4 = nHidden2
nHidden5 = nHidden3
nOutputs = nInputs
learningRate = 0.001
nEpochs = 1000
nDigits = 1
batchSize = 150
numBatches = len(Xtrain) // batchSize

X = tf.placeholder(tf.float32, shape=(None, nInputs), name="X")

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

init = tf.global_variables_initializer()

with tf.Session() as sess:
	init.run()
	for epoch in range(nEpochs):
		for batch in createMinibatch(Xtrain, batchSize, True):
			Xbatch = batch #Do it in batches
			print(Xbatch.shape)
			exit()
			sess.run(trainingOp, feed_dict={X: Xbatch})
		loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstructionLoss, latent_loss], feed_dict={X: Xtrain}) # not shown
		print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)  # not shown
	codings_rnd = np.random.normal(size=[nDigits, nHidden3])
	outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})
	plt.figure(figsize=(8,50))	
for iteration in range(nDigits):
	#plt.subplot(nDigits, 10, iteration + 1)
	print(outputs_val)
	plot_image(outputs_val[iteration])
plt.show()
