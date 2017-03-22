import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import Random
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops


#### model description here

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us



X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625]) # create symbolic variables
w_o = init_weights([625, 10])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs


optimize = tf.contrib.layers.optimize_loss(cost,global_step=tf.Variable(0),
	optimizer=tf.train.AdamOptimizer(learning_rate=0.05), learning_rate= 0.05
	)

correct_pred = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

### end of model

#train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer


### dataset, using importable mnist dataset for simplicity

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


### do actual graph computation 
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(100):
	#for start, end in zip(range(0, len(teX), 128), range(128, len(teX)+1, 128)):
		#temp_accuacy = accuracy.eval(feed_dict={X: trX, Y: trY})
		
		

		_, train_acc, train_loss, train_pred = sess.run(
                    [optimize, accuracy, cost, py_x],
                    feed_dict={
                        X: trX,
                        Y: trY
                    }
                )
		print(str(i) + " : "+ str(train_pred))