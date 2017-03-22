import numpy as np 
import tensorflow as tf 


with tf.Session() as sess:
	a = tf.zeros((10, 10))
	b = tf.ones((10,10))
	c = a * b

	print(sess.run(c))



