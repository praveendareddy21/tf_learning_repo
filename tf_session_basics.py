import numpy as np 
import tensorflow as tf 


with tf.Session() as sess:
	a = tf.zeros((10, 10))
	b = tf.ones((10,10))
	c = a * b

	print(sess.run(c))


a_ = []
b_ = []
c_ = []
for i in range(10):
	a_.append(i * i)
	b_.append(i)
	c_.append(i -1)



##### essentially, i * i - ( i * (i -1)) = i 

with tf.Session() as sess:
	a = tf.placeholder("float", [10])
	b = tf.placeholder("float", [ 10])
	c = tf.placeholder("float", [10])
	
	d = b * c
	e = a - d

	output_e = sess.run(e , feed_dict = {  ## feeds data to variables 
		a : a_, b : b_, c : c_ 
		} )
	print(output_e)

	output_e , output_d = sess.run([e, d], feed_dict = {   #for multiple outputs, use a list
		a : a_, b : b_, c : c_ 
		} )
	print(output_e)
	print(output_d)

