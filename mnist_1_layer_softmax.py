""" 
mnist using 1 layer and using softmax crosss entropy with logits
Accuracy : 0.922

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
mnist = mnist_data.read_data_sets("data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_predition = tf.matmul(x, w) + b
y_original = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_original, logits=y_predition))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(2000):
	print("Iteration ",i)
	x_batch, y_batch = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: x_batch, y_original: y_batch})

correct_pred = tf.equal(tf.argmax(y_predition, 1), tf.argmax(y_original, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print("^^^ Accuracy ^^^ : ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_original: mnist.test.labels}))
