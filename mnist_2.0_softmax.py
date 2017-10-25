"""
mnist using 4 hidden layers and using softmax crosss entropy with logits
Accuracy : 0.9762
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
mnist = mnist_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32,[None, 784])
y_original = tf.placeholder(tf.float32,[None, 10])

layer1 = 200 		#200 neurons in 1st hidden layer 
layer2 = 100
layer3 = 60
layer4 = 30
layer5 = 10

w1 = tf.Variable(tf.truncated_normal([784, layer1], stddev=0.01))
b1 = tf.Variable(tf.zeros([layer1]))
w2 = tf.Variable(tf.truncated_normal([layer1, layer2], stddev=0.01))
b2 = tf.Variable(tf.zeros([layer2]))
w3 = tf.Variable(tf.truncated_normal([layer2, layer3], stddev=0.01))
b3 = tf.Variable(tf.zeros([layer3]))
w4 = tf.Variable(tf.truncated_normal([layer3, layer4], stddev=0.01))
b4 = tf.Variable(tf.zeros([layer4]))
w5 = tf.Variable(tf.truncated_normal([layer4,layer5], stddev=0.01))
b5 = tf.Variable(tf.zeros([layer5]))

y1 = tf.nn.sigmoid(tf.matmul(x , w1) + b1)
y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
y3 = tf.nn.sigmoid(tf.matmul(y2, w3) + b3)
y4 = tf.nn.sigmoid(tf.matmul(y3, w4) + b4)
yLogits = tf.matmul(y4, w5) + b5
y = tf.nn.softmax(yLogits)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yLogits, labels=y_original))
train_step = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
	print("Iteration ", i)
	x_batch, y_batch = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: x_batch, y_original: y_batch})

correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_original,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
print("^^^ Accuracy ^^^ : ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_original: mnist.test.labels}))