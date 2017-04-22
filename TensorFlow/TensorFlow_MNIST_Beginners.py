from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                  help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()
# mnist is my input data sources - http://yann.lecun.com/exdb/mnist/
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

##### preparing the nueral network weight variables
num_features = len(mnist.train.images[0])
x = tf.placeholder(tf.float32, [None, num_features])  # 784, the number of features
W = tf.Variable(tf.zeros([num_features, 10]))
b = tf.Variable(tf.zeros([10]))  # 10 corresponds to the number of target vectors
y = tf.nn.softmax(tf.matmul(x, W) + b)

#### setting up the cost function --- Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

## The train will automatically use backpropgration to determine how varaibles affect the loss
## apply a gradient descent optimization with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# other training algos - https://www.tensorflow.org/versions/r0.12/api_docs/python/train/#optimizers

# define the operation
init = tf.global_variables_initializer()

### launch the model in a session and run the operation that initalizes the variables
sess = tf.Session()
sess.run(init)

# run the training step 1000 times!

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 100 corresponds to the training batch size
    # Using small batches of random data is called stochastic training -
    # in this case, stochastic gradient descent
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# evaluate the accuracy of the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))