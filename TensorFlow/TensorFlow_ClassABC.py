from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf

class_letters = pd.read_csv('ABC_data.csv', header=None)
class_letters = class_letters.sort_values(81, ascending=True)

##### preparing the letter data for the neural network
train_x = class_letters.loc[:, :80]
num_features = len(train_x.loc[0])
# creating the one hot encoding for each unique letter
# I coded this so it will pick up the unique letter classes
# This way when we add the letter variants, they will still be part of the same class
class_targetLetter_List = list(np.unique(class_letters[81]))
num_classes = len(class_targetLetter_List)

letter_idx = []
for letter in range(0, len(class_targetLetter_List)):
    letter_idx.append([int(i == letter) for i in range(26)])

train_y = np.array(letter_idx)

x = tf.placeholder(tf.float32, [None, num_features])
W = tf.Variable(tf.zeros([num_features, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

#### setting up the cost function --- Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, num_classes])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

## The train will automatically use backpropgration to determine how varaibles affect the loss
## apply a gradient descent optimization with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# other training algos - https://www.tensorflow.org/versions/r0.12/api_docs/python/train/#optimizers

hm_epochs = 10 # an epoch is a feed forward + back propagation
batch_size = 1 # is the number of samples to feed into the network at a time
# define the operation
init = tf.global_variables_initializer()

### launch the model in a session and run the operation that initalizes the variables
sess = tf.Session()
sess.run(init)

# run the training step 1000 times!
##### I had to change how the code iterates through batches, since we cannot use the mnist next bactch anymore
#### instead of doing batches of 100, for our example, I am going to do batches of one letter at a time
for epoch in range(hm_epochs):
    epoch_loss = 0
    i = 0

    while i < len(train_x):

        # if we wanted to, we can change this to use our random letter generator that we have been using
        start = i
        end = i + batch_size
        batch_x = np.array(train_x[start:end])
        batch_y = np.array(train_y[start:end])

        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

        i += batch_size

    print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

# evaluate the accuracy of the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## I had to change the x and y data feeds to our letter data
print(sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))