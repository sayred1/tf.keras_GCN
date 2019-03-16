#from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import sys
import tensorflow as tf
# MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data


'''

The data will be saved to --logdir, these datafiles are called even files, and they hold summary data
Type in tensorboard --logdir=summaries to activate what has been saved. These summaries can be viewed on the browser by typing in localhost:<session_id>

'''

# 5-layers NN to classify hand written digit images

def accuracy(predictions,labels):
    '''
    Accuracy of a given set of predictions of size (N x n_classes) and
    labels of size (N x n_classes)
    '''
    return np.sum(np.argmax(predictions,axis=1)==np.argmax(labels,axis=1))*100.0/labels.shape[0]

# Each image in MNIST is size 28x28 --> 784 when unwrapped

batch_size = 100
layer_ids = ['hidden1','hidden2','hidden3','hidden4','hidden5','out']
layer_sizes = [784, 500, 400, 300, 200, 100, 10]

#tf.reset_default_graph()

# Inputs and Labels
train_inputs = tf.placeholder(tf.float32, shape=[batch_size, layer_sizes[0]], name='train_inputs')
train_labels = tf.placeholder(tf.float32, shape=[batch_size, layer_sizes[-1]], name='train_labels')

# Weight and Bias definitions for each layer.
for idx, lid in enumerate(layer_ids):
    with tf.variable_scope(lid): # scope allows us to define layers
        w = tf.get_variable('weights',shape=[layer_sizes[idx], layer_sizes[idx+1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.05))
        b = tf.get_variable('bias',shape= [layer_sizes[idx+1]],
                            initializer=tf.random_uniform_initializer(-0.1,0.1))

# Upon defining input and output placeholders, as well as weights and biases per layer, we now introduce the forward pass to obtain logits (unnormalized values produced in the last layer).

# Calc logits
h = train_inputs
for lid in layer_ids:
    with tf.variable_scope(lid,reuse=True):
        w, b = tf.get_variable('weights'), tf.get_variable('bias')
        if lid != 'out':
          h = tf.nn.relu(tf.matmul(h,w)+b,name=lid+'_output')
        else:
          h = tf.nn.xw_plus_b(h,w,b,name=lid+'_output')

# make prediction
tf_predictions = tf.nn.softmax(h, name='predictions')

# calculate the loss
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_labels, logits=h),name='loss')

# optimize

tf_learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
optimizer = tf.train.MomentumOptimizer(tf_learning_rate,momentum=0.9)
grads_and_vars = optimizer.compute_gradients(tf_loss)
tf_loss_minimize = optimizer.minimize(tf_loss)

"""
Now we can start writing what we want to to our event files
"""

# use tf.name_scope to group scalars on the board. That is, scalars having the same name scope will be displayed on the same row.

with tf.name_scope('performance'):

    tf_loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary') # mean loss goes here
    tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph) # to view the mean loss, create summary scalar

    tf_accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary') # mean test accuracy
    tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph) # to view create summary scalar

# Gradient norm summary, applied at the last layer

for g,v in grads_and_vars:
    if 'hidden5' in v.name and 'weights' in v.name:
        with tf.name_scope('gradients'):
            tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
            tf_gradnorm_summary = tf.summary.scalar('grad_norm', tf_last_grad_norm)
            break

# Merge all summaries together
performance_summaries = tf.summary.merge([tf_loss_summary,tf_accuracy_summary])

# Now me must create a session to execute the above
# Then we create a summary folder
# Then a summary writer
# Initialize variables
# Load the dataset
# During Training:
# - For each epoch and batch, execulte then write gradnorm_summary (only for first batch to reduce clutter)
# - optimization then loss
# - after an entire epoch, calculate average training loss
# then validation
# then test: calc accuracy per batch
# finally, execute performance summary --> file via writer

image_size = 28
n_channels = 1
n_classes = 10
n_train = 55000
n_valid = 5000
n_test = 10000
n_epochs = 10

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # making sure Tensorflow doesn't overflow the GPU
session = tf.InteractiveSession(config=config)

# where the events are housed
if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','lr: .01')):
    os.mkdir(os.path.join('summaries','lr: .01'))

# how to write the events
summ_writer = tf.summary.FileWriter(os.path.join('summaries','.01'), session.graph)

tf.global_variables_initializer().run()

accuracy_per_epoch = []
# read in input data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)


# feed dict allows you to read the inputs into the placeholders.

for epoch in range(n_epochs):
    loss_per_epoch = []
    for i in range(n_train//batch_size): #55000//100, for i in number of batches

        # =================================== Training for one step ========================================
        batch = mnist_data.train.next_batch(batch_size)    # Get one batch of training data
        if i == 0:
            # Only for the first epoch, get the summary data
            # Otherwise, it can clutter the visualization
            l,_,gn_summ = session.run([tf_loss,tf_loss_minimize,tf_gradnorm_summary],
                                      feed_dict={train_inputs: batch[0].reshape(batch_size,image_size*image_size),
                                                 train_labels: batch[1],
                                                tf_learning_rate: .01})
            summ_writer.add_summary(gn_summ, epoch)
        else:
            # Optimize with training data
            l,_ = session.run([tf_loss,tf_loss_minimize],
                              feed_dict={train_inputs: batch[0].reshape(batch_size,image_size*image_size),
                                         train_labels: batch[1],
                                         tf_learning_rate: .01})
        loss_per_epoch.append(l)

    print('Average loss in epoch %d: %.5f'%(epoch,np.mean(loss_per_epoch)))
    avg_loss = np.mean(loss_per_epoch) # this goes into the board
    print(avg_loss.dtype)
    sys.exit()

    # ====================== Calculate the Validation Accuracy ===============================================
    valid_accuracy_per_epoch = []
    for i in range(n_valid//batch_size):
        valid_images,valid_labels = mnist_data.validation.next_batch(batch_size)
        valid_batch_predictions = session.run(
            tf_predictions,feed_dict={train_inputs: valid_images.reshape(batch_size,image_size*image_size)})
        valid_accuracy_per_epoch.append(accuracy(valid_batch_predictions,valid_labels))

    mean_v_acc = np.mean(valid_accuracy_per_epoch)
    print('\tAverage Valid Accuracy in epoch %d: %.5f'%(epoch,np.mean(valid_accuracy_per_epoch)))

    # ===================== Calculate the Test Accuracy ===============================
    accuracy_per_epoch = []
    for i in range(n_test//batch_size):
        test_images, test_labels = mnist_data.test.next_batch(batch_size)
        test_batch_predictions = session.run(
            tf_predictions,feed_dict={train_inputs: test_images.reshape(batch_size,image_size*image_size)}
        )
        accuracy_per_epoch.append(accuracy(test_batch_predictions,test_labels))

    print('\tAverage Test Accuracy in epoch %d: %.5f\n'%(epoch,np.mean(accuracy_per_epoch)))
    avg_test_accuracy = np.mean(accuracy_per_epoch) # this goes into the board

    # Execute the summaries defined above
    summ = session.run(performance_summaries, feed_dict={tf_loss_ph:avg_loss, tf_accuracy_ph:avg_test_accuracy})

    # Write the obtained summaries to the file, so it can be displayed in the TensorBoard
    summ_writer.add_summary(summ, epoch)

session.close()
