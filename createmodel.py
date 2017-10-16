#import modules 
import tensorflow as tf
import numpy as np 
import load_data
#import data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Start create and train nodel

mmmm = load_data.load_data()
training_data = mmmm[0]
test_data = mmmm[1]
sess = tf.InteractiveSession()
# Create the model
x = tf.placeholder(tf.float32, [None, 13456])   #input image from mnist.train 
y_ = tf.placeholder(tf.float32, [None, 2])   #target result from mnist 
W = tf.Variable(tf.zeros([13456, 2]))          #weight matrix 
b = tf.Variable(tf.zeros([2]))               #bias matrix 
y = tf.nn.softmax(tf.matmul(x, W) + b)        #predict result 

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def next_batch(num, data, labels):
  idx = np.arange(1 , len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[ i] for i in idx]
  labels_shuffle = [labels[ i] for i in idx]
  return np.asarray(data_shuffle), np.asarray(labels_shuffle)


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,116,116,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# Define loss and optimizer
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#with tf.Session() as sess:
    #sess.run(init_op)
for i in range(1000):
  batch_xs,batch_ys = next_batch(50,training_data[0],training_data[1])
  batch_xs = np.reshape(batch_xs,(50,2))
  batch_ys = np.reshape(batch_ys,(50,2))
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch_xs, y_: batch_ys, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    
save_path = saver.save(sess, "model2.ckpt")
print ("Model saved in file: ", save_path)
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_data[0], y_: test_data[1], keep_prob: 1.0}))


