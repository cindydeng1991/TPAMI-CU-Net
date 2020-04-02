from __future__ import division, print_function, absolute_import

import math
import os
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import gc
os.environ['CUDA_VISIBLE_DEVICES']='0'
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allocator_type = 'BFC'
# learning rate
starter_learning_rate = 0.0001

# Network Parameters
Height=64
Width=64
batch_size = 64
n=64 # number of filters
s=8 #filter size
nl=4 # number of layers
Channel=1 # 1 for grayscale images, 3 for RGB images
# tf Graph input (only pictures)
X = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Z = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
weights = []

def eta(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)

def get_u(x):
    Wx_00=tf.get_variable("Wx_00", shape=[s, s, Channel, n], initializer=tf.contrib.layers.xavier_initializer())
    lamx_00=tf.get_variable("lamx_00", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
    p1 = tf.nn.conv2d(x, Wx_00, strides=[1, 1, 1, 1], padding='SAME')
    weights.append(Wx_00)
    weights.append(lamx_00)
    tensor = eta(p1, lamx_00)
    for i in range(nl):
       Wx = tf.get_variable("Wx_%02d" % (i + 1), shape=[s, s, n, Channel], initializer=tf.contrib.layers.xavier_initializer())
       Wxx = tf.get_variable("Wxx_%02d" % (i + 1), shape=[s, s, Channel, n], initializer=tf.contrib.layers.xavier_initializer())
       lamx = tf.get_variable("lamx_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
       weights.append(Wx)
       weights.append(Wxx)
       weights.append(lamx)
       p3 = tf.nn.conv2d(tensor, Wx, strides=[1, 1, 1, 1], padding='SAME')
       p4 = tf.nn.conv2d(p3, Wxx, strides=[1, 1, 1, 1], padding='SAME')
       p5 = tf.subtract(tensor,p4)
       p6 = tf.add(p1,p5)
       tensor = eta(p6, lamx)
    return tensor
def get_v(y):
    Wy_00 = tf.get_variable("Wy_00", shape=[s, s, Channel, n], initializer=tf.contrib.layers.xavier_initializer())
    lamy_00 = tf.get_variable("lamy_00", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
    p1 = tf.nn.conv2d(y, Wy_00, strides=[1, 1, 1, 1], padding='SAME')
    weights.append(Wy_00)
    weights.append(lamy_00)
    tensor = eta(p1, lamy_00)
    for i in range(nl):
        Wy = tf.get_variable("Wy_%02d" % (i + 1), shape=[s, s, n, Channel],
                             initializer=tf.contrib.layers.xavier_initializer())
        Wyy = tf.get_variable("Wyy_%02d" % (i + 1), shape=[s, s, Channel, n],
                              initializer=tf.contrib.layers.xavier_initializer())
        lamy = tf.get_variable("lamy_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
        weights.append(Wy)
        weights.append(Wyy)
        weights.append(lamy)
        p3 = tf.nn.conv2d(tensor, Wy, strides=[1, 1, 1, 1], padding='SAME')
        p4 = tf.nn.conv2d(p3, Wyy, strides=[1, 1, 1, 1], padding='SAME')
        p5 = tf.subtract(tensor, p4)
        p6 = tf.add(p1, p5)
        tensor = eta(p6, lamy)
    return tensor

u = get_u(X)
v = get_v(Y)
Wdx=tf.get_variable("Wdx", shape=[s,s,n,Channel], initializer=tf.contrib.layers.xavier_initializer())
Wdy=tf.get_variable("Wdy", shape=[s,s,n,Channel], initializer=tf.contrib.layers.xavier_initializer())
weights.append(Wdx)
weights.append(Wdy)
p8x = tf.subtract(X, tf.nn.conv2d(u, Wdx, strides=[1, 1, 1, 1], padding='SAME'))
p8y = tf.subtract(Y, tf.nn.conv2d(v, Wdy, strides=[1, 1, 1, 1], padding='SAME'))
p9xy = tf.concat([p8x,p8y], 3)
#print(p9xy.shape)
def get_z(y):
    Wz_00 = tf.get_variable("Wz_00", shape=[s, s, 2*Channel, n], initializer=tf.contrib.layers.xavier_initializer())
    lamz_00 = tf.get_variable("lamz_00", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
    p1 = tf.nn.conv2d(y, Wz_00, strides=[1, 1, 1, 1], padding='SAME')
    weights.append(Wz_00)
    weights.append(lamz_00)
    tensor = eta(p1, lamz_00)
    for i in range(nl):
        Wz = tf.get_variable("Wz_%02d" % (i + 1), shape=[s, s, n, 2*Channel],
                             initializer=tf.contrib.layers.xavier_initializer())
        Wzz = tf.get_variable("Wzz_%02d" % (i + 1), shape=[s, s, 2*Channel, n],
                              initializer=tf.contrib.layers.xavier_initializer())
        lamz = tf.get_variable("lamz_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
        weights.append(Wz)
        weights.append(Wzz)
        weights.append(lamz)
        p3 = tf.nn.conv2d(tensor, Wz, strides=[1, 1, 1, 1], padding='SAME')
        p4 = tf.nn.conv2d(p3, Wzz, strides=[1, 1, 1, 1], padding='SAME')
        p5 = tf.subtract(tensor, p4)
        p6 = tf.add(p1, p5)
        tensor = eta(p6, lamz)
    return tensor

z = get_z(p9xy)

def decoder(u,z):
    Rez = tf.get_variable("Rez", shape=[s, s, n, Channel], initializer=tf.contrib.layers.xavier_initializer())
    Reu = tf.get_variable("Reu", shape=[s, s, n, Channel], initializer=tf.contrib.layers.xavier_initializer())
    rec_z=tf.nn.conv2d(z,Rez,strides=[1, 1, 1, 1], padding='SAME')
    rec_u = tf.nn.conv2d(u, Reu, strides=[1, 1, 1, 1], padding='SAME')
    weights.append(Rez)
    weights.append(Reu)
    Z_rec = rec_z+rec_u
    return Z_rec

f_pred = decoder(u,z)
f_true = Z
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           200000, 0.9, staircase=True)
# Define loss and optimizer, minimize the squared error
loss = 1000*tf.reduce_mean(tf.pow(f_true - f_pred, 2))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=None)
train_data_x = np.load("data_x.npy")
train_data_y = np.load("data_y.npy")
train_label = np.load("label.npy")

batch_num = len(train_data_x) //batch_size

print(len(train_data_x), batch_num)
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print('Training...')
    for ep in range(300):
        total_loss = 0
        indices = np.random.permutation(train_data_x.shape[0])
        train_data_x = train_data_x[indices,:,:,:]
        train_data_y = train_data_y[indices,:,:,:]
        train_label = train_label[indices,:,:,:]

        for idx in range(0, batch_num):
           batch_xs = train_data_x[idx*batch_size : (idx+1)*batch_size,:,:,:]
           batch_ys = train_data_y[idx*batch_size : (idx+1)*batch_size,:,:,:]
           batch_label = train_label[ idx * batch_size: (idx + 1) * batch_size,:,:,:]
           _, loss_batch = sess.run([train_step, loss], feed_dict={X: batch_xs, Y: batch_ys, Z: batch_label})
           total_loss += loss_batch
           #print(' ep, idx, loss_batch = %6d:%6d: %6.3f' % (ep,idx, loss_batch))
        print(' ep, total_loss = %6d: %6.5f' % (ep, total_loss))

        gc.collect()
        checkpoint_path = os.path.join('model', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=ep)
    print("Optimization Finished!")


