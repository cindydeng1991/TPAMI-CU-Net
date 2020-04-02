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
import time
import gc
os.environ['CUDA_VISIBLE_DEVICES']='1'
# Parameters
Height=1320
Width=1080
batch_size = 2
n=64 # number of filters
s=8 # filter size
nl=4 # number of layers
Channel=1

X = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Z = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])

def eta(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)

def get_u(x):
    Wx_00=tf.get_variable("Wx_00", shape=[s, s, Channel, n], initializer=tf.contrib.layers.xavier_initializer())
    lamx_00=tf.get_variable("lamx_00", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
    p1 = tf.nn.conv2d(x, Wx_00, strides=[1, 1, 1, 1], padding='SAME')
    tensor = eta(p1, lamx_00)
    for i in range(nl):
       Wx = tf.get_variable("Wx_%02d" % (i + 1), shape=[s, s, n, Channel], initializer=tf.contrib.layers.xavier_initializer())
       Wxx = tf.get_variable("Wxx_%02d" % (i + 1), shape=[s, s, Channel, n], initializer=tf.contrib.layers.xavier_initializer())
       lamx = tf.get_variable("lamx_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
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
    tensor = eta(p1, lamy_00)
    for i in range(nl):
        Wy = tf.get_variable("Wy_%02d" % (i + 1), shape=[s, s, n, Channel],
                             initializer=tf.contrib.layers.xavier_initializer())
        Wyy = tf.get_variable("Wyy_%02d" % (i + 1), shape=[s, s, Channel, n],
                              initializer=tf.contrib.layers.xavier_initializer())
        lamy = tf.get_variable("lamy_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
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
p8x = tf.subtract(X, tf.nn.conv2d(u, Wdx, strides=[1, 1, 1, 1], padding='SAME'))
p8y = tf.subtract(Y, tf.nn.conv2d(v, Wdy, strides=[1, 1, 1, 1], padding='SAME'))
p9xy = tf.concat([p8x,p8y], 3)
def get_z(y):
    Wz_00 = tf.get_variable("Wz_00", shape=[s, s, 2*Channel, n], initializer=tf.contrib.layers.xavier_initializer())
    lamz_00 = tf.get_variable("lamz_00", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
    p1 = tf.nn.conv2d(y, Wz_00, strides=[1, 1, 1, 1], padding='SAME')
    tensor = eta(p1, lamz_00)
    for i in range(nl):
        Wz = tf.get_variable("Wz_%02d" % (i + 1), shape=[s, s, n, 2*Channel],
                             initializer=tf.contrib.layers.xavier_initializer())
        Wzz = tf.get_variable("Wzz_%02d" % (i + 1), shape=[s, s, 2*Channel, n],
                              initializer=tf.contrib.layers.xavier_initializer())
        lamz = tf.get_variable("lamz_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())

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
    Z_rec = rec_z+rec_u
    return Z_rec

f_pred = decoder(u,z)

depth=np.load("data_d.npy")
color=np.load("data_c.npy")
saver = tf.train.Saver()
with tf.Session() as sess:
     saver.restore(sess, './model/model.ckpt-106')
     batch_xs=depth[0:2,0:Height, 0:Width, :]
     batch_ys=color[0:2,0:Height, 0:Width, :]
     start = time.time()
     Y_p=sess.run(f_pred, feed_dict={X: batch_xs, Y: batch_ys})
     end = time.time()
     print('running time: ',end-start)
     sio.savemat('result.mat', {'Y_p': Y_p})


