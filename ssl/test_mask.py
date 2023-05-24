import tensorflow as tf
import numpy as np
import copy
import random

random.seed(1234)

import mask

import matplotlib.pyplot as plt
fig = plt.figure()

idx = 1
for i in range(3):
  #x = tf.reshape(tf.range(2000, dtype=tf.float32), [1, 100, 20])
  #ref = tf.reshape(tf.range(2000, dtype=tf.float32), [1, 100, 20]) / 2
  x = tf.reshape(tf.ones(4*400, dtype=tf.float32), [4, 20, 20])
  ref = tf.reshape(tf.ones(4*400, dtype=tf.float32), [4, 20, 20]) / 2

  _res = mask.mask_tera(x, mask_ratio=0.5, mask_freq=0.5)
  print(_res)

  ax = fig.add_subplot(2*3, 2, idx)
#  ax.imshow(tf.squeeze(_res[1], 0))
  ax.imshow(_res[1][0]); idx += 1

  ax = fig.add_subplot(2*3, 2, idx)
#  ax.imshow(tf.squeeze(_res[0], 0))
  ax.imshow(_res[0][0]); idx += 1

  ax = fig.add_subplot(2*3, 2, idx)
#  ax.imshow(tf.squeeze(_res[1], 0))
  ax.imshow(_res[1][1]); idx += 1

  ax = fig.add_subplot(2*3, 2, idx)
#  ax.imshow(tf.squeeze(_res[0], 0))
  ax.imshow(_res[0][1]); idx += 1

plt.savefig('test_mask.png')
