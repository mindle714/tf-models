import tensorflow as tf
import functools
import numpy as np
from util import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class waveunet(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(waveunet, self).__init__(*args, **kwargs)
    self.layer = 8
    self.dim = 64
    self.ksize = 5

  def build(self, input_shape):
    conv_opt = dict(padding='same')

    self.conv_pre = conv1d(self.dim, self.ksize, **conv_opt)
    self.conv_post = conv1d(1, self.ksize, **conv_opt)

    self.down_convs = [conv1d(self.dim, self.ksize,
      strides=2, **conv_opt) for _ in range(self.layer)]    
    self.up_convs = [conv1dtrans(self.dim, self.ksize,
      strides=2, **conv_opt) for _ in range(self.layer)]

    self.conv_mid = conv1d(self.dim, self.ksize, **conv_opt)

  def call(self, inputs, training=None):
    x, ref = inputs

    if ref is not None:
      ref = tf_expd(ref, -1)

    x = tf_expd(x, -1)
    x = tf.nn.relu(self.conv_pre(x))

    encs = []
    for down_conv in self.down_convs:
      x = tf.nn.relu(down_conv(x))
      encs.append(x)

    x = self.conv_mid(x)

    encs = encs[::-1]
    for enc, up_conv in zip(encs, self.up_convs):
      x = tf.concat([x, enc], -1)
      x = tf.nn.relu(up_conv(x))
    
    x = self.conv_post(x)
    x = tf.math.tanh(x)

    if ref is not None:
      samp_loss = tf.math.reduce_mean((x - ref) ** 2)
      def stft_loss(

    return x
