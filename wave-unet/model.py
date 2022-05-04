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
    self.dims = [64, 64, 64, 64, 128, 128, 128, 128]
    self.ksize = 5

  def build(self, input_shape):
    conv_opt = dict(padding='same')

    self.conv_pre = conv1d(self.dims[0], self.ksize, **conv_opt)
    self.conv_post = conv1d(1, self.ksize, **conv_opt)

    self.down_convs = list(zip(
      [conv1d(self.dims[idx], 3,
        strides=2, **conv_opt) for idx in range(self.layer)],
      [conv1d(self.dims[idx], self.ksize,
        strides=1, **conv_opt) for idx in range(self.layer)]))

    self.up_convs = list(zip(
      [conv1dtrans(self.dims[::-1][idx], 3,
        strides=2, **conv_opt) for idx in range(self.layer)],
      [conv1d(self.dims[::-1][idx], self.ksize,
        strides=1, **conv_opt) for idx in range(self.layer)]))

    self.conv_mid = conv1d(self.dims[-1], self.ksize, **conv_opt)

  def call(self, inputs, training=None):
    x, ref = inputs

    x = tf_expd(x, -1)
    x = tf.nn.relu(self.conv_pre(x))

    encs = []
    for down_conv, conv in self.down_convs:
      x = tf.nn.relu(conv(x))
      encs.append(x)
      x = tf.nn.relu(down_conv(x))

    x = self.conv_mid(x)

    encs = encs[::-1]
    for enc, (up_conv, conv) in zip(encs, self.up_convs):
      x = tf.nn.relu(up_conv(x))
      x = tf.concat([x, enc], -1)
      x = tf.nn.relu(conv(x))
    
    x = self.conv_post(x)
    x = tf.math.tanh(x)
    x = tf.squeeze(x, -1)

    if ref is not None:
      samp_loss = tf.math.reduce_mean((x - ref) ** 2)

      def stft_loss(x, ref, frame_length, frame_step, fft_length):
        stft_opt = dict(frame_length=frame_length,
          frame_step=frame_step, fft_length=fft_length)
        mag_x = tf.math.abs(stft(x, **stft_opt))
        mag_ref = tf.math.abs(stft(ref, **stft_opt))

        fro_opt = dict(axis=(-2, -1), ord='fro')
        sc_loss = tf.norm(mag_x - mag_ref, **fro_opt) / (tf.norm(mag_x, **fro_opt) + 1e-9)
        sc_loss = tf.reduce_mean(sc_loss)

        mag_loss = tf.math.log(mag_x + 1e-9) - tf.math.log(mag_ref + 1e-9)
        mag_loss = tf.reduce_mean(tf.math.abs(mag_loss))

        return sc_loss + mag_loss

      spec_loss = stft_loss(x, ref, 25, 5, 1024)
      spec_loss += stft_loss(x, ref, 50, 10, 2048)
      spec_loss += stft_loss(x, ref, 10, 2, 512)

      return samp_loss + spec_loss

    return x
