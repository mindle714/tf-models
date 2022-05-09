import tensorflow as tf
import functools
import numpy as np
from util import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class pdisc(tf.keras.layers.Layer):
  def __init__(self, period, ksize, stride, **kwargs):
    super(pdisc, self).__init__(**kwargs)
    self.period = period
    self.ksize = ksize
    self.stride = stride
  
  def build(self, input_shape):
    conv_opt = dict(padding='same')

    self.convs = [
      conv2d(32, (self.ksize, 1), strides=(self.stride, 1), **conv_opt),
      conv2d(32, (self.ksize, 1), strides=(self.stride, 1), **conv_opt),
      conv2d(64, (self.ksize, 1), strides=(self.stride, 1), **conv_opt),
      conv2d(128, (self.ksize, 1), strides=(self.stride, 1), **conv_opt),
      conv2d(128, (self.ksize, 1), strides=1, **conv_opt)]

    self.conv_post = conv2d(1, (3, 1), strides=1, **conv_opt)

  def call(self, inputs, training=None):
    x = inputs

    x_shape = tf.shape(x)
    mod = tf.math.floormod(x_shape[-1], self.period)
    x = tf.pad(x, [[0, 0], [0, self.period-mod]])

    x = tf.reshape(x, [x_shape[0],
      x_shape[1]//self.period + 1, self.period])

    x = tf_expd(x, -1)
    for conv in self.convs:
      x = tf.nn.relu(conv(x))

    x = self.conv_post(x)
    x = tf.squeeze(x, -1)
    x = tf.math.reduce_mean(x, -1)
    x = tf.math.sigmoid(x)

    return x

class sdisc(tf.keras.layers.Layer):
  def __init__(self, ksize, stride, **kwargs):
    super(sdisc, self).__init__(**kwargs)
    self.ksize = ksize
    self.stride = stride
  
  def build(self, input_shape):
    conv_opt = dict(padding='same')

    self.pool = tf.keras.layers.AveragePooling1D(
      self.ksize, self.stride)

    self.convs = [
      conv1d(32, 15, strides=1, **conv_opt),
      conv1d(32, 41, strides=2, groups=4, **conv_opt),
      conv1d(32, 41, strides=2, groups=16, **conv_opt),
      conv1d(64, 41, strides=4, groups=16, **conv_opt),
      conv1d(128, 41, strides=4, groups=16, **conv_opt),
      conv1d(128, 41, strides=1, groups=16, **conv_opt),
      conv1d(128, 5, strides=1, **conv_opt)]

    self.conv_post = conv1d(1, 3, strides=1, **conv_opt)

  def call(self, inputs, training=None):
    x = inputs

    x = tf_expd(x, -1)
    x = self.pool(x)

    for conv in self.convs:
      x = tf.nn.relu(conv(x))

    x = self.conv_post(x)
    x = tf.squeeze(x, -1)
    x = tf.math.reduce_mean(x, -1)
    x = tf.math.sigmoid(x)
   
    return x

class waveunet(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(waveunet, self).__init__(*args, **kwargs)
    self.layer = 4
    self.dims = [32, 32, 32, 32]
    self.ksize = 16
    self.sublayer = 4

  def build(self, input_shape):
    conv_opt = dict(padding='same')

    self.conv_pre = conv1d(self.dims[0], self.ksize, **conv_opt)
    self.conv_post = conv1d(1, self.ksize, **conv_opt)

    self.down_convs = list(zip(
      [conv1d(self.dims[idx], 3,
        strides=2, **conv_opt) for idx in range(self.layer)],
      [[conv1d(None, self.ksize,
        strides=1, **conv_opt) for _ in range(self.sublayer)] for idx in range(self.layer)]))

    self.up_convs = list(zip(
      [conv1dtrans(self.dims[::-1][idx], 3,
        strides=2, **conv_opt) for idx in range(self.layer)],
      [[conv1d(None, self.ksize,
        strides=1, **conv_opt) for _ in range(self.sublayer)] for idx in range(self.layer)]))

    self.conv_mid = conv1d(self.dims[-1], self.ksize, **conv_opt)

  def call(self, inputs, training=None):
    x, ref = inputs

    x = tf_expd(x, -1)
    x = tf.nn.relu(self.conv_pre(x))

    encs = []
    for down_conv, convs in self.down_convs:
      for conv in convs:
        x = tf.nn.relu(conv(x)) + x
      encs.append(x)
      x = tf.nn.relu(down_conv(x))

    x = self.conv_mid(x)

    encs = encs[::-1]
    for enc, (up_conv, convs) in zip(encs, self.up_convs):
      x = tf.nn.relu(up_conv(x))
      x = tf.concat([x, enc], -1)
      for conv in convs:
        x = tf.nn.relu(conv(x)) + x
    
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

      return x, samp_loss + spec_loss

    return x, encs

class wavegan(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(wavegan, self).__init__(*args, **kwargs)
  
  def build(self, input_shape):
    self.gen = waveunet()

    self.mpdisc = [
      pdisc(2, 5, 3),
      pdisc(3, 5, 3),
      pdisc(5, 5, 3),
      pdisc(7, 5, 3),
      pdisc(11, 5, 3)]

    self.msdisc = [
      sdisc(1, 1),
      sdisc(4, 2),
      sdisc(4, 2)]

  def call(self, inputs, training=None):
    x, ref = inputs

    hyp, gen_loss = self.gen((x, ref))
    if ref is None:
      return hyp, None, None

    disc_losses = []
    for _disc in self.mpdisc + self.msdisc:
      hyp_d = _disc(hyp); ref_d = _disc(ref)

      gen_loss += tf.math.reduce_mean((1-hyp_d)**2)

      disc_loss = tf.math.reduce_mean((hyp_d)**2)
      disc_loss += tf.math.reduce_mean((1-ref_d)**2)
      disc_losses.append(disc_loss)

    disc_loss = sum(disc_losses)
    return hyp, gen_loss, disc_loss
