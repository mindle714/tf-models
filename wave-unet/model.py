import tensorflow as tf
import functools
import numpy as np
from util import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class conv1d_v2(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.conv_args = args
    self.conv_kwargs = kwargs
    super(conv1d_v2, self).__init__()

  def build(self, input_shape):
    conv_opt = dict(padding='same')

    # self.conv1 = conv1d(*self.conv_args, **self.conv_kwargs)
    # self.conv2 = conv1d(*self.conv_args, **self.conv_kwargs)
    self.conv1 = tf.keras.layers.DepthwiseConv1D(self.conv_args[1], **self.conv_kwargs)
    self.conv2 = tf.keras.layers.DepthwiseConv1D(self.conv_args[1], **self.conv_kwargs)

    dconv_kwargs = self.conv_kwargs
    dconv_kwargs["groups"] = 8
    self.dconv = conv1d(self.conv_args[0], 4, groups=8, **conv_opt)

    pconv_args = (self.conv_args[0], 1)
    self.pconv = conv1d(*pconv_args, **conv_opt)
  
  def call(self, inputs, training=None):
    x = inputs

    loss = 0.
    for conv in [self.conv1, self.conv2]:
      if len(conv.weights) > 0:
        kernel = conv.weights[0]
        # kernel = tf.reshape(kernel, [-1, tf.shape(kernel)[-1]])
        kernel = tf.reshape(kernel, [tf.shape(kernel)[0], -1])
        kernel_trans = tf.transpose(kernel)

        f = tf.signal.fft(tf.cast(kernel_trans, tf.complex64))
        _loss = tf.reduce_max(tf.nn.softmax(tf.math.abs(f), -1), -1)
        loss = loss - tf.reduce_mean(_loss)

    x = tf.nn.relu(self.conv1(x)) + tf.nn.relu(self.conv2(x))
    x = self.pconv(self.dconv(x))

    return x, loss

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
      [conv1d_v2(self.dims[idx], self.ksize,
        strides=2, **conv_opt) for idx in range(self.layer)],
      [[conv1d(None, self.ksize,
        strides=1, **conv_opt) for _ in range(self.sublayer)] for idx in range(self.layer)]))

    self.up_convs = list(zip(
      [conv1dtrans(self.dims[::-1][idx], self.ksize,
        strides=2, **conv_opt) for idx in range(self.layer)],
      [[conv1d(None, self.ksize,
        strides=1, **conv_opt) for _ in range(self.sublayer)] for idx in range(self.layer)]))

    self.conv_mid = conv1d(self.dims[-1], self.ksize, **conv_opt)

  def call(self, inputs, training=None):
    x, ref = inputs

    x = tf_expd(x, -1)
    #x, conv_pre_loss = self.conv_pre(x)
    x = self.conv_pre(x); conv_pre_loss = 0.
    x = tf.nn.relu(x)

    encs = []; down_conv_losses = []
    for down_conv, convs in self.down_convs:
      for conv in convs:
        x = tf.nn.relu(conv(x)) + x
      encs.append(x)

      x, down_conv_loss = down_conv(x)
      x = tf.nn.relu(x)
      down_conv_losses.append(down_conv_loss)

    down_conv_loss = sum(down_conv_losses)
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

      return samp_loss + spec_loss, conv_pre_loss + down_conv_loss

    return x, encs
