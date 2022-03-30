import tensorflow as tf
import functools
import numpy as np
  
tf_mean = tf.math.reduce_mean
tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

def log10(x):
  num = tf.math.log(x)
  denom = tf.math.log(tf.constant(10, dtype=num.dtype))
  return num / denom

import itertools

# orig[batch, timestep, mixture], est[..]
def si_snr(orig, est, pit=True, eps=1e-8):
  orig = orig - tf_mean(orig, 1, keepdims=True)
  est = est - tf_mean(est, 1, keepdims=True)

  if pit:
    orig = tf_expd(orig, -2)
    est = tf_expd(est, -1)

  # s_tgt = <s,s'>s/||s||^2
  s_tgt = tf_sum(orig * est, 1, keepdims=True)
  s_tgt = s_tgt * orig / (tf_sum(orig**2, 1, keepdims=True) + eps)
  
  e_ns = est - s_tgt
  snr = tf_sum(s_tgt**2, 1) / (tf_sum(e_ns**2, 1) + eps)
  snr = 10 * log10(snr)

  num_mix = snr.shape[-1]
  perms = tf.one_hot(list(itertools.permutations(range(num_mix))), num_mix)
  snr = tf_expd(snr, 1) * tf_expd(perms, 0)
  snr = tf_sum(tf_sum(snr, -1), -1)

  max_snr = tf.math.reduce_max(snr, -1) / num_mix
  loss = -tf_mean(max_snr)

  return loss

class gnorm(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(gnorm, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    dim = input_shape[-1]

    self.eps = 1e-8
    self.gamma = self.add_weight(shape=(dim), initializer="ones")
    self.beta = self.add_weight(shape=(dim), initializer="zeros")

  def call(self, inputs, training=None):
    x = inputs
    m, v = tf.nn.moments(x, axes=[0, 1], keepdims=True)
    return self.gamma * (x-m) / tf.math.sqrt(v + self.eps) + self.beta

class depconv(tf.keras.layers.Layer):
  def __init__(self, dim, kernel, dilation, *args, **kwargs):
    super(depconv, self).__init__(*args, **kwargs)
    self.dim = dim
    self.kernel = kernel
    self.dilation = dilation

  def build(self, input_shape):
    conv_opt = dict(padding='same')
    self.conv = tf.keras.layers.Conv1D(self.dim, 1, **conv_opt)
    self.dconv = tf.keras.layers.Conv1D(self.dim, self.kernel,
      groups=self.dim, dilation_rate=self.dilation, **conv_opt)

    dim = input_shape[-1]
    self.out = tf.keras.layers.Conv1D(dim, 1, **conv_opt)

    self.prelu1 = tf.keras.layers.PReLU()
    self.prelu2 = tf.keras.layers.PReLU()

    self.norm1 = gnorm()
    self.norm2 = gnorm()

  def call(self, inputs, training=None):
    x = inputs
    x = self.norm1(self.prelu1(self.conv(x)))
    x = self.norm2(self.prelu2(self.dconv(x)))
    x = self.out(x)
    return x

class tcn(tf.keras.layers.Layer):
  def __init__(self, stack, layer, out_dim, *args, **kwargs):
    super(tcn, self).__init__(*args, **kwargs)
    self.hid_dim = 128
    self.out_dim = out_dim
    self.causal = False
    self.stack = stack
    self.layer = layer
    self.kernel = 3

  def build(self, input_shape):
    self.norm = gnorm()
    self.preconv = tf.keras.layers.Conv1D(self.hid_dim, 1)

    self.convs = []
    for s in range(self.stack):
      for l in range(self.layer):
        conv = depconv(
          self.hid_dim, self.kernel, dilation=2**l)
        self.convs.append(conv)

    self.prelu = tf.keras.layers.PReLU()
    self.postconv = tf.keras.layers.Conv1D(self.out_dim, 1)
  
  def call(self, inputs, training=None):
    x = inputs
    x = self.preconv(self.norm(x))

    for conv in self.convs:
      x = conv(x)

    x = self.postconv(self.prelu(x))
    return x

class convtas(tf.keras.layers.Layer):
  def __init__(self, sr=8000, win=2, *args, **kwargs):
    super(convtas, self).__init__(*args, **kwargs)
    self.enc_dim = 512
    self.feature_dim = 128
    self.sr = sr
    self.win =int(sr * win / 1000)
    self.stride = self.win // 2
    self.layer = 8
    self.stack = 3
    self.kernel = 3
    self.causal = False

  def build(self, input_shape):
    self.encoder = tf.keras.layers.Conv1D(self.enc_dim, self.win,
      use_bias=False, strides=self.stride)

    self.tcn = tcn(self.stack, self.layer, self.enc_dim)

    self.decoder = tf.keras.layers.Conv1DTranspose(1, self.win,
      use_bias=False, strides=self.stride)

  def call(self, inputs, training=None):
    s1, s2, mix = inputs
    sm = tf.concat([tf_expd(s1, -1), tf_expd(s2, -1)], -1)

    x = self.encoder(tf_expd(mix, -1))
    mask = tf.math.sigmoid(self.tcn(x))
    x = x * mask

    x = self.decoder(x)
    return si_snr(sm, x)
