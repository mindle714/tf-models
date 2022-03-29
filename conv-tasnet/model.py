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
      use_bias=False, stride=self.stride)

    self.tcn = tcn()

    self.decoder = tf.keras.layers.Conv1DTranspose(1, self.win,
      use_bias=False, stride=self.stride)

  def call(self, inputs, training=None):
    s1, s2, mix = inputs

    x = self.encoder(mix)
    mask = tf.math.sigmoid(self.TCN(x))
    x = x * mask

    x = self.decoder(x)
    return x
