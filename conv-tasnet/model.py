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

# ref[batch, timestep, mixture], hyp[..]
def si_snr(ref, hyp, pit=True, eps=1e-8):
  ref = ref - tf_mean(ref, 1, keepdims=True)
  hyp = hyp - tf_mean(hyp, 1, keepdims=True)

  if pit:
    ref = tf_expd(ref, -2)
    hyp = tf_expd(hyp, -1)

  # s_tgt = <s,s'>s/||s||^2
  s_tgt = tf_sum(ref * hyp, 1, keepdims=True)
  s_tgt = s_tgt * ref / (tf_sum(ref**2, 1, keepdims=True) + eps)
  
  e_ns = hyp - s_tgt
  snr = tf_sum(s_tgt**2, 1) / (tf_sum(e_ns**2, 1) + eps)
  snr = 10 * log10(snr)

  if pit:
    num_mix = snr.shape[-1]
    perms = tf.one_hot(list(itertools.permutations(range(num_mix))), num_mix)
    # perms[batch, num_mix, num_mix]
    snr = tf_expd(snr, 1) * tf_expd(perms, 0)
    snr = tf_sum(tf_sum(snr, -1), -1)

    batch = tf.shape(snr)[0]
    perm_id = tf.math.argmax(snr, -1)
    perm_id_bat = tf.concat([
      tf_expd(tf.range(batch, dtype=perm_id.dtype), -1), tf_expd(perm_id, -1)], -1)

    max_snr = tf.gather_nd(snr, perm_id_bat) / num_mix
    max_perm = tf.gather(perms, perm_id)
    sort_hyp = tf.linalg.matmul(tf.squeeze(hyp, -1), max_perm)

    return max_snr, sort_hyp

  return snr, hyp

class gnorm(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(gnorm, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    dim = input_shape[-1]

    self.eps = 1e-8
    self.gamma = self.add_weight(shape=(dim),
      initializer="ones", name="gamma")
    self.beta = self.add_weight(shape=(dim),
      initializer="zeros", name="beta")

  def call(self, inputs, training=None):
    x = inputs
    m, v = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    return self.gamma * (x-m) / tf.math.sqrt(v + self.eps) + self.beta

class depconv(tf.keras.layers.Layer):
  def __init__(self, dim, kernel, dilation,
               skip=True, add_out=True, *args, **kwargs):
    super(depconv, self).__init__(*args, **kwargs)
    self.dim = dim
    self.kernel = kernel
    self.dilation = dilation
    self.skip = skip
    self.add_out = add_out

  def build(self, input_shape):
    conv_opt = dict(padding='same')

    self.conv = tf.keras.layers.Conv1D(self.dim, 1, **conv_opt)
    self.dconv = tf.keras.layers.Conv1D(self.dim, self.kernel,
      groups=self.dim, dilation_rate=self.dilation, **conv_opt)

    dim = input_shape[-1]
    if self.add_out:
      self.out = tf.keras.layers.Conv1D(dim, 1, **conv_opt, name='out')

    self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1, 2], 
      alpha_initializer=tf.constant_initializer(0.25))
    self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1, 2], 
      alpha_initializer=tf.constant_initializer(0.25))

    self.norm1 = gnorm()
    self.norm2 = gnorm()

    if self.skip:
      self.skipconv = tf.keras.layers.Conv1D(dim, 1, **conv_opt, name='skip')

  def call(self, inputs, training=None):
    x = inputs

    x = self.norm1(self.prelu1(self.conv(x)))
    x = self.norm2(self.prelu2(self.dconv(x)))
    x_res = self.out(x) if self.add_out else None

    if self.skip:
      x_skip = self.skipconv(x)
      return x_res, x_skip

    return x_res

class tcn(tf.keras.layers.Layer):
  def __init__(self, stack, layer, out_dim, num_spk, skip=True, *args, **kwargs):
    super(tcn, self).__init__(*args, **kwargs)
    self.B = 128
    self.H = 512
    self.out_dim = out_dim
    self.num_spk = num_spk
    self.causal = False
    self.stack = stack
    self.layer = layer
    self.kernel = 3
    self.skip = skip

  def build(self, input_shape):
    self.norm = gnorm()
    self.preconv = tf.keras.layers.Conv1D(self.B, 1)

    self.convs = []
    for s in range(self.stack):
      for l in range(self.layer):
        add_out = True
        if s == (self.stack-1) and l == (self.layer-1) and self.skip:
          add_out = False

        conv = depconv(
          self.H, self.kernel, dilation=2**l, skip=self.skip, add_out=add_out)
        self.convs.append(conv)

    self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2], 
      alpha_initializer=tf.constant_initializer(0.25))
    self.postconv = tf.keras.layers.Conv1D(self.out_dim*self.num_spk, 1)
  
  def call(self, inputs, training=None):
    x = inputs
    x = self.preconv(self.norm(x))

    if self.skip:
      x_skip = None
      for idx, conv in enumerate(self.convs):
        res = x
        x, x_skip_ = conv(x)
        x = x + res if idx < (len(self.convs)-1) else None
        if x_skip is None: x_skip = x_skip_
        else: x_skip = x_skip + x_skip_
      x = x_skip

    else:
      for conv in self.convs:
        res = x
        x = conv(x) + res

    x = self.postconv(self.prelu(x))
    return tf.reshape(x,
      tf.concat([tf.shape(x)[:2], [self.num_spk, self.out_dim]], 0))

class convtas(tf.keras.layers.Layer):
  def __init__(self, sr=8000, L=16, *args, **kwargs):
    super(convtas, self).__init__(*args, **kwargs)
    self.enc_dim = 512
    self.feature_dim = 128
    self.sr = sr
    self.L = L
    self.layer = 8
    self.stack = 3
    self.kernel = 3
    self.causal = False
    self.num_spk = 2

  def build(self, input_shape):
    conv_opt = dict(padding='same')
    stride = self.L // 2

    self.encoder = tf.keras.layers.Conv1D(self.enc_dim, self.L,
      use_bias=False, strides=stride, **conv_opt)

    self.tcn = tcn(self.stack, self.layer, self.enc_dim, self.num_spk)

    self.decoder = tf.keras.layers.Conv1DTranspose(1, self.L,
      use_bias=False, strides=stride, **conv_opt)

  def call(self, inputs, training=None):
    s1, s2, mix = inputs
    if s1 is not None and s2 is not None:
      sm = tf.concat([tf_expd(s1, -1), tf_expd(s2, -1)], -1)

    x = tf.nn.relu(self.encoder(tf_expd(mix, -1)))
    mask = tf.math.sigmoid(self.tcn(x))
    x = tf_expd(x, -2) * mask

    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.reshape(x, [-1, tf.shape(x)[-2], self.enc_dim])
    
    x = self.decoder(x)
    x = tf.reshape(x, [-1, self.num_spk, tf.shape(x)[-2]])
    x = tf.transpose(x, [0, 2, 1])

    if s1 is not None and s2 is not None:
      snr, sort_x = si_snr(sm, x)
      return -tf_mean(snr)

    return x
