import tensorflow as tf
import functools
import numpy as np

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

def log10(x):
  num = tf.math.log(x)
  denom = tf.math.log(tf.constant(10, dtype=num.dtype))
  return num / denom

import itertools

# ref[batch, timestep, mixture], hyp[..]
def si_snr(ref, hyp, mask=None, pit=True, eps=1e-8, return_ref=False):
  def norm_mean(e):
    if mask is not None:
      m, _ = tf.nn.weighted_moments(e, [1], mask, keepdims=True)
    else:
      m, _ = tf.nn.moments(e, [1], keepdims=True)

    return e - m

  _ref = ref
  ref = norm_mean(ref)
  hyp = norm_mean(hyp)

  if pit:
    ref = tf_expd(ref, -2)
    hyp = tf_expd(hyp, -1)

  if mask is not None:
    mask = tf_expd(mask, -1)
    # s_tgt = <s,s'>s/||s||^2
    s_tgt = tf_sum((ref * hyp) * mask, 1, keepdims=True)
    s_tgt = s_tgt * ref / (tf_sum((ref**2) * mask, 1, keepdims=True) + eps)
  
    e_ns = hyp - s_tgt
    snr = tf_sum((s_tgt**2) * mask, 1) / (tf_sum((e_ns**2) * mask, 1) + eps)
    snr = 10 * log10(snr)

  else:
    # s_tgt = <s,s'>s/||s||^2
    s_tgt = tf_sum((ref * hyp), 1, keepdims=True)
    s_tgt = s_tgt * ref / (tf_sum((ref**2), 1, keepdims=True) + eps)
  
    e_ns = hyp - s_tgt
    snr = tf_sum((s_tgt**2), 1) / (tf_sum((e_ns**2), 1) + eps)
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

    if not return_ref:
      return max_snr, sort_hyp
    else:
      return max_snr, sort_hyp, tf.linalg.matmul(_ref, max_perm)

  return snr, hyp

def stft(
  pcm, sr=16000,
  frame_length=25, frame_step=10, fft_length=None,
  window_fn=functools.partial(tf.signal.hann_window, periodic=True)):
  
  frame_length = int(frame_length * sr / 1e3)
  frame_step = int(frame_step * sr / 1e3)
  if fft_length is None:
    fft_length = int(2**(np.ceil(np.log2(frame_length))))

  return tf.signal.stft(
    pcm, frame_length=frame_length, frame_step=frame_step,
    fft_length=fft_length, window_fn=window_fn, pad_end=True)

import mel_ops

def melspec(
  pcm, sr=16000,
  frame_length=400, frame_step=160, fft_length=400,
  window_fn=functools.partial(tf.signal.hann_window, periodic=True),
  lower_edge_hertz=80.0, upper_edge_hertz=7600.0, num_mel_bins=24,
  eps=1e-10, power=2.,
  center=True, pad_mode='reflect', cmvn=True):

  if center:
    pcm = tf.pad(pcm, tf.constant(
      [[0, 0], [fft_length//2, fft_length//2]]), mode=pad_mode)

  stfts = tf.signal.stft(
      pcm,
      frame_length=frame_length,
      frame_step=frame_step,
      fft_length=fft_length,
      window_fn=window_fn, pad_end=False)

  # An energy spectrogram is the magnitude of the complex-valued STFT.
  # A float32 Tensor of shape [batch_size, ?, 257].
  magnitude_spectrograms = tf.abs(stfts)**power

  # Warp the linear-scale, magnitude spectrograms into the mel-scale.
  num_spectrogram_bins = magnitude_spectrograms.shape[-1]
  linear_to_mel_weight_matrix = (
      mel_ops.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
          upper_edge_hertz))
  mel_spectrograms = tf.tensordot(
      magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
  # Note: Shape inference for tensordot does not currently handle this case.
  mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

  log_mel_sgram = tf.math.log(eps + mel_spectrograms)

  if cmvn:
    cmvn_sgram = (log_mel_sgram - tf.math.reduce_mean(log_mel_sgram, 1, keepdims=True))
    std_cmvn = tf.math.reduce_variance(log_mel_sgram, 1, keepdims=True)
    std_n = tf.cast(tf.shape(log_mel_sgram)[1], tf.float32)
    std_cmvn = std_cmvn * std_n / (std_n-1)
    std_cmvn = tf.math.sqrt(std_cmvn)
    #cmvn_sgram /= (tf.math.reduce_std(log_mel_sgram, 1, keepdims=True) + eps)
    cmvn_sgram /= (std_cmvn + eps)
    log_mel_sgram = cmvn_sgram

  return log_mel_sgram

def mfcc(
  pcm, frame_length=400, frame_step=160, fft_length=400,
  lower_edge_hertz=80.0, upper_edge_hertz=7600.0, num_mel_bins=80,
  center=True, pad_mode='reflect'):

  if center:
    pcm = tf.pad(pcm, tf.constant(
      [[0, 0], [fft_length//2, fft_length//2]]), mode=pad_mode)

  stfts = tf.signal.stft(
    pcm, frame_length=frame_length,
    frame_step=frame_step, fft_length=fft_length)

  spectrograms = tf.abs(stfts)

  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = stfts.shape[-1].value
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :13]

  return mfccs

class conv1d(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.conv_args = args
    self.conv_kwargs = kwargs
    super(conv1d, self).__init__()

  def build(self, input_shape):
    if isinstance(input_shape, tuple): dim = input_shape[0][-1]
    else: dim = input_shape[-1]

    groups = 1.
    if "groups" in self.conv_kwargs:
      groups = tf.cast(self.conv_kwargs["groups"], tf.float32)
    minmax = groups / tf.cast(self.conv_args[1] * dim, tf.float32)
    minmax = tf.math.sqrt(minmax)

    kernel_init = tf.keras.initializers.RandomUniform(
      minval=-minmax, maxval=minmax)
    if "kernel_initializer" not in self.conv_kwargs:
      self.conv_kwargs["kernel_initializer"] = kernel_init

    if self.conv_args[0] is None: self.conv_args = (dim, self.conv_args[1])
    self.conv = tf.keras.layers.Conv1D(*self.conv_args, **self.conv_kwargs)

    if isinstance(input_shape, tuple):
      mask_args = (1, self.conv_args[1]) # must override only kernel size
      mask_kwargs = self.conv_kwargs
      mask_kwargs["kernel_initializer"] = tf.constant_initializer(1./mask_args[1])
      mask_kwargs["bias_initializer"] = "zeros"
      mask_kwargs["groups"] = 1
      mask_kwargs["trainable"] = False

      self.mask_conv = tf.keras.layers.Conv1D(*mask_args, **mask_kwargs)

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, mask = inputs
      x = self.conv(x)
      mask = self.mask_conv(mask)
      return x, mask

    x = inputs
    return self.conv(x)

class conv1dtrans(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.conv_args = args
    self.conv_kwargs = kwargs
    super(conv1dtrans, self).__init__()

  def build(self, input_shape):
    dim = self.conv_args[0]
    
    groups = 1.
    if "groups" in self.conv_kwargs:
      groups = tf.cast(self.conv_kwargs["groups"], tf.float32)
    minmax = groups / tf.cast(self.conv_args[1] * dim, tf.float32)
    minmax = tf.math.sqrt(minmax)
    
    kernel_init = tf.keras.initializers.RandomUniform(
      minval=-minmax, maxval=minmax)
    if "kernel_initializer" not in self.conv_kwargs:
      self.conv_kwargs["kernel_initializer"] = kernel_init

    if self.conv_args[0] is None: self.conv_args = (dim, self.conv_args[1])
    self.conv = tf.keras.layers.Conv1DTranspose(*self.conv_args, **self.conv_kwargs)
    
    if isinstance(input_shape, tuple):
      mask_args = (1, self.conv_args[1]) # must override only kernel size
      mask_kwargs = self.conv_kwargs
      mask_kwargs["kernel_initializer"] = "ones"
      mask_kwargs["bias_initializer"] = "zeros"
      mask_kwargs["trainable"] = False

      self.mask_conv = tf.keras.layers.Conv1DTranspose(*mask_args, **mask_kwargs)

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, mask = inputs
      x = self.conv(x)
      mask = self.mask_conv(mask)
      mask = tf.math.minimum(mask, 1.)
      return x, mask

    x = inputs
    return self.conv(x)

conv2d = tf.keras.layers.Conv2D
conv2dtrans = tf.keras.layers.Conv2DTranspose

'''
class conv2d(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.conv_args = args
    self.conv_kwargs = kwargs
    super(conv2d, self).__init__()

  def build(self, input_shape):
    if isinstance(input_shape, tuple): dim = input_shape[0][-1]
    else: dim = input_shape[-1]

    if self.conv_args[0] is None: self.conv_args = (dim, self.conv_args[1])
    self.conv = tf.keras.layers.Conv2D(*self.conv_args, **self.conv_kwargs)

  def call(self, inputs, training=None):
    x = inputs
    return self.conv(x)
'''

class gnorm(tf.keras.layers.Layer):
  def __init__(self, num_groups=1, *args, **kwargs):
    self.num_groups = num_groups
    super(gnorm, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    if isinstance(input_shape, tuple): dim = input_shape[0][-1]
    else: dim = input_shape[-1]

    self.eps = 1e-5
    self.gamma = self.add_weight(shape=(dim),
      initializer="ones", name="gamma")
    self.beta = self.add_weight(shape=(dim),
      initializer="zeros", name="beta")

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, mask = inputs
      if self.num_groups == 1:
        m, v = tf.nn.weighted_moments(x, [1, 2], mask, keepdims=True)
      else:
        x = tf.reshape(x, tf.concat([tf.shape(x)[:2], [self.num_groups, -1]], 0))
        m, v = tf.nn.weighted_moments(x, [1, 3], mask, keepdims=True)
        gamma = tf.reshape(self.gamma, [1, 1, self.num_groups, -1])
        beta = tf.reshape(self.beta, [1, 1, self.num_groups, -1])

        norm = gamma * (x-m) / tf.math.sqrt(v + self.eps) + beta
        return tf.reshape(norm, tf.concat([tf.shape(x)[:2], [-1]], 0))

    else:
      x = inputs
      if self.num_groups == 1:
        m, v = tf.nn.moments(x, [1, 2], keepdims=True)
      else:
        x = tf.reshape(x, tf.concat([tf.shape(x)[:2], [self.num_groups, -1]], 0))
        m, v = tf.nn.moments(x, [1, 3], keepdims=True)
        gamma = tf.reshape(self.gamma, [1, 1, self.num_groups, -1])
        beta = tf.reshape(self.beta, [1, 1, self.num_groups, -1])

        norm = gamma * (x-m) / tf.math.sqrt(v + self.eps) + beta
        return tf.reshape(norm, tf.concat([tf.shape(x)[:2], [-1]], 0))

    gamma = tf.reshape(self.gamma, [1, 1, -1])
    beta = tf.reshape(self.beta, [1, 1, -1])
    return self.gamma * (x-m) / tf.math.sqrt(v + self.eps) + self.beta

class cnorm(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(cnorm, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    if isinstance(input_shape, tuple): dim = input_shape[0][-1]
    else: dim = input_shape[-1]

    self.eps = 1e-8
    self.gamma = self.add_weight(shape=(dim),
      initializer="ones", name="gamma")
    self.beta = self.add_weight(shape=(dim),
      initializer="zeros", name="beta")

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, mask = inputs
      m, v = tf.nn.weighted_moments(x, [2], mask, keepdims=True)
    else:
      x = inputs
      m, v = tf.nn.moments(x, [2], keepdims=True)

    return self.gamma * (x-m) / tf.math.sqrt(v + self.eps) + self.beta

class lnorm(gnorm):
  def __init__(self, affine=True, eps=1e-5, *args, **kwargs):
    self.affine = affine
    self.eps = eps
    super(lnorm, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    if isinstance(input_shape, tuple): dim = input_shape[0][-1]
    else: dim = input_shape[-1]

    if self.affine:
      self.gamma = self.add_weight(shape=(dim),
        initializer="ones", name="gamma")
      self.beta = self.add_weight(shape=(dim),
        initializer="zeros", name="beta")

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, mask = inputs
      m, v = tf.nn.weighted_moments(x, [-1], mask, keepdims=True)

    else:
      x = inputs
      m, v = tf.nn.moments(x, [-1], keepdims=True)

    if self.affine:
      gamma = tf.reshape(self.gamma, [1, 1, -1])
      beta = tf.reshape(self.beta, [1, 1, -1])
      return self.gamma * (x-m) / tf.math.sqrt(v + self.eps) + self.beta

    return (x-m) / tf.math.sqrt(v + self.eps)

class depthconv1d(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.conv_args = args
    self.conv_kwargs = kwargs

    self.strides = 1
    if "strides" in kwargs:
      self.strides = kwargs["strides"]

    self.padding = "SAME"
    if "padding" in kwargs:
      self.padding = kwargs["padding"].upper()

    super(depthconv1d, self).__init__()
  
  def build(self, input_shape):
    dim = input_shape[-1]
    self.dim = dim

    #self.conv_kwargs["use_bias"] = False
    #self.convs = [tf.keras.layers.Conv1D(1, self.conv_args[0], 
    #    **self.conv_kwargs) for _ in range(dim)]
    '''
    minmax = 1. / tf.cast(self.conv_args[0] * dim, tf.float32)
    minmax = tf.math.sqrt(minmax)
    
    kernel_init = tf.keras.initializers.RandomUniform(
      minval=-minmax, maxval=minmax)
    '''
    kernel_init = tf.keras.initializers.GlorotUniform()

    self.kernel = self.add_weight(shape=(self.conv_args[0], dim, 1),
      initializer=kernel_init, name="kernel")
    self.bias = self.add_weight(shape=(dim), initializer="zeros", name="bias")

  def call(self, inputs, training=None):
    x = inputs

    x_chs = []
    for idx in range(self.dim):
      x_ch = tf.slice(x, [0, 0, idx], [-1, -1, 1])
      kernel_ch = tf.slice(self.kernel, [0, idx, 0], [-1, 1, -1])

      #x_ch = self.convs[idx](x_ch)
      x_ch = tf.nn.conv1d(x_ch, kernel_ch, self.strides, self.padding) 
      x_chs.append(x_ch)

    x = tf.concat(x_chs, -1)
    x = x + tf.reshape(self.bias, [1, 1, -1])

    return x
