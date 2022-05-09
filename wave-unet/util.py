import tensorflow as tf
import functools
import numpy as np

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

def mel_filterbank(
  pcm, sr=16000,
  frame_length=25, frame_step=10, fft_length=None,
  window_fn=functools.partial(tf.signal.hann_window, periodic=True),
  lower_edge_hertz=80.0, upper_edge_hertz=7600.0, num_mel_bins=24,
  log_noise_floor=1e-3):

  stfts = stft(
      pcm, sr=sr,
      frame_length=frame_length,
      frame_step=frame_step,
      fft_length=fft_length,
      window_fn=window_fn)

  # An energy spectrogram is the magnitude of the complex-valued STFT.
  # A float32 Tensor of shape [batch_size, ?, 257].
  magnitude_spectrograms = tf.abs(stfts)

  # Warp the linear-scale, magnitude spectrograms into the mel-scale.
  num_spectrogram_bins = magnitude_spectrograms.shape[-1]
  linear_to_mel_weight_matrix = (
      tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
          upper_edge_hertz))
  mel_spectrograms = tf.tensordot(
      magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
  # Note: Shape inference for tensordot does not currently handle this case.
  mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

  log_mel_sgram = tf.math.log(tf.maximum(log_noise_floor, mel_spectrograms))
  return log_mel_sgram

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

class gnorm(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(gnorm, self).__init__(*args, **kwargs)

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
      m, v = tf.nn.weighted_moments(x, [1, 2], mask, keepdims=True)
    else:
      x = inputs
      m, v = tf.nn.moments(x, [1, 2], keepdims=True)

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
