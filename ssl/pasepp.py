import tensorflow as tf
from util import *
from spec_ops import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class qrnnlayer(tf.keras.layers.Layer):
  def __init__(self, dim, *args, **kwargs):
    super(qrnnlayer, self).__init__()
    self.dim = dim

  def build(self, input_shape):
    self.linear = tf.keras.layers.Dense(3 * self.dim)

  def call(self, inputs, training=None):
    x = inputs

    x = self.linear(x)
    return x

class qrnn(tf.keras.layers.Layer):
  def __init__(self, dim, *args, **kwargs):
    super(qrnn, self).__init__()
    self.num_layers = 1
    self.dim = dim

  def build(self, input_shape):
    self.layers = [qrnnlayer(self.dim) for _ in range(self.num_layers)]

  def call(self, inputs, training=None):
    x = inputs
    
    for layer in self.layers:
      x = layer(x)

    return x

class sincconv(tf.keras.layers.Layer):
  def __init__(self, dim, ksize, stride, padding='SAME', *args, **kwargs):
    super(sincconv, self).__init__()
    self.dim = dim
    self.ksize = ksize
    self.stride = stride
    self.dilation = 1
    self.padding = padding
    self.min_low_hz = 50
    self.min_band_hz = 50
    self.sr = 16000

  def build(self, input_shape):
    self.low_hz = self.add_weight(shape=(64,), name='low_hz')
    self.band_hz = self.add_weight(shape=(64,), name='band_hz')

    self.n = 2 * np.pi * tf.range(-(self.ksize-1)/2, 0) / self.sr
    self.window = tf.linspace(0., self.ksize/2-1, self.ksize//2)
    self.window = 0.54 - 0.46 * tf.math.cos(2 * np.pi * self.window / self.ksize)

  def call(self, inputs, training=None):
    x = inputs

    low = self.min_low_hz + tf.math.abs(self.low_hz)
    high = tf.clip_by_value(low + self.min_band_hz + tf.math.abs(self.band_hz),
      self.min_low_hz, self.sr/2)
    band = tf_expd(high - low, -1)

    f_times_t_low = tf.linalg.matmul(tf_expd(low, -1), tf_expd(self.n, 0))
    f_times_t_high = tf.linalg.matmul(tf_expd(high, -1), tf_expd(self.n, 0))

    band_pass_left = tf.math.sin(f_times_t_high) - tf.math.sin(f_times_t_low)
    band_pass_left = band_pass_left / (self.n/2) * self.window
    band_pass_center = 2 * band
    band_pass_right = tf.reverse(band_pass_left, axis=[1])

    band_pass = tf.concat([band_pass_left, band_pass_center, band_pass_right], 1)
    band_pass = band_pass / (2 * band)

    _filter = tf.reshape(band_pass, [self.dim, 1, self.ksize])
    _filter = tf.transpose(_filter, [2, 1, 0])

    if self.padding.lower() == 'same':
      if self.stride > 1:
        x = tf.pad(x, [[0,0], [self.ksize//2-1, self.ksize//2], [0,0]], 'reflect')
      else:
        x = tf.pad(x, [[0,0], [self.ksize//2, self.ksize//2], [0,0]], 'reflect')

    x = tf.nn.conv1d(x, _filter, stride=self.stride, padding='VALID',
      dilations=self.dilation)

    return x

class feblock(tf.keras.layers.Layer):
  def __init__(self, dim, ksize, stride, sincnet, *args, **kwargs):
    super(feblock, self).__init__()
    self.dim = dim
    self.ksize = ksize
    self.stride = stride
    self.sincnet = sincnet

  def build(self, input_shape):
    if self.sincnet:
      self.conv = sincconv(self.dim, self.ksize, self.stride)
    else:
      self.conv = conv1d(self.dim, self.ksize, 
        strides=self.stride, padding='valid')
    
    self.norm = tf.keras.layers.BatchNormalization(
      epsilon=1e-5, momentum=0.1)
    self.prelu = tf.keras.layers.PReLU(shared_axes=[1])

  def call(self, inputs, training=None):
    x = inputs

    if not self.sincnet:
      if self.stride > 1 or self.ksize % 2 == 0:
        x = tf.pad(x, [[0,0], [self.ksize//2-1, self.ksize//2], [0,0]], 'reflect')
      else:
        x = tf.pad(x, [[0,0], [self.ksize//2, self.ksize//2], [0,0]], 'reflect')

    x = self.conv(x)
    x = self.prelu(self.norm(x, training=training)) 

    return x

class pasepp(tf.keras.layers.Layer):
  def __init__(self, n_fft, hop_len, *args, **kwargs):
    super(pasepp, self).__init__()
    self.dims = [64, 64, 128, 128, 256, 256, 512, 512]
    self.ksizes = [251, 20, 11, 11, 11, 11, 11, 11]
    self.strides = [1, 10, 2, 1, 2, 1, 2, 2]

  def build(self, input_shape):
    self.blocks = [feblock(dim, ksize, stride, sincnet=(idx==0)) \
      for idx, (dim, ksize, stride) in enumerate(
        zip(self.dims, self.ksizes, self.strides))]

#    self.denses = [conv1d(256, 1, strides=1, use_bias=False) \
#      for _ in range(len(self.blocks)-1)]
  
  def call(self, inputs, training=None):
    x = inputs

    x = tf_expd(x, -1)
    for block in self.blocks:
      x = block(x)

    return x

class pasepp_seq(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(pasepp_seq, self).__init__()

  def build(self, input_shape):
    self.pasepp = pasepp(400, 160)
  
  def call(self, inputs, training=None):
    x = inputs
    return self.pasepp(x)

class pasepp_unet(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(pasepp_unet, self).__init__()

    self.layer = 4
    self.dims = [32, 32, 32, 32]
    self.ksize = 16
    self.sublayer = 4

    self.n_fft = 400
    self.hop_len = 160

    self.k = 10
    self.c = 0.1

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.pasepp = pasepp(self.n_fft, self.hop_len)

    self.conv_r = conv1d(self.n_fft//2+1, 3, **conv_opt)
    self.conv_i = conv1d(self.n_fft//2+1, 3, **conv_opt)
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, ref = inputs

    else:
      x = inputs
      ref = None

    _in = x

    xs = self.pasepp(x)
    x = xs[-1]

    x = tf.keras.activations.gelu(x)
    x = tf.stop_gradient(x)

    x_r = tf.clip_by_value(self.conv_r(x), -self.k, self.k)
    x_i = tf.clip_by_value(self.conv_i(x), -self.k, self.k)
      
    in_pad = tf.pad(_in, tf.constant(
      [[0, 0], [self.n_fft//2, self.n_fft//2]]), mode='reflect')

    Y = tf.signal.stft(in_pad, 
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
    Yr = tf.math.real(Y); Yi = tf.math.imag(Y)

    Mr = -1. / self.c * tf.math.log(tf.nn.relu((self.k - x_r) / (self.k + x_r)) + 1e-10)
    Mi = -1. / self.c * tf.math.log(tf.nn.relu((self.k - x_i) / (self.k + x_i)) + 1e-10)

    Sr = (Mr * Yr) - (Mi * Yi)
    Si = (Mr * Yi) + (Mi * Yr)
    x = tf.signal.inverse_stft(tf.complex(Sr, Si),
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
    x = x[..., self.n_fft//2:self.n_fft//2+tf.shape(_in)[1]]

    def get_cirm(Yr, Yi, ref):
      ref_pad = tf.pad(ref, tf.constant(
        [[0, 0], [self.n_fft//2, self.n_fft//2]]), mode='reflect')

      S = tf.signal.stft(ref_pad,
        frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
      Sr = tf.math.real(S); Si = tf.math.imag(S)

      M_denom = (Yr * Yr) + (Yi * Yi)
      Mr_num = (Yr * Sr) + (Yi * Si)
      Mr = Mr_num / (M_denom + 1e-10)

      Mi_num = (Yr * Si) - (Yi * Sr)
      Mi = Mi_num / (M_denom + 1e-10)

      Cr_exp = tf.math.exp(-self.c * Mr)
      #Cr = self.k * ((1 - Cr_exp) / (1 + Cr_exp))
      Cr = self.k * (-1. + 2. * tf.math.reciprocal_no_nan(1 + Cr_exp))
        
      Ci_exp = tf.math.exp(-self.c * Mi)
      #Ci = self.k * ((1 - Ci_exp) / (1 + Ci_exp))
      Ci = self.k * (-1. + 2. * tf.math.reciprocal_no_nan(1 + Ci_exp))

      return Cr, Ci

    if ref is not None:
      x = x[..., :tf.shape(ref)[-1]]
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

      Cr, Ci = get_cirm(Yr, Yi, ref)
      cirm_loss = tf.math.reduce_mean(tf.math.abs(Cr - x_r))
      cirm_loss += tf.math.reduce_mean(tf.math.abs(Ci - x_i))

      return samp_loss + spec_loss, cirm_loss

    return x
