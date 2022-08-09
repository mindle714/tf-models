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

  # TODO slow version
  def forgetmult(self, f, x):
    fx = f * x

    fs = tf.unstack(f, axis=1)
    fxs = tf.unstack(fx, axis=1)

    xs = []; prev_h = None
    for f, h in zip(fs, fxs):
      if prev_h is not None:
        h = h + (1 - f) * prev_h
      xs.append(h)
      prev_h = h

    return tf.stack(xs, axis=1)

  def call(self, inputs, training=None):
    x = inputs

    xs = []
    xs.append(tf.zeros([tf.shape(x)[0], 1, tf.shape(x)[-1]]))
    xs.append(x[..., :-1, :])
    xs = tf.concat(xs, 1)

    x = tf.concat([x, xs], -1)
    x = self.linear(x)
    z, f, o = tf.split(x, 3, axis=-1)

    z = tf.math.tanh(z)
    f = tf.math.sigmoid(f)
    c = self.forgetmult(f, z)

    h = tf.math.sigmoid(o) * c
    return h, c[..., -1:, :]

class qrnn(tf.keras.layers.Layer):
  def __init__(self, dim, *args, **kwargs):
    super(qrnn, self).__init__()
    self.num_layers = 1
    self.dim = dim

  def build(self, input_shape):
    self.layers = [qrnnlayer(self.dim) for _ in range(self.num_layers)]

  def call(self, inputs, training=None):
    x = inputs
   
    hs = []
    for layer in self.layers:
      x, h = layer(x)
      hs.append(h)
    h = tf.concat(hs, 1)

    return x, h

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

class pasep(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(pasep, self).__init__()
    self.dims = [64, 64, 128, 128, 256, 256, 512, 512]
    self.ksizes = [251, 20, 11, 11, 11, 11, 11, 11]
    self.strides = [1, 10, 2, 1, 2, 1, 2, 2]

  def build(self, input_shape):
    self.blocks = [feblock(dim, ksize, stride, sincnet=(idx==0)) \
      for idx, (dim, ksize, stride) in enumerate(
        zip(self.dims, self.ksizes, self.strides))]

    self.denses = [conv1d(256, 1, strides=1, use_bias=False) \
      for _ in range(len(self.blocks)-1)]

    self.rnn = qrnn(512)
    self.rnn_out = conv1d(256, 1, strides=1)
    self.rnn_norm = tf.keras.layers.BatchNormalization(
      epsilon=1e-5, momentum=0.1, center=False, scale=False)
  
  def call(self, inputs, training=None):
    x = inputs
    x = tf_expd(x, -1)

    ds = []; fes = []
    for idx, block in enumerate(self.blocks):
      if self.strides[idx] > 1:
        fes.append(x)

      x = block(x, training=training)
      if idx < (len(self.blocks) - 1):
        ds.append(self.denses[idx](x))

    x, h = self.rnn(x)
    x = self.rnn_out(x)

    for d in ds:
      factor = tf.shape(d)[1] // tf.shape(x)[1]
      _d = d[..., :tf.shape(x)[1]*factor, :]
      _d = tf.reshape(_d, 
        [tf.shape(_d)[0], tf.shape(x)[1], factor, tf.shape(_d)[-1]])
      _d = tf.math.reduce_mean(_d, 2)
      x = x + _d

    x = self.rnn_norm(x, training=training)

    return x, fes

class pasep_seq(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(pasep_seq, self).__init__()

  def build(self, input_shape):
    self.pasep = pasep()
  
  def call(self, inputs, training=None):
    x = inputs
    return self.pasep(x, training=training)[0]

class pasep_unet(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(pasep_unet, self).__init__()
    self.layer = 5
    self.ksizes = [20, 11, 11, 11, 11]
    self.dims = [256, 256, 128, 128, 64]
    self.strides = [10, 2, 2, 2, 2]
    self.sublayer = 4

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.pasep = pasep()
    
    _strides = self.strides[::-1]
    self.up_convs = list(zip(
      [conv1dtrans(self.dims[::-1][idx], self.ksizes[::-1][idx],
        strides=_strides[idx], **conv_opt) for idx in range(self.layer)],
      [[conv1d(None, 3,
        strides=1, **conv_opt) for _ in range(self.sublayer)] for idx in range(self.layer)]))
    
    self.conv_post = conv1d(1, 251, **conv_opt)
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, ref = inputs

    else:
      x = inputs
      ref = None

    _in = x

    x, fes = self.pasep(x, training=training)

    x = tf.keras.activations.gelu(x)
    #x = tf.stop_gradient(x)
    
    idx = 0; fes = fes[::-1]
    for _enc, (up_conv, convs) in zip(fes, self.up_convs):
      x = tf.keras.activations.gelu(up_conv(x))
      
      #enc = self.enc_convs[idx](_enc)
      #enc = _enc
      #x = tf.concat([x, enc], -1)

      for conv in convs:
        x = tf.keras.activations.gelu(conv(x)) + x
      idx += 1
    
    x = self.conv_post(x)
    x = tf.math.tanh(x)
    x = tf.squeeze(x, -1)

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

      return samp_loss + spec_loss

    return x
