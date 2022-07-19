import tensorflow as tf
from util import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class gnormconv1d(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(gnormconv1d, self).__init__()

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.conv = tf.keras.layers.Conv1D(512, kernel_size=10, strides=5, **conv_opt)
    self.norm = gnorm(512)
    self.gelu = tf.keras.activations.gelu
  
  def call(self, inputs, training=None):
    x = inputs
    return self.gelu(self.norm(self.conv(x)))

class nonormconv1d(tf.keras.layers.Layer):
  def __init__(self, ksize, *args, **kwargs):
    self.ksize = ksize
    super(nonormconv1d, self).__init__()

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.conv = tf.keras.layers.Conv1D(512, kernel_size=self.ksize, strides=2, **conv_opt)
    self.gelu = tf.keras.activations.gelu
  
  def call(self, inputs, training=None):
    x = inputs
    return self.gelu(self.conv(x))

class pd_mask(tf.keras.layers.Layer):
  def __init__(self, period, *args, **kwargs):
    self.period = period
    super(pd_mask, self).__init__()
  
  def build(self, input_shape):
    self.dim = input_shape[-1]

  def call(self, inputs, training=None):
    x = inputs
    x_len = tf.shape(x)[1]

    mask_idx = tf.random.uniform([], maxval=self.period, dtype=tf.int32)
    _filter = tf.linalg.LinearOperatorBlockDiag([
      tf.linalg.LinearOperatorFullMatrix(tf.eye(self.dim)) if idx != mask_idx \
      else tf.linalg.LinearOperatorFullMatrix(tf.zeros([self.dim, self.dim])) \
      for idx in range(self.period)
    ]).to_dense()
    _filter = tf.reshape(_filter, [self.period, self.dim, -1])

    mask_x = tf.nn.conv1d(x, _filter, self.period, 'SAME')
    shape = tf.shape(mask_x)
    x = tf.reshape(mask_x, [shape[0], shape[1]*self.period, shape[2]//self.period])

    return x[:,:x_len,:]

class mpd_mask(tf.keras.layers.Layer):
  def __init__(self, periods=[2,3,5], *args, **kwargs):
    self.periods = periods
    super(mpd_mask, self).__init__()

  def build(self, input_shape):
    self.pds = [pd_mask(p) for p in self.periods]

  def call(self, inputs, training=None):
    x = inputs

    pd_idx = tf.random.uniform([], maxval=len(self.periods), dtype=tf.int32)

    ret_x = x
    for idx in range(len(self.pds)):
      if idx == pd_idx:
        ret_x = self.pds[idx](x)

    return ret_x

class wav2vec2(tf.keras.layers.Layer):
  def __init__(self, pretrain, *args, **kwargs):
    super(wav2vec2, self).__init__()
    self.pretrain = pretrain

  def build(self, input_shape):
    ksizes = [3, 3, 3, 3, 2, 2]
    self.conv_layers = [gnormconv1d()] + [nonormconv1d(ksizes[i]) for i in range(6)]

    if self.pretrain:
      self.masks = [mpd_mask() for _ in range(len(self.conv_layers))]
  
  def call(self, inputs, training=None):
    x = inputs

    fes = []
    for mask, conv in zip(self.masks, self.conv_layers):
      if self.pretrain:
        x = mask(x)

      fes.append(x)
      x = conv(x)

    return x, fes

class wav2vec2_unet(tf.keras.layers.Layer):
  def __init__(self, pretrain=False, *args, **kwargs):
    super(wav2vec2_unet, self).__init__()
    self.pretrain = pretrain
    self.layer = 7
    self.dims = [64 for _ in range(self.layer)]
    self.strides = [5, 2, 2, 2, 2, 2, 2]
    self.ksize = 16
    self.sublayer = 4

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.wav2vec2 = wav2vec2(self.pretrain)

    self.conv_mid = conv1d(self.dims[-1], self.ksize, **conv_opt)

    self.enc_convs = [tf.keras.layers.Dense(64) for _ in range(self.layer)]
    self.up_norms = [lnorm() for _ in range(self.layer)]
    self.up_convs = list(zip(
      [conv1dtrans(self.dims[::-1][idx], 5,
        strides=self.strides[::-1][idx], **conv_opt) for idx in range(self.layer)],
      [[conv1d(None, self.ksize,
        strides=1, **conv_opt) for _ in range(self.sublayer)] for idx in range(self.layer)]))

    self.conv_post = conv1d(1, self.ksize, **conv_opt)
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, ref = inputs
    elif self.pretrain:
      x = inputs
      ref = x
    else:
      x = inputs
      ref = None

    x = tf_expd(x, -1)
    x, fes = self.wav2vec2(x)
    x = tf.keras.activations.gelu(x)

    x = self.conv_mid(x)
   
    idx = 0; fes = fes[::-1]
    for _enc, (up_conv, convs) in zip(fes, self.up_convs):
      x = tf.keras.activations.gelu(up_conv(x))
      
      enc = self.enc_convs[idx](_enc)
      x = tf.concat([x, enc], -1)

      for conv in convs:
        x = tf.keras.activations.gelu(conv(x)) + x
      idx += 1
    
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
