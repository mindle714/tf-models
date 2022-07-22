import tensorflow as tf
from util import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims
gelu = tf.keras.activations.gelu

class gnormconv1d(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(gnormconv1d, self).__init__()

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.conv = tf.keras.layers.Conv1D(512, kernel_size=10, strides=5, **conv_opt)
    self.norm = gnorm(512)
  
  def call(self, inputs, training=None):
    x = inputs
    return gelu(self.norm(self.conv(x)))

class nonormconv1d(tf.keras.layers.Layer):
  def __init__(self, ksize, *args, **kwargs):
    self.ksize = ksize
    super(nonormconv1d, self).__init__()

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)
    self.conv = tf.keras.layers.Conv1D(512, kernel_size=self.ksize, strides=2, **conv_opt)
  
  def call(self, inputs, training=None):
    x = inputs
    return gelu(self.conv(x))

class mpd_mask(tf.keras.layers.Layer):
  def __init__(self, periods, *args, **kwargs):
    self.periods = periods + [1]
    super(mpd_mask, self).__init__()

  def call(self, inputs, training=None):
    x, pd_idx = inputs

    pds = tf.gather(self.periods, pd_idx)
    def _map_fn(e):
      pd, _x = e[0], e[1]
      seq = tf.shape(_x)[0]; dim = tf.shape(_x)[1]

      mask_pos = tf.random.uniform([], maxval=pd, dtype=tf.int32)

      mask_chunk = tf.concat([[0.], tf.ones(pd-1, dtype=tf.float32)], 0)
      mask = tf.tile(mask_chunk, [(seq // pd) + 1])
      mask = tf.concat([tf.ones(mask_pos, dtype=tf.float32), mask], 0)

      mask_zero = tf.math.reduce_sum(tf.math.abs(mask))
      mask = tf.cond(mask_zero == 0, lambda: tf.ones_like(mask), lambda: mask)

      mask = tf_expd(mask[:seq], -1)
      return mask

    mask = tf.map_fn(_map_fn, (pds, x), fn_output_signature=tf.float32) 
    return mask * x

class wav2vec2(tf.keras.layers.Layer):
  def __init__(self, pretrain, *args, **kwargs):
    self.pretrain = pretrain
    self.periods = [2,3,5,7,11]
    super(wav2vec2, self).__init__()

  def get_vocab_size(self):
    return 7 * len(self.periods)

  def get_vocab_idx(self, mask_idx, pd_idx):
    return mask_idx * len(self.periods) + pd_idx

  def build(self, input_shape):
    ksizes = [3, 3, 3, 3, 2, 2]
    self.conv_layers = [gnormconv1d()] + [nonormconv1d(ksizes[i]) for i in range(6)]

    if self.pretrain:
      self.mask = mpd_mask(self.periods)
    else:
      self.mask = None
  
  def call(self, inputs, training=None):
    x = inputs
    
    batch_size = tf.shape(x)[0]
    mask_idx = tf.random.uniform([batch_size],
      maxval=len(self.conv_layers), dtype=tf.int32)
    pd_idx = tf.random.uniform([batch_size], 
      maxval=len(self.periods), dtype=tf.int32)

    fes = []
    for idx, conv in enumerate(self.conv_layers):
      fes.append(x)
      x = conv(x)

      if self.pretrain:
        _pd_idx = tf.where(idx == mask_idx, pd_idx, len(self.periods))
        x = self.mask((x, _pd_idx))

    return x, fes, mask_idx, pd_idx 

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

    self.pre_conv_mid = conv1d(self.dims[-1], self.ksize, strides=5, **conv_opt)
    self.pre_conv_post = tf.keras.layers.Dense(
      self.wav2vec2.get_vocab_size(), use_bias=False)

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
    x, fes, mask_idx, pd_idx = self.wav2vec2(x)
    x = gelu(x)

    if self.pretrain:
      pre_x = gelu(self.pre_conv_mid(x))
      pre_x = tf.math.reduce_mean(pre_x, 1)
      pre_x = self.pre_conv_post(pre_x)

      vocab_idx = self.wav2vec2.get_vocab_idx(mask_idx, pd_idx)
      pre_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=vocab_idx, logits=pre_x)

    x = self.conv_mid(x)
   
    idx = 0; fes = fes[::-1]
    for _enc, (up_conv, convs) in zip(fes, self.up_convs):
      x = gelu(up_conv(x))
      
      enc = self.enc_convs[idx](_enc)
      x = tf.concat([x, enc], -1)

      for conv in convs:
        x = gelu(conv(x)) + x
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

      return samp_loss + spec_loss, pre_loss

    return x
