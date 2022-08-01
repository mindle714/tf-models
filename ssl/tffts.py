import tensorflow as tf
import numpy as np
from util import *
import fftlib
import scipy.signal

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

  def build(self, input_shape):
    ksizes = [3, 3, 3, 3, 2, 2]
    self.conv_layers = [gnormconv1d()] + [nonormconv1d(ksizes[i]) for i in range(6)]
  
  def call(self, inputs, training=None):
    x = inputs

    fes = []
    for idx, conv in enumerate(self.conv_layers):
      fes.append(x)
      x = conv(x)

    return x, fes 

class tconv(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.conv_args = args
    self.conv_kwargs = kwargs
    super(tconv, self).__init__()

  def build(self, input_shape):
    conv_opt = dict(padding='same')

    self.dconv = conv1d(self.conv_args[0], 4, groups=8, **conv_opt)

    pconv_args = (self.conv_args[0], 1)
    self.pconv = conv1d(*pconv_args, **conv_opt)
  
  def call(self, inputs, training=None):
    x = inputs
    x = self.pconv(self.dconv(x))

    return x 

class tffts_unet(tf.keras.layers.Layer):
  def __init__(self, pretrain=False, *args, **kwargs):
    self.flens = [512, 256, 128]
    self.hlens = [128, 64, 32]
    for flen in self.flens:
      assert 2**(np.log2(flen)) == flen
    super(tffts_unet, self).__init__()
    self.pretrain = pretrain
    self.layer = 7
    self.dims = [64 for _ in range(self.layer)]
    self.strides = [5, 2, 2, 2, 2, 2, 2]
    self.ksize = 16
    self.sublayer = 4
    assert not pretrain

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)
    
    self.tstfts = [tstft(flen, hlen) for flen, hlen in zip(self.flens, self.hlens)]
    self.itstfts = [tstft(flen, hlen) for flen, hlen in zip(self.flens, self.hlens)]
    self.tfft_rprjs = [conv2d(1, (8,8), strides=(2,2), **conv_opt) for _ in self.hlens]
    self.tfft_iprjs = [conv2d(1, (8,8), strides=(2,2), **conv_opt) for _ in self.hlens]
    self.itfft_rprjs = [conv2dtrans(1, (8,8), strides=(2,2), **conv_opt) for _ in self.flens]
    self.itfft_iprjs = [conv2dtrans(1, (8,8), strides=(2,2), **conv_opt) for _ in self.flens]

    self.wav2vec2 = wav2vec2(self.pretrain)

    self.conv_mid = conv1d(self.dims[-1], self.ksize, **conv_opt)

    self.pre_conv_mid = conv1d(self.dims[-1], self.ksize, strides=5, **conv_opt)
    self.pre_conv_post = tf.keras.layers.Dense(
#      250, use_bias=False)
      32, use_bias=False)

    self.enc_convs = [tf.keras.layers.Dense(64) for _ in range(self.layer)]
    self.up_norms = [lnorm() for _ in range(self.layer)]
    self.up_convs = list(zip(
      [conv1dtrans(self.dims[::-1][idx], 5,
        strides=self.strides[::-1][idx], **conv_opt) for idx in range(self.layer)],
      [[conv1d(None, self.ksize,
        strides=1, **conv_opt) for _ in range(self.sublayer)] for idx in range(self.layer)]))

    self.enc_post = conv1d(1+2*len(self.flens), self.ksize, **conv_opt)
    self.conv_post = conv1d(1, self.ksize, **conv_opt)

  # inputs[batch, seq]
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, ref = inputs

    elif self.pretrain:
      x = inputs
      ref = x

    else:
      x = inputs
      ref = None

    batch_size = tf.shape(x)[0]
    slen = tf.shape(x)[1]
    fxs = []

    for tstft, tfft_rprj, tfft_iprj in zip(self.tstfts, self.tfft_rprjs, self.tfft_iprjs):
      rx, ix = tstft(x)

      _fxs = [tf.reshape(tfft_rprj(tf_expd(rx, -1)), [batch_size, -1, 1]),
              tf.reshape(tfft_iprj(tf_expd(ix, -1)), [batch_size, -1, 1])]
      fxs += _fxs

    fx = tf.concat(fxs, -1)
    x = tf.concat([fx, tf_expd(x, -1)], -1)
    
    x, fes = self.wav2vec2(x)
    x = gelu(x)

    if self.pretrain:
      pre_x = gelu(self.pre_conv_mid(x))
      pre_x = tf.math.reduce_mean(pre_x, 1)
      pre_x = self.pre_conv_post(pre_x)

      pre_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=mask_idx,
        logits=pre_x)

    x = self.conv_mid(x)
   
    idx = 0; fes = fes[::-1]
    for _enc, (up_conv, convs) in zip(fes, self.up_convs):
      x = gelu(up_conv(x))
      
      enc = self.enc_convs[idx](_enc)
      x = tf.concat([x, enc], -1)

      for conv in convs:
        x = gelu(conv(x)) + x
      idx += 1
    
    x = gelu(self.enc_post(x))
    xs = tf.split(x, 1+2*len(self.flens), -1)
    x = xs[-1]; xs = [(xs[2*i], xs[2*i+1]) for i in range(len(self.flens))]

    fxs = []

    for (rx, ix), itstft, itfft_rprj, itfft_iprj, hlen in zip(
      xs, self.itstfts, self.itfft_rprjs, self.itfft_iprjs, self.hlens):

      fx_r = itfft_rprj(tf.reshape(rx, [batch_size, -1, hlen*2, 1]))
      fx_i = itfft_iprj(tf.reshape(ix, [batch_size, -1, hlen*2, 1]))
      fx_r = tf.squeeze(fx_r, -1)
      fx_i = tf.squeeze(fx_i, -1)

      fx = itstft((fx_r, fx_i), inverse=True)
      fx = tf_expd(fx[:, :slen], -1)
      fxs.append(fx)

    fx = tf.concat(fxs, -1)
    x = tf.concat([fx, x], -1)

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

      if self.pretrain:
        return samp_loss + spec_loss, pre_loss
      return samp_loss + spec_loss

    return x

class tfft(tf.keras.layers.Layer):
  def __init__(self, nfft, *args, **kwargs):
    assert 2**(np.log2(nfft)) == nfft
    self.nfft = nfft
    super(tfft, self).__init__()

  def build(self, input_shape):
    def init_window(shape, dtype=None):
      return scipy.signal.get_window('hann', shape[-1]).reshape(shape)

    self.window = self.add_weight(
      shape=(1, 1, self.nfft), initializer=init_window,
      name='window', trainable=True)

    self.rconvs = []
    self.iconvs = []

    wms_t, pn_t = fftlib.decompose_fft(self.nfft, transpose=True)
    self.perm = tf.constant(pn_t, dtype=tf.float32)

    for i in range(int(np.log2(self.nfft))):
      subdim = 2**(i+1)
      hdim = 2**(i)

      def init_real(wm, subdim):
        def _init_real(shape, dtype=None):
          swm = wm.real[:subdim, :subdim]
          swm = np.concatenate([swm, swm], 0)
          swm[hdim:hdim+subdim, hdim:] = swm[:subdim, hdim:]

          w0 = np.diag(swm[:subdim, :]).reshape([1, -1])
          w1 = np.diag(swm[hdim:hdim+subdim, :]).reshape([1, -1])
          w = np.concatenate([w0, w1], 0)

          return w
        return _init_real
      
      def init_imag(wm, subdim):
        def _init_imag(shape, dtype=None):
          swm = wm.imag[:subdim, :subdim]
          swm = np.concatenate([swm, swm], 0)
          swm[hdim:hdim+subdim, hdim:] = swm[:subdim, hdim:]

          w0 = np.diag(swm[:subdim, :]).reshape([1, -1])
          w1 = np.diag(swm[hdim:hdim+subdim, :]).reshape([1, -1])
          w = np.concatenate([w0, w1], 0)

          return w
        return _init_imag

      self.rconvs.append(self.add_weight(
        shape=(2, subdim), initializer=init_real(wms_t[i], subdim),
        name='rconvs{}'.format(i), trainable=True))
      self.iconvs.append(self.add_weight(
        shape=(2, subdim), initializer=init_imag(wms_t[i], subdim),
        name='iconvs{}'.format(i), trainable=True))

  def conv(self, x, _conv, stride):
    subdim = tf.shape(_conv)[1]
    filt = tf.concat([
      tf.concat([tf.linalg.diag(_conv[0][:subdim//2]), 
                 tf.linalg.diag(_conv[1][:subdim//2])], 0),
      tf.concat([tf.linalg.diag(_conv[0][subdim//2:]),
                 tf.linalg.diag(_conv[1][subdim//2:])], 0)
    ], 1)

    return tf.nn.conv1d(x, 
      tf.expand_dims(filt, 1), stride=stride, padding='SAME')
 
  # inputs[batch, seq, self.nfft]
  def call(self, inputs, inverse=False, training=None):
    if not inverse:
      x = inputs
      x *= self.window

      rx = x
      ix = tf.zeros_like(x)

      rx = tf.linalg.matmul(rx, self.perm)
      numf = tf.shape(rx)[1]

      for i in range(int(np.log2(self.nfft))):
        subdim = 2**(i+1)

        rx = tf.reshape(rx, [tf.shape(rx)[0], -1, 1]) 
        ix = tf.reshape(ix, [tf.shape(ix)[0], -1, 1]) 

        _rrx = self.conv(rx, self.rconvs[i], subdim)
        _iix = self.conv(ix, self.iconvs[i], subdim)
        _rix = self.conv(ix, self.rconvs[i], subdim)
        _irx = self.conv(rx, self.iconvs[i], subdim)

        rx = _rrx - _iix
        ix = _rix + _irx

        rx = tf.reshape(rx, [tf.shape(rx)[0], numf, -1])
        ix = tf.reshape(ix, [tf.shape(ix)[0], numf, -1])
    
      return rx, ix

    else:
      rx, ix = inputs
    
      rx = tf.linalg.matmul(rx, self.perm)
      ix = tf.linalg.matmul(-ix, self.perm)

      for i in range(int(np.log2(self.nfft))):
        subdim = 2**(i+1)
        rx = tf.reshape(rx, [tf.shape(rx)[0], -1, 1]) 
        ix = tf.reshape(ix, [tf.shape(ix)[0], -1, 1]) 

        _rrx = self.conv(rx, self.rconvs[i], subdim)
        _iix = self.conv(ix, self.iconvs[i], subdim)
        _rix = self.conv(ix, self.rconvs[i], subdim)
        _irx = self.conv(rx, self.iconvs[i], subdim)

        rx = _rrx - _iix
        ix = _rix + _irx

      rx = rx / self.nfft
      ix = ix / self.nfft

      return tf.math.sqrt(rx**2 + ix**2) * tf.squeeze(self.window, 0)

class tstft(tf.keras.layers.Layer):
  def __init__(self, nfft, hlen, *args, **kwargs):
    assert 2**(np.log2(nfft)) == nfft
    self.nfft = nfft
    self.hlen = hlen
    super(tstft, self).__init__()

  def build(self, input_shape):
    self.tfft = tfft(self.nfft)
 
  # inputs[batch, seq, self.nfft]
  def call(self, inputs, inverse=False, training=None):
    if not inverse:
      x = inputs
      fx = tf.signal.frame(x, frame_length=self.nfft,
        frame_step=self.hlen, pad_end=True)

      rx, ix = self.tfft(fx, inverse=inverse)
      return rx, ix

    else:
      rx, ix = inputs
      __x = self.tfft((rx, ix), inverse=inverse)
      _x = tf.signal.overlap_and_add(__x, self.hlen)

      win_sq = tf.tile(self.tfft.window, [1, 1, tf.shape(rx)[1]])
      win_sq = tf.reshape(win_sq, [1, tf.shape(rx)[1], -1])
      win_sq = win_sq ** 2
      win_sq = tf.signal.overlap_and_add(win_sq, self.hlen)

      # tf.math.maximum is used to prevent nan on back-prop
      _x = tf.where(win_sq > 1e-10, _x / tf.math.maximum(win_sq, 1e-10), _x)
      return _x
