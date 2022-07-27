import tensorflow as tf
import numpy as np
from util import *
import fftlib

class tfft(tf.keras.layers.Layer):
  def __init__(self, nfft, *args, **kwargs):
    assert 2**(np.log2(nfft)) == nfft
    self.nfft = nfft
    super(tfft, self).__init__()

  def build(self, input_shape):
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
  def call(self, inputs, training=None):
    x = inputs
    rx = x
    ix = tf.zeros_like(x)

    rx = tf.linalg.matmul(rx, self.perm)

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

    return rx, ix

class itfft(tfft):
  def __init__(self, nfft, *args, **kwargs):
    assert 2**(np.log2(nfft)) == nfft
    self.nfft = nfft
    super(itfft, self).__init__(nfft)
 
  # inputs[batch, seq, self.nfft]
  def call(self, inputs, training=None):
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

    return tf.math.sqrt(rx**2 + ix**2)

class tffts(tf.keras.layers.Layer):
  def __init__(self, flens, hlens, *args, **kwargs):
    self.flens = flens
    self.hlens = hlens
    assert len(flens) == len(hlens)
    for flen in flens:
      assert 2**(np.log2(flen)) == flen
    super(tffts, self).__init__()

  def build(self, input_shape):
    self.tffts = [tfft(flen) for flen in self.flens]
    self.itffts = [itfft(flen) for flen in self.flens]

  # inputs[batch, seq]
  def call(self, inputs, training=None):
    x = inputs

    fxs = []
    for tfft, flen, hlen in zip(self.tffts, self.flens, self.hlens):
      fx = tf.signal.frame(x, frame_length=flen,
        frame_step=hlen, pad_end=True)
      fx = tfft(fx)
      fxs.append(fx)

    #x = tf.concat(fxs, -1)
    return x
