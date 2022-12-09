import tensorflow as tf
from util import *
from spec_ops import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class attention(tf.keras.layers.Layer):
  def __init__(self, heads = 8, hdim = 64, *args, **kwargs):
    super(attention, self).__init__()

    self.heads = heads
    self.hdim = hdim
    self.max_pemb = 512

  def build(self, input_shape):
    dim = input_shape[0][-1]

    in_dim = self.hdim * self.heads
    
    self.q = tf.keras.layers.Dense(in_dim, use_bias=False)
    self.kv = tf.keras.layers.Dense(in_dim * 2, use_bias=False)
    self.out = tf.keras.layers.Dense(dim, use_bias=True)

    self.rel_pemb = tf.keras.layers.Embedding(2 * self.max_pemb + 1, self.hdim)
#    self.drop = tf.keras.layers.Dropout(rate = 0.2)
  
  def call(self, inputs, training=None):
    x, attn_mask = inputs

    q = self.q(x)
    k, v = tf.split(self.kv(x), 2, axis=-1)
   
    def reshape(e):
      e = tf.reshape(e,
        tf.concat([tf.shape(x)[:2], [self.heads, self.hdim]], 0))
      e = tf.transpose(e, [0, 2, 1, 3])
      return e

    q = reshape(q)
    k = reshape(k)
    v = reshape(v)
    
    scale = self.hdim ** -0.5
    attn_score = tf.linalg.matmul(q, k, transpose_b=True)
    attn_score *= scale

    seq = tf.range(tf.shape(x)[-2])
    dist = tf_expd(seq, -1) - tf_expd(seq, 0)
    dist = tf.clip_by_value(dist, - self.max_pemb, self.max_pemb) + self.max_pemb
    rel_pemb = self.rel_pemb(dist)
    pos_attn = tf.linalg.matmul(tf_expd(q, -2), tf.transpose(rel_pemb, [0, 2, 1]))
    pos_attn = tf.squeeze(pos_attn, -2) * scale
    attn_score += pos_attn

    attn_probs = tf.nn.softmax(attn_score, -1)
    ctx = tf.linalg.matmul(attn_probs, v)
    ctx = tf.transpose(ctx, [0, 2, 1, 3])
    ctx = tf.reshape(ctx,
      tf.concat([tf.shape(ctx)[:-2], [-1]], 0))

    x = self.out(ctx)
#    x = self.drop(x)
 
    return x

class fforward(tf.keras.layers.Layer):
  def __init__(self, mult = 4, *args, **kwargs):
    super(fforward, self).__init__()
    self.mult = mult

  def build(self, input_shape):
    dim = input_shape[-1]

    self.ff1 = tf.keras.layers.Dense(dim * self.mult)
#    self.drop = tf.keras.layers.Dropout(rate = 0.2)
    self.ff2 = tf.keras.layers.Dense(dim)
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.ff1(x)
    x = tf.keras.activations.swish(x)
#    x = self.drop(x)
    x = self.ff2(x)
#    x = self.drop(x)

    return x

class conformerconv(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(conformerconv, self).__init__()

    self.exp_factor = 2
    self.ksize = 31

  def build(self, input_shape):
    dim = input_shape[-1]

    self.lnorm = lnorm()
    self.conv = conv1d(dim * self.exp_factor * 2, 1)
    self.dconv = depthconv1d(self.ksize)
    self.bnorm = tf.keras.layers.BatchNormalization(epsilon = 1e-5, momentum = 0.9)
    self.conv_2 = conv1d(dim, 1)
#    self.drop = tf.keras.layers.Dropout(rate = 0.)
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.lnorm(x)
    x = self.conv(x)
    x, gate = tf.split(x, 2, axis=-1)
    x = x * tf.math.sigmoid(gate)
    x = self.dconv(x)
    x = self.bnorm(x)
    x = tf.keras.activations.swish(x)
    x = self.conv_2(x)
#    x = self.drop(x)

    return x

class conformer(tf.keras.layers.Layer):
  def __init__(self, 
               heads = 8, hdim = 64, ff_mult = 4,
               conv_exp_factor = 2, ksize = 31,
               *args, **kwargs):
    super(conformer, self).__init__()

    self.hdim = hdim
    self.heads = heads
    self.ff_mult = ff_mult
    self.conv_exp_factor = conv_exp_factor
    self.ksize = ksize

  def build(self, input_shape):
    dim = input_shape[-1]

    self.ff1 = fforward(mult = self.ff_mult)
    self.attn = attention(heads = self.heads, hdim = self.hdim)
    self.conv = conformerconv()
    self.ff2 = fforward(mult = self.ff_mult)

    self.attn_norm = lnorm()
    self.ff1_norm = lnorm()
    self.ff2_norm = lnorm()

    self.norm = lnorm()
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.ff1(self.ff1_norm(x)) * 0.5 + x
    x = self.attn((self.attn_norm(x), None)) + x
    x = self.conv(x) + x
    x = self.ff2(self.ff2_norm(x)) * 0.5 + x
    x = self.norm(x)

    return x

class tscb(tf.keras.layers.Layer):
  def __init__(self, channel = 64, *args, **kwargs):
    super(tscb, self).__init__()
    self.channel = 64

  def build(self, input_shape):
    dim = input_shape[-1]
    self.t_conformer = conformer(heads = 4, hdim = self.channel // 4)
    self.f_conformer = conformer(heads = 4, hdim = self.channel // 4)
  
  def call(self, inputs, training=None):
    x = inputs

    x = tf.transpose(x, [0, 2, 1, 3])
    x_shape = tf.shape(x)
    x = tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], -1))
    x = self.t_conformer(x) + x

    x = tf.reshape(x, x_shape)
    x = tf.transpose(x, [0, 2, 1, 3])
    x_shape = tf.shape(x)
    x = tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], -1))
    x = self.f_conformer(x) + x

    x = tf.reshape(x, x_shape)
    return x

class dildense(tf.keras.layers.Layer):
  def __init__(self, depth = 4, *args, **kwargs):
    super(dildense, self).__init__()

    self.depth = depth
    self.twidth = 2
    self.ksize = (self.twidth, 3)

  def build(self, input_shape):
    dim = input_shape[-1]

    self.convs = [conv2d(dim, self.ksize, 
      dilation_rate=(2 ** i, 1)) for i in range(self.depth)]
    self.norms = [inorm2d() for _ in range(self.depth)]
    self.prelus = [tf.keras.layers.PReLU(
      shared_axes=[1,2]) for _ in range(self.depth)]

  def call(self, inputs, training=None):
    x = inputs

    _x = x
    for i in range(self.depth):
      dil = 2 ** i
      pad_len = self.twidth + (dil - 1) * (self.twidth - 1) - 1

      x = tf.pad(_x, tf.constant(
        [[0, 0], [pad_len, 0], [1,1], [0, 0]]), 
        mode='constant', constant_values=0)
      x = self.convs[i](x)
      x = self.norms[i](x)
      x = self.prelus[i](x)
      _x = tf.concat([x, _x], -1)

    return x

class dencoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(dencoder, self).__init__()
    self.channels = 64

  def build(self, input_shape):
    self.conv_1 = conv2d(self.channels, (1,1), strides=(1,1)) 
    self.inorm2d = inorm2d()
    self.prelu = tf.keras.layers.PReLU(shared_axes=[1,2])

    self.dildense = dildense()
    
    self.conv_2 = conv2d(self.channels, (1,3), strides=(1,2)) 
    self.inorm2d_2 = inorm2d()
    self.prelu_2 = tf.keras.layers.PReLU(shared_axes=[1,2])
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.conv_1(x)
    x = self.inorm2d(x)
    x = self.prelu(x)

    x = self.dildense(x)

    x = tf.pad(x, tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]))
    x = self.conv_2(x)
    x = self.inorm2d_2(x)
    x = self.prelu_2(x)

    return x

class spconvtrans2d(tf.keras.layers.Layer):
  def __init__(self, dim, ksize, r = 1, *args, **kwargs):
    super(spconvtrans2d, self).__init__()

    self.dim = dim
    self.ksize = ksize
    self.r = r

  def build(self, input_shape):
    self.conv = conv2d(self.dim * self.r, self.ksize, strides=(1,1))
  
  def call(self, inputs, training=None):
    x = inputs

    x = tf.pad(x, tf.constant(
      [[0, 0], [0, 0], [1, 1], [0, 0]]))
    x = self.conv(x)

    dim = tf.shape(x)[-1]
    x = tf.reshape(x, tf.concat(
      [tf.shape(x)[:3], [self.r, dim // self.r]], -1))
    x = tf.transpose(x, [0, 1, 4, 2, 3]) # [batch, h, dim, w, r]
    x = tf.reshape(x, tf.concat(
      [tf.shape(x)[:3], [-1]], -1)) # [batch, h, dim, wr]
    x = tf.transpose(x, [0, 1, 3, 2])

    return x

class mdecoder(tf.keras.layers.Layer):
  def __init__(self, channel = 64, dim = 1, *args, **kwargs):
    super(mdecoder, self).__init__()
    self.dim = dim
    self.channel = channel

  def build(self, input_shape):
    self.dildense = dildense()
    self.subpx = spconvtrans2d(self.channel, (1, 3), 2)
    self.conv = conv2d(self.dim, (1, 2))
    self.norm = inorm2d()
    self.prelu = tf.keras.layers.PReLU(shared_axes=[1,2])
    self.conv_2 = conv2d(self.dim, (1, 1))
    self.prelu_2 = tf.keras.layers.PReLU(shared_axes=[1],
      alpha_initializer=tf.initializers.constant(-0.25))
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.dildense(x)
    x = self.subpx(x)
    x = self.conv(x)
    x = self.prelu(self.norm(x))
    x = self.conv_2(x)
    x = self.prelu_2(x)

    return x

class cdecoder(tf.keras.layers.Layer):
  def __init__(self, channel = 64, dim = 2, *args, **kwargs):
    super(cdecoder, self).__init__()
    self.dim = dim
    self.channel = channel

  def build(self, input_shape):
    self.dildense = dildense()
    self.subpx = spconvtrans2d(self.channel, (1, 3), 2)
    self.conv = conv2d(self.dim, (1, 2))
    self.norm = inorm2d()
    self.prelu = tf.keras.layers.PReLU(shared_axes=[1,2])
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.dildense(x)
    x = self.subpx(x)
    x = self.prelu(self.norm(x))
    x = self.conv(x)

    return x

class tscnet(tf.keras.layers.Layer):
  def __init__(self, n_fft, hop_len, *args, **kwargs):
    super(tscnet, self).__init__()
    self.n_fft = n_fft
    self.hop_len = hop_len

  def build(self, input_shape):
    self.denc = dencoder()
    self.tscbs = [tscb() for _ in range(4)]
    self.mdecoder = mdecoder()
    self.cdecoder = cdecoder()
  
  def call(self, inputs, training=None):
    x = inputs

    mag = tf.math.abs(x)
    phase = tf.math.angle(x)
    x = tf.concat([tf_expd(e, -1) for e in 
      [mag, tf.math.real(x), tf.math.imag(x)]], -1)

    x = self.denc(x)
    for tscb in self.tscbs:
      x = tscb(x)

    mask = self.mdecoder(x)
    out_mag = tf.squeeze(mask, -1) * mag

    x = self.cdecoder(x)
    x_r = out_mag * tf.cos(phase) + x[..., 0]
    x_i = out_mag * tf.sin(phase) + x[..., 1]

    return x_r, x_i

class cmgan(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(cmgan, self).__init__()

    self.layer = 4
    self.dims = [32, 32, 32, 32]
    self.ksize = 16
    self.sublayer = 4

    self.n_fft = 400
    self.hop_len = 100

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.tscnet = tscnet(self.n_fft, self.hop_len)

    self.conv_r = conv1d(self.n_fft//2+1, 3, **conv_opt, name='prj_r')
    self.conv_i = conv1d(self.n_fft//2+1, 3, **conv_opt, name='prj_i')
  
  def power_compress(self, x):
    mag = tf.math.abs(x)
    phase = tf.math.angle(x)
    mag **= 0.3
    return tf.complex(mag * tf.cos(phase), mag * tf.sin(phase))
  
  def power_uncompress(self, x_r, x_i):
    x = tf.complex(x_r, x_i)
    mag = tf.math.abs(x)
    phase = tf.math.angle(x)
    mag **= (1. / 0.3)
    return tf.complex(mag * tf.cos(phase), mag * tf.sin(phase))

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, ref = inputs

    else:
      x = inputs
      ref = None

    x_len = tf.shape(x)[-1]
    c = tf.math.sqrt(tf.cast(x_len, tf.float32) / 
            tf.math.reduce_sum(x ** 2., -1))
    c = tf_expd(c, -1)
    x *= c

    tot_len = tf.cast(tf.math.ceil(x_len / 100), tf.int32) * 100
    pad_len = tot_len - x_len
    x = tf.concat([x, x[:, :pad_len]], -1)

    x = tf.pad(x, tf.constant(
      [[0, 0], [self.n_fft//2, self.n_fft//2]]), mode='reflect')
    x = tf.signal.stft(x, window_fn=tf.signal.hamming_window, 
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
    
    x = self.power_compress(x)
    x_r, x_i = self.tscnet(x)
    x = self.power_uncompress(x_r, x_i)

    x = tf.signal.inverse_stft(x, window_fn=tf.signal.hamming_window,
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
    x /= c
    x = x[..., self.n_fft//2:self.n_fft//2+x_len]

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
