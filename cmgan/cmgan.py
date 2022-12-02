import tensorflow as tf
from util import *
from spec_ops import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class self_attn(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.num_heads = 12
    self.head_dim = 64
    super(self_attn, self).__init__()

  def build(self, input_shape):
    self.all_head_size = self.num_heads * self.head_dim
    self.query = tf.keras.layers.Dense(self.all_head_size, use_bias=True)
    self.key = tf.keras.layers.Dense(self.all_head_size, use_bias=True)
    self.value = tf.keras.layers.Dense(self.all_head_size, use_bias=True)
  
  def call(self, inputs, training=None):
    x, attn_mask = inputs

    mixed_q = self.query(x)
    mixed_k = self.key(x)
    mixed_v = self.value(x)
    
    def reshape(e):
      e = tf.reshape(e,
        tf.concat([tf.shape(x)[:2], [self.num_heads, self.head_dim]], 0))
      e = tf.transpose(e, [0, 2, 1, 3])
      return e

    q = reshape(mixed_q)
    k = reshape(mixed_k)
    v = reshape(mixed_v)

    attn_score = tf.linalg.matmul(q, k, transpose_b=True)
    attn_score /= np.sqrt(self.head_dim)
    attn_score += attn_mask

    attn_probs = tf.nn.softmax(attn_score, -1)
    ctx = tf.linalg.matmul(attn_probs, v)
    ctx = tf.transpose(ctx, [0, 2, 1, 3])
    ctx = tf.reshape(ctx,
      tf.concat([tf.shape(ctx)[:-2], [-1]], 0))
 
    return ctx

class attention(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(attention, self).__init__()

  def build(self, input_shape):
    dim = input_shape[0][-1]
    self.self_attn = self_attn()
    self.out = tf.keras.layers.Dense(dim, use_bias=True)
    self.lnorm = lnorm(affine=True, eps=1e-12)
  
  def call(self, inputs, training=None):
    _x, attn_mask = inputs

    x = self.self_attn((_x, attn_mask))
    x = self.out(x) + _x
    x = self.lnorm(x)

    return x

class fforward(tf.keras.layers.Layer):
  def __init__(self, mult = 4, *args, **kwargs):
    super(fforward, self).__init__()
    self.mult = mult

  def build(self, input_shape):
    dim = input_shape[0][-1]

    self.ff1 = tf.keras.layers.Dense(dim * self.mult)
    self.drop = tf.keras.layers.Dropout()
    self.ff2 = tf.keras.layers.Dense(dim)
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.ff1(x)
    x = tf.keras.activations.swish(x)
    x = self.drop(x)
    x = self.ff2(x)
    x = self.drop(x)

    return x

class conformerconv(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(conformerconv, self).__init__()
    self.exp_factor = 2

  def build(self, input_shape):
    dim = input_shape[-1]

    self.lnorm = lnorm()
    self.conv = conv1d(dim * exp_factor * 2, 1)
    self.glu = glu()
    self.dconv = depthconv1d()
    self.bnorm = tf.keras.layers.BatchNormalization()
    self.conv_2 = conv1d(dim, 1)
    self.drop = tf.keras.layers.Dropout()
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.lnorm(x)
    x = self.conv(x)
    x = self.glu(x)
    x = self.dconv(x)
    x = self.bnorm(x)
    x = self.conv_2(x)
    x = self.drop(x)

    return x

class conformer(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(conformer, self).__init__()

  def build(self, input_shape):
    dim = input_shape[0][-1]

    self.ff1 = fforward()
    self.attn = attention()
    self.conv = conformerconv()
    self.ff2 = fforward()

    self.attn_norm = lnorm()
    self.ff1_norm = lnorm()
    self.ff2_norm = lnorm()

    self.norm = lnorm()
  
  def call(self, inputs, training=None):
    x, mask = inputs

    x = self.ff1_norm(self.ff1(x)) * 0.5 + x
    x = self.attn_norm(self.attn((x, mask))) + x
    x = self.conv(x) + x
    x = self.ff2_norm(self.ff2(x)) * 0.5 + x
    x = self.norm(x)

    return x

class tscb(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(tscb, self).__init__()

  def build(self, input_shape):
    dim = input_shape[-1]
    self.t_conformer = conformer()
    self.f_conformer = conformer()
  
  def call(self, inputs, training=None):
    x = inputs

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

class tscnet(tf.keras.layers.Layer):
  def __init__(self, n_fft, hop_len, *args, **kwargs):
    super(tscnet, self).__init__()
    self.n_fft = n_fft
    self.hop_len = hop_len

  def build(self, input_shape):
    self.denc = dencoder()
    self.tscbs = [tscb() for _ in range(4)]
  
  def call(self, inputs, training=None):
    x = inputs

    mag = tf.math.abs(x)
    phase = tf.math.angle(x)
    x = tf.concat([tf_expd(e, -1) for e in 
      [mag, tf.math.real(x), tf.math.imag(x)]], -1)

    x = self.denc(x)
    for tscb in self.tscbs:
      x = tscb(x)

    return x

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

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, ref = inputs

    else:
      x = inputs
      ref = None

    _in = x

    x_len = tf.shape(x)[-1]
    c = tf.math.sqrt(tf.cast(x_len, tf.float32) / 
            tf.math.reduce_sum(x ** 2., -1))
    x *= c

    tot_len = tf.cast(tf.math.ceil(x_len / 100), tf.int32) * 100
    pad_len = tot_len - x_len
    x = tf.concat([x, x[:, :pad_len]], -1)
    x = tf.pad(x, tf.constant(
      [[0, 0], [self.n_fft//2, self.n_fft//2]]), mode='reflect')

    x = tf.signal.stft(x, window_fn=tf.signal.hamming_window, 
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
    x = self.power_compress(x)

    x = self.tscnet(x)
    return x
