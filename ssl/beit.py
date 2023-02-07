import tensorflow as tf
from util import *
from spec_ops import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class patchemb(tf.keras.layers.Layer):
  def __init__(self, hdim = 768, *args, **kwargs):
    self.hdim = hdim
    super(patchemb, self).__init__()

  def build(self, input_shape):
    self.proj = conv2d(self.hdim, (16, 16), strides=(16, 16))
    self.cls_token = self.add_weight(shape=[1, 1, self.hdim], trainable=True)

  def call(self, inputs, training=None):
    x = inputs

    x = self.proj(x)
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])
    x = tf.transpose(x, [0, 2, 1])

    cls_token = tf.tile(self.cls_token, [tf.shape(x)[0], 1, 1])
    x = tf.concat([cls_token, x], axis=1)

    return x

class self_attn(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.num_heads = 12
    self.head_dim = 64
    self.wsize = (14, 14)
    super(self_attn, self).__init__()

  def build(self, input_shape):
    self.all_head_size = self.num_heads * self.head_dim
    self.query = tf.keras.layers.Dense(self.all_head_size, use_bias=True)
    self.key = tf.keras.layers.Dense(self.all_head_size, use_bias=False)
    self.value = tf.keras.layers.Dense(self.all_head_size, use_bias=True)

    num_dist = (2 * self.wsize[0] - 1) * (2 * self.wsize[1] - 1) + 3
    self.rel_bias_tbl = self.add_weight(shape=(num_dist, self.num_heads), trainable=True)

    coords_h = np.arange(self.wsize[0])
    coords_w = np.arange(self.wsize[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
    coords = coords.reshape((coords.shape[0], -1))

    rel_coords = coords[:, :, None] - coords[:, None, :]
    rel_coords = rel_coords.transpose(1, 2, 0)
    rel_coords[:, :, 0] += self.wsize[0] - 1
    rel_coords[:, :, 1] += self.wsize[1] - 1
    rel_coords[:, :, 0] *= 2 * self.wsize[1] - 1

    rel_idx = np.zeros((self.wsize[0] * self.wsize[1] + 1,) * 2, dtype=np.int)
    rel_idx[1:, 1:] = rel_coords.sum(-1)
    rel_idx[0, 0:] = num_dist - 3 
    rel_idx[0:, 0] = num_dist - 2
    rel_idx[0, 0] = num_dist - 1
    self.rel_idx = rel_idx.flatten()
  
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
    if attn_mask is not None:
      attn_score += attn_mask

    rel_bias = tf.reshape(tf.gather(self.rel_bias_tbl, self.rel_idx),
        [self.wsize[0] * self.wsize[1] + 1,] * 2 + [-1])
    attn_score += tf_expd(tf.transpose(rel_bias, (2, 0, 1)), 0)

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
  
  def call(self, inputs, training=None):
    _x, attn_mask = inputs

    x = self.self_attn((_x, attn_mask))
    x = self.out(x)
    return x

class enclayer(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.init_scale = 0.1    
    super(enclayer, self).__init__()

  def build(self, input_shape):
    self.lnorm = lnorm(affine=True, eps=1e-12)
    self.atten = attention()

    dim = input_shape[-1]
    self.lambda_1 = self.add_weight(shape=[dim], 
        initializer=tf.constant_initializer(self.init_scale))

    self.lnorm2 = lnorm(affine=True, eps=1e-12)
    self.inter = tf.keras.layers.Dense(3072, use_bias=True)
    self.out = tf.keras.layers.Dense(dim, use_bias=True)

    self.lambda_2 = self.add_weight(shape=[dim],
        initializer=tf.constant_initializer(self.init_scale))

  def call(self, inputs, training=None):
    x = inputs

    _x = x
    x = self.lnorm(x)
    x = self.atten((x, None))
    x *= self.lambda_1
    x = x + _x

    _x = x
    x = self.lnorm2(x)
    x = self.inter(x)
    x = tf.keras.activations.gelu(x)
    x = self.out(x)
    x *= self.lambda_2
    x = x + _x

    return x

class encoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(encoder, self).__init__()

  def build(self, input_shape):
    self.layers = [enclayer() for _ in range(12)]
  
  def call(self, inputs, training=None):
    x = inputs

    for i, layer in enumerate(self.layers):
      x = layer(x)

    return x

class beit(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(beit, self).__init__()

  def build(self, input_shape):
    self.pemb = patchemb()
    self.enc = encoder()
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.pemb(x)
    x = self.enc(x)
    return x

class beit_seq(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(beit_seq, self).__init__()

  def build(self, input_shape):
    self.beit = beit()
  
  def call(self, inputs, training=None):
    x = inputs
    return self.beit(x)
