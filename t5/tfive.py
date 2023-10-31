import tensorflow as tf
import math

def rel_pos_bucket(rel_pos, bidirec=True, num_buckets=32, max_dist=128):
  ret = 0
  n = -rel_pos
  if bidirec:
    num_buckets //= 2
    ret += tf.cast(tf.math.less(n, 0), tf.int32) * num_buckets
    n = tf.math.abs(n)
  else:
    n = tf.math.maximum(n, 0)
  # n in the range [0, inf)
  max_exact = num_buckets // 2
  is_small = tf.math.less(n, max_exact)
  val_if_large = max_exact + tf.cast(
    tf.math.log(tf.cast(n, tf.float32) / max_exact)
    / math.log(max_dist / max_exact) * (num_buckets - max_exact), tf.int32)
  val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
  ret += tf.where(is_small, n, val_if_large)
  return ret

class rms_norm(tf.keras.layers.Layer):
  def __init__(self, eps=1e-6, *args, **kwargs):
    super(rms_norm, self).__init__(*args, **kwargs)
    self.eps = eps

  def build(self, input_shape):
    dim = input_shape[-1]
    self.scale = self.add_weight(shape=dim, name="scale", initializer="ones")
  
  def call(self, inputs, training=None):
    x = inputs
    var = tf.math.reduce_mean(tf.square(x), -1, keepdims=True)

    return x * tf.math.rsqrt(var + self.eps) * self.scale 

class sublayer(tf.keras.layers.Layer):
  def __init__(self, layer_cls, *args, **kwargs):
    super(sublayer, self).__init__(*args, **kwargs)
    self.layer_cls = layer_cls

  def build(self, input_shape):
    self.layer = self.layer_cls()
    self.layer_norm = rms_norm()

  def call(self, inputs, training=None):
    rel_bias = None
    x_len = None

    if isinstance(inputs, tuple) and len(inputs) == 3:
      x, x_len, rel_bias = inputs
    
    elif isinstance(inputs, tuple) and len(inputs) == 2:
      x, x_len = inputs

    else:
      x = inputs

    _in = x

    x = self.layer_norm(x)
    if x_len is not None:
      mask = tf.cast(tf.sequence_mask(x_len, tf.shape(x)[1]), x.dtype)
      x = x * tf.expand_dims(mask, -1)

    if isinstance(self.layer, attention):
      x = self.layer((x, rel_bias))
    else:
      x = self.layer(x)

    x += _in
    return x

class attention(tf.keras.layers.Layer):
  def __init__(self, num_heads=8, *args, **kwargs):
    super(attention, self).__init__(*args, **kwargs)
    self.num_heads = num_heads

  def build(self, input_shape):
    if isinstance(input_shape, tuple):
      dim = input_shape[0][-1]

    else:
      dim = input_shape[-1]

    self.head_dim = dim // self.num_heads
    
    self.q = self.add_weight(shape=(512, 512), name="q")
    self.k = tf.keras.layers.Dense(512, use_bias=False, name="k")
    self.v = tf.keras.layers.Dense(512, use_bias=False, name="v")
    self.out = tf.keras.layers.Dense(512, use_bias=False, name="out")

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple) and len(inputs) == 2:
      x, rel_bias = inputs

    else:
      x = inputs
      rel_bias = None

    q = tf.linalg.matmul(x, self.q)
    k = self.k(x)
    v = self.v(x)
    
    def reshape(e):
      e = tf.reshape(e,
        tf.concat([tf.shape(e)[:2], [self.num_heads, self.head_dim]], 0))
      e = tf.transpose(e, [0, 2, 1, 3])
      e = tf.reshape(e,
        tf.concat([[-1], tf.shape(e)[-2:]], 0))
      return e

    q = reshape(q)
    k = reshape(k)
    v = reshape(v)

    attn_weights = tf.linalg.matmul(q, k, transpose_b=True)
    if rel_bias is not None:
      attn_weights += tf.transpose(rel_bias, [0, 2, 1])

    attn_probs = tf.nn.softmax(attn_weights, -1)
    attn_output = tf.linalg.matmul(attn_probs, v)
    
    attn_output = tf.reshape(attn_output,
      tf.concat([[-1, self.num_heads], tf.shape(attn_output)[-2:]], 0))
    attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
    attn_output = tf.reshape(attn_output,
      tf.concat([tf.shape(attn_output)[:-2], [-1]], 0))

    attn_output = self.out(attn_output)
    return attn_output

class encdec_attention(tf.keras.layers.Layer):
  def __init__(self, num_heads=8, *args, **kwargs):
    super(encdec_attention, self).__init__(*args, **kwargs)
    self.num_heads = num_heads

  def build(self, input_shape):
    if isinstance(input_shape, tuple):
      dim = input_shape[0][-1]

    else:
      dim = input_shape[-1]

    self.head_dim = dim // self.num_heads
    
    self.q = self.add_weight(shape=(512, 512), name="q")
    self.k = tf.keras.layers.Dense(512, use_bias=False, name="k")
    self.v = tf.keras.layers.Dense(512, use_bias=False, name="v")
    self.out = tf.keras.layers.Dense(512, use_bias=False, name="out")

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple) and len(inputs) == 2:
      x, rel_bias = inputs

    else:
      x = inputs
      rel_bias = None

    q = tf.linalg.matmul(x, self.q)
    k = self.k(x)
    v = self.v(x)
    
    def reshape(e):
      e = tf.reshape(e,
        tf.concat([tf.shape(e)[:2], [self.num_heads, self.head_dim]], 0))
      e = tf.transpose(e, [0, 2, 1, 3])
      e = tf.reshape(e,
        tf.concat([[-1], tf.shape(e)[-2:]], 0))
      return e

    q = reshape(q)
    k = reshape(k)
    v = reshape(v)

    attn_weights = tf.linalg.matmul(q, k, transpose_b=True)
    if rel_bias is not None:
      attn_weights += tf.transpose(rel_bias, [0, 2, 1])

    attn_probs = tf.nn.softmax(attn_weights, -1)
    attn_output = tf.linalg.matmul(attn_probs, v)
    
    attn_output = tf.reshape(attn_output,
      tf.concat([[-1, self.num_heads], tf.shape(attn_output)[-2:]], 0))
    attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
    attn_output = tf.reshape(attn_output,
      tf.concat([tf.shape(attn_output)[:-2], [-1]], 0))

    attn_output = self.out(attn_output)
    return attn_output

class denserelu(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(denserelu, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    if isinstance(input_shape, tuple):
      dim = input_shape[0][-1]

    else:
      dim = input_shape[-1]

    self.wi = tf.keras.layers.Dense(2048, use_bias=False, name="wi") 
    self.wo = tf.keras.layers.Dense(dim, use_bias=False, name="wo") 

  def call(self, inputs, training=None):
    x = inputs

    x = self.wi(x)
    x = tf.nn.relu(x)
    x = self.wo(x) 

    return x

class t5_encoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(t5_encoder, self).__init__(*args, **kwargs)
    self.bucket_dim = 32

  def build(self, input_shape):
    self.rel_bias = self.add_weight(shape=(8, 32), name="rel_bias")
    self.sublayers = [
      sublayer(attention), sublayer(denserelu),
      sublayer(attention), sublayer(denserelu),
      sublayer(attention), sublayer(denserelu),
      sublayer(attention), sublayer(denserelu),
      sublayer(attention), sublayer(denserelu),
      sublayer(attention), sublayer(denserelu),
    ]
    self.layer_norm = rms_norm()
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple) and len(inputs) == 2:
      x, x_len = inputs

    else:
      x = inputs
      x_len = None

    rel_pos = tf.reshape(tf.range(128), (-1, 1)) - tf.reshape(tf.range(128), (1, -1))
    rp_bucket = rel_pos_bucket(rel_pos)
    rel_bias = tf.gather(self.rel_bias, rp_bucket, axis=1)
    if x_len is not None:
      mask = tf.cast(tf.sequence_mask(x_len, tf.shape(x)[1]), x.dtype)
      rel_bias += (1. - tf.expand_dims(mask, -1)) * (-1e9) 

    for sublayer in self.sublayers:
      if sublayer.layer_cls is attention:
        x = sublayer((x, x_len, rel_bias))
      else:
        x = sublayer((x, x_len))

    x = self.layer_norm(x)
    return x

class t5_decoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(t5_decoder, self).__init__(*args, **kwargs)
    self.bucket_dim = 32

  def build(self, input_shape):
    self.rel_bias = self.add_weight(shape=(8, 32), name="rel_bias")
    self.sublayers = [
      sublayer(attention), sublayer(encdec_attention), sublayer(denserelu),
      sublayer(attention), sublayer(encdec_attention), sublayer(denserelu),
      sublayer(attention), sublayer(encdec_attention), sublayer(denserelu),
      sublayer(attention), sublayer(encdec_attention), sublayer(denserelu),
      sublayer(attention), sublayer(encdec_attention), sublayer(denserelu),
      sublayer(attention), sublayer(encdec_attention), sublayer(denserelu),
    ]
    self.layer_norm = rms_norm()
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple) and len(inputs) == 2:
      x, x_len = inputs

    else:
      x = inputs
      x_len = None

    rel_pos = tf.reshape(tf.range(32), (-1, 1)) - tf.reshape(tf.range(32), (1, -1))
    rp_bucket = rel_pos_bucket(rel_pos)
    rel_bias = tf.gather(self.rel_bias, rp_bucket, axis=1)
    if x_len is not None:
      mask = tf.cast(tf.sequence_mask(x_len, tf.shape(x)[1]), x.dtype)
      rel_bias += (1. - tf.expand_dims(mask, -1)) * (-1e9) 

    for sublayer in self.sublayers:
      if sublayer.layer_cls is attention:
        x = sublayer((x, x_len, rel_bias))
      else:
        x = sublayer((x, x_len))

    x = self.layer_norm(x)
    return x

class t5(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(t5, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.embed = self.add_weight(shape=(32128, 512), name="embed")
    self.encoder = t5_encoder()
    self.decoder = t5_decoder()
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple) and len(inputs) == 4:
      x, x_len, y, y_len = inputs

    elif isinstance(inputs, tuple) and len(inputs) == 2:
      x, x_len = inputs
      y = None; y_len = None

    else:
      x = inputs
      x_len = None
      y = None; y_len = None
    
    x = tf.gather(self.embed, x)
    x = self.encoder((x, x_len))

    if y is not None:
      y = tf.gather(self.embed, y)
      y = self.decoder((y, y_len))

    return x
