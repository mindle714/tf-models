import tensorflow as tf
from util import *
import mask
from tf_seq2seq_losses import classic_ctc_loss as _ctc_loss

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims
gelu = tf.keras.activations.gelu

class gnormconv1d(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(gnormconv1d, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(512, kernel_size=10, strides=5, use_bias=False)
    self.norm = gnorm(512)
  
  def call(self, inputs, training=None):
    x = inputs
    return gelu(self.norm(self.conv(x)))

class lnormconv1d(tf.keras.layers.Layer):
  def __init__(self, ksize, stride=2, *args, **kwargs):
    self.ksize = ksize
    self.stride = stride
    super(lnormconv1d, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(
      512, kernel_size=self.ksize, strides=self.stride, use_bias=False)
    self.norm = lnorm()
  
  def call(self, inputs, training=None):
    x = inputs
    return gelu(self.norm(self.conv(x)))

class nonormconv1d(tf.keras.layers.Layer):
  def __init__(self, ksize, stride=2, *args, **kwargs):
    self.ksize = ksize
    self.stride = stride
    super(nonormconv1d, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(
      512, kernel_size=self.ksize, strides=self.stride, use_bias=False)
  
  def call(self, inputs, training=None):
    x = inputs
    return gelu(self.conv(x))

class featencoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(featencoder, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    ksizes = [10, 3, 3, 3, 3, 2, 2]
    strides = [5, 2, 2, 2, 2, 2, 2]
    self.conv_layers = [
      lnormconv1d(ksizes[i], stride=strides[i]) for i in range(len(ksizes))]
  
  def call(self, inputs, training=None):
    x = inputs

    for conv in self.conv_layers:
      x = conv(x)

    return x

class featproj(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(featproj, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.norm = lnorm()
    self.proj = tf.keras.layers.Dense(768, use_bias=True)
    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
  
  def call(self, inputs, training=None):
    x = inputs
    return self.dropout(self.proj(self.norm(x)))

class posconvemb(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(posconvemb, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.convs = [tf.keras.layers.Conv1D(768,
      kernel_size=19, strides=1, groups=16) for _ in range(5)]
    self.norm = lnorm(affine=False)
  
  def call(self, inputs, training=None):
    x = inputs

    for conv in self.convs:
      shape = [tf.shape(x)[0], 9, tf.shape(x)[-1]]
      pad = tf.zeros(shape)
      x_pad = tf.concat([pad, x, pad], 1)
      x = gelu(self.norm(conv(x_pad)))

    return x

class attention(tf.keras.layers.Layer):
  def __init__(self, num_heads=12, dim=768, *args, **kwargs):
    self.num_heads = num_heads
    self.dim = dim
    super(attention, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    if isinstance(input_shape, tuple):
      dim = input_shape[0][-1]

    else:
      dim = input_shape[-1]

    self.head_dim = dim // self.num_heads
    self.scaling = self.head_dim ** -0.5
    self.k_proj = tf.keras.layers.Dense(self.dim, use_bias=True)
    self.v_proj = tf.keras.layers.Dense(self.dim, use_bias=True)
    self.q_proj = tf.keras.layers.Dense(self.dim, use_bias=True)
    self.out_proj = tf.keras.layers.Dense(self.dim, use_bias=True)
    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, attn_mask = inputs

    else:
      x = inputs
      attn_mask = None

    def reshape(e):
      e = tf.reshape(e,
        tf.concat([tf.shape(x)[:2], [self.num_heads, self.head_dim]], 0))
      e = tf.transpose(e, [0, 2, 1, 3])
      e = tf.reshape(e,
        tf.concat([[-1], tf.shape(e)[-2:]], 0))
      return e

    q = reshape(self.q_proj(x) * self.scaling)
    k = reshape(self.k_proj(x))
    v = reshape(self.v_proj(x))

    attn_weights = tf.linalg.matmul(q, k, transpose_b=True)
    if attn_mask is not None:
      attn_mask = tf_expd(tf_expd(attn_mask, 1), 1)
      attn_mask = tf.tile(attn_mask, [1, self.num_heads, 1, 1])
      attn_mask = tf.reshape(attn_mask,
        tf.concat([[-1], tf.shape(attn_mask)[-2:]], 0))
      attn_weights += attn_mask

    attn_weights = tf.nn.softmax(attn_weights, -1)
    attn_probs = self.dropout(attn_weights)
    attn_output = tf.linalg.matmul(attn_probs, v)

    attn_output = tf.reshape(attn_output,
      tf.concat([[-1, self.num_heads], tf.shape(attn_output)[-2:]], 0))
    attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
    attn_output = tf.reshape(attn_output,
      tf.concat([tf.shape(attn_output)[:-2], [-1]], 0))

    attn_output = self.out_proj(attn_output)
    return attn_output

class feedforward(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(feedforward, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.in_dense = tf.keras.layers.Dense(3072, use_bias=True)
    self.out_dense = tf.keras.layers.Dense(input_shape[-1], use_bias=True)
    #self.in_dropout = tf.keras.layers.Dropout(0)
    #self.out_dropout = tf.keras.layers.Dropout(0)
    self.in_dropout = tf.identity
    self.out_dropout = tf.identity
  
  def call(self, inputs, training=None):
    x = inputs
    x = self.in_dropout(gelu(self.in_dense(x)))
    x = self.out_dropout(self.out_dense(x))
    return x

class enclayer(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(enclayer, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.atten = attention()
    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
    self.norm = lnorm()
    self.feed = feedforward()
    self.out_norm = lnorm()

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, attn_mask = inputs

    else:
      x = inputs
      attn_mask = None

    x_attn = self.atten((x, attn_mask))
    x_attn = self.dropout(x_attn)
    x = x + x_attn
    x = self.norm(x)
    x = x + self.feed(x)
    x = self.out_norm(x)
    return x

class encoder(tf.keras.layers.Layer):
  def __init__(self, num_enc_layer, *args, **kwargs):
    super(encoder, self).__init__(*args, **kwargs)
    self.num_enc_layer = num_enc_layer

  def build(self, input_shape):
    self.emb = posconvemb()
    self.norm = lnorm()
    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
    self.layers = [enclayer() for _ in range(self.num_enc_layer)]
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, attn_mask = inputs

    else:
      x = inputs
      attn_mask = None

    x = x + self.emb(x)
    x = self.norm(x)
    x = self.dropout(x)

    encs = []
    for i, layer in enumerate(self.layers):
      encs.append(x)
      x = layer((x, attn_mask))

    encs.append(x)
    return encs

class data2vec(tf.keras.layers.Layer):
  def __init__(self,
               num_enc_layer=1,
               *args, **kwargs):
    super(data2vec, self).__init__(*args, **kwargs)
   
    self.num_enc_layer = num_enc_layer

  def build(self, input_shape):
    self.fe = featencoder()
    self.fp = featproj()
    self.enc = encoder(self.num_enc_layer)
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple) and len(inputs) == 2:
      x, attn_mask = inputs

    else:
      x = inputs
      attn_mask = None

    x = self.fe(tf_expd(x, -1))
    x, x_feat = self.fp(x)

    encs = self.enc((x, attn_mask))
    return encs, x_feat

class data2vec_seq(tf.keras.layers.Layer):
  def __init__(self, num_enc_layer=12, *args, **kwargs):
    super(data2vec_seq, self).__init__(*args, **kwargs)
    self.num_enc_layer = num_enc_layer

  def build(self, input_shape):
    self.data2vec = data2vec(self.num_enc_layer)

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple) and len(inputs) == 2:
      x, attn_mask = inputs

    else:
      x = inputs
      attn_mask = None

    encs, x_feat = self.data2vec((x, attn_mask), training=training)
    return encs, x_feat

class data2vec_phone(tf.keras.layers.Layer):
  def __init__(self, 
               num_enc_layer=12, num_class=74, 
               use_last=False, use_layers=12,
               single_output=False,
               *args, **kwargs):
    super(data2vec_phone, self).__init__(*args, **kwargs)
    
    self.num_enc_layer = num_enc_layer
    self.num_class = num_class
    self.use_last = use_last
    self.use_layers = use_layers
    self.single_output = single_output

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.data2vec = data2vec_seq(self.num_enc_layer)
    
    self.proj = tf.keras.layers.Dense(256, use_bias=True)
    self.linear = tf.keras.layers.Dense(self.num_class, use_bias=True)
    
  def call(self, inputs, training=None, stop_grad=False, ctc=True):
    if isinstance(inputs, tuple) and len(inputs) == 4:
      x, ref, x_len, ref_len = inputs

      max_x_len = mask.get_feat_extract_output_length(tf.shape(x)[1])
      x_len = mask.get_feat_extract_output_length(x_len)
      attn_mask = tf.sequence_mask(tf.squeeze(x_len, -1), max_x_len)
      attn_mask = 1. - tf.cast(attn_mask, dtype=tf.float32)
      attn_mask *= -1e9

    else:
      x = inputs
      x_len = None
      ref = None; ref_len = None
      attn_mask = None

    encs, _ = self.data2vec((x, attn_mask), training=training)
    assert len(encs) == (self.num_enc_layer + 1)

    if self.use_last:
      x = encs[-1]
    else:
      x = sum(encs[:(self.use_layers+1)])

    if stop_grad:
      x = tf.stop_gradient(x)

    x = gelu(x)
    
    x = self.proj(x)

    if self.single_output:
      x = tf.math.reduce_mean(x, axis=1, keepdims=True)

    # TODO in s3prl, no activation between two linear layers
    x = self.linear(x)

    if ref is not None:
      if ctc:
        ctc_loss = _ctc_loss(
          tf.cast(ref, tf.int32), x, 
          tf.squeeze(tf.cast(ref_len, tf.int32), -1), 
          tf.squeeze(tf.cast(x_len, tf.int32), -1), 
          blank_index = 0)

        return ctc_loss

      else:
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          tf.cast(ref, tf.int32), x)
        if self.single_output:
          return tf.math.reduce_mean(ce_loss)

        _ref_len = tf.squeeze(ref_len, -1)
        ce_mask = tf.sequence_mask(_ref_len, tf.shape(x)[1])
        ce_mask = tf.cast(ce_mask, x.dtype)

        ce_loss = ce_loss * ce_mask
        # instead of sample-wise masking, do batch-wise
        ce_loss = tf.math.reduce_sum(ce_loss)
        ce_loss /= (tf.math.reduce_sum(ce_mask) + 1e-9)

        return ce_loss

    return x
