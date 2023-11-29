import tensorflow as tf
from util import *
import mask
import math
from tf_seq2seq_losses import classic_ctc_loss as _ctc_loss

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims
gelu = tf.keras.activations.gelu

def rel_pos_bucket(rel_pos, bidirec=True, num_buckets=320, max_dist=800):
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

class gnormconv1d(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(gnormconv1d, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(512, kernel_size=10, strides=5, use_bias=False)
    self.norm = gnorm(512)
  
  def call(self, inputs, training=None):
    x = inputs
    return gelu(self.norm(self.conv(x)))

class nonormconv1d(tf.keras.layers.Layer):
  def __init__(self, ksize, *args, **kwargs):
    self.ksize = ksize
    super(nonormconv1d, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(512,
      kernel_size=self.ksize, strides=2, use_bias=False)
  
  def call(self, inputs, training=None):
    x = inputs
    return gelu(self.conv(x))

class featencoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(featencoder, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    ksizes = [3, 3, 3, 3, 2, 2]
    self.conv_layers = [gnormconv1d()] + [nonormconv1d(ksizes[i]) for i in range(6)]
  
  def call(self, inputs, training=None, eps=1e-10):
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
    x_norm = self.norm(x)
    return self.dropout(self.proj(x_norm)), x_norm

class posconvemb(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(posconvemb, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(768, 
      kernel_size=128, strides=1, groups=16)
  
  def call(self, inputs, training=None):
    x = inputs
    x_pad = tf.pad(x, tf.constant([[0, 0], [64, 64], [0, 0]]), "CONSTANT")
    return gelu(self.conv(x_pad)[:,:-1,:])

class attention(tf.keras.layers.Layer):
  def __init__(self, num_heads=12, dim=768, *args, **kwargs):
    super(attention, self).__init__(*args, **kwargs)
    
    self.num_heads = num_heads
    self.dim = dim

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

    self.grep_linear = tf.keras.layers.Dense(8, use_bias=True)
    self.grep_a = self.add_weight(
      shape=(self.num_heads, 1, 1), name="grep_a")

    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, attn_mask, rel_bias = inputs

    else:
      x = inputs
      attn_mask = None
      rel_bias = None
    
    x_k = x; x_q = x

    def reshape(e):
      e = tf.reshape(e,
        tf.concat([tf.shape(x_k)[:2], [self.num_heads, self.head_dim]], 0))
      e = tf.transpose(e, [0, 2, 1, 3])
      e = tf.reshape(e,
        tf.concat([[-1], tf.shape(e)[-2:]], 0))
      return e

    q = reshape(self.q_proj(x_q))
    k = reshape(self.k_proj(x_k))
    v = reshape(self.v_proj(x_q))

    x_h = reshape(x)
    gate_ab = self.grep_linear(x_h)
    gate_ab = tf.reshape(gate_ab, tf.concat([tf.shape(gate_ab)[:2], [2, 4]], 0))
    gate_a, gate_b = tf.split(
      tf.nn.sigmoid(tf.math.reduce_sum(gate_ab, -1)), 2, axis=-1)

    gate_a_1 = gate_a * (gate_b * self.grep_a - 1.) + 2.
    rel_bias = gate_a_1 * tf.transpose(rel_bias, [0, 2, 1])

    attn_weights = tf.linalg.matmul(q, k, transpose_b=True) * self.scaling
    attn_weights += rel_bias

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
      x, attn_mask, rel_bias = inputs

    else:
      x = inputs
      attn_mask = None
      rel_bias = None

    x_attn = self.atten((x, attn_mask, rel_bias))
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
    self.num_buckets = 320

  def build(self, input_shape):
    self.emb = posconvemb()
    self.norm = lnorm()
    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
    self.layers = [enclayer() for _ in range(self.num_enc_layer)]
    
    self.rel_bias = self.add_weight(
      shape=(12, self.num_buckets), name="rel_bias")
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, attn_mask = inputs

    else:
      x = inputs
      attn_mask = None

    if len(self.layers) > 0:
      x = x + self.emb(x)
      x = self.norm(x)
      x = self.dropout(x)

    x_max_len = tf.shape(x)[1]
    rel_pos = tf.reshape(tf.range(x_max_len), (-1, 1)) - \
            tf.reshape(tf.range(x_max_len), (1, -1))
    rp_bucket = rel_pos_bucket(rel_pos)

    rel_bias = tf.gather(self.rel_bias, rp_bucket, axis=1)
    rel_bias = tf.tile(rel_bias, (tf.shape(x)[0], 1, 1)) 

    encs = []
    for i, layer in enumerate(self.layers):
      encs.append(x)
      x = layer((x, attn_mask, rel_bias))

    encs.append(x)
    return encs

class wavlm(tf.keras.layers.Layer):
  def __init__(self,
               num_enc_layer=1,
               *args, **kwargs):
    super(wavlm, self).__init__(*args, **kwargs)
   
    self.num_enc_layer = num_enc_layer

  def build(self, input_shape):
    self.fe = featencoder()
    self.fp = featproj()

    self.masked_spec_embed = self.add_weight(
      shape=(768,), name="masked_spec_embed")
    self.enc = encoder(self.num_enc_layer)
  
  def call(self, inputs, training=None):
    mask_time_indices = None
    attn_mask = None

    if isinstance(inputs, tuple) and len(inputs) == 3:
      x, mask_time_indices, attn_mask = inputs
    
    elif isinstance(inputs, tuple) and len(inputs) == 2:
      x, attn_mask = inputs

    else:
      x = inputs

    x = self.fe(tf_expd(x, -1))
    x, x_feat = self.fp(x)

    if mask_time_indices is not None:
      _mask = tf_expd(mask_time_indices, -1)
      x = self.masked_spec_embed * _mask + x * (1. - _mask)

    encs = self.enc((x, attn_mask))
    return encs, x_feat

class wavlm_seq(tf.keras.layers.Layer):
  def __init__(self, num_enc_layer=12, *args, **kwargs):
    super(wavlm_seq, self).__init__(*args, **kwargs)
    self.num_enc_layer = num_enc_layer

  def build(self, input_shape):
    self.wavlm = wavlm(self.num_enc_layer)

    self.final_proj = tf.keras.layers.Dense(256, use_bias=True, name="final_proj")
    self.labels_embs = self.add_weight(
      shape=(504, 256), name="labels_embs")
    
    self.cossim = tf.keras.losses.CosineSimilarity(axis=-1, reduction='none')
    self.temperature = 0.1

    self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

  def call(self, inputs, training=None):
    if isinstance(inputs, tuple) and len(inputs) == 4:
      x, hb_idx, mask_time_indices, attn_mask = inputs
      
      encs, _ = self.wavlm((x, mask_time_indices, attn_mask), training=training)
      x = encs[-1]

      x = self.final_proj(x)
      qx_feat = tf.gather(self.labels_embs, hb_idx, axis=0)

      neg_qx_feat = tf_expd(tf_expd(self.labels_embs, 1), 1)
      neg_qx_feat = tf.tile(neg_qx_feat, [1, tf.shape(x)[0], tf.shape(x)[1], 1])
      
      qx_logits = -self.cossim(x, tf_expd(qx_feat, 0)) / self.temperature
      neg_qx_logits = -self.cossim(x, neg_qx_feat) / self.temperature

      neg_is_pos = tf.math.reduce_all(qx_feat == neg_qx_feat, -1)
      if tf.math.reduce_any(neg_is_pos):
        neg_qx_logits = tf.where(neg_is_pos,
          tf.ones_like(neg_qx_logits) * (-np.inf), neg_qx_logits)

      logits = tf.concat([qx_logits, neg_qx_logits], 0)
      logits = tf.transpose(logits, [2, 1, 0])
      logits = tf.reshape(logits, [-1, tf.shape(logits)[-1]])

      mask_loss = tf.transpose(mask_time_indices, [1, 0])
      mask_loss = tf.reshape(mask_loss, [-1])
      cont_loss = self.cce(tf.zeros_like(mask_loss), logits)
      cont_loss = tf.math.reduce_sum(cont_loss * mask_loss)

      return cont_loss

    if isinstance(inputs, tuple) and len(inputs) == 2:
      x, attn_mask = inputs

    else:
      x = inputs
      attn_mask = None

    encs, x_feat = self.wavlm((x, attn_mask), training=training)
    return encs, x_feat

class wavlm_phone(tf.keras.layers.Layer):
  def __init__(self, 
               num_enc_layer=12, num_class=74, 
               use_last=False, use_layers=12,
               mask_prob=0.1, mask_len=10,
               single_output=False,
               *args, **kwargs):
    super(wavlm_phone, self).__init__(*args, **kwargs)
    
    self.num_enc_layer = num_enc_layer
    self.num_class = num_class
    self.use_last = use_last
    self.use_layers = use_layers
    self.mask_prob = mask_prob
    self.mask_len = mask_len
    self.min_masks = 2
    self.single_output = single_output

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.wavlm = wavlm_seq(self.num_enc_layer)
    
    self.proj = tf.keras.layers.Dense(256, use_bias=True)
    self.linear = tf.keras.layers.Dense(self.num_class, use_bias=True)
    
  def call(self, inputs, training=None,
           ssl_loss=False, stop_grad=False, ctc=True,
           return_feat=False):
    if isinstance(inputs, tuple) and len(inputs) == 4:
      x, ref, _x_len, ref_len = inputs

      max_x_len = mask.get_feat_extract_output_length(tf.shape(x)[1])
      x_len = mask.get_feat_extract_output_length(_x_len)
      attn_mask = tf.sequence_mask(tf.squeeze(x_len, -1), max_x_len)
      attn_mask = 1. - tf.cast(attn_mask, dtype=tf.float32)
      attn_mask *= -1e9

    else:
      x = inputs
      _x_len = None; x_len = None
      ref = None; ref_len = None
      attn_mask = None

    if ssl_loss:
      x_feat = x; x_feat_len = _x_len
      batch_size = x_feat.shape[0]
      
      max_x_feat_len = mask.get_feat_extract_output_length(tf.shape(x_feat)[1])
      x_feat_len = mask.get_feat_extract_output_length(x_feat_len)
      feat_attn_mask = tf.sequence_mask(tf.squeeze(x_feat_len, -1), max_x_feat_len)

      mask_time_indices = mask.compute_mask_indices(
        batch_size, max_x_feat_len,
        tf.cast(feat_attn_mask, tf.int32),
        self.mask_prob, self.mask_len, self.min_masks)
        
      feat_attn_mask = 1. - tf.cast(feat_attn_mask, dtype=tf.float32)
      feat_attn_mask *= -1e9

      seq_loss = self.wavlm(
        (x_feat, ref, mask_time_indices, feat_attn_mask), training=training)

      return seq_loss

    encs, _ = self.wavlm((x, attn_mask), training=training)
    assert len(encs) == (self.num_enc_layer + 1)
    
    if return_feat:
      return encs

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
