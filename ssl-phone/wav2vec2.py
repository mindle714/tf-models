import tensorflow as tf
from util import *
import gumbel
import mask
from tf_seq2seq_losses import classic_ctc_loss as _ctc_loss

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims
gelu = tf.keras.activations.gelu

def _normalize_wav_decibel(wav, target_level=-25):
  rms = (tf.math.reduce_mean(wav**2, -1, keepdims=True))**0.5
  scalar = (10 ** (target_level / 20)) / (rms + 1e-10)
  wav = wav * scalar
  return wav

def sample_negative_indices(batch_size, seq_len, 
                            mask_time_indices = None, num_neg = 100):
  if mask_time_indices is None:
    mask_time_indices = tf.ones((batch_size, seq_len))
  else:
    mask_time_indices = tf.cast(mask_time_indices, tf.int32)

  neg_indices = []
  for idx in range(batch_size):
    high = tf.math.reduce_sum(mask_time_indices[idx]) - 1
    mapped_masked_indices = tf.squeeze(tf.where(mask_time_indices[idx]), 1)

    feat_indices = tf.expand_dims(tf.range(high + 1, dtype=tf.int32) , -1)
    feat_indices = tf.tile(feat_indices, [1, num_neg])

    sampled_indices = tf.random.uniform(
      (high + 1, num_neg), 0, high, dtype=tf.int32)
    sampled_indices = tf.where(sampled_indices >= feat_indices,
      sampled_indices + 1, sampled_indices)

    _updates = tf.gather(mapped_masked_indices, sampled_indices)      
    _neg_indices = tf.scatter_nd(
      tf.where(mask_time_indices[idx]), _updates, [seq_len, num_neg])
    _neg_indices += tf.cast(idx * seq_len, _neg_indices.dtype)
    neg_indices.append(_neg_indices)

  neg_indices = tf.concat([
    tf.expand_dims(e, 0) for e in neg_indices], 0)
  return neg_indices

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
    
    x_k = x; x_q = x

    def reshape(e):
      e = tf.reshape(e,
        tf.concat([tf.shape(x_k)[:2], [self.num_heads, self.head_dim]], 0))
      e = tf.transpose(e, [0, 2, 1, 3])
      e = tf.reshape(e,
        tf.concat([[-1], tf.shape(e)[-2:]], 0))
      return e

    q = reshape(self.q_proj(x_q) * self.scaling)
    k = reshape(self.k_proj(x_k))
    v = reshape(self.v_proj(x_q))

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

    if len(self.layers) > 0:
      x = x + self.emb(x)
      x = self.norm(x)
      x = self.dropout(x)

    encs = []
    for i, layer in enumerate(self.layers):
      encs.append(x)
      x = layer((x, attn_mask))

    encs.append(x)
    return encs

class wav2vec2(tf.keras.layers.Layer):
  def __init__(self,
               num_enc_layer=1,
               norm_wav=False,
               *args, **kwargs):
    super(wav2vec2, self).__init__(*args, **kwargs)
   
    self.num_enc_layer = num_enc_layer
    self.norm_wav = norm_wav

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
      x, mask_time_indices = inputs

    else:
      x = inputs

    if self.norm_wav:
      x = _normalize_wav_decibel(x)

    x = self.fe(tf_expd(x, -1))
    x, x_feat = self.fp(x)

    if mask_time_indices is not None:
      _mask = tf_expd(mask_time_indices, -1)
      x = self.masked_spec_embed * _mask + x * (1. - _mask)

    encs = self.enc((x, attn_mask))
    return encs, x_feat

class wav2vec2_seq(tf.keras.layers.Layer):
  def __init__(self, num_enc_layer=12, 
               norm_wav=False, *args, **kwargs):
    super(wav2vec2_seq, self).__init__(*args, **kwargs)

    self.num_enc_layer = num_enc_layer
    self.norm_wav = norm_wav

  def build(self, input_shape):
    self.wav2vec2 = wav2vec2(self.num_enc_layer, norm_wav=self.norm_wav)

    self.project_hid = tf.keras.layers.Dense(256, use_bias=True)
    self.project_q = tf.keras.layers.Dense(256, use_bias=True)
    self.quantizer = gumbel.gumbelq()

    self.cossim = tf.keras.losses.CosineSimilarity(axis=-1, reduction='none')
    self.temperature = 0.1

    self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  
  def call(self, inputs, training=None, ssl_only=False):
    mask_time_indices = None
    sampled_negative_indices = None
    skip_encs = False
    attn_mask = None

    if ssl_only:
      assert isinstance(inputs, tuple) and len(inputs) == 4
      x, mask_time_indices, sampled_negative_indices, attn_mask = inputs

      encs, x_feat = self.wav2vec2((x, mask_time_indices, attn_mask), training=training)
      x = encs[-1]
    
      x = self.project_hid(x)
      return x
    
    if isinstance(inputs, tuple) and len(inputs) == 5:
      x, qx_feat, mask_time_indices, sampled_negative_indices, attn_mask = inputs
      
      encs, x_feat = self.wav2vec2((x, mask_time_indices, attn_mask), training=training)
      x = encs[-1]
    
      x = self.project_hid(x)
    
      batch_size, seq_len, hdim = tf.shape(qx_feat)[0], tf.shape(qx_feat)[1], tf.shape(qx_feat)[2]

      neg_qx_feat = tf.gather(
        tf.reshape(qx_feat, [-1, hdim]),
        tf.reshape(sampled_negative_indices, [-1]))

      neg_qx_feat = tf.reshape(neg_qx_feat, [batch_size, seq_len, -1, hdim])
      neg_qx_feat = tf.transpose(neg_qx_feat, [2, 0, 1, 3])

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

    elif isinstance(inputs, tuple) and len(inputs) == 4:
      if isinstance(inputs[0], tuple):
        (encs, x_feat), mask_time_indices, sampled_negative_indices, attn_mask = inputs
        skip_encs = True

      else:
        x, mask_time_indices, sampled_negative_indices, attn_mask = inputs

    elif isinstance(inputs, tuple) and len(inputs) == 3:
      if isinstance(inputs[0], tuple):
        (encs, x_feat), mask_time_indices, sampled_negative_indices = inputs
        skip_encs = True

      else:
        x, mask_time_indices, sampled_negative_indices = inputs

    elif isinstance(inputs, tuple) and len(inputs) == 2:
      x, attn_mask = inputs

    else:
      x = inputs

    if not skip_encs:
      encs, x_feat = self.wav2vec2((x, mask_time_indices, attn_mask), training=training)
      if sampled_negative_indices is None:
        return encs, x_feat

    x = encs[-1]

    x = self.project_hid(x)
    qx_feat, perp = self.quantizer((x_feat, mask_time_indices), training=training)
    qx_feat = self.project_q(qx_feat)

    batch_size, seq_len, hdim = tf.shape(qx_feat)[0], tf.shape(qx_feat)[1], tf.shape(qx_feat)[2]

    neg_qx_feat = tf.gather(
      tf.reshape(qx_feat, [-1, hdim]),
      tf.reshape(sampled_negative_indices, [-1]))

    neg_qx_feat = tf.reshape(neg_qx_feat, [batch_size, seq_len, -1, hdim])
    neg_qx_feat = tf.transpose(neg_qx_feat, [2, 0, 1, 3])

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

    num_codevectors = 640
    div_loss = ((num_codevectors - perp) / num_codevectors)
    div_loss *= tf.math.reduce_sum(mask_time_indices)

    loss = cont_loss + 0.1 * div_loss 
      
    return loss

class wav2vec2_phone(tf.keras.layers.Layer):
  def __init__(self, 
               num_enc_layer=12, num_class=74, 
               use_last=False, use_layers=12, num_neg=100,
               mask_prob=0.05, mask_len=10,
               single_output=False,
               norm_wav=False,
               *args, **kwargs):
    super(wav2vec2_phone, self).__init__(*args, **kwargs)
    
    self.num_enc_layer = num_enc_layer
    self.num_class = num_class
    self.use_last = use_last
    self.use_layers = use_layers
    self.num_neg = num_neg
    self.mask_prob = mask_prob
    self.mask_len = mask_len
    self.min_masks = 2
    self.single_output = single_output
    self.norm_wav = norm_wav

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.wav2vec2 = wav2vec2_seq(self.num_enc_layer, norm_wav=self.norm_wav)
    
    self.proj = tf.keras.layers.Dense(256, use_bias=True)
    self.linear = tf.keras.layers.Dense(self.num_class, use_bias=True)
    
  def call(self, inputs, training=None,
           ssl_loss=False, stop_grad=False, ctc=True,
           ssl_only=False, ssl_only_ewc=False):
    if ssl_only_ewc:
      assert isinstance(inputs, tuple) and len(inputs) == 2
      x_feat, x_feat_len = inputs
      batch_size = x_feat.shape[0]
      
      max_x_feat_len = mask.get_feat_extract_output_length(tf.shape(x_feat)[1])
      x_feat_len = mask.get_feat_extract_output_length(x_feat_len)
      feat_attn_mask = tf.sequence_mask(tf.squeeze(x_feat_len, -1), max_x_feat_len)

      mask_time_indices = mask.compute_mask_indices(
        batch_size, max_x_feat_len,
        tf.cast(feat_attn_mask, tf.int32),
        self.mask_prob, self.mask_len, self.min_masks)

      sampled_negative_indices = sample_negative_indices(
        batch_size, max_x_feat_len, mask_time_indices, self.num_neg)
        
      feat_attn_mask = 1. - tf.cast(feat_attn_mask, dtype=tf.float32)
      feat_attn_mask *= -1e9

      seq_loss = self.wav2vec2(
        (x_feat, mask_time_indices, sampled_negative_indices, feat_attn_mask), training=training)

      return seq_loss

    if ssl_only:
      assert isinstance(inputs, tuple) and len(inputs) == 2
      x_feat, x_feat_len = inputs
      batch_size = x_feat.shape[0]
      
      max_x_feat_len = mask.get_feat_extract_output_length(tf.shape(x_feat)[1])
      x_feat_len = mask.get_feat_extract_output_length(x_feat_len)
      feat_attn_mask = tf.sequence_mask(tf.squeeze(x_feat_len, -1), max_x_feat_len)

      mask_time_indices = mask.compute_mask_indices(
        batch_size, max_x_feat_len,
        tf.cast(feat_attn_mask, tf.int32),
        self.mask_prob, self.mask_len, self.min_masks)

      sampled_negative_indices = sample_negative_indices(
        batch_size, max_x_feat_len, mask_time_indices, self.num_neg)
        
      feat_attn_mask = 1. - tf.cast(feat_attn_mask, dtype=tf.float32)
      feat_attn_mask *= -1e9
        
      _x = self.wav2vec2(
        (x_feat, mask_time_indices, sampled_negative_indices, feat_attn_mask),
        ssl_only=True, training=training)

      return mask_time_indices, sampled_negative_indices, _x
   
    mask_time_indices = None
    sampled_negative_indices = None
    qx_feat = None
    neg_qx_feat = None
    perp = None
    ema_x_feat = None

    if isinstance(inputs, tuple) and len(inputs) == 9:
      x, x_feat, ref, x_len, x_feat_len, ref_len, mask_time_indices, sampled_negative_indices, ema_x_feat = inputs
      attn_mask = None

    elif isinstance(inputs, tuple) and (len(inputs) == 6 or len(inputs) == 4):
      if len(inputs) == 6:
        x, x_feat, ref, x_len, x_feat_len, ref_len = inputs

      else:
        x, ref, x_len, ref_len = inputs
        x_feat = x; x_feat_len = x_len

      max_x_len = mask.get_feat_extract_output_length(tf.shape(x)[1])
      x_len = mask.get_feat_extract_output_length(x_len)
      attn_mask = tf.sequence_mask(tf.squeeze(x_len, -1), max_x_len)
      attn_mask = 1. - tf.cast(attn_mask, dtype=tf.float32)
      attn_mask *= -1e9

    else:
      x = inputs
      x_len = None
      x_feat = None; x_feat_len = None
      ref = None; ref_len = None
      attn_mask = None

    encs, _ = self.wav2vec2((x, attn_mask), training=training)
    assert len(encs) == (self.num_enc_layer + 1)

    if self.use_last:
      x = encs[-1]
    else:
      x = sum(encs[:(self.use_layers+1)])

    if self.single_output:
      if x_len is not None:
        attn_mask = tf.sequence_mask(tf.squeeze(x_len, -1), tf.shape(x)[1])
        attn_mask = tf.cast(attn_mask, dtype=x.dtype)
        x_sum = tf.math.reduce_sum(tf_expd(attn_mask, -1) * x, axis=1, keepdims=True)
        x = x_sum / tf_expd(tf.cast(x_len, tf.float32), -1)

      else:
        x = tf.math.reduce_mean(x, axis=1, keepdims=True)

    if stop_grad:
      x = tf.stop_gradient(x)

    x = gelu(x)
    
    x = self.proj(x)

    # TODO in s3prl, no activation between two linear layers
    x = self.linear(x)

    if ref is not None:
      seq_loss = 0.

      if ssl_loss:
        batch_size = x_feat.shape[0]
      
        max_x_feat_len = mask.get_feat_extract_output_length(tf.shape(x_feat)[1])
        x_feat_len = mask.get_feat_extract_output_length(x_feat_len)
        feat_attn_mask = tf.sequence_mask(tf.squeeze(x_feat_len, -1), max_x_feat_len)

        if mask_time_indices is None:
          mask_time_indices = mask.compute_mask_indices(
            batch_size, max_x_feat_len,
            tf.cast(feat_attn_mask, tf.int32),
            self.mask_prob, self.mask_len, self.min_masks)

          sampled_negative_indices = sample_negative_indices(
            batch_size, max_x_feat_len, mask_time_indices, self.num_neg)
        
          feat_attn_mask = 1. - tf.cast(feat_attn_mask, dtype=tf.float32)
          feat_attn_mask *= -1e9

          seq_loss = self.wav2vec2(
            (x_feat, mask_time_indices, sampled_negative_indices, feat_attn_mask), training=training)

        else: 
          feat_attn_mask = 1. - tf.cast(feat_attn_mask, dtype=tf.float32)
          feat_attn_mask *= -1e9

          seq_loss = self.wav2vec2(
            (x_feat, ema_x_feat, mask_time_indices, sampled_negative_indices, feat_attn_mask), training=training)

      if ctc:
        ctc_loss = _ctc_loss(
          tf.cast(ref, tf.int32), x, 
          tf.squeeze(tf.cast(ref_len, tf.int32), -1), 
          tf.squeeze(tf.cast(x_len, tf.int32), -1), 
          blank_index = 0)

        return ctc_loss, seq_loss

      else:
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          tf.cast(ref, tf.int32), x)
        if self.single_output:
          return tf.math.reduce_mean(ce_loss), seq_loss

        _ref_len = tf.squeeze(ref_len, -1)
        ce_mask = tf.sequence_mask(_ref_len, tf.shape(x)[1])
        ce_mask = tf.cast(ce_mask, x.dtype)

        ce_loss = ce_loss * ce_mask
        # instead of sample-wise masking, do batch-wise
        ce_loss = tf.math.reduce_sum(ce_loss)
        ce_loss /= (tf.math.reduce_sum(ce_mask) + 1e-9)

        return ce_loss, seq_loss

    return x
