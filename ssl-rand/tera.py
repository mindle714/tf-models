import tensorflow as tf
from util import *
from spec_ops import *
from mask import mask_tera
from tf_seq2seq_losses import classic_ctc_loss as _ctc_loss

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class inputrep(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.hidden_size = 768
    super(inputrep, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.spec_transform = tf.keras.layers.Dense(768, use_bias=True)
    self.lnorm = lnorm(affine=True, eps=1e-12)

  def get_sinusoid_table(self, seq_len, hidden_size):
    X, Y = tf.meshgrid(tf.range(hidden_size), tf.range(seq_len))
    sinusoid_table = tf.cast(Y, tf.float64) / tf.math.pow(tf.cast(10000., tf.float64), 2 * (X // 2) / hidden_size)
    sinusoid_table = tf.cast(sinusoid_table, tf.float32)
    
    _x = tf.reshape(tf.transpose(sinusoid_table, [1,0]), [-1, 2, seq_len])
    sinusoid_table = tf.concat([
      tf.math.sin(_x[:,0,:]), tf.math.cos(_x[:,1,:])
    ], 1)
    sinusoid_table = tf.transpose(tf.reshape(sinusoid_table, [-1, seq_len]), [1,0])
    sinusoid_table = sinusoid_table[:seq_len, :]

    return sinusoid_table
  
  def call(self, inputs, training=None):
    x = inputs

    pos_enc = self.get_sinusoid_table(tf.shape(x)[1], self.hidden_size) 

    x = self.spec_transform(x) + pos_enc
    x = self.lnorm(x)
    return x

class self_attn(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.num_heads = 12
    self.head_dim = 64
    super(self_attn, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.all_head_size = self.num_heads * self.head_dim
    self.query = tf.keras.layers.Dense(self.all_head_size, use_bias=True)
    self.key = tf.keras.layers.Dense(self.all_head_size, use_bias=True)
    self.value = tf.keras.layers.Dense(self.all_head_size, use_bias=True)
    self.inv_hdim = 1 / np.sqrt(self.head_dim)
  
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
    attn_score *= self.inv_hdim #/= np.sqrt(self.head_dim)
    attn_score += attn_mask

    attn_probs = tf.nn.softmax(attn_score, -1)
    ctx = tf.linalg.matmul(attn_probs, v)
    ctx = tf.transpose(ctx, [0, 2, 1, 3])
    ctx = tf.reshape(ctx,
      tf.concat([tf.shape(ctx)[:-2], [-1]], 0))
 
    return ctx

class attention(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(attention, self).__init__(*args, **kwargs)

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

class enclayer(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(enclayer, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    dim = input_shape[0][-1]
    self.atten = attention()
    self.inter = tf.keras.layers.Dense(dim*4, use_bias=True)
    self.out = tf.keras.layers.Dense(dim, use_bias=True)
    self.lnorm = lnorm(affine=True, eps=1e-12)

  def call(self, inputs, training=None):
    x, attn_mask = inputs

    x = self.atten((x, attn_mask))
    _x = tf.keras.activations.gelu(self.inter(x))

    x = self.out(_x) + x
    x = self.lnorm(x)

    return x

class encoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(encoder, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.layers = [enclayer() for _ in range(3)]
  
  def call(self, inputs, training=None):
    x, attn_mask = inputs

    encs = []
    for i, layer in enumerate(self.layers):
      encs.append(x)
      x = layer((x, attn_mask))
    encs.append(x)

    return encs

class tera(tf.keras.layers.Layer):
  def __init__(self, n_fft, hop_len, *args, **kwargs):
    super(tera, self).__init__(*args, **kwargs)
    self.n_fft = n_fft
    self.hop_len = hop_len

  def build(self, input_shape):
    self.fe = inputrep()
    self.enc = encoder()
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple) and len(inputs) == 2:
      x, x_len = inputs
      attn_mask = tf.sequence_mask(tf.squeeze(x_len, -1), tf.shape(x)[1])
      attn_mask = tf.cast(attn_mask, dtype=tf.float32)
      attn_mask = 1. - attn_mask
      attn_mask *= -1e9
      attn_mask = tf_expd(tf_expd(attn_mask, 1), 1)

    else:
      x = inputs
      attn_mask = tf.zeros([1, 1, 1, tf.shape(x)[1]])

    x = self.fe(x)
    x = self.enc((x, attn_mask))

    return x

class pred_head(tf.keras.layers.Layer):
  def __init__(self, out_dim=80, *args, **kwargs):
    self.out_dim = out_dim
    super(pred_head, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    dim = input_shape[-1]
    self.dense = tf.keras.layers.Dense(dim)
    self.lnorm = lnorm(affine=True, eps=1e-12)
    self.out = tf.keras.layers.Dense(self.out_dim)
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.dense(x)
    x = tf.keras.activations.gelu(x)
    x = self.lnorm(x)
    x = self.out(x)

    return x

class tera_seq(tf.keras.layers.Layer):
  def __init__(self, n_fft=400, hop_len=160, *args, **kwargs):
    self.n_fft = n_fft
    self.hop_len = hop_len
    super(tera_seq, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.tera = tera(self.n_fft, self.hop_len)
    self.spechead = pred_head()
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple) and len(inputs) == 2:
      x, x_len = inputs
      _x = self.tera((x, x_len))

    else:
      x = inputs
      _x = self.tera(x)

    x = self.spechead(_x[-1])
    return _x, x

def stft_loss(x, ref, frame_length, frame_step, fft_length):
  stft_opt = dict(frame_length=frame_length,
    frame_step=frame_step, fft_length=fft_length)

  mag_x = tf.math.abs(stft(x, **stft_opt))
  mag_ref = tf.math.abs(stft(ref, **stft_opt))

  fro_opt = dict(axis=(-2, -1), ord='fro')
  sc_loss = tf.norm(mag_x - mag_ref, **fro_opt) / (tf.norm(mag_x, **fro_opt) + 1e-9)
  sc_loss = tf.reduce_mean(sc_loss)

  mag_loss = tf.math.log(mag_x + 1e-9) - tf.math.log(mag_ref + 1e-9)
  mag_loss = tf.reduce_mean(tf.math.abs(mag_loss), [-1, -2])

  return sc_loss + mag_loss

class tera_phone(tf.keras.layers.Layer):
  def __init__(self, num_class=74, use_last=False,
               single_output=False, *args, **kwargs):
    super(tera_phone, self).__init__(*args, **kwargs)

    self.n_fft = 400
    self.hop_len = 160
    self.num_class = num_class
    self.use_last = use_last
    self.single_output = single_output

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.tera = tera_seq(self.n_fft, self.hop_len)

    self.proj = tf.keras.layers.Dense(256, use_bias=True)
    self.linear = tf.keras.layers.Dense(self.num_class, use_bias=True)
  
  def call(self, inputs, training=None, 
           ssl_loss=False, stop_grad=False, ctc=True):
    mask_label = None
    if isinstance(inputs, tuple) and len(inputs) == 4:
      x, ref, x_len, ref_len = inputs
    
    elif isinstance(inputs, tuple) and len(inputs) == 2:
      x, x_len = inputs
      ref = None; ref_len = None

    else:
      x = inputs
      x_len = None
      ref = None; ref_len = None

    if ssl_loss:
      x_feat = x; x_feat_len = x_len

      if mask_label is None:
        x_mask, mask_label = mask_tera(x_feat, x_feat_len)
        mask_label = tf.cast(mask_label, tf.float32)
      else:
        x_mask = (1. - mask_label) * x_feat

      _, seq_out = self.tera((x_mask, x_feat_len))
      seq_loss = tf.math.abs(mask_label * (x_feat - seq_out))

      seq_loss = tf.math.reduce_sum(seq_loss, [-1, -2])
      denom = tf.math.reduce_sum(mask_label, [-1, -2])
      seq_loss /= (denom + 1e-9)

      return seq_loss

    if x_len is not None:
      xs, _ = self.tera((x, x_len), training=training)
    else:
      xs, _ = self.tera(x, training=training)

    if self.use_last:
      x = xs[-1]
    else:
      x = sum(xs)
    
    if stop_grad:
      x = tf.stop_gradient(x)

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
          blank_index=0)

        return ctc_loss

      else:
        _ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          tf.cast(ref, tf.int32), x)
        if self.single_output:
          return tf.math.reduce_mean(_ce_loss)

        _ref_len = tf.squeeze(ref_len, -1)
        ce_mask = tf.sequence_mask(_ref_len, tf.shape(x)[1])
        ce_mask = tf.cast(ce_mask, x.dtype)

        ce_loss = _ce_loss * ce_mask
        # instead of sample-wise masking, do batch-wise
        ce_loss = tf.math.reduce_sum(ce_loss)
        ce_loss /= (tf.math.reduce_sum(ce_mask) + 1e-9)

        return ce_loss

    return x
