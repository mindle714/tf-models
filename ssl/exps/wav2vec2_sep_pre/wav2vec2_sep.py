import tensorflow as tf
from util import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class gnormconv1d(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(gnormconv1d, self).__init__()

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(512, kernel_size=10, strides=5, use_bias=False)
    self.norm = gnorm(512)
    self.gelu = tf.keras.activations.gelu
  
  def call(self, inputs, training=None):
    x = inputs
    return self.gelu(self.norm(self.conv(x)))

class nonormconv1d(tf.keras.layers.Layer):
  def __init__(self, ksize, *args, **kwargs):
    self.ksize = ksize
    super(nonormconv1d, self).__init__()

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(512, kernel_size=self.ksize, strides=2, use_bias=False)
    self.gelu = tf.keras.activations.gelu
  
  def call(self, inputs, training=None):
    x = inputs
    return self.gelu(self.conv(x))

class featencoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(featencoder, self).__init__()

  def build(self, input_shape):
    ksizes = [3, 3, 3, 3, 2, 2]
    self.conv_layers = [gnormconv1d()] + [nonormconv1d(ksizes[i]) for i in range(6)]
  
  def call(self, inputs, training=None):
    x = inputs

    fes = []
    for conv in self.conv_layers:
      fes.append(x)
      x = conv(x)

    return x, fes

class featproj(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(featproj, self).__init__()

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
    super(posconvemb, self).__init__()

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(768, 
      kernel_size=128, strides=1, groups=16)
    self.gelu = tf.keras.activations.gelu
  
  def call(self, inputs, training=None):
    x = inputs
    shape = [tf.shape(x)[0], 64, tf.shape(x)[-1]]
    pad = tf.zeros(shape)
    x_pad = tf.concat([pad, x, pad], 1)
    return self.gelu(self.conv(x_pad)[:,:-1,:])

class attention(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.num_heads = 12 
    super(attention, self).__init__()

  def build(self, input_shape):
    dim = input_shape[-1]
    self.head_dim = dim // self.num_heads
    self.scaling = self.head_dim ** -0.5
    self.k_proj = tf.keras.layers.Dense(768, use_bias=True)
    self.v_proj = tf.keras.layers.Dense(768, use_bias=True)
    self.q_proj = tf.keras.layers.Dense(768, use_bias=True)
    self.out_proj = tf.keras.layers.Dense(768, use_bias=True)
    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
  
  def call(self, inputs, training=None):
    x = inputs

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
    super(feedforward, self).__init__()

  def build(self, input_shape):
    self.in_dense = tf.keras.layers.Dense(3072, use_bias=True)
    self.out_dense = tf.keras.layers.Dense(input_shape[-1], use_bias=True)
    self.gelu = tf.keras.activations.gelu
    #self.in_dropout = tf.keras.layers.Dropout(0)
    #self.out_dropout = tf.keras.layers.Dropout(0)
    self.in_dropout = tf.identity
    self.out_dropout = tf.identity
  
  def call(self, inputs, training=None):
    x = inputs
    x = self.in_dropout(self.gelu(self.in_dense(x)))
    x = self.out_dropout(self.out_dense(x))
    return x

class enclayer(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(enclayer, self).__init__()

  def build(self, input_shape):
    self.atten = attention()
    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
    self.norm = lnorm()
    self.feed = feedforward()
    self.out_norm = lnorm()

  def call(self, inputs, training=None):
    x = inputs
    x_attn = self.atten(x)
    x_attn = self.dropout(x_attn)
    x = x + x_attn
    x = self.norm(x)
    x = x + self.feed(x)
    x = self.out_norm(x)
    return x

class encoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(encoder, self).__init__()

  def build(self, input_shape):
    self.emb = posconvemb()
    self.norm = lnorm()
    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
    self.layers = [enclayer() for _ in range(12)]
  
  def call(self, inputs, training=None):
    x = inputs
    x = x + self.emb(x)
    x = self.norm(x)
    x = self.dropout(x)

    encs = []
    for layer in self.layers:
      encs.append(x)
      x = layer(x)
    return x, encs

class wav2vec2(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(wav2vec2, self).__init__()

  def build(self, input_shape):
    self.fe = featencoder()
    #self.fp = featproj()
    #self.enc = encoder()
  
  def call(self, inputs, training=None):
    x = inputs
    x, fes = self.fe(x)
    #x = self.fp(x)
    #x, encs = self.enc(x)
    return x, fes#, encs

class wav2vec2_seq(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(wav2vec2_seq, self).__init__()

  def build(self, input_shape):
    self.wav2vec2 = wav2vec2()
    self.projector = tf.keras.layers.Dense(256, use_bias=True)
    self.classifier = tf.keras.layers.Dense(12, use_bias=True)
  
  def call(self, inputs, training=None):
    x = inputs
    x, fes, encs = self.wav2vec2(x)
    x = self.projector(x)
    x = tf.math.reduce_mean(x, 1)
    x = self.classifier(x)
    return x

class wav2vec2_unet(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(wav2vec2_unet, self).__init__()
    self.layer = 7
    self.dims = [64 for _ in range(self.layer)]
    self.strides = [5, 2, 2, 2, 2, 2, 2]
    self.ksize = 16
    self.sublayer = 4

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)
    self.ref_len = input_shape[0][1]

    self.wav2vec2 = wav2vec2()

    self.conv_mid = conv1d(self.dims[-1], self.ksize, **conv_opt)

    self.enc_convs = [tf.keras.layers.Dense(64) for _ in range(self.layer)]
    self.up_norms = [lnorm() for _ in range(self.layer)]
    self.up_convs = list(zip(
      [conv1dtrans(self.dims[::-1][idx], 5,
        strides=self.strides[::-1][idx], **conv_opt) for idx in range(self.layer)],
      [[conv1d(None, self.ksize,
        strides=1, **conv_opt) for _ in range(self.sublayer)] for idx in range(self.layer)]))
    #self.up_int_convs = [
    #  conv1dtrans(self.dims[::-1][idx], 5,
    #    strides=self.strides[::-1][idx], **conv_opt) for idx in range(self.layer)]

    self.conv_post = conv1d(2, self.ksize, **conv_opt)
  
  def call(self, inputs, training=None):
    sm = None

    if isinstance(inputs, tuple):
      s1, s2, x = inputs
      sm = tf.concat([tf_expd(s1, -1), tf_expd(s2, -1)], -1)

    else:
      x = inputs

    x = tf_expd(x, -1)
    x, fes = self.wav2vec2(x)
    x = tf.keras.activations.gelu(x)
    #x = tf.stop_gradient(x)

    x = self.conv_mid(x)
   
    idx = 0; fes = fes[::-1]
    #for _enc, (up_conv, convs) in zip(encs, self.up_convs):
    for _enc, (up_conv, convs) in zip(fes, self.up_convs):
      x = tf.keras.activations.gelu(up_conv(x))
      
      enc = self.enc_convs[idx](_enc)
      #for _idx in range(idx+1):
      #  enc = tf.keras.activations.gelu(self.up_int_convs[_idx](enc))
      #enc = self.up_norms[idx](enc)

      #x = tf.concat([x, enc], -1)
      #x = x + enc[:,:tf.shape(x)[1],:]
      pad = tf.shape(enc)[1] - tf.shape(x)[1]
      lpad = pad // 2
      rpad = pad - lpad
      x = tf.concat([
        tf.zeros_like(x)[:,:lpad,:],
        x,
        tf.zeros_like(x)[:,:rpad,:]], 1)
      x = tf.concat([x, enc], -1)

      #x = tf.keras.activations.gelu(up_conv(x))
      for conv in convs:
        x = tf.keras.activations.gelu(conv(x)) + x
      idx += 1
    
    x = self.conv_post(x)
    x = tf.math.tanh(x)

    #pad = tf.shape(x)[1] - self.ref_len
    #lpad = pad // 2
    #rpad = pad - lpad
    #x = x[:, lpad:-rpad]

    if sm is not None:
      snr, sort_x = si_snr(sm, x)
      return -tf.math.reduce_mean(snr)

    return x
