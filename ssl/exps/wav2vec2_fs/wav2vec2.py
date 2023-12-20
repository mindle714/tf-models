import tensorflow as tf
from util import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims
NUM_ENC_LAYER = 3

class gnormconv1d(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(gnormconv1d, self).__init__(*args, **kwargs)

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
    super(nonormconv1d, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(512, kernel_size=self.ksize, strides=2, use_bias=False)
    self.gelu = tf.keras.activations.gelu
  
  def call(self, inputs, training=None):
    x = inputs
    return self.gelu(self.conv(x))

class featencoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(featencoder, self).__init__(*args, **kwargs)

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
    super(attention, self).__init__(*args, **kwargs)

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
    super(feedforward, self).__init__(*args, **kwargs)

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
    super(enclayer, self).__init__(*args, **kwargs)

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
  def __init__(self, num_enc_layer, beta_r, beta_m, *args, **kwargs):
    super(encoder, self).__init__(*args, **kwargs)
   
    self.num_enc_layer = num_enc_layer
    self.beta_r = beta_r
    self.beta_m = beta_m

  def build(self, input_shape):
    self.emb = posconvemb()
    self.norm = lnorm()
    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
    self.layers = [enclayer() for _ in range(self.num_enc_layer)]
    
    self.factors = [self.add_weight(shape=(),
      initializer="zeros", name="factor_{}".format(i), trainable=False) \
      for i in range(len(self.layers))]
    
    self.axis = [0, 1]
    x_shape = input_shape[0]
    shape = [x_shape[e] for e in range(len(x_shape)) if e not in self.axis]
    
    self.ref_r = [self.add_weight(shape=shape,
      initializer="ones", name="ref_r_{}".format(i), trainable=False) \
      for i in range(len(self.layers))]
    self.ns_r = [self.add_weight(shape=shape,
      initializer="ones", name="ns_r_{}".format(i), trainable=False) \
      for i in range(len(self.layers))]

    self.x_m = [self.add_weight(shape=shape,
      initializer="zeros", name="x_m_{}".format(i), trainable=False) \
      for i in range(len(self.layers))]
    self.ref_m = [self.add_weight(shape=shape,
      initializer="zeros", name="ref_m_{}".format(i), trainable=False) \
      for i in range(len(self.layers))]
    self.ns_m = [self.add_weight(shape=shape,
      initializer="zeros", name="ns_m_{}".format(i), trainable=False) \
      for i in range(len(self.layers))]
    
    self.moving_avg_step = self.add_weight(shape=(),
      initializer="zeros", name="step", trainable=False)

  def mv_delta(self, step, e, v, momentum, debias=False):
    delta = (e - v) * (1. - momentum) 

    if debias:
      denom = 1. - tf.math.pow(momentum, step)
      delta = e - e / denom 

    return delta
  
  def call(self, inputs, training=None, eps=1e-10):
    _encs = None
    
    if isinstance(inputs, tuple):
      x, _encs = inputs
    
    else:
      x = inputs
    
    if training:
      self.moving_avg_step.assign_add(1, use_locking=True)

    x = x + self.emb(x)
    x = self.norm(x)
    x = self.dropout(x)

    encs = []
    _x = x

    for i, layer in enumerate(self.layers):
      if i >= NUM_ENC_LAYER: break

      if training and (_encs is not None):
        def reshape(e):
          for ax in self.axis:
            e = tf_expd(e, ax)
          return e

        mean_x = tf.math.reduce_mean(_x, self.axis)
        std_x = tf.math.reduce_std(_x, self.axis)

        #self.moving_avg(
        #  self.moving_avg_step, self.x_m[i],
        #  mean_x, self.beta_m[i])
        x_m = self.x_m[i] - self.mv_delta(self.moving_avg_step,
          self.x_m[i], mean_x, self.beta_m[i])

        mean_ref = tf.math.reduce_mean(_encs[i], self.axis)
        std_ref = tf.math.reduce_std(_encs[i], self.axis)

        #self.moving_avg(
        #  self.moving_avg_step, self.ref_r[i],
        #  std_ref / (std_x + eps), self.beta_r[i])
        #self.moving_avg(
        #  self.moving_avg_step, self.ref_m[i],
        #  mean_ref, self.beta_m[i])
        ref_r = self.ref_r[i] - self.mv_delta(self.moving_avg_step, 
          self.ref_r[i], std_ref / (std_x + eps), self.beta_r[i])
        ref_m = self.ref_m[i] - self.mv_delta(self.moving_avg_step, 
          self.ref_m[i], mean_ref, self.beta_m[i])

        norm_x = reshape(ref_r) * x + reshape(ref_m - ref_r * x_m)
        delta = tf.stop_gradient(norm_x) - x
      
        '''
        mean_ns = tf.math.reduce_mean(_x, self.axis)
        std_ns = tf.math.reduce_std(_x, self.axis)

        self.moving_avg(
          self.moving_avg_step, self.ns_r[i],
          std_ns / (std_x + eps), self.beta_r)
        self.moving_avg(
          self.moving_avg_step, self.ns_m[i],
          mean_ns, self.beta_m)
          
        ns_r = reshape(self.ns_r[i])
        ns_d = reshape(self.ns_m[i] - self.ns_r[i] * self.x_m[i])

        norm_x = ns_r * x + ns_d
        delta_ns = norm_x - x
        '''

        x += self.factors[i] * delta

        self.x_m[i].assign(x_m, use_locking=True)
        
        self.ref_r[i].assign(ref_r, use_locking=True)
        self.ref_m[i].assign(ref_m, use_locking=True)
        
        '''
        self.ns_r[i].assign(ns_r, use_locking=True)
        self.ns_m[i].assign(ns_m, use_locking=True)
        '''
      
      encs.append(x)

      x = layer(x)
      _x = layer(_x)

    encs.append(x)
    return encs

class wav2vec2(tf.keras.layers.Layer):
  def __init__(self,
               num_enc_layer=3,
               beta_r=[0.999, 0.999, 0.999],
               beta_m=[0.9, 0.9, 0.9],
               *args, **kwargs):
    super(wav2vec2, self).__init__(*args, **kwargs)
   
    self.num_enc_layer = num_enc_layer
    self.beta_r = beta_r
    self.beta_m = beta_m

  def build(self, input_shape):
    self.fe = featencoder()
    self.fp = featproj()
    self.enc = encoder(self.num_enc_layer, self.beta_r, self.beta_m)
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, _encs = inputs

    else:
      x = inputs
      _encs = None

    x, fes = self.fe(tf_expd(x, -1))
    x = self.fp(x)
    encs = self.enc((x, _encs))
    return fes, encs

class wav2vec2_seq(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(wav2vec2_seq, self).__init__(*args, **kwargs)

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
  def __init__(self, 
               beta_r=[0.999, 0.999, 0.999], 
               beta_m=[0.9, 0.9, 0.9],
               *args, **kwargs):
    super(wav2vec2_unet, self).__init__(*args, **kwargs)
    
    self.beta_r = beta_r
    self.beta_m = beta_m

    self.layer = 7
    self.dims = [64 for _ in range(self.layer)]
    self.strides = [5, 2, 2, 2, 2, 2, 2]
    self.ksize = 16
    self.sublayer = 4

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)
    self.ref_len = input_shape[0][1]

    self.wav2vec2_frozen = wav2vec2(name='wav2vec2_frozen')
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

    self.conv_post = conv1d(1, self.ksize, **conv_opt)
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, ref = inputs

    else:
      x = inputs
      ref = None

    fs = None
    if ref is not None:
      _, fs = self.wav2vec2_frozen(ref, training=False)
      fs = [tf.stop_gradient(f) for f in fs]

    fes, encs = self.wav2vec2((x, fs))
    x = encs[NUM_ENC_LAYER]

    x = tf.keras.activations.gelu(x)
    #x = tf.stop_gradient(x)

    x = self.conv_mid(x)
   
    idx = 0; fes = fes[::-1]
    for _enc, (up_conv, convs) in zip(fes, self.up_convs):
      x = tf.keras.activations.gelu(up_conv(x))
      
      enc = self.enc_convs[idx](_enc)

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
    x = tf.squeeze(x, -1)

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
