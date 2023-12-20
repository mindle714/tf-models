import tensorflow as tf
from util import *

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
  def __init__(self, beta_r, beta_m, *args, **kwargs):
    super(featencoder, self).__init__(*args, **kwargs)
    
    self.beta_r = beta_r
    self.beta_m = beta_m

  def build(self, input_shape):
    if isinstance(input_shape, tuple):
      x_shape = input_shape[0]
    
    else:
      x_shape = input_shape

    ksizes = [3, 3, 3, 3, 2, 2]
    self.conv_layers = [gnormconv1d()] + [nonormconv1d(ksizes[i]) for i in range(6)]
    
    self.factors = [self.add_weight(shape=(),
      initializer="zeros", name="factor_{}".format(i), trainable=False) \
      for i in range(6)]
    
    self.axis = [0, 1]
    #shape = [x_shape[e] for e in range(len(x_shape)) if e not in self.axis]
    shape=[512]
    
    self.ref_r = [self.add_weight(shape=shape,
      initializer="ones", name="ref_r_{}".format(i), trainable=False) \
      for i in range(6)]
    self.ns_r = [self.add_weight(shape=shape,
      initializer="ones", name="ns_r_{}".format(i), trainable=False) \
      for i in range(6)]

    self.x_m = [self.add_weight(shape=shape,
      initializer="zeros", name="x_m_{}".format(i), trainable=False) \
      for i in range(6)]
    self.ref_m = [self.add_weight(shape=shape,
      initializer="zeros", name="ref_m_{}".format(i), trainable=False) \
      for i in range(6)]
    self.ns_m = [self.add_weight(shape=shape,
      initializer="zeros", name="ns_m_{}".format(i), trainable=False) \
      for i in range(6)]
    
    self.moving_avg_step = self.add_weight(shape=(),
      initializer="zeros", name="step", trainable=False)
  
  def moving_avg(self, step, e, v, momentum, debias=False):
    delta = (e - v) * (1. - momentum) 
    e.assign_sub(delta, use_locking=True)

    if debias:
      denom = 1. - tf.math.pow(momentum, step)
      e.assign(e / denom, use_locking=True)

    return delta
  
  def call(self, inputs, training=None, eps=1e-10):
    _encs = None
    
    if isinstance(inputs, tuple):
      x, _encs = inputs
    
    else:
      x = inputs
    
    if training:
      self.moving_avg_step.assign_add(1, use_locking=True)

    fes = [x]
    x = self.conv_layers[0](x)

    _x = x

    for i, conv in enumerate(self.conv_layers[1:]):
      if training and (_encs is not None) and i < len(self.beta_m):
        def reshape(e):
          for ax in self.axis:
            e = tf_expd(e, ax)
          return e

        mean_x = tf.math.reduce_mean(_x, self.axis)
        std_x = tf.math.reduce_std(_x, self.axis)

        self.moving_avg(
          self.moving_avg_step, self.x_m[i],
          mean_x, self.beta_m[i])

        mean_ref = tf.math.reduce_mean(_encs[i], self.axis)
        std_ref = tf.math.reduce_std(_encs[i], self.axis)

        self.moving_avg(
          self.moving_avg_step, self.ref_r[i],
          std_ref / (std_x + eps), self.beta_r[i])
        self.moving_avg(
          self.moving_avg_step, self.ref_m[i],
          mean_ref, self.beta_m[i])

        ref_r = reshape(self.ref_r[i])
        ref_d = reshape(self.ref_m[i] - self.ref_r[i] * self.x_m[i])

        norm_x = ref_r * x + ref_d
        delta = norm_x - x
      
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

        #delta_comb = tf.stop_gradient((delta + delta_ns) / 2.)
        #delta_comb = tf.stop_gradient(
        #  self.factors[i] * delta + (1. - self.factors[i]) * delta_ns)
        #x += self.factors[i] * delta_comb
        x += self.factors[i] * tf.stop_gradient(delta)

      fes.append(x)

      x = conv(x)
      _x = conv(_x)

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
  
  def call(self, inputs, training=None):
    x = inputs
    #shape = [tf.shape(x)[0], 64, tf.shape(x)[-1]]
    #pad = tf.zeros(shape)
    #x_pad = tf.concat([pad, x, pad], 1)
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
      x_k, x_q = inputs

    else:
      x = inputs
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
    x = inputs
    x_attn = self.atten(x)
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
    x = inputs

    if len(self.layers) > 0:
      x = x + self.emb(x)
      x = self.norm(x)
      x = self.dropout(x)

    encs = []
    for i, layer in enumerate(self.layers):
      encs.append(x)
      x = layer(x)

    encs.append(x)
    return encs

class wavlm(tf.keras.layers.Layer):
  def __init__(self,
               num_enc_layer=1,
               beta_r=[0.999, 0.999, 0.999],
               beta_m=[0.9, 0.9, 0.9],
               *args, **kwargs):
    super(wavlm, self).__init__(*args, **kwargs)
   
    self.num_enc_layer = num_enc_layer
    self.beta_r = beta_r
    self.beta_m = beta_m

  def build(self, input_shape):
    self.fe = featencoder(self.beta_r, self.beta_m)
    self.fp = featproj()
    self.enc = encoder(self.num_enc_layer)
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, _fes = inputs

    else:
      x = inputs
      _fes = None

    x, fes = self.fe((tf_expd(x, -1), _fes))
    x = self.fp(x)
    encs = self.enc(x)
    return fes, encs

class wavlm_seq(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(wavlm_seq, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.wavlm = wavlm()
    self.projector = tf.keras.layers.Dense(256, use_bias=True)
    self.classifier = tf.keras.layers.Dense(12, use_bias=True)
  
  def call(self, inputs, training=None):
    x = inputs
    x, fes, encs = self.wavlm(x)
    x = self.projector(x)
    x = tf.math.reduce_mean(x, 1)
    x = self.classifier(x)
    return x

class wavlm_unet(tf.keras.layers.Layer):
  def __init__(self, 
               num_enc_layer=1,
               beta_r=[0.999, 0.999, 0.999], 
               beta_m=[0.9, 0.9, 0.9],
               *args, **kwargs):
    super(wavlm_unet, self).__init__(*args, **kwargs)
    
    self.num_enc_layer = num_enc_layer
    self.beta_r = beta_r
    self.beta_m = beta_m

    self.layer = 7
    self.dims = [64 for _ in range(self.layer)]
    self.strides = [5, 2, 2, 2, 2, 2, 2]
    self.ksizes = [10, 5, 5, 5, 5, 5, 5] #[10, 3, 3, 3, 3, 2, 2]
    self.sublayer = 4

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    if isinstance(input_shape, tuple):
      self.ref_len = input_shape[0][1]
    
    else:
      self.ref_len = input_shape[1]

    self.wavlm_frozen = wavlm(self.num_enc_layer, name='wavlm_frozen')
    self.wavlm = wavlm(self.num_enc_layer, self.beta_r, self.beta_m)
    
    self.up_convs = []
    for idx in range(self.layer):
      dim = self.dims[::-1][idx]
      ksize = self.ksizes[::-1][idx]
      stride = self.strides[::-1][idx]

      self.up_convs.append((
        conv1dtrans(dim, ksize, strides=stride, **conv_opt),
        conv1d(dim, 1, strides=1, use_bias=True),
        [
          conv1d(None, 16, strides=1, **conv_opt) \
            for _idx in range(self.sublayer)
        ]
      ))
    
    self.conv_post = conv1d(1, 16, **conv_opt)
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, ref = inputs

    else:
      x = inputs
      ref = None

    _fes = None
    if ref is not None:
      _fes, _ = self.wavlm_frozen(ref, training=False)
      _fes = [tf.stop_gradient(f) for f in _fes]

    fes, encs = self.wavlm((x, _fes))
    x = encs[-1]

    x = gelu(x)
    #x = tf.stop_gradient(x)
    
    def pad(e, ref):
      pad = tf.shape(ref)[1] - tf.shape(e)[1]
      lpad = pad // 2
      rpad = pad - lpad
      e = tf.pad(e, tf.concat([[[0, 0]], [[lpad, rpad]], [[0, 0]]], 0), "CONSTANT")
      return e

    idx = 0; fes = fes[::-1]
    for _enc, (up_conv, conv1x1, convs) in zip(fes, self.up_convs):
      x = gelu(up_conv(x))
      x = pad(x, _enc)
      
      _enc = gelu(conv1x1(_enc))
      x = tf.concat([_enc, x], -1)

      for conv in convs:
        x = gelu(conv(x)) + x

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
