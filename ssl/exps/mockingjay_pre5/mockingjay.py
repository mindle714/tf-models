import tensorflow as tf
from util import *
from spec_ops import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

def _normalize_wav_decibel(wav, target_level=-25):
  rms = (tf.math.reduce_mean(wav**2))**0.5
  scalar = (10 ** (target_level / 20)) / (rms + 1e-10)
  wav = wav * scalar
  return wav

class inputrep(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.hidden_size = 768
    super(inputrep, self).__init__()

  def build(self, input_shape):
    #self.seq_len = input_shape[1]
    #self.pos_enc = get_sinusoid_table(self.hidden_size)[:self.seq_len]
    #self.pos_enc = tf_expd(tf.cast(self.pos_enc, tf.float32), 0)

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

class enclayer(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(enclayer, self).__init__()

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
    super(encoder, self).__init__()

  def build(self, input_shape):
    self.layers = [enclayer() for _ in range(5)]
  
  def call(self, inputs, training=None):
    x, attn_mask = inputs

    encs = []
    for i, layer in enumerate(self.layers):
      encs.append(x)
      x = layer((x, attn_mask))
    encs.append(x)

    return encs

class mockingjay(tf.keras.layers.Layer):
  def __init__(self, n_fft, hop_len, *args, **kwargs):
    super(mockingjay, self).__init__()
    self.n_fft = n_fft
    self.hop_len = hop_len

  def build(self, input_shape):
    self.fe = inputrep()
    self.enc = encoder()
  
  def call(self, inputs, training=None):
    x = inputs

    x = _normalize_wav_decibel(x)
    x = melspec(x, num_mel_bins=80,
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft,
      lower_edge_hertz=0., upper_edge_hertz=8000.)
    x = self.fe(x)

    attn_mask = tf.zeros([1, 1, 1, tf.shape(x)[1]])
    x = self.enc((x, attn_mask))

    return x

class mockingjay_seq(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(mockingjay_seq, self).__init__()

  def build(self, input_shape):
    self.mockingjay = mockingjay(400, 160)
  
  def call(self, inputs, training=None):
    x = inputs
    return self.mockingjay(x)

class mockingjay_unet(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(mockingjay_unet, self).__init__()

    self.layer = 4
    self.dims = [32, 32, 32, 32]
    self.ksize = 16
    self.sublayer = 4

    self.n_fft = 400
    self.hop_len = 160

    self.k = 10
    self.c = 0.1

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.mockingjay = mockingjay(self.n_fft, self.hop_len)

    self.conv_r = conv1d(self.n_fft//2+1, 3, **conv_opt)
    self.conv_i = conv1d(self.n_fft//2+1, 3, **conv_opt)
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, ref = inputs

    else:
      x = inputs
      ref = None

    _in = x

    xs = self.mockingjay(x)
    x = xs[-1]

    x = tf.keras.activations.gelu(x)
    #x = tf.stop_gradient(x)

    x_r = tf.clip_by_value(self.conv_r(x), -self.k, self.k)
    x_i = tf.clip_by_value(self.conv_i(x), -self.k, self.k)
      
    in_pad = tf.pad(_in, tf.constant(
      [[0, 0], [self.n_fft//2, self.n_fft//2]]), mode='reflect')

    Y = tf.signal.stft(in_pad, 
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
    Yr = tf.math.real(Y); Yi = tf.math.imag(Y)

    Mr = -1. / self.c * tf.math.log(tf.nn.relu((self.k - x_r) / (self.k + x_r)) + 1e-10)
    Mi = -1. / self.c * tf.math.log(tf.nn.relu((self.k - x_i) / (self.k + x_i)) + 1e-10)

    Sr = (Mr * Yr) - (Mi * Yi)
    Si = (Mr * Yi) + (Mi * Yr)
    x = tf.signal.inverse_stft(tf.complex(Sr, Si),
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
    x = x[..., self.n_fft//2:self.n_fft//2+tf.shape(_in)[1]]

    def get_cirm(Yr, Yi, ref):
      ref_pad = tf.pad(ref, tf.constant(
        [[0, 0], [self.n_fft//2, self.n_fft//2]]), mode='reflect')

      S = tf.signal.stft(ref_pad,
        frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
      Sr = tf.math.real(S); Si = tf.math.imag(S)

      M_denom = (Yr * Yr) + (Yi * Yi)
      Mr_num = (Yr * Sr) + (Yi * Si)
      Mr = Mr_num / (M_denom + 1e-10)

      Mi_num = (Yr * Si) - (Yi * Sr)
      Mi = Mi_num / (M_denom + 1e-10)

      Cr_exp = tf.math.exp(-self.c * Mr)
      #Cr = self.k * ((1 - Cr_exp) / (1 + Cr_exp))
      Cr = self.k * (-1. + 2. * tf.math.reciprocal_no_nan(1 + Cr_exp))
        
      Ci_exp = tf.math.exp(-self.c * Mi)
      #Ci = self.k * ((1 - Ci_exp) / (1 + Ci_exp))
      Ci = self.k * (-1. + 2. * tf.math.reciprocal_no_nan(1 + Ci_exp))

      return Cr, Ci

    if ref is not None:
      x = x[..., :tf.shape(ref)[-1]]
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

      Cr, Ci = get_cirm(Yr, Yi, ref)
      cirm_loss = tf.math.reduce_mean(tf.math.abs(Cr - x_r))
      cirm_loss += tf.math.reduce_mean(tf.math.abs(Ci - x_i))

      return samp_loss + spec_loss, cirm_loss

    return x
