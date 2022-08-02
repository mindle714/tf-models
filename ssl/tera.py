import tensorflow as tf
from util import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

def _normalize_wav_decibel(wav, target_level=-25):
  rms = (tf.math.reduce_mean(wav**2))**0.5
  scalar = (10 ** (target_level / 20)) / (rms + 1e-10)
  wav = wav * scalar
  return wav

def get_sinusoid_table(hidden_size):
  def _cal_angle(position, hid_idx):
    return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)
  def _get_posi_angle_vec(position):
    return [_cal_angle(position, hid_j) for hid_j in range(hidden_size)]
  sinusoid_table = np.array([_get_posi_angle_vec(pos_i) for pos_i in range(32000)])
  sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
  sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
  return sinusoid_table

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

class inputrep(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.hidden_size = 768
    super(inputrep, self).__init__()

  def build(self, input_shape):
    self.seq_len = input_shape[1]
    self.pos_enc = get_sinusoid_table(self.hidden_size)[:self.seq_len]
    self.pos_enc = tf_expd(tf.cast(self.pos_enc, tf.float32), 0)

    self.spec_transform = tf.keras.layers.Dense(768, use_bias=True)
    self.lnorm = lnorm(affine=True, eps=1e-12)
  
  def call(self, inputs, training=None):
    x = inputs
    #return self.lnorm(self.spec_transform(x))
    x = self.spec_transform(x) + self.pos_enc
    x = self.lnorm(x)
    return x

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
    '''
    dim = input_shape[-1]
    self.head_dim = dim // self.num_heads
    self.scaling = self.head_dim ** -0.5
    self.k_proj = tf.keras.layers.Dense(768, use_bias=True)
    self.v_proj = tf.keras.layers.Dense(768, use_bias=True)
    self.q_proj = tf.keras.layers.Dense(768, use_bias=True)
    self.out_proj = tf.keras.layers.Dense(768, use_bias=True)
    #self.dropout = tf.keras.layers.Dropout(0)
    self.dropout = tf.identity
    '''
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
    '''
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
    '''

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
    '''
    x_attn = self.atten(x)
    x_attn = self.dropout(x_attn)
    x = x + x_attn
    x = self.norm(x)
    x = x + self.feed(x)
    x = self.out_norm(x)
    return x
    '''

class encoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(encoder, self).__init__()

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
      
    '''
    x = x + self.emb(x)
    x = self.norm(x)
    x = self.dropout(x)

    encs = []
    for layer in self.layers:
      encs.append(x)
      x = layer(x)
    return x, encs
    '''

class tera(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(tera, self).__init__()

  def build(self, input_shape):
    self.fe = inputrep()
    self.enc = encoder()
    #self.fp = featproj()
  
  def call(self, inputs, training=None):
    x = inputs

    x = _normalize_wav_decibel(x)
    x = melspec(x, num_mel_bins=80,
      lower_edge_hertz=0., upper_edge_hertz=8000.)
    x = self.fe(x)

    attn_mask = tf.zeros([1, 1, 1, tf.shape(x)[1]])
    x = self.enc((x, attn_mask))

    return x

class tera_seq(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(tera_seq, self).__init__()

  def build(self, input_shape):
    self.tera = tera()
    self.projector = tf.keras.layers.Dense(256, use_bias=True)
    self.classifier = tf.keras.layers.Dense(12, use_bias=True)
  
  def call(self, inputs, training=None):
    x = inputs
    '''
    x, fes = self.tera(x)
    x = self.projector(x)
    x = tf.math.reduce_mean(x, 1)
    x = self.classifier(x)
    return x
    '''
    return self.tera(x)

class tera_unet(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(tera_unet, self).__init__()
    self.layer = 7
    self.dims = [64 for _ in range(self.layer)]
    self.strides = [5, 2, 2, 2, 2, 2, 2]
    self.ksize = 16
    self.sublayer = 4

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)
    self.ref_len = input_shape[0][1]

    self.tera = tera()

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
    x, ref = inputs

    x = tf_expd(x, -1)
    x, fes = self.tera(x)
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
    x = tf.squeeze(x, -1)

    #pad = tf.shape(x)[1] - self.ref_len
    #lpad = pad // 2
    #rpad = pad - lpad
    #x = x[:, lpad:-rpad]

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
