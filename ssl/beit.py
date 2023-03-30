import tensorflow as tf
from util import *
from spec_ops import *

import librosa
import gammatone
import cqt

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

def pow_to_db(e, eps=1e-10):
  return tf.math.log(e + 1e-10)

def minmax_norm(e):
  e_trans = tf.transpose(e, [3, 0, 1, 2])
  e_trans = tf.reshape(e_trans, [tf.shape(e_trans)[0], -1])
  e_max = tf.math.reduce_max(e_trans, -1)
  e_min = tf.math.reduce_min(e_trans, -1)

  e_norm = e - (e_max + e_min) / 2
  e_norm /= (e_max - e_min)
  e_norm += 0.5
  return e_norm

class patchemb(tf.keras.layers.Layer):
  def __init__(self, hdim = 768, *args, **kwargs):
    self.hdim = hdim
    super(patchemb, self).__init__()

  def build(self, input_shape):
    #self.proj = conv2d(self.hdim, (16, 16), strides=(16, 16))
    self.cls_token = self.add_weight(shape=[1, 1, self.hdim], trainable=True, name='cls_token')

  def call(self, inputs, training=None):
    x = inputs

    #x = self.proj(x)
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])
    x = tf.transpose(x, [0, 2, 1])

    cls_token = tf.tile(self.cls_token, [tf.shape(x)[0], 1, 1])
    x = tf.concat([cls_token, x], axis=1)

    return x

class self_attn(tf.keras.layers.Layer):
  def __init__(self, wsize, *args, **kwargs):
    self.num_heads = 12
    self.head_dim = 64
    self.wsize = wsize
    super(self_attn, self).__init__()

  def build(self, input_shape):
    self.all_head_size = self.num_heads * self.head_dim
    self.query = tf.keras.layers.Dense(self.all_head_size, use_bias=True)
    self.key = tf.keras.layers.Dense(self.all_head_size, use_bias=False)
    self.value = tf.keras.layers.Dense(self.all_head_size, use_bias=True)

    num_dist = (2 * self.wsize[0] - 1) * (2 * self.wsize[1] - 1) + 3
    self.rel_bias_tbl = self.add_weight(shape=(num_dist, self.num_heads), trainable=True, name='rel_bias_tbl')

    coords_h = np.arange(self.wsize[0])
    coords_w = np.arange(self.wsize[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
    coords = coords.reshape((coords.shape[0], -1))

    rel_coords = coords[:, :, None] - coords[:, None, :]
    rel_coords = rel_coords.transpose(1, 2, 0)
    rel_coords[:, :, 0] += self.wsize[0] - 1
    rel_coords[:, :, 1] += self.wsize[1] - 1
    rel_coords[:, :, 0] *= 2 * self.wsize[1] - 1

    rel_idx = np.zeros((self.wsize[0] * self.wsize[1] + 1,) * 2, dtype=np.int)
    rel_idx[1:, 1:] = rel_coords.sum(-1)
    rel_idx[0, 0:] = num_dist - 3 
    rel_idx[0:, 0] = num_dist - 2
    rel_idx[0, 0] = num_dist - 1
    self.rel_idx = rel_idx.flatten()
  
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
    if attn_mask is not None:
      attn_score += attn_mask

    rel_bias = tf.reshape(tf.gather(self.rel_bias_tbl, self.rel_idx),
        [self.wsize[0] * self.wsize[1] + 1,] * 2 + [-1])
    attn_score += tf_expd(tf.transpose(rel_bias, (2, 0, 1)), 0)

    attn_probs = tf.nn.softmax(attn_score, -1)
    ctx = tf.linalg.matmul(attn_probs, v)
    ctx = tf.transpose(ctx, [0, 2, 1, 3])
    ctx = tf.reshape(ctx,
      tf.concat([tf.shape(ctx)[:-2], [-1]], 0))
 
    return ctx

class attention(tf.keras.layers.Layer):
  def __init__(self, wsize, *args, **kwargs):
    self.wsize = wsize
    super(attention, self).__init__()

  def build(self, input_shape):
    dim = input_shape[0][-1]

    self.self_attn = self_attn(self.wsize)
    self.out = tf.keras.layers.Dense(dim, use_bias=True)
  
  def call(self, inputs, training=None):
    _x, attn_mask = inputs

    x = self.self_attn((_x, attn_mask))
    x = self.out(x)
    return x

class enclayer(tf.keras.layers.Layer):
  def __init__(self, wsize, *args, **kwargs):
    self.init_scale = 0.1
    self.wsize = wsize
    super(enclayer, self).__init__()

  def build(self, input_shape):
    self.lnorm = lnorm(affine=True, eps=1e-12)
    self.atten = attention(self.wsize)

    dim = input_shape[-1]
    self.lambda_1 = self.add_weight(shape=[dim], 
        initializer=tf.constant_initializer(self.init_scale), name='lambda_1')

    self.lnorm2 = lnorm(affine=True, eps=1e-12)
    self.inter = tf.keras.layers.Dense(3072, use_bias=True)
    self.out = tf.keras.layers.Dense(dim, use_bias=True)

    self.lambda_2 = self.add_weight(shape=[dim],
        initializer=tf.constant_initializer(self.init_scale), name='lambda_2')

  def call(self, inputs, training=None):
    x = inputs

    _x = x
    x = self.lnorm(x)
    x = self.atten((x, None))
    x *= self.lambda_1
    x = x + _x

    _x = x
    x = self.lnorm2(x)
    x = self.inter(x)
    x = tf.keras.activations.gelu(x)
    x = self.out(x)
    x *= self.lambda_2
    x = x + _x

    return x

class encoder(tf.keras.layers.Layer):
  def __init__(self, wsize, *args, **kwargs):
    self.wsize = wsize
    super(encoder, self).__init__()

  def build(self, input_shape):
    self.layers = [enclayer(self.wsize) for _ in range(3)]
  
  def call(self, inputs, training=None):
    x = inputs

    encs = []
    for i, layer in enumerate(self.layers):
      encs.append(x)
      x = layer(x)
    encs.append(x)

    return encs

class beit(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(beit, self).__init__()

  def build(self, input_shape):
    self.pemb = patchemb()
#    self.enc = encoder([input_shape[1]//16, input_shape[2]//16])
    self.enc = encoder([input_shape[1], input_shape[2]])
  
  def call(self, inputs, training=None):
    x = inputs

    x = self.pemb(x)
    x = self.enc(x)
    return x

class beit_seq(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(beit_seq, self).__init__()

  def build(self, input_shape):
    self.beit = beit()
  
  def call(self, inputs, training=None):
    x = inputs
    return self.beit(x)

class beit_unet(tf.keras.layers.Layer):
  def __init__(self, in_types=[1,2,3], *args, **kwargs):
    super(beit_unet, self).__init__()

    self.in_types = in_types
    self.sr = 16000
    self.n_fft = 400
    self.hop_len = 80

    self.k = 10
    self.c = 0.1

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.mel_fb = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=128)
    self.gm_fb = gammatone.gammatone(sr=self.sr, n_fft=self.n_fft, n_bins=128)

    self.beit = beit(self.n_fft, self.hop_len)
    in_dim = len(self.in_types)

    self.down_convs = list(zip(
      [conv2d(dim, (2, 2), strides=(2, 2)) for dim in [in_dim, 16, 64, 256]],
      [[
        conv2d(None, 4, strides=1, padding='same') for _ in range(2)
      ] for dim in [in_dim, 16, 64, 256]]
    ))

    self.mid_conv = tf.keras.layers.Dense(768)
    self.mid_norm = lnorm()

    self.up_convs = list(zip(
      [conv2dtrans(dim, (2, 2), strides=(2, 2)) for dim in [256, 64, 16, in_dim]],
      [[
        conv2d(None, 4, strides=1, padding='same') for _ in range(2)
      ] for dim in [256, 64, 16, in_dim]]
    ))

    self.conv_r = conv1d(1, 3, **conv_opt, name='prj_r')
    self.conv_i = conv1d(1, 3, **conv_opt, name='prj_i')
  
  def call(self, inputs, training=None):
    if isinstance(inputs, tuple):
      x, ref = inputs

    else:
      x = inputs
      ref = None

    _in = x
    
    pcqt_pow = tf_expd(cqt.pcqt(x, sr=self.sr, 
      hop_length=self.hop_len, n_bins=128, power=2), -1)
    
    x = tf.signal.stft(x,
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)

    x_pow = tf.abs(x) ** 2

    mel_pow = tf_expd(tf.einsum("...tf,mf->...tm", x_pow, self.mel_fb), -1)
    gm_pow = tf_expd(tf.einsum("...tf,mf->...tm", x_pow, self.gm_fb), -1)

    pcqt_pow = tf.image.resize(pcqt_pow, tf.shape(x_pow)[1:3])
    mel_pow = tf.image.resize(mel_pow, tf.shape(x_pow)[1:3])
    gm_pow = tf.image.resize(gm_pow, tf.shape(x_pow)[1:3])

    x_pow = tf_expd(x_pow, -1)

    zero_db = tf.zeros_like(x_pow)
    x_db = pow_to_db(x_pow)
    mel_db = pow_to_db(mel_pow)
    gm_db = pow_to_db(gm_pow)
    pcqt_db = pow_to_db(pcqt_pow)

    dbs = [zero_db, x_db, mel_db, gm_db, pcqt_db]
    ins = [dbs[e] for e in self.in_types]
    x = tf.concat(ins, -1)

    fdim = self.n_fft//2 + 1
    x_len = tf.shape(x)[1]
    pad_len = fdim - (x_len % fdim)
    pad = tf.zeros([tf.shape(x)[0], pad_len, fdim, tf.shape(x)[-1]])
    x_pad = tf.concat([x, pad], 1)

    batch_size = tf.shape(x_pad)[0]
    num_patches = tf.shape(x_pad)[1] // fdim

    x_pad = tf.transpose(x_pad, [2, 3, 0, 1])
    x_pad = tf.reshape(x_pad,
      tf.concat([tf.shape(x_pad)[:3], [fdim, num_patches]], 0))
    x_pad = tf.transpose(x_pad, [0, 1, 3, 2, 4])
    x_pad = tf.reshape(x_pad,
      tf.concat([tf.shape(x_pad)[:3], [batch_size * num_patches]], 0))
    x_pad = tf.transpose(x_pad, [3, 2, 0, 1])

    x = tf.image.resize(x_pad, [224, 224])
    x = minmax_norm(x)

    encs = []
    for (down_conv, convs) in self.down_convs:
      for conv in convs:
        x = tf.keras.activations.gelu(conv(x)) + x
      encs.append(x)
      x = tf.keras.activations.gelu(down_conv(x))

    x = self.mid_norm(self.mid_conv(x))

    xs = self.beit(x)
    x = xs[-1]

    x = tf.keras.activations.gelu(x)
    #x = tf.stop_gradient(x)

    x = tf.transpose(x, [0, 2, 1])
    x = x[..., 1:]
    x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], 14, 14])
    x = tf.transpose(x, [0, 2, 3, 1])
    # x[batch_size, width, height, dim]

    encs = encs[::-1]
    for enc, (up_conv, convs) in zip(encs, self.up_convs):
      x = tf.keras.activations.gelu(up_conv(x))
      x = tf.concat([x, enc], -1)
      for conv in convs:
        x = tf.keras.activations.gelu(conv(x)) + x
    
    x = tf.image.resize(x, [fdim, fdim])

    x = tf.transpose(x, [2, 3, 1, 0])
    x = tf.reshape(x,
      tf.concat([tf.shape(x)[:3], [batch_size, num_patches]], 0))
    x = tf.transpose(x, [0, 1, 3, 2, 4])
    x = tf.reshape(x,
      tf.concat([tf.shape(x)[:3], [fdim * num_patches]], 0))
    x = x[..., :x_len]
    x = tf.transpose(x, [2, 3, 0, 1])

    x_r = tf.squeeze(self.conv_r(x), -1)
    x_i = tf.squeeze(self.conv_i(x), -1)

    x_r = tf.clip_by_value(x_r, -self.k, self.k)
    x_i = tf.clip_by_value(x_i, -self.k, self.k)
      
    #in_pad = tf.pad(_in, tf.constant(
    #  [[0, 0], [self.n_fft//2, self.n_fft//2]]), mode='reflect')

    Y = tf.signal.stft(_in,#in_pad, 
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
    Yr = tf.math.real(Y); Yi = tf.math.imag(Y)

    Mr = -1. / self.c * tf.math.log(tf.nn.relu((self.k - x_r) / (self.k + x_r)) + 1e-10)
    Mi = -1. / self.c * tf.math.log(tf.nn.relu((self.k - x_i) / (self.k + x_i)) + 1e-10)

    Sr = (Mr * Yr) - (Mi * Yi)
    Si = (Mr * Yi) + (Mi * Yr)
    x = tf.signal.inverse_stft(tf.complex(Sr, Si),
      frame_length=self.n_fft, frame_step=self.hop_len, fft_length=self.n_fft)
    #x = x[..., self.n_fft//2:self.n_fft//2+tf.shape(_in)[1]]
    x = tf.pad(x, [[0, 0], [0, tf.shape(_in)[1] - tf.shape(x)[1]]])

    def get_cirm(Yr, Yi, ref):
      #ref_pad = tf.pad(ref, tf.constant(
      #  [[0, 0], [self.n_fft//2, self.n_fft//2]]), mode='reflect')

      S = tf.signal.stft(ref,#ref_pad,
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
