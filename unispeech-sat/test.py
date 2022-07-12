import soundfile
pcm, _ = soundfile.read("/home/hejung/speech-commands/TEST_SET/no/97f4c236_nohash_0.wav")

import util
import tensorflow as tf

import torch
m = torch.load("/home/hejung/transformers/examples/pytorch/audio-classification/unispeech-sat-base-ft-keyword-spotting/checkpoint-31930/pytorch_model.bin")

class gnormconv1d(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(gnormconv1d, self).__init__()

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv1D(512, kernel_size=10, strides=5, use_bias=False)
    self.norm = util.gnorm(512)
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
    for conv in self.conv_layers:
      x = conv(x)
    return x

class featproj(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(featproj, self).__init__()

  def build(self, input_shape):
    self.norm = util.lnorm()
    self.proj = tf.keras.layers.Dense(768, use_bias=True)
    self.dropout = tf.keras.layers.Dropout(0)
  
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
  
  def call(self, inputs, training=None):
    x = inputs
    q = self.q_proj(x) * self.scaling
    k = tf.reshape(self.k_proj(x),
      tf.concat([tf.shape(x)[:2], [self.num_heads, self.head_dim]], 0))
    k = tf.transpose(k, [0, 2, 1, 3])
    v = tf.reshape(self.v_proj(x),
      tf.concat([tf.shape(x)[:2], [self.num_heads, self.head_dim]], 0))
    v = tf.transpose(v, [0, 2, 1, 3])
    return x

class enclayer(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(enclayer, self).__init__()

  def build(self, input_shape):
    self.atten = attention()
  
  def call(self, inputs, training=None):
    x = inputs
    return self.atten(x)

class encoder(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(encoder, self).__init__()

  def build(self, input_shape):
    self.emb = posconvemb()
    self.norm = util.lnorm()
    self.dropout = tf.keras.layers.Dropout(0)
    self.layers = [enclayer() for _ in range(12)]
  
  def call(self, inputs, training=None):
    x = inputs
    x = x + self.emb(x)
    x = self.norm(x)
    x = self.dropout(x)
    for layer in self.layers:
      x = layer(x)
    return x

fe = featencoder()
fp = featproj()
enc = encoder()

import numpy as np
_in = np.reshape(pcm, [1, -1, 1])
_tmp = enc(fp(fe(_in)))

for i, conv in enumerate(fe.conv_layers):
  prefix = 'unispeech_sat.feature_extractor.conv_layers'
  w = m['{}.{}.conv.weight'.format(prefix, i)].transpose(2,0).cpu().numpy()
  conv.conv.set_weights([w])
  if i == 0:
    w = m['{}.{}.layer_norm.weight'.format(prefix, i)].cpu().numpy()
    b = m['{}.{}.layer_norm.bias'.format(prefix, i)].cpu().numpy()
    conv.norm.gamma.assign(w)
    conv.norm.beta.assign(b)

prefix = 'unispeech_sat.feature_projection'
w = m['{}.layer_norm.weight'.format(prefix)]
b = m['{}.layer_norm.bias'.format(prefix)]
fp.norm.gamma.assign(w.cpu().numpy())
fp.norm.beta.assign(b.cpu().numpy())
w = m['{}.projection.weight'.format(prefix)]
b = m['{}.projection.bias'.format(prefix)]
fp.proj.set_weights([w.transpose(1,0).cpu().numpy(), b.cpu().numpy()])

prefix = 'unispeech_sat.encoder'
w_g = m['{}.pos_conv_embed.conv.weight_g'.format(prefix)].cpu().numpy()
w_g = np.reshape(w_g, [-1, 1, 1])
w_v = m['{}.pos_conv_embed.conv.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
b = m['{}.pos_conv_embed.conv.bias'.format(prefix)].cpu().numpy()
enc.emb.conv.set_weights([w, b])
w = m['{}.layer_norm.weight'.format(prefix)]
b = m['{}.layer_norm.bias'.format(prefix)]
enc.norm.gamma.assign(w.cpu().numpy())
enc.norm.beta.assign(b.cpu().numpy())

print(enc(fp(fe(_in))))
