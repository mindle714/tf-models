import tensorflow as tf

import torch
m = torch.load("/home/hejung/data2vec-audio-base-960h/pytorch_model.bin")

from data2vec import *

model = data2vec_phone(num_class=50)

import numpy as np
pcm = np.zeros(128000)
_in = np.reshape(pcm, [1, -1])
_ref = np.zeros([1, 399])
_tmp = model((_in, _ref, np.ones([1, 1])*128000, np.ones([1, 1])*399))

def load_norm(prefix, e):
  w = m['{}.weight'.format(prefix, i)].cpu().numpy()
  b = m['{}.bias'.format(prefix, i)].cpu().numpy()
  e.gamma.assign(w)
  e.beta.assign(b)

def load_affine(prefix, e):
  w = m['{}.weight'.format(prefix)]
  bname = '{}.bias'.format(prefix)
  if bname in m:
    b = m[bname]
    e.set_weights([w.transpose(1,0).cpu().numpy(), b.cpu().numpy()])
  else:
    e.set_weights([w.transpose(1,0).cpu().numpy()])

def load_conv(prefix, e):
  w = m['{}.weight'.format(prefix)]
  bname = '{}.bias'.format(prefix)
  if bname in m:
    b = m[bname]
    e.set_weights([w.transpose(2,0).cpu().numpy(), b.cpu().numpy()])
  else:
    e.set_weights([w.transpose(2,0).cpu().numpy()])

for i, layer in enumerate(model.data2vec.fe.conv_layers):
  prefix = 'data2vec_audio.feature_extractor.conv_layers'
  load_conv('{}.{}.conv'.format(prefix, i), layer.conv)
  load_norm('{}.{}.layer_norm'.format(prefix, i), layer.norm)

prefix = 'data2vec_audio.feature_projection'
load_norm('{}.layer_norm'.format(prefix), model.data2vec.fp.norm)
load_affine('{}.projection'.format(prefix), model.data2vec.fp.proj)

prefix = 'data2vec_audio.encoder'
for i, conv in enumerate(model.data2vec.enc.emb.convs):
  load_conv('{}.pos_conv_embed.layers.{}.conv'.format(prefix, i), conv)
load_norm('{}.layer_norm'.format(prefix), model.data2vec.enc.norm)

for i, layer in enumerate(model.data2vec.enc.layers):
  prefix = 'data2vec_audio.encoder.layers.{}'.format(i)
  load_affine('{}.attention.q_proj'.format(prefix), layer.atten.q_proj)
  load_affine('{}.attention.k_proj'.format(prefix), layer.atten.k_proj)
  load_affine('{}.attention.v_proj'.format(prefix), layer.atten.v_proj)
  load_affine('{}.attention.out_proj'.format(prefix), layer.atten.out_proj)
 
  load_affine('{}.feed_forward.intermediate_dense'.format(prefix), layer.feed.in_dense)
  load_affine('{}.feed_forward.output_dense'.format(prefix), layer.feed.out_dense)
  
  load_norm('{}.layer_norm'.format(prefix), layer.norm)
  load_norm('{}.final_layer_norm'.format(prefix), layer.out_norm)

ckpt = tf.train.Checkpoint(model)
ckpt.write("data2vec_timit.ckpt")
