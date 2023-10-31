import tensorflow as tf

import torch
m = torch.load("/home/hejung/wav2vec2-base/pytorch_model.bin")

from wav2vec2 import *

model = wav2vec2_phone(num_class=50)

import numpy as np
pcm = np.zeros(128000)
_in = np.reshape(pcm, [1, -1])
_ref = np.zeros([1, 399])
_tmp = model((_in, _ref, np.ones([1, 1])*128000, np.ones([1, 1])*399), ssl_loss=True)

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

for i, conv in enumerate(model.wav2vec2.wav2vec2.fe.conv_layers):
  prefix = 'wav2vec2.feature_extractor.conv_layers'
  load_conv('{}.{}.conv'.format(prefix, i), conv.conv)
  if i == 0:
    load_norm('{}.{}.layer_norm'.format(prefix, i), conv.norm)

prefix = 'wav2vec2.feature_projection'
load_norm('{}.layer_norm'.format(prefix), model.wav2vec2.wav2vec2.fp.norm)
load_affine('{}.projection'.format(prefix), model.wav2vec2.wav2vec2.fp.proj)

prefix = 'wav2vec2.encoder'
w_g = m['{}.pos_conv_embed.conv.weight_g'.format(prefix)].cpu().numpy()
w_g = np.reshape(w_g, [-1, 1, 1])
w_v = m['{}.pos_conv_embed.conv.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
b = m['{}.pos_conv_embed.conv.bias'.format(prefix)].cpu().numpy()
model.wav2vec2.wav2vec2.enc.emb.conv.set_weights([w, b])
load_norm('{}.layer_norm'.format(prefix), model.wav2vec2.wav2vec2.enc.norm)

for i, layer in enumerate(model.wav2vec2.wav2vec2.enc.layers):
  prefix = 'wav2vec2.encoder.layers.{}'.format(i)
  load_affine('{}.attention.q_proj'.format(prefix), layer.atten.q_proj)
  load_affine('{}.attention.k_proj'.format(prefix), layer.atten.k_proj)
  load_affine('{}.attention.v_proj'.format(prefix), layer.atten.v_proj)
  load_affine('{}.attention.out_proj'.format(prefix), layer.atten.out_proj)
 
  load_affine('{}.feed_forward.intermediate_dense'.format(prefix), layer.feed.in_dense)
  load_affine('{}.feed_forward.output_dense'.format(prefix), layer.feed.out_dense)
  
  load_norm('{}.layer_norm'.format(prefix), layer.norm)
  load_norm('{}.final_layer_norm'.format(prefix), layer.out_norm)

load_affine('project_hid', model.wav2vec2.project_hid)
load_affine('project_q', model.wav2vec2.project_q)

load_affine('quantizer.weight_proj', model.wav2vec2.quantizer.weight_proj)
w = m['quantizer.codevectors'].cpu().numpy()
model.wav2vec2.quantizer.codevectors.assign(w)
w = m['wav2vec2.masked_spec_embed'].cpu().numpy()
model.wav2vec2.wav2vec2.masked_spec_embed.assign(w)

ckpt = tf.train.Checkpoint(model)
ckpt.write("wav2vec2_timit_v2.ckpt")
