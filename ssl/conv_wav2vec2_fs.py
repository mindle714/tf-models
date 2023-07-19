import tensorflow as tf

import torch
m = torch.load("/home/hejung/wav2vec2-base/pytorch_model.bin")

from wav2vec2 import *

model = wav2vec2_unet()

import numpy as np
pcm = np.zeros(16000)
_in = np.reshape(pcm, [1, -1])
_tmp = model((_in, _in))
_tmp = model.wav2vec2((_in, _in))
_tmp = model.wav2vec2_frozen((_in, _in))

def load_norm(prefix, e):
  w = m['{}.weight'.format(prefix)].cpu().numpy()
  b = m['{}.bias'.format(prefix)].cpu().numpy()
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

def load_model(mdl):
  for i, conv in enumerate(mdl.fe.conv_layers):
    prefix = 'wav2vec2.feature_extractor.conv_layers'
    load_conv('{}.{}.conv'.format(prefix, i), conv.conv)
    if i == 0:
      load_norm('{}.{}.layer_norm'.format(prefix, i), conv.norm)

  prefix = 'wav2vec2.feature_projection'
  load_norm('{}.layer_norm'.format(prefix), mdl.fp.norm)
  load_affine('{}.projection'.format(prefix), mdl.fp.proj)

  if len(mdl.enc.layers) > 0:
    prefix = 'wav2vec2.encoder'
    w_g = m['{}.pos_conv_embed.conv.weight_g'.format(prefix)].cpu().numpy()
    w_g = np.reshape(w_g, [-1, 1, 1])
    w_v = m['{}.pos_conv_embed.conv.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
    w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
    b = m['{}.pos_conv_embed.conv.bias'.format(prefix)].cpu().numpy()
    mdl.enc.emb.conv.set_weights([w, b])
    load_norm('{}.layer_norm'.format(prefix), mdl.enc.norm)

  for i, layer in enumerate(mdl.enc.layers):
    prefix = 'wav2vec2.encoder.layers.{}'.format(i)
    load_affine('{}.attention.q_proj'.format(prefix), layer.atten.q_proj)
    load_affine('{}.attention.k_proj'.format(prefix), layer.atten.k_proj)
    load_affine('{}.attention.v_proj'.format(prefix), layer.atten.v_proj)
    load_affine('{}.attention.out_proj'.format(prefix), layer.atten.out_proj)
 
    load_affine('{}.feed_forward.intermediate_dense'.format(prefix), layer.feed.in_dense)
    load_affine('{}.feed_forward.output_dense'.format(prefix), layer.feed.out_dense)
  
    load_norm('{}.layer_norm'.format(prefix), layer.norm)
    load_norm('{}.final_layer_norm'.format(prefix), layer.out_norm)

load_model(model.wav2vec2)
load_model(model.wav2vec2_frozen)

ckpt = tf.train.Checkpoint(model)
#ckpt.write("wav2vec2_fs_base.ckpt")
#ckpt.write("wav2vec2_v2_fs_base.ckpt")
#ckpt.write("wav2vec2_v3_fs_base.ckpt")
#ckpt.write("wav2vec2_v4_fs_base.ckpt")
#ckpt.write("wav2vec2_v5_fs_base.ckpt")
#ckpt.write("wav2vec2_v6_fs_base.ckpt")
#ckpt.write("wav2vec2_v7_fs_base.ckpt")
#ckpt.write("wav2vec2_v8_fs_base.ckpt")
#ckpt.write("wav2vec2_v9_fs_base.ckpt")
#ckpt.write("wav2vec2_v10_fs_base.ckpt")
#ckpt.write("wav2vec2_v11_fs_base.ckpt")
#ckpt.write("wav2vec2_v12_fs_base.ckpt")
#ckpt.write("wav2vec2_v13_fs_base.ckpt")
#ckpt.write("wav2vec2_v14_fs_base.ckpt")
#ckpt.write("wav2vec2_v15_fs_base.ckpt")
#ckpt.write("wav2vec2_tow3_fs_base.ckpt")
#ckpt.write("wav2vec2_tow3_v2_fs_base.ckpt")
#ckpt.write("wav2vec2_tow3_v3_fs_base.ckpt")
#ckpt.write("wav2vec2_gate_base.ckpt")
#ckpt.write("wav2vec2_nogate_base.ckpt")
ckpt.write("wav2vec2_sm_base.ckpt")
