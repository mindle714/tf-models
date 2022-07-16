import soundfile
#pcm, _ = soundfile.read("/home/hejung/speech-commands/TEST_SET/no/97f4c236_nohash_0.wav")
pcm, _ = soundfile.read("/home/hejung/speech-commands/TEST_SET/up/9e2ce5e3_nohash_2.wav")

import tensorflow as tf

import torch
m = torch.load("/home/hejung/transformers/examples/pytorch/audio-classification/data2vec-audio-base-ft-keyword-spotting/checkpoint-25544/pytorch_model.bin")

from data2vec import *

model = data2vec_seq()

import numpy as np
_in = np.reshape(pcm, [1, -1, 1])
_tmp = model(_in)

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
'''
#w_g = m['{}.pos_conv_embed.conv.weight_g'.format(prefix)].cpu().numpy()
#w_g = np.reshape(w_g, [-1, 1, 1])
#w_v = m['{}.pos_conv_embed.conv.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
#w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
w = m['{}.pos_conv_embed.conv.weight'.format(prefix)].cpu().numpy()
b = m['{}.pos_conv_embed.conv.bias'.format(prefix)].cpu().numpy()
model.data2vec.enc.emb.conv.set_weights([w, b])
'''
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

load_affine('projector', model.projector)
load_affine('classifier', model.classifier)

print(model(_in))
