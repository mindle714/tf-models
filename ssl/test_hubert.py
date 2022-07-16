import soundfile
#pcm, _ = soundfile.read("/home/hejung/speech-commands/TEST_SET/no/97f4c236_nohash_0.wav")
pcm, _ = soundfile.read("/home/hejung/speech-commands/TEST_SET/up/9e2ce5e3_nohash_2.wav")

import tensorflow as tf

import torch
m = torch.load("/home/hejung/transformers/examples/pytorch/audio-classification/hubert-base-ft-keyword-spotting/checkpoint-12772/pytorch_model.bin")

from hubert import *

model = hubert_seq()

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

for i, conv in enumerate(model.hubert.fe.conv_layers):
  prefix = 'hubert.feature_extractor.conv_layers'
  load_conv('{}.{}.conv'.format(prefix, i), conv.conv)
  if i == 0:
    load_norm('{}.{}.layer_norm'.format(prefix, i), conv.norm)

prefix = 'hubert.feature_projection'
load_norm('{}.layer_norm'.format(prefix), model.hubert.fp.norm)
load_affine('{}.projection'.format(prefix), model.hubert.fp.proj)

prefix = 'hubert.encoder'
w_g = m['{}.pos_conv_embed.conv.weight_g'.format(prefix)].cpu().numpy()
w_g = np.reshape(w_g, [-1, 1, 1])
w_v = m['{}.pos_conv_embed.conv.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
b = m['{}.pos_conv_embed.conv.bias'.format(prefix)].cpu().numpy()
model.hubert.enc.emb.conv.set_weights([w, b])
load_norm('{}.layer_norm'.format(prefix), model.hubert.enc.norm)

for i, layer in enumerate(model.hubert.enc.layers):
  prefix = 'hubert.encoder.layers.{}'.format(i)
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
