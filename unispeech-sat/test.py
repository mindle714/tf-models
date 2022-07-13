import soundfile
pcm, _ = soundfile.read("/home/hejung/speech-commands/TEST_SET/no/97f4c236_nohash_0.wav")

import tensorflow as tf

import torch
m = torch.load("/home/hejung/transformers/examples/pytorch/audio-classification/unispeech-sat-base-ft-keyword-spotting/checkpoint-31930/pytorch_model.bin")

from unisat import *

model = unisat_seq()

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

for i, conv in enumerate(model.unisat.fe.conv_layers):
  prefix = 'unispeech_sat.feature_extractor.conv_layers'
  load_conv('{}.{}.conv'.format(prefix, i), conv.conv)
  if i == 0:
    load_norm('{}.{}.layer_norm'.format(prefix, i), conv.norm)

prefix = 'unispeech_sat.feature_projection'
load_norm('{}.layer_norm'.format(prefix), model.unisat.fp.norm)
load_affine('{}.projection'.format(prefix), model.unisat.fp.proj)

prefix = 'unispeech_sat.encoder'
w_g = m['{}.pos_conv_embed.conv.weight_g'.format(prefix)].cpu().numpy()
w_g = np.reshape(w_g, [-1, 1, 1])
w_v = m['{}.pos_conv_embed.conv.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
b = m['{}.pos_conv_embed.conv.bias'.format(prefix)].cpu().numpy()
model.unisat.enc.emb.conv.set_weights([w, b])
load_norm('{}.layer_norm'.format(prefix), model.unisat.enc.norm)

for i, layer in enumerate(model.unisat.enc.layers):
  prefix = 'unispeech_sat.encoder.layers.{}'.format(i)
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
