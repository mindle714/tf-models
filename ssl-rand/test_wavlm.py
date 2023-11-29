import tensorflow as tf

import torch
m = torch.load("/home/hejung/wavlm-base-unilm/WavLM-Base.pt")
m = m['model']

from wavlm import *

model = wavlm_phone(num_class=50)

import numpy as np
pcm = np.zeros(128000)
_in = np.reshape(pcm, [1, -1])
_ref = np.zeros([1, 399], dtype=np.int32)
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

for i, conv in enumerate(model.wavlm.wavlm.fe.conv_layers):
  prefix = 'feature_extractor.conv_layers'
  load_conv('{}.{}.0'.format(prefix, i), conv.conv)
  if i == 0:
    load_norm('{}.{}.2'.format(prefix, i), conv.norm)

load_norm('layer_norm', model.wavlm.wavlm.fp.norm)
load_affine('post_extract_proj', model.wavlm.wavlm.fp.proj)

prefix = 'encoder'
w_g = m['{}.pos_conv.0.weight_g'.format(prefix)].cpu().numpy()
w_g = np.reshape(w_g, [-1, 1, 1])
w_v = m['{}.pos_conv.0.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
b = m['{}.pos_conv.0.bias'.format(prefix)].cpu().numpy()
model.wavlm.wavlm.enc.emb.conv.set_weights([w, b])

load_norm('{}.layer_norm'.format(prefix), model.wavlm.wavlm.enc.norm)

w = m['encoder.layers.0.self_attn.relative_attention_bias.weight'].cpu().numpy().T
model.wavlm.wavlm.enc.rel_bias.assign(w)

for i, layer in enumerate(model.wavlm.wavlm.enc.layers):
  prefix = 'encoder.layers.{}'.format(i)
  load_affine('{}.self_attn.q_proj'.format(prefix), layer.atten.q_proj)
  load_affine('{}.self_attn.k_proj'.format(prefix), layer.atten.k_proj)
  load_affine('{}.self_attn.v_proj'.format(prefix), layer.atten.v_proj)
  load_affine('{}.self_attn.out_proj'.format(prefix), layer.atten.out_proj)
 
  load_affine('{}.fc1'.format(prefix), layer.feed.in_dense)
  load_affine('{}.fc2'.format(prefix), layer.feed.out_dense)
  
  load_norm('{}.self_attn_layer_norm'.format(prefix), layer.norm)
  load_norm('{}.final_layer_norm'.format(prefix), layer.out_norm)

  load_affine('{}.self_attn.grep_linear'.format(prefix), layer.atten.grep_linear)
  w = m['{}.self_attn.grep_a'.format(prefix)].cpu().numpy().squeeze(0)
  layer.atten.grep_a.assign(w)

w = m['mask_emb'].cpu().numpy()
model.wavlm.wavlm.masked_spec_embed.assign(w)

import soundfile
pcm = np.expand_dims(soundfile.read("/home/hejung/tf-models/utils/SA1_norm.wav")[0], 0).astype(np.float32)
_tmp = model(pcm, return_feat=True)
print(_tmp[-1])
