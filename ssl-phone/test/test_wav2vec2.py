import soundfile
import numpy as np

pcm, _ = soundfile.read("/home/hejung/s3prl/s3prl/downstream/speech_commands/dummy_data/train/yes/01d22d03_nohash_0.wav")
pcm2 = pcm / 2.
pcms = np.concatenate([pcm.reshape([1, -1]), pcm2.reshape([1, -1])], 0).astype(np.float32)

import tensorflow as tf

import torch
m = torch.load("/home/hejung/wav2vec2-base/pytorch_model.bin")

from wav2vec2 import *

model = wav2vec2_seq()

_in = pcms
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

for i, conv in enumerate(model.wav2vec2.fe.conv_layers):
  prefix = 'wav2vec2.feature_extractor.conv_layers'
  load_conv('{}.{}.conv'.format(prefix, i), conv.conv)
  if i == 0:
    load_norm('{}.{}.layer_norm'.format(prefix, i), conv.norm)

prefix = 'wav2vec2.feature_projection'
load_norm('{}.layer_norm'.format(prefix), model.wav2vec2.fp.norm)
load_affine('{}.projection'.format(prefix), model.wav2vec2.fp.proj)

prefix = 'wav2vec2.encoder'
w_g = m['{}.pos_conv_embed.conv.weight_g'.format(prefix)].cpu().numpy()
w_g = np.reshape(w_g, [-1, 1, 1])
w_v = m['{}.pos_conv_embed.conv.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
b = m['{}.pos_conv_embed.conv.bias'.format(prefix)].cpu().numpy()
model.wav2vec2.enc.emb.conv.set_weights([w, b])
load_norm('{}.layer_norm'.format(prefix), model.wav2vec2.enc.norm)

for i, layer in enumerate(model.wav2vec2.enc.layers):
  prefix = 'wav2vec2.encoder.layers.{}'.format(i)
  load_affine('{}.attention.q_proj'.format(prefix), layer.atten.q_proj)
  load_affine('{}.attention.k_proj'.format(prefix), layer.atten.k_proj)
  load_affine('{}.attention.v_proj'.format(prefix), layer.atten.v_proj)
  load_affine('{}.attention.out_proj'.format(prefix), layer.atten.out_proj)
 
  load_affine('{}.feed_forward.intermediate_dense'.format(prefix), layer.feed.in_dense)
  load_affine('{}.feed_forward.output_dense'.format(prefix), layer.feed.out_dense)
  
  load_norm('{}.layer_norm'.format(prefix), layer.norm)
  load_norm('{}.final_layer_norm'.format(prefix), layer.out_norm)

load_affine('project_hid', model.project_hid)
load_affine('project_q', model.project_q)

load_affine('quantizer.weight_proj', model.quantizer.weight_proj)
w = m['quantizer.codevectors'].cpu().numpy()
model.quantizer.codevectors.assign(w)
w = m['wav2vec2.masked_spec_embed'].cpu().numpy()
model.wav2vec2.masked_spec_embed.assign(w)

mask_time_indices = np.load('mask_time_indices.npy').astype(np.float32)
sampled_negative_indices = np.load('sampled_negative_indices.npy')

x, qx_feat = model((_in, mask_time_indices, sampled_negative_indices), training=True)
print(x, qx_feat)

'''
(venv3.7-tera) hejung@speech:~/tf-models/ssl$ python3 test_wav2vec2.py
tf.Tensor(
[[[-0.03795013  0.18967038  0.17773671 ...  0.22049077  0.2822556
    0.33677945]
  [-0.00936582  0.41093287 -0.02825753 ...  0.4727972   0.1705111
    0.26599422]
  [ 0.06454878  0.18859263 -0.02218433 ...  0.14479631  0.19384265
    0.2460265 ]
  ...
  [-0.1849951   0.14248125 -0.14958894 ...  0.19302145  0.14552265
    0.27958643]
  [-0.10210146  0.41023663 -0.08990705 ...  0.0936659   0.1708027
    0.2306513 ]
  [ 0.00899731  0.21980184  0.18803358 ...  0.04501326  0.19920312
    0.35360476]]], shape=(1, 49, 768), dtype=float32)

pushd ~/s3prl/s3prl
python3 -m pdb run_downstream.py -m train -n ExpName -k /home/hejung/wav2vec2-base/pytorch_model.bin -g /home/hejung/wav2vec2-base/config.json -d speech_commands
(Pdb) b /home/hejung/venv3.7-tera/lib/python3.7/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py:1079
(Pdb) c
'''
