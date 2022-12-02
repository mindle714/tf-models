import soundfile
pcm, _ = soundfile.read("dummy_one/test/noisy/p257_434.wav")
assert _ == 16000

import tensorflow as tf
from cmgan import *

model = cmgan()

import torch
m = torch.load("/home/hejung/CMGAN/src/best_ckpt/ckpt")

import numpy as np
_in = np.reshape(pcm, [1, -1])
_tmp = model(_in)
print(_tmp)

w = m['dense_encoder.conv_1.0.weight'].cpu().numpy()
b = m['dense_encoder.conv_1.0.bias'].cpu().numpy()
model.tscnet.denc.conv_1.set_weights([w.transpose([2, 3, 1, 0]), b])

w = m['dense_encoder.conv_1.1.weight'].cpu().numpy()
b = m['dense_encoder.conv_1.1.bias'].cpu().numpy()
model.tscnet.denc.inorm2d.gamma.assign(w)
model.tscnet.denc.inorm2d.beta.assign(b)

w = m['dense_encoder.conv_1.2.weight'].cpu().numpy()
model.tscnet.denc.prelu.set_weights([w.reshape([1, 1, -1])])

prefix = 'dense_encoder.dilated_dense'
for i in range(len(model.tscnet.denc.dildense.convs)):
  w = m['{}.conv{}.weight'.format(prefix, i+1)].cpu().numpy()
  b = m['{}.conv{}.bias'.format(prefix, i+1)].cpu().numpy()
  model.tscnet.denc.dildense.convs[i].set_weights([w.transpose([2, 3, 1, 0]), b])
  
  w = m['{}.norm{}.weight'.format(prefix, i+1)].cpu().numpy()
  b = m['{}.norm{}.bias'.format(prefix, i+1)].cpu().numpy()
  model.tscnet.denc.dildense.norms[i].gamma.assign(w)
  model.tscnet.denc.dildense.norms[i].beta.assign(b)
 
  w = m['{}.prelu{}.weight'.format(prefix, i+1)].cpu().numpy()
  model.tscnet.denc.dildense.prelus[i].set_weights([w.reshape([1, 1, -1])])

w = m['dense_encoder.conv_2.0.weight'].cpu().numpy()
b = m['dense_encoder.conv_2.0.bias'].cpu().numpy()
model.tscnet.denc.conv_2.set_weights([w.transpose([2, 3, 1, 0]), b])

w = m['dense_encoder.conv_2.1.weight'].cpu().numpy()
b = m['dense_encoder.conv_2.1.bias'].cpu().numpy()
model.tscnet.denc.inorm2d_2.gamma.assign(w)
model.tscnet.denc.inorm2d_2.beta.assign(b)

w = m['dense_encoder.conv_2.2.weight'].cpu().numpy()
model.tscnet.denc.prelu_2.set_weights([w.reshape([1, 1, -1])])

print(model(_in))
import sys
sys.exit()

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

#load_affine('projector', model.projector)
#load_affine('classifier', model.classifier)

print(model(_in))

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
