import soundfile
import numpy as np

pcm, _ = soundfile.read("01d22d03_nohash_0.wav")
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

import mask

#mask_time_indices = np.load('mask_time_indices.npy').astype(np.float32)
seq_len = mask.get_feat_extract_output_length(_in.shape[1])
mask_time_indices = mask.compute_mask_indices(
  _in.shape[0], seq_len, None, 0.2, 2, 0)  

#sampled_negative_indices = np.load('sampled_negative_indices.npy')
sampled_negative_indices = sample_negative_indices(
  _in.shape[0], seq_len, 100, mask_time_indices)  

x, qx_feat, loss = model((_in, mask_time_indices, sampled_negative_indices), training=True)
print(x, qx_feat, loss)

'''
(venv3-s3prl) hejung@speecht7:~/tf-models/ssl-phone/test$ python3 test_wav2vec2.py
tf.Tensor(
[[[-0.08230415  0.19224644  1.0171125  ...  0.60573643 -0.2818361
   -1.7405095 ]
  [-0.11546403 -0.32955498  1.3976142  ...  0.7780607  -0.23100913
   -1.5106542 ]
  [-0.4624009  -0.72181714  0.86816823 ... -0.07241634 -0.0333305
   -0.83358896]
  ...
  [-0.40031075  3.5778522   0.685996   ...  0.00991929  1.0833775
   -1.1881367 ]
  [ 0.16795045  0.33314243  1.3374524  ...  0.8260322  -0.26096678
   -1.0532446 ]
  [-0.2612179  -0.5210311   1.6374518  ...  0.43463713 -0.09315532
   -1.1548278 ]]

 [[ 0.40840015 -0.24011052  0.12556718 ...  0.08366838 -0.29980266
   -1.8062729 ]
  [ 0.64280236 -0.94143355  0.4831649  ... -0.4857757  -0.45735702
   -1.1713867 ]
  [ 0.39589936 -0.8151152   0.12698975 ... -0.65558887 -0.376135
   -1.0569324 ]
  ...
  [ 0.42744908  0.3861401   0.7101313  ...  0.31882718  0.19559437
   -1.3239496 ]
  [ 0.6670534  -0.6872303   0.6887257  ...  0.5492828  -0.09047058
   -1.0273757 ]
  [ 0.65042174 -0.47396213  0.6384514  ...  0.31780222 -0.22380091
   -1.6411686 ]]], shape=(2, 49, 256), dtype=float32) tf.Tensor(
[[[-0.20424592  1.0620682  -0.06415138 ...  0.03070011  0.17831248
   -0.306218  ]
  [ 0.48933423  0.5720519  -0.46116647 ... -0.5115502  -0.20376638
    0.5186587 ]
  [-0.11333267  0.07002556 -0.15774763 ...  0.06021518 -0.02813273
   -0.9968629 ]
  ...
  [ 0.2748624   0.35755044 -0.39926007 ... -0.730418   -0.09184477
    0.5876132 ]
  [ 0.05799696  0.3457167   0.13298586 ... -0.27164966 -0.04360084
    0.2299884 ]
  [-0.07032934 -0.06146909  0.19768064 ... -0.31404576 -0.09929797
    0.15501016]]

 [[-0.32734597  0.4567889  -0.1352315  ...  0.06446771  0.22887649
   -0.16870238]
  [ 0.48933423  0.5720519  -0.46116647 ... -0.5115502  -0.20376638
    0.5186587 ]
  [-0.04037524 -0.00808815 -0.00314383 ...  0.05927289  0.03996353
   -0.3618939 ]
  ...
  [ 0.2748624   0.35755044 -0.39926007 ... -0.730418   -0.09184477
    0.5876132 ]
  [-0.18174382  0.05832914  0.3339933  ...  0.02448292  0.1848859
   -0.0645766 ]
  [ 0.17478843  0.33719546  0.10228443 ...  0.21498401 -0.11323874

pushd ~/tf-models/ssl-phone/test/fairseq
python3 -m pdb test_wav2vec2_pre.py
(Pdb) b /home/hejung/venv3-s3prl/lib/python3.8/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py:1554
(Pdb) c
'''
