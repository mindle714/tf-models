import tensorflow as tf
from tensorflow.python.training import py_checkpoint_reader

reader     = py_checkpoint_reader.NewCheckpointReader(
  "/home/hejung/text-to-text-transfer-transformer/tmp/models/small/model.ckpt-1000000")

dtype_map  = reader.get_variable_to_dtype_map()
shape_map  = reader.get_variable_to_shape_map()

state_dict = {v: reader.get_tensor(v) for v in shape_map}

import torch
m = torch.load("/home/hejung/wav2vec2-base/pytorch_model.bin")

import model
_model = model.wav2vec2_t5_phone(num_class=50)

import numpy as np
pcm = np.zeros(16000)
_in = np.reshape(pcm, [1, -1])
_ = _model(_in)

w2v_size = 0

def load_norm(prefix, e):
  global w2v_size
  w = m['{}.weight'.format(prefix, i)].cpu().numpy()
  b = m['{}.bias'.format(prefix, i)].cpu().numpy()
  e.gamma.assign(w)
  e.beta.assign(b)
  w2v_size += np.size(w) + np.size(b)

def load_affine(prefix, e):
  global w2v_size
  w = m['{}.weight'.format(prefix)]
  bname = '{}.bias'.format(prefix)
  if bname in m:
    b = m[bname]
    e.set_weights([w.transpose(1,0).cpu().numpy(), b.cpu().numpy()])
    w2v_size += np.size(w.cpu().numpy()) + np.size(b.cpu().numpy())
  else:
    e.set_weights([w.transpose(1,0).cpu().numpy()])
    w2v_size += np.size(w.cpu().numpy())

def load_conv(prefix, e):
  global w2v_size
  w = m['{}.weight'.format(prefix)]
  bname = '{}.bias'.format(prefix)
  if bname in m:
    b = m[bname]
    e.set_weights([w.transpose(2,0).cpu().numpy(), b.cpu().numpy()])
    w2v_size += np.size(w.cpu().numpy()) + np.size(b.cpu().numpy())
  else:
    e.set_weights([w.transpose(2,0).cpu().numpy()])
    w2v_size += np.size(w.cpu().numpy())

for i, conv in enumerate(_model.wav2vec2_t5.fe.conv_layers):
  prefix = 'wav2vec2.feature_extractor.conv_layers'
  load_conv('{}.{}.conv'.format(prefix, i), conv.conv)
  if i == 0:
    load_norm('{}.{}.layer_norm'.format(prefix, i), conv.norm)

prefix = 'wav2vec2.feature_projection'
load_norm('{}.layer_norm'.format(prefix), _model.wav2vec2_t5.fp.norm)
load_affine('{}.projection'.format(prefix), _model.wav2vec2_t5.fp.proj)

print("w2v size {}".format(w2v_size))
t5_size = 0

# encoder
rel_bias = state_dict['encoder/block_000/layer_000/SelfAttention/relative_attention_bias']
_model.wav2vec2_t5.encoder.rel_bias.assign(rel_bias)
t5_size += np.size(rel_bias)

assert len(_model.wav2vec2_t5.encoder.sublayers) == 12
for idx in range(len(_model.wav2vec2_t5.encoder.sublayers) // 2):
  prefix = 'encoder/block_{:03d}/layer_000'.format(idx)
  sublayer = _model.wav2vec2_t5.encoder.sublayers[2 * idx]

  scale = state_dict['{}/layer_norm/scale'.format(prefix)]
  sublayer.layer_norm.scale.assign(scale)
  t5_size += np.size(scale)

  q = state_dict['{}/SelfAttention/q'.format(prefix)]
  sublayer.layer.q.assign(q)
  k = state_dict['{}/SelfAttention/k'.format(prefix)]
  sublayer.layer.k.set_weights([k])
  v = state_dict['{}/SelfAttention/v'.format(prefix)]
  sublayer.layer.v.set_weights([v])
  out = state_dict['{}/SelfAttention/o'.format(prefix)]
  sublayer.layer.out.set_weights([out])
  t5_size += np.size(q) + np.size(k) + np.size(v) + np.size(out)

  prefix = 'encoder/block_{:03d}/layer_001'.format(idx)
  sublayer = _model.wav2vec2_t5.encoder.sublayers[2 * idx + 1]

  scale = state_dict['{}/layer_norm/scale'.format(prefix)]
  sublayer.layer_norm.scale.assign(scale)
  t5_size += np.size(scale)

  wi = state_dict['{}/DenseReluDense/wi/kernel'.format(prefix)]
  wo = state_dict['{}/DenseReluDense/wo/kernel'.format(prefix)]
  sublayer.layer.wi.set_weights([wi])
  sublayer.layer.wo.set_weights([wo])
  t5_size += np.size(wi) + np.size(wo)
  
scale = state_dict['encoder/final_layer_norm/scale']
_model.wav2vec2_t5.encoder.layer_norm.scale.assign(scale)
t5_size += np.size(scale)

print("t5 size {}".format(t5_size))
