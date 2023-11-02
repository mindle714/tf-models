import tensorflow as tf
from tensorflow.python.training import py_checkpoint_reader

reader     = py_checkpoint_reader.NewCheckpointReader(
  "/home/hejung/text-to-text-transfer-transformer/tmp/models/small/model.ckpt-1000000")

dtype_map  = reader.get_variable_to_dtype_map()
shape_map  = reader.get_variable_to_shape_map()

state_dict = {v: reader.get_tensor(v) for v in shape_map}

import torch
m = torch.load("/home/hejung/wav2vec2-base/pytorch_model.bin")

import model_v2
_model = model_v2.wav2vec2_t5_phone(num_class=50)

import numpy as np
pcm = np.zeros(16000)
_in = np.reshape(pcm, [1, -1])
_ = _model(_in)

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

for i, conv in enumerate(_model.wav2vec2_t5.fe.conv_layers):
  prefix = 'wav2vec2.feature_extractor.conv_layers'
  load_conv('{}.{}.conv'.format(prefix, i), conv.conv)
  if i == 0:
    load_norm('{}.{}.layer_norm'.format(prefix, i), conv.norm)

prefix = 'wav2vec2.feature_projection'
load_norm('{}.layer_norm'.format(prefix), _model.wav2vec2_t5.fp.norm)
load_affine('{}.projection'.format(prefix), _model.wav2vec2_t5.fp.proj)

prefix = 'wav2vec2.encoder'
w_g = m['{}.pos_conv_embed.conv.weight_g'.format(prefix)].cpu().numpy()
w_g = np.reshape(w_g, [-1, 1, 1])
w_v = m['{}.pos_conv_embed.conv.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
b = m['{}.pos_conv_embed.conv.bias'.format(prefix)].cpu().numpy()
_model.wav2vec2_t5.encoder_2.emb.conv.set_weights([w, b])
load_norm('{}.layer_norm'.format(prefix), _model.wav2vec2_t5.encoder_2.norm)

for i, layer in enumerate(_model.wav2vec2_t5.encoder_2.layers):
  prefix = 'wav2vec2.encoder.layers.{}'.format(i)
  load_affine('{}.attention.q_proj'.format(prefix), layer.atten.q_proj)
  load_affine('{}.attention.k_proj'.format(prefix), layer.atten.k_proj)
  load_affine('{}.attention.v_proj'.format(prefix), layer.atten.v_proj)
  load_affine('{}.attention.out_proj'.format(prefix), layer.atten.out_proj)
 
  load_affine('{}.feed_forward.intermediate_dense'.format(prefix), layer.feed.in_dense)
  load_affine('{}.feed_forward.output_dense'.format(prefix), layer.feed.out_dense)
  
  load_norm('{}.layer_norm'.format(prefix), layer.norm)
  load_norm('{}.final_layer_norm'.format(prefix), layer.out_norm)

# encoder
rel_bias = state_dict['encoder/block_000/layer_000/SelfAttention/relative_attention_bias']
_model.wav2vec2_t5.encoder_1.rel_bias.assign(rel_bias)

assert len(_model.wav2vec2_t5.encoder_1.sublayers) == 12
for idx in range(len(_model.wav2vec2_t5.encoder_1.sublayers) // 2):
  prefix = 'encoder/block_{:03d}/layer_000'.format(idx)
  sublayer = _model.wav2vec2_t5.encoder_1.sublayers[2 * idx]

  scale = state_dict['{}/layer_norm/scale'.format(prefix)]
  sublayer.layer_norm.scale.assign(scale)

  q = state_dict['{}/SelfAttention/q'.format(prefix)]
  sublayer.layer.q.assign(q)
  k = state_dict['{}/SelfAttention/k'.format(prefix)]
  sublayer.layer.k.set_weights([k])
  v = state_dict['{}/SelfAttention/v'.format(prefix)]
  sublayer.layer.v.set_weights([v])
  out = state_dict['{}/SelfAttention/o'.format(prefix)]
  sublayer.layer.out.set_weights([out])

  prefix = 'encoder/block_{:03d}/layer_001'.format(idx)
  sublayer = _model.wav2vec2_t5.encoder_1.sublayers[2 * idx + 1]

  scale = state_dict['{}/layer_norm/scale'.format(prefix)]
  sublayer.layer_norm.scale.assign(scale)

  wi = state_dict['{}/DenseReluDense/wi/kernel'.format(prefix)]
  wo = state_dict['{}/DenseReluDense/wo/kernel'.format(prefix)]
  sublayer.layer.wi.set_weights([wi])
  sublayer.layer.wo.set_weights([wo])
  
scale = state_dict['encoder/final_layer_norm/scale']
_model.wav2vec2_t5.encoder_1.layer_norm.scale.assign(scale)

ckpt = tf.train.Checkpoint(_model)
ckpt.write("model_v2.ckpt")
