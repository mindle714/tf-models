import tensorflow as tf
from tensorflow.python.training import py_checkpoint_reader

reader     = py_checkpoint_reader.NewCheckpointReader(
  "/home/hejung/text-to-text-transfer-transformer/tmp/models/small/model.ckpt-1000000")

dtype_map  = reader.get_variable_to_dtype_map()
shape_map  = reader.get_variable_to_shape_map()

state_dict = {v: reader.get_tensor(v) for v in shape_map}

import tfive
model = tfive.t5()

import numpy as np
#_in = np.array([22377, 822, 10, 213, 19, 8, 10283, 13767, 1069, 58, 1])
_in = np.array([22377, 822, 10, 113, 1944, 528, 7, 7, 2617, 16, 388, 
    45, 2983, 63, 4033, 58, 1]) 
_in_len = np.array([_in.shape[0]]).reshape((-1, 1))
_in = np.concatenate([_in, np.zeros(128 - _in.shape[0], dtype=_in.dtype)])
_in = _in.reshape((1, -1))
assert _in.shape == (1, 128)

_out = np.array([0, 108, 3496, 26, 3, 189, 127, 6992]) 
_out_len = np.array([_out.shape[0]]).reshape((-1, 1))
_out = np.concatenate([_out, np.zeros(32 - _out.shape[0], dtype=_out.dtype)])
_out = _out.reshape((1, -1))
assert _out.shape == (1, 32)

_in = np.tile(_in, [2, 1])
_in_len = np.tile(_in_len, [2, 1])
_out = np.tile(_out, [2, 1])
_out_len = np.tile(_out_len, [2, 1])

_ = model((_in, _out, _in_len, _out_len))

emb = state_dict['shared/embedding']
model.embed.assign(emb)

# encoder
rel_bias = state_dict['encoder/block_000/layer_000/SelfAttention/relative_attention_bias']
model.encoder.rel_bias.assign(rel_bias)

assert len(model.encoder.sublayers) == 12
for idx in range(len(model.encoder.sublayers) // 2):
  prefix = 'encoder/block_{:03d}/layer_000'.format(idx)
  sublayer = model.encoder.sublayers[2 * idx]

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
  sublayer = model.encoder.sublayers[2 * idx + 1]

  scale = state_dict['{}/layer_norm/scale'.format(prefix)]
  sublayer.layer_norm.scale.assign(scale)

  wi = state_dict['{}/DenseReluDense/wi/kernel'.format(prefix)]
  wo = state_dict['{}/DenseReluDense/wo/kernel'.format(prefix)]
  sublayer.layer.wi.set_weights([wi])
  sublayer.layer.wo.set_weights([wo])
  
scale = state_dict['encoder/final_layer_norm/scale']
model.encoder.layer_norm.scale.assign(scale)

# decoder
rel_bias = state_dict['decoder/block_000/layer_000/SelfAttention/relative_attention_bias']
model.decoder.rel_bias.assign(rel_bias)

assert len(model.decoder.sublayers) == 18
for idx in range(len(model.decoder.sublayers) // 3):
  # SelfAttention
  prefix = 'decoder/block_{:03d}/layer_000'.format(idx)
  sublayer = model.decoder.sublayers[3 * idx]

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

  # EncDecAttention
  prefix = 'decoder/block_{:03d}/layer_001'.format(idx)
  sublayer = model.decoder.sublayers[3 * idx + 1]

  scale = state_dict['{}/layer_norm/scale'.format(prefix)]
  sublayer.layer_norm.scale.assign(scale)

  q = state_dict['{}/EncDecAttention/q'.format(prefix)]
  sublayer.layer.q.assign(q)
  k = state_dict['{}/EncDecAttention/k'.format(prefix)]
  sublayer.layer.k.set_weights([k])
  v = state_dict['{}/EncDecAttention/v'.format(prefix)]
  sublayer.layer.v.set_weights([v])
  out = state_dict['{}/EncDecAttention/o'.format(prefix)]
  sublayer.layer.out.set_weights([out])

  # DenseReluDense
  prefix = 'decoder/block_{:03d}/layer_002'.format(idx)
  sublayer = model.decoder.sublayers[3 * idx + 2]

  scale = state_dict['{}/layer_norm/scale'.format(prefix)]
  sublayer.layer_norm.scale.assign(scale)

  wi = state_dict['{}/DenseReluDense/wi/kernel'.format(prefix)]
  wo = state_dict['{}/DenseReluDense/wo/kernel'.format(prefix)]
  sublayer.layer.wi.set_weights([wi])
  sublayer.layer.wo.set_weights([wo])
  
scale = state_dict['decoder/final_layer_norm/scale']
model.decoder.layer_norm.scale.assign(scale)

out = model((_in, _out, _in_len, _out_len))
print(out)
'''
tf.Tensor(
[[[-21.167686   -9.563934  -14.535332  ... -41.624435  -41.80423
   -41.711132 ]
  [-32.608505   -8.9398985 -10.864493  ... -46.030415  -46.15585
   -46.02235  ]
  [-28.371763   -2.1518     -7.616701  ... -41.80992   -41.862537
   -41.853703 ]
  ...
  [ -6.0878167  -4.0208836  -7.0202265 ... -39.149033  -39.167557
   -39.121254 ]
  [ -6.1926956  -4.054224   -7.0777802 ... -39.367405  -39.385197
   -39.337067 ]
  [ -6.1926956  -4.054224   -7.0777802 ... -39.367405  -39.385197
   -39.337067 ]]], shape=(1, 32, 32128), dtype=float32)
'''
