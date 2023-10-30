import tensorflow as tf
from tensorflow.python.training import py_checkpoint_reader

reader     = py_checkpoint_reader.NewCheckpointReader(
  "/home/hejung/text-to-text-transfer-transformer/tmp/models/small/model.ckpt-1025000")

dtype_map  = reader.get_variable_to_dtype_map()
shape_map  = reader.get_variable_to_shape_map()

state_dict = { v: reader.get_tensor(v) for v in shape_map}

import tfive
model = tfive.t5()

import numpy as np
_in = np.array([22377, 822, 10, 213, 19, 8, 10283, 13767, 1069, 58, 1])
_in = np.concatenate([_in, np.zeros(128 - _in.shape[0], dtype=_in.dtype)])
_in = _in.reshape((1, -1))
assert _in.shape == (1, 128)
_in_len = np.array([11])

_ = model((_in, _in_len))

emb = state_dict['shared/embedding']
model.embed.assign(emb)

rel_bias = state_dict['encoder/block_000/layer_000/SelfAttention/relative_attention_bias']
model.rel_bias.assign(rel_bias)

assert len(model.sublayers) == 12
for idx in range(len(model.sublayers) // 2):
  prefix = 'encoder/block_{:03d}/layer_000'.format(idx)
  sublayer = model.sublayers[2 * idx]

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
  sublayer = model.sublayers[2 * idx + 1]

  scale = state_dict['{}/layer_norm/scale'.format(prefix)]
  sublayer.layer_norm.scale.assign(scale)

  wi = state_dict['{}/DenseReluDense/wi/kernel'.format(prefix)]
  wo = state_dict['{}/DenseReluDense/wo/kernel'.format(prefix)]
  sublayer.layer.wi.set_weights([wi])
  sublayer.layer.wo.set_weights([wo])
  
scale = state_dict['encoder/final_layer_norm/scale']
model.layer_norm.scale.assign(scale)

out = model((_in, _in_len))
print(out)
