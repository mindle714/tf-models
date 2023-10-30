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

prefix = 'encoder/block_000/layer_000'
scale = state_dict['{}/layer_norm/scale'.format(prefix)]
model.layer_norm.scale.assign(scale)

q = state_dict['{}/SelfAttention/q'.format(prefix)]
model.atten.q.assign(q)
k = state_dict['{}/SelfAttention/k'.format(prefix)]
model.atten.k.set_weights([k])
v = state_dict['{}/SelfAttention/v'.format(prefix)]
model.atten.v.set_weights([v])

rel_bias = state_dict['{}/SelfAttention/relative_attention_bias'.format(prefix)]
model.rel_bias.assign(rel_bias)

out = model((_in, _in_len))
print(out)
