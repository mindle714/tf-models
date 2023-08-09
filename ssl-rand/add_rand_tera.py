import tensorflow as tf
from tera import *

# copied from keras/initializers/initializers_v2.py
def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.

  Args:
    shape: Integer shape tuple or TF tensor shape.

  Returns:
    A tuple of integer scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)

# timit
model = tera_phone(num_class=50, use_last=True)

import numpy as np
spec = np.zeros(1701*80)
_in = np.reshape(spec, [1,1701,80])
_tmp = model(_in)

def print_stat(e):
  w, b = e.get_weights()
  fan_in, fan_out = _compute_fans(w.shape)
  print("{:.6f}".format(np.var(w)), "{:.6f}".format(2/(fan_in + fan_out)))

print_stat(model.tera.tera.fe.spec_transform)
# TODO what about layer norm weights?

for i in range(3):
  print_stat(model.tera.tera.enc.layers[i].atten.self_attn.query)
  print_stat(model.tera.tera.enc.layers[i].atten.self_attn.key)
  print_stat(model.tera.tera.enc.layers[i].atten.self_attn.value)
  
  print_stat(model.tera.tera.enc.layers[i].atten.out)
  # TODO lnorm
  
  print_stat(model.tera.tera.enc.layers[i].inter)
  print_stat(model.tera.tera.enc.layers[i].out)

print("load...")
ckpt = tf.train.Checkpoint(model)
ckpt.read("tera_timit.ckpt")

def add_rand(e, suffix = None):
  w, b = e.get_weights()
  fan_in, fan_out = _compute_fans(w.shape)
  if suffix is None:
    print("{:.6f}".format(np.var(w)), "{:.6f}".format(2/(fan_in + fan_out)), \
            "{:.6f}".format(np.var(w) / (2/(fan_in + fan_out))))
  else:
    print("{:.6f}".format(np.var(w)), "{:.6f}".format(2/(fan_in + fan_out)), \
            "{:.6f}".format(np.var(w) / (2/(fan_in + fan_out))), suffix)

  w_flat = w.flatten()
  idxs = np.argsort(np.abs(w_flat))
#  idxs = idxs[:int(idxs.shape[0] * 0.7)]
#  idxs = idxs[:int(idxs.shape[0] * 0.5)]
#  idxs = idxs[:int(idxs.shape[0] * 0.3)]
  idxs = idxs[:int(idxs.shape[0] * 0.2)]

  w_mask = np.ones_like(w_flat)
  w_mask[idxs] = 0
  w_mask = w_mask.reshape(w.shape)

  init = tf.keras.initializers.GlorotUniform(seed = 1234)
  w_rand = init(shape = w.shape)
  w_rand = w_mask * w + (1 - w_mask) * w_rand

  e.set_weights([w_rand, b])

add_rand(model.tera.tera.fe.spec_transform)
# TODO what about layer norm weights?

for i in range(3):
  add_rand(model.tera.tera.enc.layers[i].atten.self_attn.query, "query[{}]".format(i))
  add_rand(model.tera.tera.enc.layers[i].atten.self_attn.key, "key[{}]".format(i))
  add_rand(model.tera.tera.enc.layers[i].atten.self_attn.value, "value[{}]".format(i))
  
  add_rand(model.tera.tera.enc.layers[i].atten.out)
  # TODO lnorm
  
  add_rand(model.tera.tera.enc.layers[i].inter)
  add_rand(model.tera.tera.enc.layers[i].out)
  # TODO lnorm

#ckpt.write("tera_timit_rand.ckpt")
#ckpt.write("tera_timit_rand0_5.ckpt")
#ckpt.write("tera_timit_rand0_3.ckpt")
ckpt.write("tera_timit_rand0_2.ckpt")
