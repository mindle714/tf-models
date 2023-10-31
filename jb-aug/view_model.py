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
#model = tera_phone(num_class=50, use_last=True)
model = tera_phone(use_last=False)

import numpy as np
spec = np.zeros(1701*80)
_in = np.reshape(spec, [1,1701,80])
_tmp = model(_in)

ckpt = tf.train.Checkpoint(model)
ckpt.read("exps/libri_tera_pre_rand0.3/model-3000.ckpt")

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
