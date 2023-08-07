import tensorflow as tf

import torch
m = torch.load("/home/hejung/pase/FE_e199.ckpt", map_location=torch.device('cpu'))

from pasep import *

model = pasep_unet()

import numpy as np
pcm = np.zeros(16000)
_in = np.reshape(pcm, [1, -1])
_tmp = model(_in)

for i in range(8):
  prefix = 'blocks.{}'.format(i)

  if i == 0:
    w = m[prefix + '.conv.low_hz_'].cpu().numpy().reshape([-1])
    model.pasep.blocks[i].conv.low_hz.assign(w)

    w = m[prefix + '.conv.band_hz_'].cpu().numpy().reshape([-1])
    model.pasep.blocks[i].conv.band_hz.assign(w)

  else:
    w = m[prefix + '.conv.weight'].cpu().transpose(2,0).numpy()
    b = m[prefix + '.conv.bias'].cpu().numpy()
    model.pasep.blocks[i].conv.set_weights([w, b])

  w = m[prefix + '.norm.weight'].cpu().numpy()
  b = m[prefix + '.norm.bias'].cpu().numpy()
  mean = m[prefix + '.norm.running_mean'].cpu().numpy()
  var = m[prefix + '.norm.running_var'].cpu().numpy()
  model.pasep.blocks[i].norm.set_weights([w, b, mean, var])

  w = m[prefix + '.act.weight'].cpu().numpy().reshape([1, -1])
  model.pasep.blocks[i].prelu.set_weights([w])

for i in range(7):
  prefix = 'denseskips.{}'.format(i)
  w = m[prefix + '.weight'].cpu().transpose(2,0).numpy()
  model.pasep.denses[i].set_weights([w])

w = m['rnn.layers.0.linear.weight'].cpu().transpose(1,0).numpy()
b = m['rnn.layers.0.linear.bias'].cpu().numpy()
model.pasep.rnn.layers[0].linear.set_weights([w, b])

w = m['W.weight'].cpu().transpose(2,0).numpy()
b = m['W.bias'].cpu().numpy()
model.pasep.rnn_out.set_weights([w, b])

mean = m['norm_out.running_mean'].cpu().numpy()
var = m['norm_out.running_var'].cpu().numpy()
model.pasep.rnn_norm.set_weights([mean, var])

ckpt = tf.train.Checkpoint(model)
ckpt.write("pasep_base4.ckpt")
