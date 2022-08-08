import soundfile
pcm, _ = soundfile.read("/home/hejung/tmp/clean.wav")

import tensorflow as tf

import torch
m = torch.load("/home/hejung/pase/FE_e199.ckpt", map_location=torch.device('cpu'))

from pasepp import *

model = pasepp_seq()

import numpy as np
_in = np.reshape(pcm, [1, -1])
_tmp = model(_in)

for i in range(8):
  prefix = 'blocks.{}'.format(i)

  if i == 0:
    w = m[prefix + '.conv.low_hz_'].cpu().numpy().reshape([-1])
    model.pasepp.blocks[i].conv.low_hz.assign(w)

    w = m[prefix + '.conv.band_hz_'].cpu().numpy().reshape([-1])
    model.pasepp.blocks[i].conv.band_hz.assign(w)

  else:
    w = m[prefix + '.conv.weight'].cpu().transpose(2,0).numpy()
    b = m[prefix + '.conv.bias'].cpu().numpy()
    model.pasepp.blocks[i].conv.set_weights([w, b])

  w = m[prefix + '.norm.weight'].cpu().numpy()
  b = m[prefix + '.norm.bias'].cpu().numpy()
  mean = m[prefix + '.norm.running_mean'].cpu().numpy()
  var = m[prefix + '.norm.running_var'].cpu().numpy()
  model.pasepp.blocks[i].norm.set_weights([w, b, mean, var])

  w = m[prefix + '.act.weight'].cpu().numpy().reshape([1, -1])
  model.pasepp.blocks[i].prelu.set_weights([w])

#w = m['denseskips.0.weight'].cpu().numpy()
#model.pasepp.denses[0].set_weights([w])

_out = model(_in)
print(_out)

import sys
sys.exit()

