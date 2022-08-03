import tensorflow as tf

import torch
# ModuleNotFoundError: No module named 'optimizers' occurs without below
import sys
import s3prl.optimizers
sys.modules["optimizers"] = s3prl.optimizers
m = torch.load("/home/hejung/tera_960hr/states-1000000.ckpt")

from tera import *

model = tera_unet()

import numpy as np
pcm = np.zeros(16000)
_in = np.reshape(pcm, [1, -1])
_tmp = model((_in, None))

w = m['Transformer']['input_representations.spec_transform.weight'].cpu().numpy()
b = m['Transformer']['input_representations.spec_transform.bias'].cpu().numpy()
model.tera.fe.spec_transform.set_weights([w.transpose(1,0), b])

w = m['Transformer']['input_representations.LayerNorm.weight'].cpu().numpy()
b = m['Transformer']['input_representations.LayerNorm.bias'].cpu().numpy()
model.tera.fe.lnorm.gamma.assign(w)
model.tera.fe.lnorm.beta.assign(b)

for i in range(3):
  prefix = 'encoder.layer.{}'.format(i)
  w = m['Transformer'][prefix + '.attention.self.query.weight'].cpu().numpy()
  b = m['Transformer'][prefix + '.attention.self.query.bias'].cpu().numpy()
  model.tera.enc.layers[i].atten.self_attn.query.set_weights([w.transpose(1,0), b])

  w = m['Transformer'][prefix + '.attention.self.key.weight'].cpu().numpy()
  b = m['Transformer'][prefix + '.attention.self.key.bias'].cpu().numpy()
  model.tera.enc.layers[i].atten.self_attn.key.set_weights([w.transpose(1,0), b])

  w = m['Transformer'][prefix + '.attention.self.value.weight'].cpu().numpy()
  b = m['Transformer'][prefix + '.attention.self.value.bias'].cpu().numpy()
  model.tera.enc.layers[i].atten.self_attn.value.set_weights([w.transpose(1,0), b])

  w = m['Transformer'][prefix + '.attention.output.dense.weight'].cpu().numpy()
  b = m['Transformer'][prefix + '.attention.output.dense.bias'].cpu().numpy()
  model.tera.enc.layers[i].atten.out.set_weights([w.transpose(1,0), b])

  w = m['Transformer'][prefix + '.attention.output.LayerNorm.weight'].cpu().numpy()
  b = m['Transformer'][prefix + '.attention.output.LayerNorm.bias'].cpu().numpy()
  model.tera.enc.layers[i].atten.lnorm.gamma.assign(w)
  model.tera.enc.layers[i].atten.lnorm.beta.assign(b)

  w = m['Transformer'][prefix + '.intermediate.dense.weight'].cpu().numpy()
  b = m['Transformer'][prefix + '.intermediate.dense.bias'].cpu().numpy()
  model.tera.enc.layers[i].inter.set_weights([w.transpose(1,0), b])

  w = m['Transformer'][prefix + '.output.dense.weight'].cpu().numpy()
  b = m['Transformer'][prefix + '.output.dense.bias'].cpu().numpy()
  model.tera.enc.layers[i].out.set_weights([w.transpose(1,0), b])

  w = m['Transformer'][prefix + '.output.LayerNorm.weight'].cpu().numpy()
  b = m['Transformer'][prefix + '.output.LayerNorm.bias'].cpu().numpy()
  model.tera.enc.layers[i].lnorm.gamma.assign(w)
  model.tera.enc.layers[i].lnorm.beta.assign(b)

ckpt = tf.train.Checkpoint(root=model)
ckpt.write("tera_base.ckpt")
