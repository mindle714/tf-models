import numpy as np
_in = np.zeros((1, 16000))

from beit import *

model = beit_unet()

_tmp = model(_in)

import torch
m = torch.load("/home/hejung/beit-base-patch16-224-pt22k-ft22k/pytorch_model.bin")

w = m['beit.embeddings.patch_embeddings.projection.weight'].cpu().numpy()
b = m['beit.embeddings.patch_embeddings.projection.bias'].cpu().numpy()
model.beit.pemb.proj.set_weights([w.transpose(2, 3, 1, 0), b])

cls = m['beit.embeddings.cls_token'].cpu().numpy()
model.beit.pemb.cls_token.assign(cls)

for i in range(len(model.beit.enc.layers)):
  prefix = 'beit.encoder.layer.{}'.format(i) 
  w = m['{}.layernorm_before.weight'.format(prefix)].cpu().numpy()
  b = m['{}.layernorm_before.bias'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].lnorm.gamma.assign(w)
  model.beit.enc.layers[i].lnorm.beta.assign(b)

  w = m['{}.attention.attention.query.weight'.format(prefix)].cpu().numpy()
  b = m['{}.attention.attention.query.bias'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].atten.self_attn.query.set_weights([w.transpose(1,0), b])

  w = m['{}.attention.attention.key.weight'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].atten.self_attn.key.set_weights([w.transpose(1,0)])

  w = m['{}.attention.attention.value.weight'.format(prefix)].cpu().numpy()
  b = m['{}.attention.attention.value.bias'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].atten.self_attn.value.set_weights([w.transpose(1,0), b])

  w = m['{}.attention.attention.relative_position_bias.relative_position_bias_table'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].atten.self_attn.rel_bias_tbl.assign(w)

  w = m['{}.attention.output.dense.weight'.format(prefix)].cpu().numpy()
  b = m['{}.attention.output.dense.bias'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].atten.out.set_weights([w.transpose(1,0), b])

  w = m['{}.lambda_1'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].lambda_1.assign(w)

  w = m['{}.layernorm_after.weight'.format(prefix)].cpu().numpy()
  b = m['{}.layernorm_after.bias'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].lnorm2.gamma.assign(w)
  model.beit.enc.layers[i].lnorm2.beta.assign(b)

  w = m['{}.intermediate.dense.weight'.format(prefix)].cpu().numpy()
  b = m['{}.intermediate.dense.bias'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].inter.set_weights([w.transpose(1,0), b])

  w = m['{}.output.dense.weight'.format(prefix)].cpu().numpy()
  b = m['{}.output.dense.bias'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].out.set_weights([w.transpose(1,0), b])

  w = m['{}.lambda_2'.format(prefix)].cpu().numpy()
  model.beit.enc.layers[i].lambda_2.assign(w)

ckpt = tf.train.Checkpoint(model)
ckpt.write("beit_base.ckpt")
