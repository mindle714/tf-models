from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

import numpy as np
image = np.array(image.resize((224, 224), resample=2)).astype(np.float32) / 255.
#image = np.array(image).astype(np.float32) / 255.

mean = np.array([0.5, 0.5, 0.5]); std = np.array([0.5, 0.5, 0.5])
image = (image - mean[None, None, :]) / std[None, None, :]

from beit import *

model = beit_seq()

_in = np.expand_dims(image, 0)
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

print(model(_in)[-1])
print(_in.shape)
print(model(_in)[-1].shape)

'''
hejung@speech:~/tf-models/ssl/test$ python3 test_beit.py
tf.Tensor(
[[[ 8.87673    -3.9018948   2.3814676  ...  1.8723376  -0.50816524
   15.941325  ]
  [-3.6175113  -1.3950799   1.3528472  ...  1.5904075  12.46461
    5.2512197 ]
  [-0.3781953   1.4861003   1.8029202  ... -1.6616406  10.191763
    3.3016973 ]
  ...
  [-5.682966   -4.769623    2.2346373  ...  9.273927    6.7740855
    5.452335  ]
  [-5.9637523  -2.441523    1.4188249  ...  5.2562013  10.248429
    0.8246787 ]
  [ 3.015912    3.96977    -0.93758416 ...  5.199343    1.1881006
   -0.29633462]]], shape=(1, 197, 768), dtype=float32)

(venv3.10) hejung@speech:~/beit-base-patch16-224-pt22k-ft22k$ python3 -m pdb test.py
(Pdb) b /home/hejung/venv3.10/lib/python3.10/site-packages/transformers-4.21.0.dev0-py3.10.egg/transformers/models/beit/modeling_beit.py:538
(Pdb) c
(Pdb) p hidden_states
tensor([[[ 8.8768, -3.9019,  2.3815,  ...,  1.8724, -0.5082, 15.9413],
         [-3.6175, -1.3951,  1.3529,  ...,  1.5905, 12.4646,  5.2512],
         [-0.3782,  1.4861,  1.8030,  ..., -1.6616, 10.1916,  3.3017],
         ...,
         [-5.6831, -4.7696,  2.2346,  ...,  9.2740,  6.7741,  5.4523],
         [-5.9638, -2.4415,  1.4188,  ...,  5.2562, 10.2484,  0.8247],
         [ 3.0160,  3.9697, -0.9376,  ...,  5.1993,  1.1881, -0.2963]]],
       grad_fn=<AddBackward0>)
'''
