import soundfile
pcm, _ = soundfile.read("/home/hejung/s3prl/s3prl/downstream/speech_commands/dummy_data/train/yes/01d22d03_nohash_0.wav")

import tensorflow as tf

import torch
# ModuleNotFoundError: No module named 'optimizers' occurs without below
import sys
import s3prl.optimizers
sys.modules["optimizers"] = s3prl.optimizers
m = torch.load("/home/hejung/tera_960hr/states-1000000.ckpt")

from tera import *

model = tera_seq()

import numpy as np
_in = np.reshape(pcm, [1, -1])
_tmp = model(_in)

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

print(model(_in)[-1])

'''
(venv3.7-tera) hejung@speech:~/tf-models/ssl$ python3 test_tera.py
tf.Tensor(
[[[-0.11399778 -0.16911498  0.12274108 ...  0.02805673 -0.15087901
   -0.2363666 ]
  [-0.09442227 -0.2462019   0.11173932 ...  0.01314869 -0.0904882
   -0.29043987]
  [-0.06143013 -0.14151919  0.1611378  ...  0.00757908 -0.17190604
   -0.190993  ]
  ...
  [-0.21367435 -0.14971769  0.00995845 ... -0.00646792 -0.31736594
   -0.3090821 ]
  [-0.179086   -0.15870209  0.05857538 ... -0.04198653  0.13603775
   -0.28368437]
  [-0.2914045  -0.07896738  0.02597807 ... -0.04358101  0.46148384
   -0.15250033]]], shape=(1, 101, 768), dtype=float32)

pushd ~/s3prl/s3prl
python3 -m pdb run_downstream.py -m train -n ExpName -u tera -d speech_commands
(Pdb) b /home/hejung/venv3.7-tera/lib/python3.7/site-packages/s3prl/upstream/mockingjay/model.py:348
(Pdb) c
(Pdb) c
(Pdb) c
[Featurizer] - Take a list of 4 features and weighted sum them.
[Featurizer] - The selected feature hidden_states's downsample rate is 160
overall:   0%|                                                     | 0/200000
(Pdb) p all_encoder_layers[-1]
tensor([[[-0.1140, -0.1691,  0.1227,  ...,  0.0281, -0.1509, -0.2364],
         [-0.0944, -0.2462,  0.1117,  ...,  0.0132, -0.0905, -0.2905],
         [-0.0614, -0.1415,  0.1611,  ...,  0.0076, -0.1719, -0.1910],
         ...,
         [-0.2137, -0.1497,  0.0100,  ..., -0.0065, -0.3174, -0.3091],
         [-0.1791, -0.1587,  0.0586,  ..., -0.0420,  0.1360, -0.2837],
         [-0.2914, -0.0790,  0.0260,  ..., -0.0436,  0.4615, -0.1525]]],
       device='cuda:0')
'''
