import soundfile
pcm, _ = soundfile.read("dummy_one/test/noisy/p257_434.wav")
assert _ == 16000

import tensorflow as tf
from cmgan import *

model = cmgan()

import torch
m = torch.load("/home/hejung/CMGAN/src/best_ckpt/ckpt")

import numpy as np
_in = np.reshape(pcm, [1, -1])
_tmp = model(_in)
print(_tmp)

def load_dildense(prefix, _model):
  for i in range(len(_model.convs)):
    w = m['{}.conv{}.weight'.format(prefix, i+1)].cpu().numpy()
    b = m['{}.conv{}.bias'.format(prefix, i+1)].cpu().numpy()
    _model.convs[i].set_weights([w.transpose([2, 3, 1, 0]), b])
  
    w = m['{}.norm{}.weight'.format(prefix, i+1)].cpu().numpy()
    b = m['{}.norm{}.bias'.format(prefix, i+1)].cpu().numpy()
    _model.norms[i].gamma.assign(w)
    _model.norms[i].beta.assign(b)
 
    w = m['{}.prelu{}.weight'.format(prefix, i+1)].cpu().numpy()
    _model.prelus[i].set_weights([w.reshape([1, 1, -1])])

def load_dencoder(prefix, _model):
  w = m['{}.conv_1.0.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv_1.0.bias'.format(prefix)].cpu().numpy()
  _model.conv_1.set_weights([w.transpose([2, 3, 1, 0]), b])

  w = m['{}.conv_1.1.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv_1.1.bias'.format(prefix)].cpu().numpy()
  _model.inorm2d.gamma.assign(w)
  _model.inorm2d.beta.assign(b)

  w = m['{}.conv_1.2.weight'.format(prefix)].cpu().numpy()
  _model.prelu.set_weights([w.reshape([1, 1, -1])])

  load_dildense('{}.dilated_dense'.format(prefix), _model.dildense)

  w = m['{}.conv_2.0.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv_2.0.bias'.format(prefix)].cpu().numpy()
  _model.conv_2.set_weights([w.transpose([2, 3, 1, 0]), b])

  w = m['{}.conv_2.1.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv_2.1.bias'.format(prefix)].cpu().numpy()
  _model.inorm2d_2.gamma.assign(w)
  _model.inorm2d_2.beta.assign(b)

  w = m['{}.conv_2.2.weight'.format(prefix)].cpu().numpy()
  _model.prelu_2.set_weights([w.reshape([1, 1, -1])])

def load_conformer(prefix, _model):
  w = m['{}.ff1.fn.fn.net.0.weight'.format(prefix)].cpu().numpy()
  b = m['{}.ff1.fn.fn.net.0.bias'.format(prefix)].cpu().numpy()
  _model.ff1.ff1.set_weights([w.transpose([1, 0]), b])

  w = m['{}.ff1.fn.fn.net.3.weight'.format(prefix)].cpu().numpy()
  b = m['{}.ff1.fn.fn.net.3.bias'.format(prefix)].cpu().numpy()
  _model.ff1.ff2.set_weights([w.transpose([1, 0]), b])

  w = m['{}.ff1.fn.norm.weight'.format(prefix)].cpu().numpy()
  b = m['{}.ff1.fn.norm.bias'.format(prefix)].cpu().numpy()
  _model.ff1_norm.gamma.assign(w)
  _model.ff1_norm.beta.assign(b)

  w = m['{}.attn.norm.weight'.format(prefix)].cpu().numpy()
  b = m['{}.attn.norm.bias'.format(prefix)].cpu().numpy()
  _model.attn_norm.gamma.assign(w)
  _model.attn_norm.beta.assign(b)

  w = m['{}.attn.fn.to_q.weight'.format(prefix)].cpu().numpy()
  _model.attn.q.set_weights([w.transpose([1, 0])])

  w = m['{}.attn.fn.to_kv.weight'.format(prefix)].cpu().numpy()
  _model.attn.kv.set_weights([w.transpose([1, 0])])

  w = m['{}.attn.fn.rel_pos_emb.weight'.format(prefix)].cpu().numpy()
  _model.attn.rel_pemb.set_weights([w])

  w = m['{}.attn.fn.to_out.weight'.format(prefix)].cpu().numpy()
  b = m['{}.attn.fn.to_out.bias'.format(prefix)].cpu().numpy()
  _model.attn.out.set_weights([w.transpose([1, 0]), b])

  w = m['{}.conv.net.0.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv.net.0.bias'.format(prefix)].cpu().numpy()
  _model.conv.lnorm.gamma.assign(w)
  _model.conv.lnorm.beta.assign(b)

  w = m['{}.conv.net.2.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv.net.2.bias'.format(prefix)].cpu().numpy()
  _model.conv.conv.set_weights([w.transpose([2, 1, 0]), b])

  w = m['{}.conv.net.4.conv.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv.net.4.conv.bias'.format(prefix)].cpu().numpy()
  _model.conv.dconv.set_weights([w.transpose([2, 0, 1]), b])

  w = m['{}.conv.net.5.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv.net.5.bias'.format(prefix)].cpu().numpy()
  rm = m['{}.conv.net.5.running_mean'.format(prefix)].cpu().numpy()
  rv = m['{}.conv.net.5.running_var'.format(prefix)].cpu().numpy()
  _model.conv.bnorm.set_weights([w, b, rm, rv])

  w = m['{}.conv.net.7.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv.net.7.bias'.format(prefix)].cpu().numpy()
  _model.conv.conv_2.set_weights([w.transpose([2, 1, 0]), b])

  w = m['{}.ff2.fn.fn.net.0.weight'.format(prefix)].cpu().numpy()
  b = m['{}.ff2.fn.fn.net.0.bias'.format(prefix)].cpu().numpy()
  _model.ff2.ff1.set_weights([w.transpose([1, 0]), b])

  w = m['{}.ff2.fn.fn.net.3.weight'.format(prefix)].cpu().numpy()
  b = m['{}.ff2.fn.fn.net.3.bias'.format(prefix)].cpu().numpy()
  _model.ff2.ff2.set_weights([w.transpose([1, 0]), b])

  w = m['{}.ff2.fn.norm.weight'.format(prefix)].cpu().numpy()
  b = m['{}.ff2.fn.norm.bias'.format(prefix)].cpu().numpy()
  _model.ff2_norm.gamma.assign(w)
  _model.ff2_norm.beta.assign(b)

  w = m['{}.post_norm.weight'.format(prefix)].cpu().numpy()
  b = m['{}.post_norm.bias'.format(prefix)].cpu().numpy()
  _model.norm.gamma.assign(w)
  _model.norm.beta.assign(b)

def load_tscb(prefix, _model):
  load_conformer('{}.time_conformer'.format(prefix), _model.t_conformer)
  load_conformer('{}.freq_conformer'.format(prefix), _model.f_conformer)

def load_spconvtrans2d(prefix, _model):
  w = m['{}.conv.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv.bias'.format(prefix)].cpu().numpy()
  _model.conv.set_weights([w.transpose([2, 3, 1, 0]), b])

def load_mdecoder(prefix, _model):
  load_dildense('{}.dense_block'.format(prefix), _model.dildense)
  load_spconvtrans2d('{}.sub_pixel'.format(prefix), _model.subpx)

  w = m['{}.conv_1.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv_1.bias'.format(prefix)].cpu().numpy()
  _model.conv.set_weights([w.transpose([2, 3, 1, 0]), b])

  w = m['{}.norm.weight'.format(prefix)].cpu().numpy()
  b = m['{}.norm.bias'.format(prefix)].cpu().numpy()
  _model.norm.gamma.assign(w)
  _model.norm.beta.assign(b)
    
  w = m['{}.prelu.weight'.format(prefix)].cpu().numpy()
  _model.prelu.set_weights([w.reshape([1, 1, -1])])

  w = m['{}.final_conv.weight'.format(prefix)].cpu().numpy()
  b = m['{}.final_conv.bias'.format(prefix)].cpu().numpy()
  _model.conv_2.set_weights([w.transpose([2, 3, 1, 0]), b])
    
  w = m['{}.prelu_out.weight'.format(prefix)].cpu().numpy()
  _model.prelu_2.set_weights([w.reshape([1, -1, 1])])

def load_cdecoder(prefix, _model):
  load_dildense('{}.dense_block'.format(prefix), _model.dildense)
  load_spconvtrans2d('{}.sub_pixel'.format(prefix), _model.subpx)

  w = m['{}.conv.weight'.format(prefix)].cpu().numpy()
  b = m['{}.conv.bias'.format(prefix)].cpu().numpy()
  _model.conv.set_weights([w.transpose([2, 3, 1, 0]), b])

  w = m['{}.norm.weight'.format(prefix)].cpu().numpy()
  b = m['{}.norm.bias'.format(prefix)].cpu().numpy()
  _model.norm.gamma.assign(w)
  _model.norm.beta.assign(b)
    
  w = m['{}.prelu.weight'.format(prefix)].cpu().numpy()
  _model.prelu.set_weights([w.reshape([1, 1, -1])])

def load_tscnet(prefix, _model):
  load_dencoder('dense_encoder', _model.denc)
  for i in range(len(_model.tscbs)):
    load_tscb('TSCB_{}'.format(i+1), _model.tscbs[i])
  load_mdecoder('mask_decoder', _model.mdecoder)
  load_cdecoder('complex_decoder', _model.cdecoder)

load_tscnet('', model.tscnet)

print(model(_in))

'''
hejung@speech:~/tf-models/ssl$ python3 test_cmgan.py
tf.Tensor(
[[-0.00904491 -0.03413181 -0.03042892 ...  0.10910118  0.1264352
   0.14365053]], shape=(1, 33342), dtype=float32)
tf.Tensor(
[[-0.00291189 -0.00660475 -0.00511546 ...  0.00640525  0.00628492
   0.00696982]], shape=(1, 33342), dtype=float32)

ssh t2
pushd ~/CMGAN/src
(venv3.7-cmgan) hejung@speecht2:~/CMGAN/src$ CUDA_VISIBLE_DEVICES=1 python3 -m pdb evaluation.py --test_dir dummy_one/test/ --model_path best_ckpt/ckpt
(Pdb) b 49
(Pdb) c
(Pdb) p est_audio
array([-0.00224336, -0.00505821, -0.00389482, ...,  0.00409852,
        0.0040251 ,  0.00446793], dtype=float32)
(It seems that OLA strategy is different between torch and tensorflow)
'''
