import tensorflow as tf
from tera import *
assert 'compat' not in tf.__file__

import torch
m_wav2vec2 = torch.load("/home/hejung/wav2vec2-base/pytorch_model.bin")

# ModuleNotFoundError: No module named 'optimizers' occurs without below
import sys
import s3prl.optimizers
assert "optimizers" not in sys.modules
sys.modules["optimizers"] = s3prl.optimizers
m_tera = torch.load("/home/hejung/tera_960hr/states-1000000.ckpt")

model = tera_unet()

import numpy as np
pcm = np.zeros(16000)
_in = np.reshape(pcm, [1, -1])
_tmp = model(_in)
_tmp = model.tera(_in)
_tmp = model.wav2vec2(_in.reshape([1, -1, 1]))

def load_tera(m, mdl):
  w = m['Transformer']['input_representations.spec_transform.weight'].cpu().numpy()
  b = m['Transformer']['input_representations.spec_transform.bias'].cpu().numpy()
  mdl.fe.spec_transform.set_weights([w.transpose(1,0), b])

  w = m['Transformer']['input_representations.LayerNorm.weight'].cpu().numpy()
  b = m['Transformer']['input_representations.LayerNorm.bias'].cpu().numpy()
  mdl.fe.lnorm.gamma.assign(w)
  mdl.fe.lnorm.beta.assign(b)

  for i in range(3):
    prefix = 'encoder.layer.{}'.format(i)
    w = m['Transformer'][prefix + '.attention.self.query.weight'].cpu().numpy()
    b = m['Transformer'][prefix + '.attention.self.query.bias'].cpu().numpy()
    mdl.enc.layers[i].atten.self_attn.query.set_weights([w.transpose(1,0), b])

    w = m['Transformer'][prefix + '.attention.self.key.weight'].cpu().numpy()
    b = m['Transformer'][prefix + '.attention.self.key.bias'].cpu().numpy()
    mdl.enc.layers[i].atten.self_attn.key.set_weights([w.transpose(1,0), b])

    w = m['Transformer'][prefix + '.attention.self.value.weight'].cpu().numpy()
    b = m['Transformer'][prefix + '.attention.self.value.bias'].cpu().numpy()
    mdl.enc.layers[i].atten.self_attn.value.set_weights([w.transpose(1,0), b])

    w = m['Transformer'][prefix + '.attention.output.dense.weight'].cpu().numpy()
    b = m['Transformer'][prefix + '.attention.output.dense.bias'].cpu().numpy()
    mdl.enc.layers[i].atten.out.set_weights([w.transpose(1,0), b])

    w = m['Transformer'][prefix + '.attention.output.LayerNorm.weight'].cpu().numpy()
    b = m['Transformer'][prefix + '.attention.output.LayerNorm.bias'].cpu().numpy()
    mdl.enc.layers[i].atten.lnorm.gamma.assign(w)
    mdl.enc.layers[i].atten.lnorm.beta.assign(b)

    w = m['Transformer'][prefix + '.intermediate.dense.weight'].cpu().numpy()
    b = m['Transformer'][prefix + '.intermediate.dense.bias'].cpu().numpy()
    mdl.enc.layers[i].inter.set_weights([w.transpose(1,0), b])

    w = m['Transformer'][prefix + '.output.dense.weight'].cpu().numpy()
    b = m['Transformer'][prefix + '.output.dense.bias'].cpu().numpy()
    mdl.enc.layers[i].out.set_weights([w.transpose(1,0), b])

    w = m['Transformer'][prefix + '.output.LayerNorm.weight'].cpu().numpy()
    b = m['Transformer'][prefix + '.output.LayerNorm.bias'].cpu().numpy()
    mdl.enc.layers[i].lnorm.gamma.assign(w)
    mdl.enc.layers[i].lnorm.beta.assign(b)

def load_wav2vec2(m, mdl):
  def load_norm(prefix, e):
    w = m['{}.weight'.format(prefix, i)].cpu().numpy()
    b = m['{}.bias'.format(prefix, i)].cpu().numpy()
    e.gamma.assign(w)
    e.beta.assign(b)
  
  def load_affine(prefix, e):
    w = m['{}.weight'.format(prefix)]
    bname = '{}.bias'.format(prefix)
    if bname in m:
      b = m[bname]
      e.set_weights([w.transpose(1,0).cpu().numpy(), b.cpu().numpy()])
    else:
      e.set_weights([w.transpose(1,0).cpu().numpy()])
  
  def load_conv(prefix, e):
    w = m['{}.weight'.format(prefix)]
    bname = '{}.bias'.format(prefix)
    if bname in m:
      b = m[bname]
      e.set_weights([w.transpose(2,0).cpu().numpy(), b.cpu().numpy()])
    else:
      e.set_weights([w.transpose(2,0).cpu().numpy()])
  
  for i, conv in enumerate(mdl.fe.conv_layers):
    prefix = 'wav2vec2.feature_extractor.conv_layers'
    load_conv('{}.{}.conv'.format(prefix, i), conv.conv)
    if i == 0:
      load_norm('{}.{}.layer_norm'.format(prefix, i), conv.norm)
  
  '''
  prefix = 'wav2vec2.feature_projection'
  load_norm('{}.layer_norm'.format(prefix), mdl.fp.norm)
  load_affine('{}.projection'.format(prefix), mdl.fp.proj)
  
  prefix = 'wav2vec2.encoder'
  w_g = m['{}.pos_conv_embed.conv.weight_g'.format(prefix)].cpu().numpy()
  w_g = np.reshape(w_g, [-1, 1, 1])
  w_v = m['{}.pos_conv_embed.conv.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
  w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
  b = m['{}.pos_conv_embed.conv.bias'.format(prefix)].cpu().numpy()
  mdl.enc.emb.conv.set_weights([w, b])
  load_norm('{}.layer_norm'.format(prefix), mdl.enc.norm)
  
  for i, layer in enumerate(mdl.enc.layers):
    prefix = 'wav2vec2.encoder.layers.{}'.format(i)
    load_affine('{}.attention.q_proj'.format(prefix), layer.atten.q_proj)
    load_affine('{}.attention.k_proj'.format(prefix), layer.atten.k_proj)
    load_affine('{}.attention.v_proj'.format(prefix), layer.atten.v_proj)
    load_affine('{}.attention.out_proj'.format(prefix), layer.atten.out_proj)
   
    load_affine('{}.feed_forward.intermediate_dense'.format(prefix), layer.feed.in_dense)
    load_affine('{}.feed_forward.output_dense'.format(prefix), layer.feed.out_dense)
    
    load_norm('{}.layer_norm'.format(prefix), layer.norm)
    load_norm('{}.final_layer_norm'.format(prefix), layer.out_norm)
  '''

load_tera(m_tera, model.tera_gen)
load_tera(m_tera, model.tera)
load_wav2vec2(m_wav2vec2, model.wav2vec2)

ckpt = tf.train.Checkpoint(model)
ckpt.write("tera_wav2vec2_mfloss_base.ckpt")
