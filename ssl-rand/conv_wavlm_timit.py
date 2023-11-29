import tensorflow as tf

import torch
m = torch.load("/home/hejung/wavlm-base-unilm/WavLM-Base.pt")
m = m['model']

from wavlm import *

model = wavlm_phone(num_class=50)

import numpy as np
pcm = np.zeros(128000)
_in = np.reshape(pcm, [1, -1])
_ref = np.zeros([1, 399], dtype=np.int32)
_tmp = model((_in, _ref, np.ones([1, 1])*128000, np.ones([1, 1])*399), ssl_loss=True)

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

for i, conv in enumerate(model.wavlm.wavlm.fe.conv_layers):
  prefix = 'feature_extractor.conv_layers'
  load_conv('{}.{}.0'.format(prefix, i), conv.conv)
  if i == 0:
    load_norm('{}.{}.2'.format(prefix, i), conv.norm)

load_norm('layer_norm', model.wavlm.wavlm.fp.norm)
load_affine('post_extract_proj', model.wavlm.wavlm.fp.proj)

prefix = 'encoder'
w_g = m['{}.pos_conv.0.weight_g'.format(prefix)].cpu().numpy()
w_g = np.reshape(w_g, [-1, 1, 1])
w_v = m['{}.pos_conv.0.weight_v'.format(prefix)].transpose(2,0).cpu().numpy()
w = tf.nn.l2_normalize(w_v, axis=[1,2]) * w_g
b = m['{}.pos_conv.0.bias'.format(prefix)].cpu().numpy()
model.wavlm.wavlm.enc.emb.conv.set_weights([w, b])

load_norm('{}.layer_norm'.format(prefix), model.wavlm.wavlm.enc.norm)

w = m['encoder.layers.0.self_attn.relative_attention_bias.weight'].cpu().numpy().T
model.wavlm.wavlm.enc.rel_bias.assign(w)

for i, layer in enumerate(model.wavlm.wavlm.enc.layers):
  prefix = 'encoder.layers.{}'.format(i)
  load_affine('{}.self_attn.q_proj'.format(prefix), layer.atten.q_proj)
  load_affine('{}.self_attn.k_proj'.format(prefix), layer.atten.k_proj)
  load_affine('{}.self_attn.v_proj'.format(prefix), layer.atten.v_proj)
  load_affine('{}.self_attn.out_proj'.format(prefix), layer.atten.out_proj)
 
  load_affine('{}.fc1'.format(prefix), layer.feed.in_dense)
  load_affine('{}.fc2'.format(prefix), layer.feed.out_dense)
  
  load_norm('{}.self_attn_layer_norm'.format(prefix), layer.norm)
  load_norm('{}.final_layer_norm'.format(prefix), layer.out_norm)

  load_affine('{}.self_attn.grep_linear'.format(prefix), layer.atten.grep_linear)
  w = m['{}.self_attn.grep_a'.format(prefix)].cpu().numpy().squeeze(0)
  layer.atten.grep_a.assign(w)

w = m['mask_emb'].cpu().numpy()
model.wavlm.wavlm.masked_spec_embed.assign(w)

# instead of loading final_proj and label_embs,
# random initialize and train
import os
import glob
tfrec = "timit_hb_feat_v2/"
tfrec_list = glob.glob(os.path.join(tfrec, "train-*.tfrecord"))

def parse_func(pcm_len, txt_len, idx_len):
  desc = {
    'pcm': tf.io.FixedLenFeature([pcm_len], tf.float32),
    'pcm_len': tf.io.FixedLenFeature([1], tf.int64),
  }
  desc['txt'] = tf.io.FixedLenFeature([txt_len], tf.int64)
  desc['txt_len'] = tf.io.FixedLenFeature([1], tf.int64)
  desc['hb_idx'] = tf.io.FixedLenFeature([idx_len], tf.int64)
  desc['hb_idx_len'] = tf.io.FixedLenFeature([1], tf.int64)

  def _parse_func(ex):
    e = tf.io.parse_single_example(ex, desc)
    return e
  return _parse_func

def gen_lth(tfrec_list, _len, txt_len, idx_len,
            batch_size = 16, seed = 1234):
  dataset = tf.data.TFRecordDataset(tfrec_list)
  dataset = dataset.shuffle(batch_size*100, seed=seed, reshuffle_each_iteration=True)
  dataset = dataset.repeat()

  dataset = dataset.map(parse_func(_len, txt_len, idx_len),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.batch(batch_size,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset
  
lth_dataset = gen_lth(tfrec_list, 
  128000, 399, 399,
  batch_size=4, seed=1234)
  
opt_lth = tf.keras.optimizers.SGD(learning_rate=0.001)

weights = model.trainable_weights
weights = [e for e in weights if 'final_proj' in e.name or 'labels_embs' in e.name]

assert len(weights) == 3
accum_step = 4
accum_grads = [tf.Variable(tf.zeros_like(e)) for e in model.trainable_weights \
        if 'final_proj' in e.name or 'labels_embs' in e.name]

@tf.function
def run_lth_step(step, pcm, hb_idx,
                 samp_len, hb_idx_len,
                 accum=False, neg=False):
  with tf.GradientTape() as tape:
    tape.watch(weights)
    sloss = model(
      (pcm, hb_idx, samp_len, hb_idx_len), ssl_loss=True)
    
    sloss = tf.math.reduce_mean(sloss)
    if neg: neg_sloss = -sloss
    else: neg_sloss = sloss
    
  grads = tape.gradient(neg_sloss, weights)
  grads, _ = tf.clip_by_global_norm(grads, 5.)
    
  for idx, grad in enumerate(grads):
    if grad is None: continue
    accum_grads[idx].assign_add(grad)

  if not accum:
    for idx, grad in enumerate(grads):
      if grad is None: continue
      accum_grads[idx].assign(accum_grads[idx]/accum_step)

    opt_lth.apply_gradients(zip(accum_grads, weights))
            
    for idx, grad in enumerate(grads):
      if grad is None: continue
      accum_grads[idx].assign(tf.zeros_like(grad))

  return sloss
  
for idx, data in enumerate(lth_dataset):
    if idx > 10000: break

    _in_arg = [data["pcm"], data["hb_idx"],
               data["pcm_len"], data["hb_idx_len"]]

    loss = run_lth_step(
      tf.cast(idx, tf.int64), *_in_arg, accum = not (idx % accum_step == 0))

    if idx % 100 == 0:
      print("lth-gstep[{}] loss[{}]".format(idx, loss))

ckpt = tf.train.Checkpoint(model)
ckpt.write("wavlm_timit_v2.ckpt")
