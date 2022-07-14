import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
modelname = "unisat_pre_v2"
epoch = 1000

import sys
sys.path.insert(0, os.path.abspath(os.path.join("exps", modelname)))

import unisat
if os.path.dirname(unisat.__file__).split("/")[-1] != modelname:
  sys.exit("unisat is loaded from {}".format(unisat.__file__))

import numpy as np
opt_weight = os.path.join("exps", modelname, "adam-{}-weight.npy".format(epoch))
if not os.path.isfile(opt_weight): 
  sys.exit('adam not exist')
opt_weight = np.load(opt_weight, allow_pickle=True)

opt_cfg = os.path.join("exps", modelname, "adam-{}-config.npy".format(epoch))
opt_cfg = np.load(opt_cfg, allow_pickle=True).flatten()[0]
  
m = unisat.unisat_unet()
_in = np.zeros((1, 32000), dtype=np.float32)
_ = m((_in, None))

ckpt = tf.train.Checkpoint(m)
ckpt.read(os.path.join("exps", modelname, "model-{}.ckpt".format(epoch)))

opt = tf.keras.optimizers.Adam.from_config(opt_cfg)
m_vars = m.trainable_weights
zero_grads = [tf.zeros_like(w) for w in m_vars]
opt.apply_gradients(zip(zero_grads, m_vars))
opt.set_weights(opt_weight)

for var in opt.variables():
  if 'iter' in var.name: continue
  tensor = var.numpy()
#  print(var.name + "{}".format(np.mean(tensor)))
  print(var.name + " {}".format(np.abs(np.mean(tensor))))
