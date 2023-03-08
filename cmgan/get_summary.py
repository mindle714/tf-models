import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
modelname = "base_ljs_paug_ratio0.5_v12"
epoch = 40000

import sys
sys.path.insert(0, os.path.abspath(os.path.join("exps", modelname)))

import model
if os.path.dirname(model.__file__).split("/")[-1] != modelname:
  sys.exit("model is loaded from {}".format(model.__file__))

import numpy as np
opt_weight = os.path.join("exps", modelname, "adam-{}-weight.npy".format(epoch))
if not os.path.isfile(opt_weight): 
  sys.exit('adam not exist')
opt_weight = np.load(opt_weight, allow_pickle=True)

opt_cfg = os.path.join("exps", modelname, "adam-{}-config.npy".format(epoch))
opt_cfg = np.load(opt_cfg, allow_pickle=True).flatten()[0]
  
m = model.waveunet()
_in = np.zeros((16, 16384), dtype=np.float32)
_ = m((_in, None))

ckpt = tf.train.Checkpoint(m)
ckpt.read(os.path.join("exps", modelname, "model-{}.ckpt".format(epoch)))

kernel = m.down_convs[0][0].conv1.weights[0]
bias = m.down_convs[0][0].conv1.weights[1]

print("{} {}".format(np.mean(kernel), np.var(kernel)))
print("{} {}".format(np.mean(bias), np.var(bias)))
