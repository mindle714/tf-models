import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-list", type=str, required=False,
  default="/home/speech/wsj0/8k_tt_min.list") 
parser.add_argument("--save-result", action="store_true") 
args = parser.parse_args()

import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

expdir = os.path.abspath(os.path.dirname(args.ckpt))
sys.path.insert(0, expdir)
expname = expdir.split("/")[-1]
epoch = os.path.basename(args.ckpt).replace(".", "-").split("-")[1]

import model
if os.path.dirname(model.__file__) != expdir:
  sys.exit("model is loaded from {}".format(model.__file__))

import json
exp_args = os.path.join(expdir, "ARGS")
with open(exp_args, "r") as f:
  jargs = json.loads(f.readlines()[-1])

  with open(os.path.join(jargs["tfrec"], "ARGS")) as f2:
    jargs2 = json.loads(f2.readlines()[-1])
    samp_len = int(jargs2["samp_len"])

import numpy as np
m = model.convtas()
_in = np.zeros((1, samp_len), dtype=np.float32)
_ = m((None, None, _in), training=False)

import tensorflow as tf
ckpt = tf.train.Checkpoint(m)
ckpt.read(args.ckpt)
    
opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
flops = tf.compat.v1.profiler.profile(tf.compat.v1.get_default_graph(), options=opts).total_float_ops
print(flops)

