import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-list", type=str, required=True) 
args = parser.parse_args()

import os
import sys

expdir = os.path.abspath(os.path.dirname(args.ckpt))
sys.path.insert(0, expdir)

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

import warnings
import soundfile
import tqdm

evals = [e.strip() for e in open(args.eval_list, "r").readlines()]
evals = evals[:5]

pcount = 0
for idx, _line in tqdm.tqdm(enumerate(evals), total=len(evals)):
  if len(_line.split()) != 3:
    warnings.warn("failed to parse {} at line {}".format(_line, idx))
    continue

  s1, s2, mix = [soundfile.read(e)[0] for e in _line.split()]
  assert s1.shape[0] == s2.shape[0] and s2.shape[0] == mix.shape[0]
    
  out = m((None, None, np.expand_dims(mix, 0)), training=False)
  soundfile.write("eval-{}-mix.wav".format(idx), mix, 8000)
  soundfile.write("eval-{}-s1.wav".format(idx), out[0,:,0], 8000)
  soundfile.write("eval-{}-s2.wav".format(idx), out[0,:,1], 8000)
