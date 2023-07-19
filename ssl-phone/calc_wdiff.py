import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, required=True) 
args = parser.parse_args()

import os
from os.path import join
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import sys
expdir = os.path.abspath(args.exp)
sys.path.insert(0, expdir)
expname = expdir.split("/")[-1]

traincmd = open(os.path.join(expdir, "ARGS"), "r").readlines()[0].strip()
timit = ("--timit" in traincmd)
import json
_json = open(os.path.join(expdir, "ARGS"), "r").readlines()[-1].strip()
_json = json.loads(_json)
optim_ckpt = _json["warm_start"]

if os.path.exists(join(expdir, "tera.py")):
  import tera
  if os.path.dirname(tera.__file__) != expdir:
    sys.exit("tera is loaded from {}".format(tera.__file__))
  if timit:
    m = tera.tera_phone(num_class=50)
    m_frozen = tera.tera_phone(num_class=50)
  else:
    m = tera.tera_phone()
    m_frozen = tera.tera_phone()
  is_tera = True

elif os.path.exists(join(expdir, "wav2vec2.py")):
  import wav2vec2
  if os.path.dirname(wav2vec2.__file__) != expdir:
    sys.exit("wav2vec2 is loaded from {}".format(wav2vec2.__file__))
  if timit:
    m = wav2vec2.wav2vec2_phone(num_class=50)
    m_frozen = wav2vec2.wav2vec2_phone(num_class=50)
  else:
    m = wav2vec2.wav2vec2_phone()
    m_frozen = wav2vec2.wav2vec2_phone()
  is_tera = False

else:
  assert False, "Invalid experiment path {}".format(expdir)

import numpy as np
if is_tera:
  if timit:
    _in = np.zeros((1, 801, 80), dtype=np.float32)
  else:
    _in = np.zeros((1, 1701, 80), dtype=np.float32)
else:
  if timit:
    _in = np.zeros((1, 128000), dtype=np.float32)
  else:
    _in = np.zeros((1, 272000), dtype=np.float32)
_ = m(_in, training=False)
_ = m_frozen(_in, training=False)

ckpt_frozen = tf.train.Checkpoint(m_frozen)
ckpt_frozen.read(optim_ckpt)

opt_weights = m_frozen.trainable_weights
opt_weights_exc = [e for e in m_frozen.trainable_weights if 'tera_phone/dense' not in e.name]

import glob
ckpt_paths = glob.glob(os.path.join(expdir, "model-*.ckpt.index"))

results = []
for ckpt_path in ckpt_paths:
  ckpt_path = ckpt_path.replace(".index", "")
  epoch = int(ckpt_path.split("/")[-1].replace("model-", "").replace(".ckpt", ""))

  ckpt = tf.train.Checkpoint(m)
  ckpt.read(ckpt_path)

  weights = m.trainable_weights
  weights_exc = [e for e in m.trainable_weights if 'tera_phone/dense' not in e.name]

  diff = 0
  for var, var_f in zip(weights, opt_weights): 
    diff += tf.math.reduce_sum((var - var_f)**2)
  diff = tf.math.sqrt(diff)

  diff_exc = 0
  for var, var_f in zip(weights_exc, opt_weights_exc): 
    diff_exc += tf.math.reduce_sum((var - var_f)**2)
  diff_exc = tf.math.sqrt(diff_exc)

  results.append((epoch, "{} {} {}".format(ckpt_path.split("/")[-1], diff, diff_exc)))

results = sorted(results, key=lambda e: e[0])
results = [e[1] for e in results]
print(results)
