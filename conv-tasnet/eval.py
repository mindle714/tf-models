import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-list", type=str, required=False,
  default="/home/speech/wsj0/8k_tt_min.list") 
args = parser.parse_args()

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

import warnings
import soundfile
import tqdm

evals = [e.strip() for e in open(args.eval_list, "r").readlines()]

pcount = 0; snr_tot = 0
#for idx, _line in tqdm.tqdm(enumerate(evals), total=len(evals)):
for idx, _line in enumerate(evals):
  if len(_line.split()) != 3:
    warnings.warn("failed to parse {} at line {}".format(_line, idx))
    continue

  s1, s2, mix = [soundfile.read(e)[0] for e in _line.split()]
  assert s1.shape[0] == s2.shape[0] and s2.shape[0] == mix.shape[0]

  mix = np.expand_dims(mix, 0).astype(np.float32)
  ref = np.concatenate([e[np.newaxis,:,np.newaxis] for e in [s1, s2]], -1)
  ref = ref.astype(np.float32)

  def pad(pcm, mod=512):
    if pcm.shape[-1] % mod != 0:
      pcm = np.concatenate([pcm, np.zeros((1, mod-pcm.shape[-1]%mod))], -1)
      return pcm
    return pcm

  hyp = m((None, None, pad(mix)), training=False)
  _, sort_hyp = model.si_snr(ref, hyp[:,:ref.shape[1],:])

  ref1, ref2 = np.split(ref, 2, -1)
  hyp1, hyp2 = np.split(sort_hyp, 2, -1)

  snr1, _ = model.si_snr(ref1, hyp1, False)
  snr2, _ = model.si_snr(ref2, hyp2, False)

  mix = np.expand_dims(mix, -1)
  snr1_mix, _ = model.si_snr(ref1, mix, False)
  snr2_mix, _ = model.si_snr(ref2, mix, False)
  snr = ((snr1 - snr1_mix) + (snr2 - snr2_mix)) / 2.

  snr_tot += np.squeeze(snr)

print("{}-{}\t{}".format(expname, epoch, snr_tot / len(evals)))
