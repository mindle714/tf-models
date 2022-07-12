import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-dir", type=str, required=True)
parser.add_argument("--save-result", action="store_true") 
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

evalname = [e for e in args.eval_dir.split("/") if e][-1]
eval_list = os.path.join(args.eval_dir, "pcm_ref.list")
assert os.path.isfile(eval_list)

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
m = model.waveunet()
_in = np.zeros((1, samp_len), dtype=np.float32)
_ = m((_in, None), training=False)

ckpt = tf.train.Checkpoint(m)
ckpt.read(args.ckpt)

import warnings
import soundfile
import tqdm

resname = "{}-{}-{}".format(evalname, expname, epoch)
evals = [e.strip().split() for e in open(eval_list, "r").readlines()]
with open("{}.eval".format(resname), "w") as f:
  pcount = 0; snr_tot = 0

  for idx, (_pcm, _ref) in enumerate(evals):
    pcm, _ = soundfile.read(_pcm)
    ref, _ = soundfile.read(_ref)
    pcm_len = pcm.shape[0]

    ref = ref.reshape([1,ref.shape[0],1]).astype(np.float32)
    pcm = np.expand_dims(pcm, 0).astype(np.float32)

    def pad(pcm, mod=16384):
      if pcm.shape[-1] % mod != 0:
        pcm = np.concatenate([pcm, np.zeros((1, mod-pcm.shape[-1]%mod))], -1)
        return pcm
      return pcm

    hyp, encs = m((pad(pcm), None), training=False)
    hyp = np.squeeze(hyp)[:pcm_len]

    import librosa
    import librosa.display
    for eidx, enc in enumerate(encs):
      import matplotlib.pyplot as plt
      fig = plt.figure(figsize=(19.2, 4.8))
      librosa.display.specshow(np.log(np.squeeze(enc)+1e-9).T, x_axis='time', y_axis='linear')
      plt.colorbar()
      plt.savefig("{}_hyp_enc{}_{}.png".format(resname, eidx, idx))

    if idx < 2:
      import viz
      if 'ljs' in args.eval_dir: sr = 22050
      else: sr = 16000

      soundfile.write("{}_orig_{}.wav".format(resname, idx), np.squeeze(pcm), sr)
      soundfile.write("{}_hyp_{}.wav".format(resname, idx), hyp, sr)
      soundfile.write("{}_ref_{}.wav".format(resname, idx), np.squeeze(ref), sr)
      viz.plot_spec(np.squeeze(pcm), "{}_orig_{}".format(resname, idx), sr)
      viz.plot_spec(hyp, "{}_hyp_{}".format(resname, idx), sr)
      viz.plot_spec(np.squeeze(ref), "{}_ref_{}".format(resname, idx), sr)
    else:
      break

