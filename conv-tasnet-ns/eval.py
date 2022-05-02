import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-list", type=str, required=False, default="test.list") 
parser.add_argument("--noise-list", type=str, required=False, default="ns_test.list") 
parser.add_argument("--save-result", action="store_true") 
args = parser.parse_args()

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
_ = m((_in, None), training=False)

ckpt = tf.train.Checkpoint(m)
ckpt.read(args.ckpt)

import warnings
import soundfile
import tqdm

def add_noise(pcm, noise, snr_db):
  if pcm.shape[0] >= noise.shape[0]:
    noise = np.repeat(noise, (pcm.shape[0]//noise.shape[0]+1))
    noise = noise[:pcm.shape[0]]
  else:
    pos = np.random.randint(0, noise.shape[0]-pcm.shape[0]+1)
    noise = noise[pos:pos+pcm.shape[0]]

  pcm_en = np.mean(pcm**2)
  noise_en = np.maximum(np.mean(noise**2), 1e-9)
  snr_en = 10.**(snr_db/10.)

  noise *= np.sqrt(pcm_en / (snr_en * noise_en))
  pcm += noise
  noise_pcm_en = np.maximum(np.mean(pcm**2), 1e-9)
  pcm *= np.sqrt(pcm_en / noise_pcm_en)

  return pcm

evals = [e.strip() for e in open(args.eval_list, "r").readlines()]
noises = [e.strip() for e in open(args.noise_list, "r").readlines()]
 
if len(evals) > len(noises):
  noises = noises * (len(evals)//len(noises)+1)
noises = noises[:len(evals)]

with open("{}-{}.eval2".format(expname, epoch), "w") as f:
  pcount = 0; snr_tot = 0

  for idx, _line in enumerate(evals):
    pcm, _ = soundfile.read(_line)
    ref = np.copy(pcm)

    noise, _ = soundfile.read(noises[idx])
    pcm = add_noise(pcm, noise, 20)

    ref = ref.reshape([1,ref.shape[0],1]).astype(np.float32)
    pcm = np.expand_dims(pcm, 0).astype(np.float32)

    def pad(pcm, mod=8):
      if pcm.shape[-1] % mod != 0:
        pcm = np.concatenate([pcm, np.zeros((1, mod-pcm.shape[-1]%mod))], -1)
        return pcm
      return pcm

    hyp = m((pad(pcm), None), training=False)
    _, sort_hyp = model.si_snr(ref, hyp[:,:ref.shape[1],:])
    if idx < 3:
      soundfile.write("{}_orig_{}.wav".format(expname, idx), np.squeeze(pcm), 16000)
      soundfile.write("{}_hyp_{}.wav".format(expname, idx), np.squeeze(sort_hyp), 16000)
      soundfile.write("{}_ref_{}.wav".format(expname, idx), np.squeeze(ref), 16000)

    snr, _ = model.si_snr(ref, sort_hyp, pit=False)

    #pcm = np.expand_dims(pcm, -1)
    #snr_pcm, _ = model.si_snr(ref, pcm, pit=False)
    #snr = (snr - snr_pcm)

    snr_tot += np.squeeze(snr)
    print("si-snr {} len {}".format(snr, ref.shape[1]))
    f.write("si-snr {} len {}\n".format(snr, ref.shape[1]))
    f.flush()

print("{}-{}\t{}".format(expname, epoch, snr_tot / len(evals)))
