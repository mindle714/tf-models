import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-list", type=str, required=False, 
  default="/data/hejung/wsj0/8k_tt_min.list")
parser.add_argument("--save-result", action="store_true") 
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

assert os.path.isfile(args.eval_list)

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

if expname.startswith("wav2vec2_sep"):
  import wav2vec2_sep
  if os.path.dirname(wav2vec2_sep.__file__) != expdir:
    sys.exit("wav2vec2_sep is loaded from {}".format(wav2vec2_sep.__file__))
  m = wav2vec2_sep.wav2vec2_unet()

elif expname.startswith("data2vec_sep"):
  import data2vec_sep
  if os.path.dirname(data2vec_sep.__file__) != expdir:
    sys.exit("data2vec_sep is loaded from {}".format(data2vec_sep.__file__))
  m = data2vec_sep.data2vec_unet()

elif expname.startswith("tera_sep"):
  import tera_sep
  if os.path.dirname(tera_sep.__file__) != expdir:
    sys.exit("tera_sep is loaded from {}".format(tera_sep.__file__))
  m = tera_sep.tera_unet()

elif expname.startswith("mockingjay_sep"):
  import mockingjay_sep
  if os.path.dirname(mockingjay_sep.__file__) != expdir:
    sys.exit("mockingjay_sep is loaded from {}".format(mockingjay_sep.__file__))
  m = mockingjay_sep.mockingjay_unet()

else:
  import model
  if os.path.dirname(model.__file__) != expdir:
    sys.exit("model is loaded from {}".format(model.__file__))
  m = model.wav2vec2_unet()

import numpy as np
_in = np.zeros((1, 32000), dtype=np.float32)
_ = m((_in, _in, _in), training=False)

ckpt = tf.train.Checkpoint(m)
ckpt.read(args.ckpt)

import warnings
import soundfile
import librosa
import tqdm
from pesq import pesq
from composite import composite 
from pystoi import stoi
from util import si_snr

def eval(_s1, _s2, _mix):
  s1, _ = librosa.load(_s1, sr=None)
  s2, _ = librosa.load(_s2, sr=None)
  mix, _ = librosa.load(_mix, sr=None)
  assert s1.shape[0] == s2.shape[0] and s2.shape[0] == mix.shape[0]

  mix = np.expand_dims(mix, 0).astype(np.float32)
  ref = np.concatenate([e[np.newaxis,:,np.newaxis] for e in [s1, s2]], -1)
  ref = ref.astype(np.float32)

  def pad(pcm, mod=32000):
    if pcm.shape[-1] % mod != 0:
      pcm = np.concatenate([pcm, np.zeros((1, mod-pcm.shape[-1]%mod))], -1)
      return pcm

  hyp = m(pad(mix), training=False)
  _, sort_hyp = si_snr(ref, hyp[:,:ref.shape[1],:])

  ref1, ref2 = np.split(ref, 2, -1)
  hyp1, hyp2 = np.split(sort_hyp, 2, -1)

  snr1, _ = si_snr(ref1, hyp1, pit=False)
  snr2, _ = si_snr(ref2, hyp2, pit=False)

  mix = np.expand_dims(mix, -1)
  snr1_mix, _ = si_snr(ref1, mix, pit=False)
  snr2_mix, _ = si_snr(ref2, mix, pit=False)
  snr = ((snr1 - snr1_mix) + (snr2 - snr2_mix)) / 2.
  snr = np.squeeze(snr)

  return hyp1, hyp2, mix, ref1, ref2, snr

resname = "{}-{}".format(expname, epoch)
evals = [e.strip().split() for e in open(args.eval_list, "r").readlines()]

if args.save_result:
  output = os.path.join("eval_output", expname)
  os.makedirs(output, exist_ok=True)

  for idx, (_s1, _s2, _mix) in enumerate(evals):
    hyp1, hyp2, mix, ref1, ref2, snr = eval(_s1, _s2, _mix)

    soundfile.write(os.path.join(output, "{}-{}-mix-{}.wav".format(expname, epoch, idx)), mix[0,:,0], 8000)
    soundfile.write(os.path.join(output, "{}-{}-s1-{}.wav".format(expname, epoch, idx)), hyp1[0,:,0], 8000)
    soundfile.write(os.path.join(output, "{}-{}-s2-{}.wav".format(expname, epoch, idx)), hyp2[0,:,0], 8000)

else:
  with open(os.path.join("results", "{}.eval".format(resname)), "w") as f:
    pcount = 0; snr_tot = 0

    for idx, (_s1, _s2, _mix) in enumerate(evals):
      hyp1, hyp2, mix, ref1, ref2, snr = eval(_s1, _s2, _mix)

      f.write("{}\n".format(snr))
      f.flush()

      snr_tot += snr

    f.write("final: {}\n".format(snr_tot / len(evals)))
