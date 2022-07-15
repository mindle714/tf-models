import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-list", type=str, required=False, default="/home/hejung/voicebank-demand/testset.list")
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

import unisat
if os.path.dirname(unisat.__file__) != expdir:
  sys.exit("unisat is loaded from {}".format(unisat.__file__))

import numpy as np
m = unisat.unisat_unet()
_in = np.zeros((1, 32000), dtype=np.float32)
_ = m((_in, None), training=False)

ckpt = tf.train.Checkpoint(m)
ckpt.read(args.ckpt)

import warnings
import soundfile
import librosa
import tqdm
from pesq import pesq
from pystoi import stoi

resname = "{}-{}".format(expname, epoch)
evals = [e.strip().split() for e in open(args.eval_list, "r").readlines()]
with open("{}.eval".format(resname), "w") as f:
  pcount = 0; pesq_tot = 0; stoi_tot = 0

  for idx, (_ref, _pcm) in enumerate(evals):
    pcm, _ = librosa.load(_pcm, sr=16000)
    ref, _ = librosa.load(_ref, sr=16000)
    pcm_len = pcm.shape[0]

    ref = ref.reshape([1,ref.shape[0],1]).astype(np.float32)
    pcm = np.expand_dims(pcm, 0).astype(np.float32)

    def pad(pcm, mod=32000):
      if pcm.shape[-1] % mod != 0:
        pcm = np.concatenate([pcm, np.zeros((1, mod-pcm.shape[-1]%mod))], -1)
        return pcm
#      return pcm

    hyp  = m((pad(pcm), None), training=False)
    hyp = np.squeeze(hyp)[:pcm_len]
    pcm = np.squeeze(pcm)
    ref = np.squeeze(ref)

    _pesq = pesq(16000, ref, hyp)
    _stoi = stoi(ref, hyp, 16000, extended=False)
    #print("{}".format(_pesq))
    f.write("{} {}\n".format(_pesq, _stoi))
    pesq_tot += _pesq
    stoi_tot += _stoi

    if args.save_result:
      soundfile.write("{}_orig_{}.wav".format(resname, idx), pcm, 16000)
      soundfile.write("{}_hyp_{}.wav".format(resname, idx), hyp, 16000)
      soundfile.write("{}_ref_{}.wav".format(resname, idx), ref, 16000)

  #print("final: {}".format(pesq_tot / len(evals)))
  f.write("final: {} {}\n".format(pesq_tot / len(evals), stoi_tot / len(evals)))
