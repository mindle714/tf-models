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

if expname.startswith("unisat"):
  import unisat
  if os.path.dirname(unisat.__file__) != expdir:
    sys.exit("unisat is loaded from {}".format(unisat.__file__))
  m = unisat.unisat_unet()

elif expname.startswith("wavlm"):
  import wavlm
  if os.path.dirname(wavlm.__file__) != expdir:
    sys.exit("wavlm is loaded from {}".format(wavlm.__file__))
  m = wavlm.wavlm_unet()

elif expname.startswith("wav2vec2"):
  import wav2vec2
  if os.path.dirname(wav2vec2.__file__) != expdir:
    sys.exit("wav2vec2 is loaded from {}".format(wav2vec2.__file__))
  m = wav2vec2.wav2vec2_unet()

elif expname.startswith("hubert"):
  import hubert
  if os.path.dirname(hubert.__file__) != expdir:
    sys.exit("hubert is loaded from {}".format(hubert.__file__))
  m = hubert.hubert_unet()

elif expname.startswith("data2vec"):
  import data2vec
  if os.path.dirname(data2vec.__file__) != expdir:
    sys.exit("data2vec is loaded from {}".format(data2vec.__file__))
  m = data2vec.data2vec_unet()

else:
  assert False, "undefined model type {}".format(expname)

import numpy as np
_in = np.zeros((1, 32000), dtype=np.float32)
_ = m((_in, None), training=False)

ckpt = tf.train.Checkpoint(m)
ckpt.read(args.ckpt)

import warnings
import soundfile
import librosa
import tqdm
from pesq import pesq
from composite import composite 
from pystoi import stoi

def eval(_pcm, _ref):
  pcm, _ = librosa.load(_pcm, sr=16000)
  ref, _ = librosa.load(_ref, sr=16000)
  pcm_len = pcm.shape[0]

  ref = ref.reshape([1,ref.shape[0],1]).astype(np.float32)
  pcm = np.expand_dims(pcm, 0).astype(np.float32)

  def pad(pcm, mod=32000):
    if pcm.shape[-1] % mod != 0:
      pcm = np.concatenate([pcm, np.zeros((1, mod-pcm.shape[-1]%mod))], -1)
      return pcm

  hyp  = m((pad(pcm), None), training=False)
  hyp = np.squeeze(hyp)[:pcm_len]
  pcm = np.squeeze(pcm)
  ref = np.squeeze(ref)

  return hyp, pcm, ref

resname = "{}-{}".format(expname, epoch)
evals = [e.strip().split() for e in open(args.eval_list, "r").readlines()]

if args.save_result:
  for idx, (_ref, _pcm) in enumerate(evals):
    hyp, pcm, ref = eval(_pcm, _ref)

    soundfile.write("{}_orig_{}.wav".format(resname, idx), pcm, 16000)
    soundfile.write("{}_hyp_{}.wav".format(resname, idx), hyp, 16000)
    soundfile.write("{}_ref_{}.wav".format(resname, idx), ref, 16000)

else:
  with open("{}.eval".format(resname), "w") as f:
    pcount = 0
    pesq_tot = 0; stoi_tot = 0
    csig_tot = 0; cbak_tot = 0
    covl_tot = 0; segsnr_tot = 0

    for idx, (_ref, _pcm) in enumerate(evals):
      hyp, pcm, ref = eval(_pcm, _ref)

      _pesq = pesq(16000, ref, hyp)
      _stoi = stoi(ref, hyp, 16000, extended=False)
      csig, cbak, covl, segsnr = composite(ref, hyp, 16000)

      f.write("{} {} {} {} {} {}\n".format(
        _pesq, _stoi, csig, cbak, covl, segsnr))
      f.flush()

      pesq_tot += _pesq
      stoi_tot += _stoi
      csig_tot += csig
      cbak_tot += cbak
      covl_tot += covl
      segsnr_tot += segsnr

    f.write("final: {} {} {} {} {} {}\n".format(
      pesq_tot / len(evals), stoi_tot / len(evals),
      csig_tot / len(evals), cbak_tot / len(evals),
      covl_tot / len(evals), segsnr_tot / len(evals)))
