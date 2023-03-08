import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-list", type=str, required=False, 
  default="/data/hejung/voicebank-demand/testset.list")
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

if expname.startswith("tffts"):
  import tffts
  if os.path.dirname(tffts.__file__) != expdir:
    sys.exit("tffts is loaded from {}".format(tffts.__file__))
  m = tffts.tffts_unet()

else:
  assert False

import numpy as np
_in = np.zeros((1, 32000), dtype=np.float32)
_ = m((_in, None), training=False)

ckpt = tf.train.Checkpoint(m)
ckpt.read(args.ckpt)

import warnings
import soundfile
import librosa
import librosa.display

pcm, _ = soundfile.read("clean.wav")
f = librosa.stft(pcm, n_fft=512, hop_length=256)
db = librosa.amplitude_to_db(np.abs(f), ref=np.max)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4.8, 4.8))

ax = fig.add_subplot(4, 1, 1)
librosa.display.specshow(db, x_axis='time', y_axis='linear',
  sr=16000, hop_length=256, ax=ax)
#plt.colorbar()

ax = fig.add_subplot(4, 1, 2)

_in = pcm.reshape([1, -1])
pfft_r, pfft_i = m.tstfts[0](_in)
pdb = librosa.amplitude_to_db(np.sqrt(pfft_r**2 + pfft_i**2), ref=np.max)
pdb = pdb[..., :pdb.shape[-1]//2+1]
pdb = pdb.transpose([0,2,1]).squeeze(0)

librosa.display.specshow(pdb, x_axis='time', y_axis='linear',
  sr=16000, hop_length=256, ax=ax)

ax = fig.add_subplot(4, 1, 3)

_in = pcm.reshape([1, -1])
pfft_r, pfft_i = m.tstfts[1](_in)
pdb = librosa.amplitude_to_db(np.sqrt(pfft_r**2 + pfft_i**2), ref=np.max)
pdb = pdb[..., :pdb.shape[-1]//2+1]
pdb = pdb.transpose([0,2,1]).squeeze(0)

librosa.display.specshow(pdb, x_axis='time', y_axis='linear',
  sr=16000, hop_length=256, ax=ax)

ax = fig.add_subplot(4, 1, 4)

_in = pcm.reshape([1, -1])
pfft_r, pfft_i = m.tstfts[2](_in)
pdb = librosa.amplitude_to_db(np.sqrt(pfft_r**2 + pfft_i**2), ref=np.max)
pdb = pdb[..., :pdb.shape[-1]//2+1]
pdb = pdb.transpose([0,2,1]).squeeze(0)

librosa.display.specshow(pdb, x_axis='time', y_axis='linear',
  sr=16000, hop_length=256, ax=ax)

plt.savefig("pfft.png")
