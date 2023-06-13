import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-list", type=str, required=False,
  default="/home/hejung/wsj0/8k_tt_min.list") 
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

import warnings
import soundfile
import tqdm

evals = [e.strip() for e in open(args.eval_list, "r").readlines()]

with open("{}-{}.eval".format(expname, epoch), "w") as f:
  pcount = 0; snr_tot = 0

  for idx, _line in enumerate(evals):
    if len(_line.split()) != 3:
      warnings.warn("failed to parse {} at line {}".format(_line, idx))
      continue

    s1, s2, mix = [soundfile.read(e)[0] for e in _line.split()]
    assert s1.shape[0] == s2.shape[0] and s2.shape[0] == mix.shape[0]

    mix = np.expand_dims(mix, 0).astype(np.float32)
    ref = np.concatenate([e[np.newaxis,:,np.newaxis] for e in [s1, s2]], -1)
    ref = ref.astype(np.float32)

    def pad(pcm, mod=8):
      if pcm.shape[-1] % mod != 0:
        pcm = np.concatenate([pcm, np.zeros((1, mod-pcm.shape[-1]%mod))], -1)
        return pcm
      return pcm

    hyp = m((None, None, pad(mix)), training=False)
    _, sort_hyp = model.si_snr(ref, hyp[:,:ref.shape[1],:])

    ref1, ref2 = np.split(ref, 2, -1)
    hyp1, hyp2 = np.split(sort_hyp, 2, -1)

    snr1, _ = model.si_snr(ref1, hyp1, pit=False)
    snr2, _ = model.si_snr(ref2, hyp2, pit=False)

    mix = np.expand_dims(mix, -1)
    snr1_mix, _ = model.si_snr(ref1, mix, pit=False)
    snr2_mix, _ = model.si_snr(ref2, mix, pit=False)
    snr = ((snr1 - snr1_mix) + (snr2 - snr2_mix)) / 2.

    snr_tot += np.squeeze(snr)
    print("si-snr {} len {}".format(snr, ref.shape[1]))
    f.write("si-snr {} len {}\n".format(snr, ref.shape[1]))
    f.flush()
  
    if args.save_result:
      soundfile.write("{}-{}-mix-{}.wav".format(expname, epoch, idx), mix[0,:,0], 8000)
      soundfile.write("{}-{}-s1-{}.wav".format(expname, epoch, idx), hyp1[0,:,0], 8000)
      soundfile.write("{}-{}-s2-{}.wav".format(expname, epoch, idx), hyp2[0,:,0], 8000)

      import librosa
      import librosa.display

      def plot_spec(pcm, suffix, frame_sec=0.02):
        f = librosa.stft(pcm, n_fft=int(8000*frame_sec))
        db = librosa.amplitude_to_db(np.abs(f), ref=np.max)
      
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(19.2, 4.8))
        librosa.display.specshow(db, x_axis='time', y_axis='linear',
          sr=8000, hop_length=int(8000*frame_sec)//4)
        plt.colorbar()
        plt.savefig("{}-{}-{}.png".format(expname, epoch, suffix))

      plot_spec(mix[0,:,0], "mix-{}".format(idx))
      plot_spec(hyp1[0,:,0], "s1-{}".format(idx))
      plot_spec(hyp2[0,:,0], "s2-{}".format(idx))

print("{}-{}\t{}".format(expname, epoch, snr_tot / len(evals)))
