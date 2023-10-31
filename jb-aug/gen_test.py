import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test-list", type=str, required=True) 
parser.add_argument("--noise-list", type=str, required=True) 
parser.add_argument("--snr", type=int, required=True)
parser.add_argument("--samp-rate", type=int, required=False, default=16000)
parser.add_argument("--output", type=str, required=True) 
args = parser.parse_args()

import os
import sys
import json
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR

args_file = os.path.join(args.output, "ARGS")
if os.path.isdir(args.output):
  msg = 'directory {} exists. Do you want to proceed?'.format(args.output)
  cont = input("%s (y/N) " % msg).lower() == 'y'
  if not cont: sys.exit(0)
  if os.path.isfile(args_file):
    os.chmod(args_file, S_IWUSR|S_IREAD)

os.makedirs(args.output, exist_ok=True)
test_list = [e.strip() for e in open(args.test_list).readlines()]
noise_list = [e.strip() for e in open(args.noise_list).readlines()]

import numpy as np
import copy
import librosa
import soundfile
import tqdm

noise_idx = 0
with open(os.path.join(args.output, "test.wav.txt"), "w") as f:

  for bidx in tqdm.tqdm(range(len(test_list))):
    def parse(e):
      pcm_path = e.split()[0] 
      pcm, _ = librosa.load(pcm_path, sr=args.samp_rate)

      txt = e.split()[1:]

      out_path = "_".join(pcm_path.split("/")[-3:]).split(".")[0]

      if not os.path.isfile(os.path.join(args.output, "{}.wav".format(out_path))):
        out_path = os.path.join(args.output, "{}.wav".format(out_path))

      else:
        out_idx = 0
        while True:
          if not os.path.isfile(os.path.join(args.output, "{}_{}.wav".format(out_idx, out_path))):
            out_path = os.path.join(args.output, "{}_{}.wav".format(out_idx, out_path))
            break
          out_idx += 1

      return pcm, txt, out_path

    pcm, txt, out_path = parse(test_list[bidx])

    def sig_pow(e):
      return np.mean(e**2)

    while True:
      noise, _ = librosa.load(noise_list[noise_idx], sr=args.samp_rate)
      noise_idx = (noise_idx + 1) % len(noise_list)
      if sig_pow(noise) != 0: break

    if pcm.shape[0] > noise.shape[0]:
      ns_repeat = pcm.shape[0] // noise.shape[0] + (pcm.shape[0] % noise.shape[0] != 0)
      noise = np.tile(noise, ns_repeat)[:pcm.shape[0]]

    else:
      noise_pos = np.random.randint(0, noise.shape[0] - pcm.shape[0] + 1) 
      noise = noise[noise_pos:noise_pos + pcm.shape[0]]
      assert noise.shape[0] == pcm.shape[0]

    scale = np.sqrt(sig_pow(pcm) / (np.power(10, args.snr / 10) * sig_pow(noise)))
    pcm_noise = pcm + scale * noise
    soundfile.write(out_path, pcm_noise, args.samp_rate)

    f.write("{} {}\n".format(os.path.abspath(out_path), " ".join(txt)))

with open(args_file, "w") as f:
  f.write(" ".join([sys.executable] + sys.argv))
  f.write("\n")
  f.write(json.dumps(vars(args)))
os.chmod(args_file, S_IREAD|S_IRGRP|S_IROTH)

import types
specs = [val.__spec__ for name, val in sys.modules.items() \
  if isinstance(val, types.ModuleType) and not ('_main_' in name)]
origins = [spec.origin for spec in specs if spec is not None]
origins = [e for e in origins if e is not None and os.getcwd() in e]

import shutil
for origin in origins + [os.path.abspath(__file__), args.test_list]:
  shutil.copy(origin, args.output, follow_symlinks=True)
