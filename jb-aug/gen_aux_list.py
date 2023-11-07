import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-list", type=str, required=True) 
parser.add_argument("--samp-rate", type=int, required=False, default=16000)
parser.add_argument("--noise-list", type=str, required=True)
parser.add_argument("--min-snr", type=int, required=False, default=10)
parser.add_argument("--max-snr", type=int, required=False, default=10)
parser.add_argument("--output", type=str, required=True) 
args = parser.parse_args()

import os
import sys
import json
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

args_file = os.path.join(args.output, "ARGS")
if os.path.isdir(args.output):
  msg = 'directory {} exists. Do you want to proceed?'.format(args.output)
  cont = input("%s (y/N) " % msg).lower() == 'y'
  if not cont: sys.exit(0)
  if os.path.isfile(args_file):
    os.chmod(args_file, S_IWUSR|S_IREAD)

os.makedirs(os.path.join(args.output, "noise"), exist_ok=True)
train_list = [e.strip() for e in open(args.train_list).readlines()]

import random

noise_list = [e.strip() for e in open(args.noise_list).readlines()]
if len(noise_list) > len(train_list):
  random.shuffle(noise_list)
  noise_list = noise_list[:len(train_list)]
elif len(noise_list) < len(train_list):
  rep = len(train_list) // len(noise_list)
  noise_list = noise_list * (rep + 1)
  random.shuffle(noise_list)
  noise_list = noise_list[:len(train_list)]
else:
  print("Same noise, train list; use as it is")

assert len(train_list) == len(noise_list)

import warnings
import numpy as np
import copy
import librosa
import soundfile

snrs = []; ns = []
for idx, (e, n) in enumerate(zip(train_list, noise_list)):
  pcm_path = e.split()[0] 
  pcm, _ = librosa.load(pcm_path, sr=args.samp_rate)
    
  snr_db = np.random.uniform(args.min_snr, args.max_snr)
  snrs.append(snr_db)

  noise, _ = librosa.load(n, sr=args.samp_rate)

  if pcm.shape[0] > noise.shape[0]:
    ns_repeat = pcm.shape[0] // noise.shape[0] + int(pcm.shape[0] % noise.shape[0] != 0)
    noise = np.tile(noise, ns_repeat)[:pcm.shape[0]]

  else:
    noise_pos = np.random.randint(0, noise.shape[0] - pcm.shape[0] + 1)
    noise = noise[noise_pos:noise_pos + pcm.shape[0]]
    
  assert noise.shape[0] == pcm.shape[0]
  noise_path = os.path.join(args.output, "noise", "{}.wav".format(idx))
  soundfile.write(noise_path, noise, args.samp_rate)
  ns.append(os.path.abspath(noise_path))

with open(os.path.join(args.output, "noise.list"), "w") as f:
  for snr, n in zip(snrs, ns):
    f.write("{} {}\n".format(snr, n))

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
for origin in origins + [os.path.abspath(__file__), args.train_list]:
  shutil.copy(origin, args.output, follow_symlinks=True)
