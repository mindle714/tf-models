import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test-list", type=str, required=True) 
parser.add_argument("--noise-list", nargs="+", default=[]) 
parser.add_argument("--noise-ratio", type=float, required=False, default=0.)
parser.add_argument("--min-snr", type=int, required=False, default=0)
parser.add_argument("--max-snr", type=int, required=False, default=10)
parser.add_argument("--lpf-ratio", type=float, required=False, default=0.)
parser.add_argument("--lpf-min-thres", type=float, required=False, default=0.)
parser.add_argument("--lpf-max-thres", type=float, required=False, default=0.)
parser.add_argument("--paug-ratio", type=float, required=False, default=0.)
parser.add_argument("--paug-min-freq", type=float, required=False, default=0.)
parser.add_argument("--paug-max-freq", type=float, required=False, default=1.)
parser.add_argument("--hop-len", type=int, required=False, default=16000)
parser.add_argument("--no-remainder", action='store_true')
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--apply-jointb", action='store_true') 
args = parser.parse_args()

if args.noise_ratio < 0. or args.noise_ratio > 1.: sys.exit(0)
if args.lpf_ratio < 0. or args.lpf_ratio > 1.: sys.exit(0)
if args.paug_ratio < 0. or args.paug_ratio > 1.: sys.exit(0)

if args.apply_jointb:
  import jointbilatFil
  import librosa

  def magnitude(e): return np.abs(e)
  def phase(e): return np.arctan2(e.imag, e.real)
  def polar(mag, phase):
    return mag * (np.cos(phase) + np.sin(phase) * 1j)

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

if len(args.noise_list) > 0:
  noise_list = []
  for f in args.noise_list:
    noise_list += [e.strip() for e in open(f).readlines()]

  import random
  random.shuffle(noise_list)
  if len(test_list) > len(noise_list):
    noise_list = noise_list * (len(test_list)//len(noise_list)+1)
  noise_list = noise_list[:len(test_list)]

  with open(os.path.join(args.output, "noise.list"), "w") as f:
    for _list in noise_list:
      f.write("{}\n".format(_list))

import warnings
import tensorflow as tf
import multiprocessing
import numpy as np
import copy
import soundfile
import tqdm
import lpf
import paug

def add_noise(pcm, noise, snr_db):
  ns_pcm = copy.deepcopy(pcm)

  if args.apply_jointb:
    f_orig = librosa.stft(ns_pcm)
    m_orig = magnitude(f_orig)
    m_orig = librosa.amplitude_to_db(m_orig)

  pcm_en = np.mean(ns_pcm**2)
  noise_en = np.maximum(np.mean(noise**2), 1e-9)
  snr_en = 10.**(snr_db/10.)

  noise *= np.sqrt(pcm_en / (snr_en * noise_en))
  ns_pcm += noise
  noise_pcm_en = np.maximum(np.mean(ns_pcm**2), 1e-9)
  ns_pcm *= np.sqrt(pcm_en / noise_pcm_en)

  if args.apply_jointb:
    f_ns = librosa.stft(ns_pcm)
    m_ns = magnitude(f_ns); ph_ns = phase(f_ns)
    m_ns = librosa.amplitude_to_db(m_ns)

    m_ns_new = jointbilatFil.jointBilateralFilter(
      np.expand_dims(m_ns, -1), np.expand_dims(m_orig, -1))
    m_ns_new = np.squeeze(m_ns_new, -1)
    m_ns_new = librosa.db_to_amplitude(m_ns_new)

    ns_pcm = librosa.istft(polar(m_ns_new, ph_ns), length=ns_pcm.shape[0])

  return ns_pcm

def get_feat(_pcm,
             do_add_noise, noise, snr_db, 
             do_lpf, lpf_thres,
             do_paug):

  _ref = copy.deepcopy(_pcm)
  if do_add_noise: _pcm = add_noise(_pcm, noise, snr_db)
  if do_lpf: _pcm = lpf.lowpass(_pcm, lpf_thres) 
  if do_paug: _pcm = paug.pattern_mask(_pcm,
    fmin=args.paug_min_freq, fmax=args.paug_max_freq,
    ratio=0.5, enable_overlap=False)

  feats = {'pcm': _pcm, 'ref': _ref}
  return feats

hop_len = args.hop_len
num_process = 8
pcm_refs = []

for bidx in tqdm.tqdm(range(len(test_list)//num_process+1)):
  blist = test_list[bidx*num_process:(bidx+1)*num_process]
  if len(blist) == 0: break
  nlist = noise_list[bidx*num_process:(bidx+1)*num_process]

  pcms = [soundfile.read(e)[0] for e in blist]

  do_add_noises = np.random.choice([True, False],
    len(blist), p=[args.noise_ratio, 1-args.noise_ratio])
  if args.noise_list is None: do_add_noises = [False for _ in nlist]
  snr_dbs = np.random.uniform(args.min_snr, args.max_snr, len(blist))
  noises = [soundfile.read(e)[0] for e in nlist]

  do_lpfs = np.random.choice([True, False],
    len(blist), p=[args.lpf_ratio, 1-args.lpf_ratio])
  lpf_thress = np.random.uniform(args.lpf_min_thres, 
    args.lpf_max_thres, len(blist))

  do_paugs = np.random.choice([True, False],
    len(blist), p=[args.paug_ratio, 1-args.paug_ratio])
  
  for nidx in range(len(noises)):
    pcm = pcms[nidx]; noise = noises[nidx]
    if pcm.shape[0] >= noise.shape[0]:
      noise = np.repeat(noise, (pcm.shape[0]//noise.shape[0]+1))
      noise = noise[:pcm.shape[0]]
    else:
      pos = np.random.randint(0, noise.shape[0]-pcm.shape[0]+1)
      noise = noise[pos:pos+pcm.shape[0]]
    noises[nidx] = noise

  with multiprocessing.Pool(num_process) as pool:
    exs = pool.starmap(get_feat,
      zip(pcms, do_add_noises, noises, snr_dbs, do_lpfs, lpf_thress, do_paugs))

  for idx, ex in enumerate(exs):
    pcm_path = os.path.join(args.output, 
      "pcm_{}.wav".format(bidx*num_process + idx))
    ref_path = os.path.join(args.output,
      "ref_{}.wav".format(bidx*num_process + idx))

    soundfile.write(pcm_path, ex["pcm"], 16000) 
    soundfile.write(ref_path, ex["ref"], 16000)
    pcm_refs.append((os.path.abspath(pcm_path), os.path.abspath(ref_path)))

with open(os.path.join(args.output, "pcm_ref.list"), "w") as f:
  for pcm, ref in pcm_refs:
    f.write("{} {}\n".format(pcm, ref))

with open(args_file, "w") as f:
  f.write(" ".join([sys.executable] + sys.argv))
  f.write("\n")
  f.write(json.dumps(vars(args)))
os.chmod(args_file, S_IREAD|S_IRGRP|S_IROTH)

import shutil
for origin in [os.path.abspath(__file__), args.test_list]:
  shutil.copy(origin, args.output)
