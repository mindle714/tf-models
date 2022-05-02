import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-list", type=str, required=True) 
parser.add_argument("--noise-list", nargs="+", default=[]) 
parser.add_argument("--noise-ratio", type=float, required=False, default=0.)
parser.add_argument("--min-snr", type=int, required=False, default=0)
parser.add_argument("--max-snr", type=int, required=False, default=10)
parser.add_argument("--num-chunks", type=int, required=False, default=100)
parser.add_argument("--samp-len", type=int, required=False, default=32000)
parser.add_argument("--hop-len", type=int, required=False, default=16000)
parser.add_argument("--no-remainder", action='store_true')
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--apply-jointb", action='store_true') 
args = parser.parse_args()

if args.noise_ratio < 0. or args.noise_ratio > 1.: sys.exit(0)

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
train_list = [e.strip() for e in open(args.train_list).readlines()]

if len(args.noise_list) > 0:
  noise_list = []
  for f in args.noise_list:
    noise_list += [e.strip() for e in open(f).readlines()]

  import random
  random.shuffle(noise_list)
  if len(train_list) > len(noise_list):
    noise_list = noise_list * (len(train_list)//len(noise_list)+1)
  noise_list = noise_list[:len(train_list)]

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

def get_feat(_pcm, _samp_len, do_add_noise, noise, snr_db):
  _ref = copy.deepcopy(_pcm)
  if do_add_noise: _pcm = add_noise(_pcm, noise, snr_db)

  ref_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_ref))
  pcm_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_pcm))
  len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[_samp_len]))

  feats = {'pcm': pcm_feat, 'ref': ref_feat, 'samp_len': len_feat}
  ex = tf.train.Example(features=tf.train.Features(feature=feats))
  return ex.SerializeToString()

def get_feats(pcm, do_add_noise, noise, snr_db):
  exs = []
  num_seg = max((pcm.shape[0] - samp_len) // hop_len + 1, 0)

  for pcm_idx in range(num_seg):
    _pcm = pcm[pcm_idx*hop_len: pcm_idx*hop_len+samp_len]
    _noise = noise[pcm_idx*hop_len: pcm_idx*hop_len+samp_len]
    ex = get_feat(_pcm, samp_len, do_add_noise, _noise, snr_db)
    exs.append(ex)

  rem_len = pcm[num_seg*hop_len:].shape[0]
  if (not args.no_remainder) and rem_len > 0:
    def pad(_in):
      return np.concatenate([_in,
        np.zeros(samp_len-_in.shape[0], dtype=_in.dtype)], 0)

    _pcm = pad(pcm[num_seg*hop_len:])
    _noise = pad(noise[num_seg*hop_len:])
    ex = get_feat(_pcm, rem_len, do_add_noise, _noise, snr_db)
    exs.append(ex)

  return exs

num_chunks = min(len(train_list), args.num_chunks)
writers = [tf.io.TFRecordWriter(os.path.join(
    args.output, "train-{}.tfrecord".format(idx))) for idx in range(num_chunks)]

chunk_idx = 0; chunk_lens = [0 for _ in range(num_chunks)]
hop_len = args.hop_len; samp_len = args.samp_len
num_process = 8

for bidx in tqdm.tqdm(range(len(train_list)//num_process+1)):
  blist = train_list[bidx*num_process:(bidx+1)*num_process]
  if len(blist) == 0: break
  nlist = noise_list[bidx*num_process:(bidx+1)*num_process]

  pcms = [soundfile.read(e)[0] for e in blist]
  do_add_noises = np.random.choice([True, False],
    len(blist), p=[args.noise_ratio, 1-args.noise_ratio])
  if args.noise_list is None: do_add_noises = [False for _ in nlist]
  snr_dbs = np.random.uniform(args.min_snr, args.max_snr, len(blist))
  noises = [soundfile.read(e)[0] for e in nlist]
  
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
    exs = pool.starmap(get_feats,
      zip(pcms, do_add_noises, noises, snr_dbs))

  for ex in exs:
    for _ex in ex:
      writers[chunk_idx].write(_ex)
      chunk_lens[chunk_idx] += 1
      chunk_idx = (chunk_idx+1) % num_chunks

for writer in writers:
  writer.close()

args.chunk_lens = chunk_lens

with open(args_file, "w") as f:
  f.write(" ".join([sys.executable] + sys.argv))
  f.write("\n")
  f.write(json.dumps(vars(args)))
os.chmod(args_file, S_IREAD|S_IRGRP|S_IROTH)

import shutil
for origin in [os.path.abspath(__file__), args.train_list]:
  shutil.copy(origin, args.output)
