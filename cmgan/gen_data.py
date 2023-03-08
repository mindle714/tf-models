import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-list", type=str, required=True) 
parser.add_argument("--num-chunks", type=int, required=False, default=100)
parser.add_argument("--samp-rate", type=int, required=False, default=16000)
parser.add_argument("--samp-len", type=int, required=False, default=32000)
parser.add_argument("--hop-len", type=int, required=False, default=16000)
parser.add_argument("--no-remainder", action='store_true')
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
train_list = [e.strip() for e in open(args.train_list).readlines()]

import warnings
import tensorflow as tf
import multiprocessing
import numpy as np
import copy
import soundfile
import librosa
import tqdm

def get_feat(_ref, _pcm, _samp_len):
  ref_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_ref))
  pcm_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_pcm))
  len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[_samp_len]))

  feats = {'pcm': pcm_feat, 'ref': ref_feat, 'samp_len': len_feat}
  ex = tf.train.Example(features=tf.train.Features(feature=feats))
  return ex.SerializeToString()

def get_feats(ref, pcm): 
  exs = []
  num_seg = max((pcm.shape[0] - samp_len) // hop_len + 1, 0)

  for pcm_idx in range(num_seg):
    _ref = ref[pcm_idx*hop_len: pcm_idx*hop_len+samp_len]
    _pcm = pcm[pcm_idx*hop_len: pcm_idx*hop_len+samp_len]
    ex = get_feat(_ref, _pcm, samp_len)
    exs.append(ex)

  rem_len = pcm[num_seg*hop_len:].shape[0]
  if (not args.no_remainder) and rem_len > 0:
    def pad(_in):
      return np.concatenate([_in,
        np.zeros(samp_len-_in.shape[0], dtype=_in.dtype)], 0)

    _ref = pad(ref[num_seg*hop_len:])
    _pcm = pad(pcm[num_seg*hop_len:])
    ex = get_feat(_ref, _pcm, rem_len)
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

  refs = [librosa.load(e.split()[0], sr=args.samp_rate)[0] for e in blist]
  pcms = [librosa.load(e.split()[1], sr=args.samp_rate)[0] for e in blist]

  with multiprocessing.Pool(num_process) as pool:
    exs = pool.starmap(get_feats, zip(refs, pcms))

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

import types
specs = [val.__spec__ for name, val in sys.modules.items() \
  if isinstance(val, types.ModuleType) and not ('_main_' in name)]
origins = [spec.origin for spec in specs if spec is not None]
origins = [e for e in origins if e is not None and os.getcwd() in e]

import shutil
for origin in origins + [os.path.abspath(__file__), args.train_list]:
  shutil.copy(origin, args.output, follow_symlinks=True)
