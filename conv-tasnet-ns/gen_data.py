import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-list", type=str, required=True) 
parser.add_argument("--num-chunks", type=int, required=False, default=100)
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
import numpy as np
import soundfile
import tqdm

num_chunks = min(len(train_list), args.num_chunks)
writers = [tf.io.TFRecordWriter(os.path.join(
    args.output, "train-{}.tfrecord".format(idx))) for idx in range(num_chunks)]

chunk_idx = 0; chunk_lens = [0 for _ in range(num_chunks)]

for idx, _list in tqdm.tqdm(enumerate(train_list), total=len(train_list)):
  if len(_list.split()) != 3:
    warnings.warn("failed to parse {} at line {}".format(_list, idx))
    continue

  s1, s2, mix = [soundfile.read(e)[0] for e in _list.split()]
  assert s1.shape[0] == s2.shape[0] and s2.shape[0] == mix.shape[0]

  hop_len = args.hop_len
  samp_len = args.samp_len
  num_seg = max((s1.shape[0] - samp_len) // hop_len + 1, 0)

  def get_feat(_s1, _s2, _mix, _samp_len):
    s1_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_s1))
    s2_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_s2))
    mix_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_mix))
    len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[_samp_len]))

    feats = {'s1': s1_feat, 's2': s2_feat, 'mix': mix_feat, 'samp_len': len_feat}
    return tf.train.Example(features=tf.train.Features(feature=feats))

  for pcm_idx in range(num_seg):
    _s1 = s1[pcm_idx*hop_len: pcm_idx*hop_len+samp_len]
    _s2 = s2[pcm_idx*hop_len: pcm_idx*hop_len+samp_len]
    _mix = mix[pcm_idx*hop_len: pcm_idx*hop_len+samp_len]

    ex = get_feat(_s1, _s2, _mix, samp_len)
    writers[chunk_idx].write(ex.SerializeToString())

    chunk_lens[chunk_idx] += 1
    chunk_idx = (chunk_idx+1) % num_chunks

  rem_len = s1[num_seg*hop_len:].shape[0]
  if (not args.no_remainder) and rem_len > 0:
    def pad(_in):
      return np.concatenate([_in,
        np.zeros(samp_len-_in.shape[0], dtype=_in.dtype)], 0)

    _s1 = pad(s1[num_seg*hop_len:])
    _s2 = pad(s2[num_seg*hop_len:])
    _mix = pad(mix[num_seg*hop_len:])

    ex = get_feat(_s1, _s2, _mix, rem_len)
    writers[chunk_idx].write(ex.SerializeToString())

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
for origin in [os.path.abspath(__file__)]:
  shutil.copy(origin, args.output)
