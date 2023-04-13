import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-list", type=str, required=True) 
parser.add_argument("--num-chunks", type=int, required=False, default=100)
parser.add_argument("--samp-rate", type=int, required=False, default=16000)
parser.add_argument("--samp-len", type=int, required=False, default=272000)
parser.add_argument("--text-len", type=int, required=False, default=272)
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

def get_feats(pcm, txt): 
  if pcm.shape[0] > args.samp_len: return None

  def pad(_in, _len, val):
    return np.concatenate([_in,
      np.ones(_len-len(_in), dtype=_in.dtype) * val], 0)

  pcm_len = len(pcm); txt_len = len(txt)
  _pcm = pad(pcm, args.samp_len, 0)
  _txt = pad(np.array(txt), args.text_len, 1)

  pcm_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_pcm))
  txt_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=_txt))
  pcm_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[pcm_len]))
  txt_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[txt_len]))

  feats = {'pcm': pcm_feat, 'txt': txt_feat, 
          'pcm_len': pcm_len_feat, 'txt_len': txt_len_feat}
  ex = tf.train.Example(features=tf.train.Features(feature=feats))
  return ex.SerializeToString()

num_chunks = min(len(train_list), args.num_chunks)
writers = [tf.io.TFRecordWriter(os.path.join(
    args.output, "train-{}.tfrecord".format(idx))) for idx in range(num_chunks)]

chunk_idx = 0; chunk_lens = [0 for _ in range(num_chunks)]
num_process = 8

for bidx in tqdm.tqdm(range(len(train_list)//num_process+1)):
  blist = train_list[bidx*num_process:(bidx+1)*num_process]
  if len(blist) == 0: break

  pcms = [librosa.load(e.split()[0], sr=args.samp_rate)[0] for e in blist]
  txts = [[int(_e) for _e in e.split()[1:]] for e in blist]

  with multiprocessing.Pool(num_process) as pool:
    exs = pool.starmap(get_feats, zip(pcms, txts))

  for ex in exs:
    if ex is None: continue
    writers[chunk_idx].write(ex)
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
