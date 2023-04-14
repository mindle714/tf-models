import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-list", type=str, required=True) 
parser.add_argument("--samp-list", type=str, required=False, default=None) 
parser.add_argument("--num-chunks", type=int, required=False, default=100)
parser.add_argument("--samp-rate", type=int, required=False, default=16000)
parser.add_argument("--samp-len", type=int, required=False, default=272000)
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

if args.samp_list is not None:
  samp_list = [int(e.strip()) for e in open(args.samp_list).readlines()]
  assert len(train_list) == len(samp_list)
  train_list = list(zip(train_list, samp_list))
  train_list = sorted(train_list, key=lambda e: e[1], reverse=True)

  idx = 0
  while idx < len(train_list):
    if train_list[idx][1] <= args.samp_len: break
    idx += 1
  train_list = train_list[idx:]

  _train_list = []
  bidx = 0; pidx = len(train_list)-1

  _train = [train_list[bidx][0]]
  _len = train_list[bidx][1]; _text_len = len(train_list[bidx][0].split()) - 1
  
  _train_list = []
  max_text_len = 0

  while bidx < pidx:
    if (_len + train_list[pidx][1]) <= args.samp_len:
      _train.append(train_list[pidx][0])
      _len += train_list[pidx][1]
      _text_len += len(train_list[pidx][0].split()) - 1
      pidx -= 1

    else:
      _train_list.append(_train)
      max_text_len = max(max_text_len, _text_len)
      bidx += 1
      _train = [train_list[bidx][0]]
      _len = train_list[bidx][1]
      _text_len = len(train_list[bidx][0].split()) - 1

  _train_list.append(_train)
  max_text_len = max(max_text_len, _text_len)
  train_list = _train_list
  print("Maximum text length : {}".format(max_text_len))
  args.text_len = max_text_len

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
  _txt = pad(np.array(txt), max_text_len, 0)

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
      
  def parse(e):
    pcm_path = e.split()[0] 
    pcm, _ = librosa.load(pcm_path, sr=args.samp_rate)

    txt = e.split()[1:]
    txt = [int(e) for e in txt]
    return pcm, txt

  pcms = []; txts = []
  for e in blist:
    if isinstance(e, list):
      _pcms = []; _txts = []
      _rms = None

      for idx, _e in enumerate(e):
        pcm, txt = parse(_e)

        rms = np.mean(pcm**2)**0.5
        if _rms is None:
          _rms = rms
        else:
          pcm *= (_rms / rms)

        if idx < (len(e) - 1):
          if txt[-1] == -1:
            txt = txt[:-1]

        _pcms = np.concatenate([_pcms, pcm], 0)
        _txts += txt

      pcms.append(_pcms)
      txts.append(_txts)

    else:
      pcm, txt = parse(e)
      pcms.append(pcm)
      txts.append(txt)

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
