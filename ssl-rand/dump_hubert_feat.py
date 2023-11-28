import os
import tqdm
import random
import librosa
import numpy as np
import tensorflow as tf

seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-list", type=str, required=True) 
parser.add_argument("--num-chunks", type=int, required=False, default=100)
parser.add_argument("--samp-rate", type=int, required=False, default=16000)
parser.add_argument("--samp-len", type=int, required=False, default=128000)
parser.add_argument("--text-len", type=int, required=False, default=399)
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--hubert-ckpt", type=str, required=True, default=None)
parser.add_argument("--km-mdl", type=str, required=False, default=None)
args = parser.parse_args()

import sys
import json
import types
import hashlib
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR

override = False
km_mdl = None

args_file = os.path.join(args.output, "ARGS")
if os.path.isdir(args.output):
  msg = 'directory {} exists. Do you want to proceed?'.format(args.output)
  cont = input("%s (y/N) " % msg).lower() == 'y'
  if not cont: sys.exit(0)
  if os.path.isfile(args_file):
    os.chmod(args_file, S_IWUSR|S_IREAD)
    
    old_hasher = hashlib.md5() 
    old_hasher.update(open(args_file, "r").readlines()[-1].strip().encode("utf-8"))
    old_hash = str(old_hasher.hexdigest())

    new_hasher = hashlib.md5()
    new_hasher.update(json.dumps(vars(args)).encode("utf-8"))
    new_hash = str(new_hasher.hexdigest())
    override = (old_hash == new_hash)

os.makedirs(args.output, exist_ok=True)
train_list = [e.strip() for e in open(args.train_list).readlines()]

feat_np_path = os.path.join(args.output, "feat.npy")
km_mdl_path = os.path.join(args.output, "km.mdl")

force_km_mdl = False
if args.km_mdl is not None:
  force_km_mdl = True
  os.symlink(args.km_mdl, km_mdl_path)

if override and os.path.isfile(feat_np_path) and not os.path.isfile(km_mdl_path):
  feat_np = np.load(feat_np_path)

  from sklearn.cluster import MiniBatchKMeans
  km_mdl = MiniBatchKMeans(
    n_clusters = 500,
    init = "k-means++",
    max_iter = 100,
    batch_size = 10000,
    verbose = 1,
    compute_labels = False,
    tol = 0.0,
    max_no_improvement = 100,
    init_size = None,
    n_init = 20,
    reassignment_ratio = 0.0)

  km_mdl.fit(feat_np)
  inertia = -km_mdl.score(feat_np) / len(feat_np)
  print("total intertia: %.5f", inertia)

  import joblib
  joblib.dump(km_mdl, km_mdl_path)
  sys.exit(0)

if force_km_mdl or (override and os.path.isfile(km_mdl_path)):
  import joblib
  km_mdl = joblib.load(km_mdl_path)
  c_np = km_mdl.cluster_centers_.transpose()
  cnorm_np = (c_np ** 2).sum(0, keepdims = True)

  num_chunks = min(len(train_list), args.num_chunks)
  writers = [tf.io.TFRecordWriter(os.path.join(
    args.output, "train-{}.tfrecord".format(idx))) for idx in range(num_chunks)]

  chunk_idx = 0; chunk_lens = [0 for _ in range(num_chunks)]
  args.num_clusters = c_np.shape[-1] 

if not override:
  with open(args_file, "w") as f:
    f.write(" ".join([sys.executable] + sys.argv))
    f.write("\n")
    f.write(json.dumps(vars(args)))
  os.chmod(args_file, S_IREAD|S_IRGRP|S_IROTH)

import parse_data
import glob

import hubert
m = hubert.hubert_phone(num_class=50, use_last=False, use_layers=9)

_in = np.zeros((16, args.samp_len), dtype=np.float32)
_ref = np.zeros((16, args.text_len), dtype=np.int32)
_in_len = np.ones((16, 1), dtype=np.int32) * args.samp_len
_ref_len = np.ones((16, 1), dtype=np.int32) * args.text_len

_ = m((_in, _ref, _in_len, _ref_len), training = True, ctc = True)

specs = [val.__spec__ for name, val in sys.modules.items() \
  if isinstance(val, types.ModuleType) and not ('_main_' in name)]
origins = [spec.origin for spec in specs if spec is not None]
origins = [e for e in origins if e is not None and os.getcwd() in e]

if not override:
  import shutil
  for origin in origins + [os.path.abspath(__file__)]:
    shutil.copy(origin, args.output)

@tf.function
def run_feat(pcm, txt, samp_len, txt_len):
  feat = m(
    (pcm, txt, samp_len, txt_len),
    training = False, ctc = False, return_feat=True)

  return feat

ckpt = tf.train.Checkpoint(m)
ckpt.read(args.hubert_ckpt)#.assert_consumed()

def get_feats(pcms, txts): 
  def pad(_in, _len, val):
    return np.concatenate([_in,
      np.ones(_len-len(_in), dtype=_in.dtype) * val], 0)

  pcms_pad = []; txts_pad = []
  pcms_len = []; txts_len = []

  for pcm, txt in zip(pcms, txts):
    pcm_len = len(pcm); txt_len = len(txt)

    _pcm = pad(pcm, args.samp_len, 0)
    _txt = pad(np.array(txt), args.text_len, 0)

    pcms_pad.append([_pcm]); txts_pad.append([_txt])
    pcms_len.append([pcm_len]); txts_len.append([txt_len])

  pcms_pad = np.concatenate(pcms_pad, 0)
  txts_pad = np.concatenate(txts_pad, 0)
  pcms_len = np.concatenate(pcms_len, 0).reshape([-1, 1])
  txts_len = np.concatenate(txts_len, 0).reshape([-1, 1])

  feats = run_feat(pcms_pad, txts_pad, pcms_len, txts_len) 
  feats = feats[9].numpy()

  bfeats = []
  for feat, txt in zip(feats, txts_len):
    txt = int(txt)
    feat = feat[:txt, :]
    bfeats.append(feat)

  return np.concatenate(bfeats, 0), bfeats

def get_km_feat(pcm, txt, hb_feat): 
  def pad(_in, _len, val):
    return np.concatenate([_in,
      np.ones(_len-len(_in), dtype=_in.dtype) * val], 0)

  pcm_len = len(pcm); txt_len = len(txt)

  _pcm = pad(pcm, args.samp_len, 0)
  _txt = pad(np.array(txt), args.text_len, 0)

  txt_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=_txt))
  txt_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[txt_len]))
  feats = {'txt': txt_feat, 'txt_len': txt_len_feat}

  pcm_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_pcm))
  pcm_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[pcm_len]))
  feats['pcm'] = pcm_feat; feats['pcm_len'] = pcm_len_feat

  dist = (
    (hb_feat ** 2).sum(1, keepdims = True) 
    - 2 * np.matmul(hb_feat, c_np) 
    + cnorm_np
  )
  idx = np.argmin(dist, axis = 1)
  idx_len = len(idx)

  idx = pad(idx, args.text_len, -1)
  idx_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=idx))
  idx_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[idx_len]))
  feats['hb_idx'] = idx_feat; feats['hb_idx_len'] = idx_len_feat

  ex = tf.train.Example(features=tf.train.Features(feature=feats))
  return ex.SerializeToString()

feat_nps = []
for bidx in tqdm.tqdm(range(len(train_list)//16+1)):
  blist = train_list[bidx*16:(bidx+1)*16]
  if len(blist) == 0: break
  
  def parse(e):
    pcm_path = e.split()[0] 
    pcm, _ = librosa.load(pcm_path, sr=args.samp_rate)

    txt = e.split()[1:]
    txt = [int(e) for e in txt]
    return pcm, txt

  pcms = []; txts = []
  for e in blist:
    pcm, txt = parse(e)
    pcms.append(pcm)
    txts.append(txt)
  
  feats, bfeats = get_feats(pcms, txts)
  if km_mdl is None:
    feat_nps.append(feats)

  else:
    exs = []
    for pcm, txt, feat in zip(pcms, txts, bfeats):
      exs.append(get_km_feat(pcm, txt, feat))
  
    for ex in exs:
      if ex is None: continue
      writers[chunk_idx].write(ex)
      chunk_lens[chunk_idx] += 1
      chunk_idx = (chunk_idx+1) % num_chunks

if km_mdl is None:
  feat_np = np.concatenate(feat_nps, 0)
  np.save(os.path.join(args.output, "feat"), feat_np)

else:
  for writer in writers:
    writer.close()
  writer.close()
