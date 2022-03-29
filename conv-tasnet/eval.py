import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-list", type=str, required=True) 
parser.add_argument("--eval-type", type=str, required=False,
  default="id", choices=["id", "vr"])
parser.add_argument("--print-norm", action="store_true")
args = parser.parse_args()

import os
import sys

expdir = os.path.abspath(os.path.dirname(args.ckpt))
sys.path.insert(0, expdir)

import model
if os.path.dirname(model.__file__) != expdir:
  sys.exit("model is loaded from {}".format(model.__file__))

import json
exp_args = os.path.join(expdir, "ARGS")
with open(exp_args, "r") as f:
  jargs = json.loads(f.readlines()[-1])
  vocab = {e.strip():idx for idx, e in enumerate(open(jargs["vocab"]).readlines())}

  with open(os.path.join(jargs["tfrec"], "ARGS")) as f2:
    jargs2 = json.loads(f2.readlines()[-1])
    samp_len = int(jargs2["samp_len"])

import numpy as np
m = model.tdnn(len(vocab))
_ = m((np.zeros((1, samp_len), dtype=np.float32), 
  np.zeros((1, 1), dtype=np.int32)), training=False)

import tensorflow as tf
ckpt = tf.train.Checkpoint(m)
ckpt.read(args.ckpt)

import sklearn.manifold
import sklearn.decomposition
import sklearn.discriminant_analysis
import warnings
import soundfile
import tqdm

evals = [e.strip() for e in open(args.eval_list, "r").readlines()]

if args.eval_type == "id":
  pcount = 0
  for idx, _line in tqdm.tqdm(enumerate(evals), total=len(evals)):
    if len(_line.split()) != 2:
      warnings.warn("failed to parse {} at line {}".format(_line, idx))
      continue

    spk, wav = _line.split()
    if args.eval_type == "id" and spk not in vocab:
      warnings.warn("vocab {} missing {}".format(args.vocab, spk))
      continue

    pcm, sr = soundfile.read(wav)
    out, _ = m((np.expand_dims(pcm, 0), None), training=False)
    pcount += int(np.argmax(out) == vocab[spk])

  def get_size(bsize):
    if bsize > (1024*1024):
      return "{:.1f}MB".format(bsize / float(1024*1024))
    return "{:.1f}KB".format(bsize / float(1024))

  mdl_size = os.path.getsize(args.ckpt + ".data-00000-of-00001")
  print("{}({}) overall pass {}/{}({:.3f}%)".format(
    args.ckpt, get_size(mdl_size),
    pcount, len(evals), float(pcount)/(len(evals))*100))
    
else:
  xvecs = {}
  for idx, _line in enumerate(evals):
    if len(_line.split()) < 3:
      warnings.warn("failed to parse {} at line {}".format(_line, idx))
      continue

    _lines = _line.split()
    fa = _lines[0]; ta = _lines[1]; enrolls = _lines[2:]
    
    def get_xvec(wav):
      pcm, sr = soundfile.read(wav)
      _, emb = m((np.expand_dims(pcm, 0), None), training=False)
      return np.squeeze(emb, 0)

    def cos_sim(e1, e2):
      denom = np.sqrt(np.sum(e1**2)) * np.sqrt(np.sum(e2**2))
      return np.sum(e1*e2) / denom

    enroll_xvec = [get_xvec(e) for e in enrolls]
    spk = os.path.basename(enrolls[0]).split("_")[0]
    if spk not in xvecs: xvecs[spk] = []
    xvecs[spk] += enroll_xvec

    enroll_xvec = sum(enroll_xvec) / len(enrolls)
    fa_xvec = get_xvec(fa); ta_xvec = get_xvec(ta)

    print("{} target".format(cos_sim(enroll_xvec, ta_xvec)))
    print("{} nontarget".format(cos_sim(enroll_xvec, fa_xvec)))

  xvecs_val = []; xvecs_tgt = []; xvecs_idx = []; accum = 0
  for idx, spk in enumerate(xvecs):
    xvecs_val += xvecs[spk]
    xvecs_tgt += [idx for _ in range(len(xvecs[spk]))]

    xvecs_idx.append((accum, accum + len(xvecs[spk])))
    accum += len(xvecs[spk])

  xvecs_tsne = sklearn.manifold.TSNE(n_components=2)
  xvecs_tsne = xvecs_tsne.fit_transform(xvecs_val)

  import matplotlib.pyplot as plt
  for beg, end in xvecs_idx:
    plt.scatter(xvecs_tsne[beg:end][:,0], xvecs_tsne[beg:end][:,1])
  plt.legend(xvecs.keys(), loc='upper right')

  expname = expdir.split("/")[-1]
  epoch = os.path.basename(args.ckpt).replace(".", "-").split("-")[1]
  plt.savefig('{}-{}-vr.png'.format(expname, epoch))
  plt.clf()

  if args.print_norm:
    def l2_norm(e):
      return e / np.linalg.norm(e, axis=-1, keepdims=True)

    xvecs_pca = sklearn.decomposition.PCA(n_components=2)
    xvecs_pca = xvecs_pca.fit_transform(xvecs_val)
    # xvecs_pca = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
    # xvecs_pca = xvecs_pca.fit_transform(xvecs_val, xvecs_tgt)

    scs = []
    for beg, end in xvecs_idx:
      xvecs_norm = l2_norm(xvecs_pca[beg:end])
      sc = plt.scatter(xvecs_norm[:,0], xvecs_norm[:,1])
      color = sc.get_facecolor()
      scs.append(sc)

      xvecs_mean = np.mean(xvecs_norm, 0, keepdims=True)
      plt.quiver(np.zeros(1), np.zeros(1), xvecs_mean[:,0], xvecs_mean[:,1],
              angles='xy', scale_units='xy', scale=1, color=color)
    plt.legend(scs, xvecs.keys(), loc='upper right')
    plt.savefig('{}-{}-vr-norm.png'.format(expname, epoch))
