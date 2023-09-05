import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--beam-size", type=int, required=False, default=0)
parser.add_argument("--chunk-len", type=int, required=False, default=272000)
parser.add_argument("--pad-short", action="store_true")
args = parser.parse_args()

import os
from os.path import join
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import sys
expdir = os.path.abspath(os.path.dirname(args.ckpt))
sys.path.insert(0, expdir)
expname = expdir.split("/")[-1]
epoch = os.path.basename(args.ckpt).replace(".", "-").split("-")[1]

traincmd = open(os.path.join(expdir, "ARGS"), "r").readlines()[0].strip()
timit = ("--timit" in traincmd)
speech_command = ("--speech-command" in traincmd)
voxceleb = ("--voxceleb" in traincmd)
# none of these options coincide; checked in train.py

if timit:
  args.eval_list = "/data/hejung/timit/test.wav.phone"
elif speech_command:
  args.eval_list = "/data/hejung/speech-commands/test.v1.wav.key"
elif voxceleb:
  args.eval_list = "/data/hejung/vox1/test.wav.key"
else:
  args.eval_list = "/data/hejung/librispeech/test-clean.flac.phone"

assert os.path.isfile(args.eval_list)

if os.path.exists(join(expdir, "tera.py")):
  import tera
  if os.path.dirname(tera.__file__) != expdir:
    sys.exit("tera is loaded from {}".format(tera.__file__))
  if timit:
    m = tera.tera_phone(num_class=50)
  else:
    m = tera.tera_phone()
  is_tera = True

elif os.path.exists(join(expdir, "wav2vec2.py")):
  import wav2vec2
  if os.path.dirname(wav2vec2.__file__) != expdir:
    sys.exit("wav2vec2 is loaded from {}".format(wav2vec2.__file__))
  if timit:
    m = wav2vec2.wav2vec2_phone(num_class=50)
  else:
    m = wav2vec2.wav2vec2_phone()
  is_tera = False

else:
  assert False, "Invalid experiment path {}".format(expdir)

import numpy as np
if is_tera:
  if timit:
    _in = np.zeros((1, 801, 80), dtype=np.float32)
  else:
    _in = np.zeros((1, 1701, 80), dtype=np.float32)
else:
  if timit:
    _in = np.zeros((1, 128000), dtype=np.float32)
  else:
    _in = np.zeros((1, 272000), dtype=np.float32)
_ = m(_in, training=False)

ckpt = tf.train.Checkpoint(m)
ckpt.read(args.ckpt)

import warnings
import soundfile
import librosa
import tqdm
import metric
import parse_data
from text import WordTextEncoder
  
def softmax(x):
  z = x - np.max(x, -1, keepdims=True)
  num = np.exp(z)
  denom = np.sum(num, -1, keepdims=True)
  return num / denom

def eval(_pcm, chunk_len):
  pcm, _ = librosa.load(_pcm, sr = 16000)
  pcm_len = pcm.shape[0]
  if chunk_len is None:
    chunk_len = pcm_len

  hyps = []
  for idx in range(int(np.ceil(pcm_len / chunk_len))):
    _pcm = pcm[idx * chunk_len : (idx+1) * chunk_len]
    _pcm_len = _pcm.shape[0]

    if _pcm_len < chunk_len:
      if args.pad_short:
        _pcm = np.concatenate([_pcm, 
          np.zeros(chunk_len - _pcm_len, dtype=_pcm.dtype)], -1)

      else:
        if _pcm_len < 200: continue # if > n_fft//2, error in reflect pad

    if is_tera:
      spec_dict = parse_data.conv_spec(
        {'pcm': np.expand_dims(_pcm, 0).astype(np.float32), 'pcm_len':chunk_len})

      _hyp = m(spec_dict['spec'], training=False)

    else:
      _hyp = m(np.expand_dims(_pcm, 0).astype(np.float32), training=False)

    hyps.append(_hyp)

  hyp = np.concatenate(hyps, 1)

  if timit:
    return [str(e) for e in np.argmax(np.squeeze(hyp, 0), -1)]

  def greedy(hyp):
    truns = []; prev = 0
    for idx in hyp:
      if idx != prev:
        if prev != 0: truns.append(prev)
      prev = idx
    if prev != 0: truns.append(prev)
    return tokenizer.decode(truns)
  
  if args.beam_size < 1:
    maxids = np.argmax(np.squeeze(hyp, 0), -1)
    return greedy(maxids)
    
  else:
    logits = softmax(np.squeeze(hyp, 0))
    beams = [((1., 0.), [])]

    for t in range(logits.shape[0]):
      new_beams = []

      for bidx in range(len(beams)):
        (bprob, nbprob), y = beams[bidx]
        (c_bprob, c_nbprob), c_y = ((0., 0.), y)

        if len(y) > 0:
          ye = y[-1]
          c_nbprob = nbprob * logits[t][ye] # repeat last

          for e in beams:
            if y[:-1] == e[1]:
              # blank is needed to append "b" to "~b"
              prefix_prob = e[0][0] if (len(e[1]) > 0 and ye == e[1][-1]) else sum(e[0])
              c_nbprob += prefix_prob * logits[t][ye] # expand from prefix
           
        c_bprob = (bprob + nbprob) * logits[t][0] # blank -> 0
        new_beams.append(((c_bprob, c_nbprob), c_y))

        for k in range(1, logits[t].shape[-1]):
          (c_bprob, c_nbprob), c_y = ((0., 0.), y + [k])
          prefix_prob = bprob if (len(y) > 0 and k == y[-1]) else bprob + nbprob 

          c_nbprob = prefix_prob * logits[t][k]
          new_beams.append(((c_bprob, c_nbprob), c_y))

      beams = sorted(new_beams, 
        key=lambda e: sum(e[0])/(((5+len(e[1]))**0.5)/(6**0.5)), reverse=True)
      beams = beams[:args.beam_size]

    return tokenizer.decode(beams[0][1])

resname = "{}-{}".format(expname, epoch)
evals = [e.strip() for e in open(args.eval_list, "r").readlines()]
pers = []

if timit:
  with open(join("results", "{}.eval".format(resname)), "w") as f:
    for idx, pcm_ref in enumerate(evals):
      _pcm = pcm_ref.split()[0]
      _ref = [int(e) for e in pcm_ref.split()[1:]]
      
      hyp = eval(_pcm, None)
      _per = metric.per([" ".join(hyp)], 
              [" ".join([str(e) for e in _ref])])
      pers.append(_per)

      f.write("{} {}\n".format(_per, " ".join(hyp)))
      f.flush()

    f.write("final: {}\n".format(np.mean(pers)))

else:
  path = "/data/hejung/librispeech"
  lexicon = [
    join(path, "lexicon/librispeech-lexicon-200k-g2p.txt"),
    join(path, "lexicon/librispeech-lexicon-allothers-g2p.txt")
  ]
  tokenizer = WordTextEncoder.load_from_file(
    join(path, "vocab/phoneme.txt"))

  with open(join("results", "{}.eval".format(resname)), "w") as f:
    for idx, pcm_ref in enumerate(evals):
      _pcm = pcm_ref.split()[0]
      _ref = [int(e) for e in pcm_ref.split()[1:]]

      hyp = eval(_pcm, args.chunk_len)
      _per = metric.per([hyp], [tokenizer.decode(_ref)])
      pers.append(_per)

      f.write("{} {}\n".format(_per, " ".join(hyp)))
      f.flush()

    f.write("final: {}\n".format(np.mean(pers)))
