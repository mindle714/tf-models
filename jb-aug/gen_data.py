import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-list", type=str, required=True) 
parser.add_argument("--num-chunks", type=int, required=False, default=100)
parser.add_argument("--samp-rate", type=int, required=False, default=16000)
parser.add_argument("--samp-len", type=int, required=False, default=272000)
parser.add_argument("--text-len", type=int, required=False, default=None)
parser.add_argument("--no-spec", action='store_true')
parser.add_argument("--norm", action='store_true')
parser.add_argument("--crop-middle", action='store_true')

parser.add_argument("--noise-list", type=str, required=False, default=None) 
parser.add_argument("--min-snr", type=int, required=False, default=10)
parser.add_argument("--max-snr", type=int, required=False, default=10)
parser.add_argument("--ignore-prev-snr", action='store_true')

parser.add_argument("--apply-jointb", action='store_true') 
parser.add_argument("--reverse-jointb", action='store_true') 
parser.add_argument("--apply-guide", action='store_true') 
parser.add_argument("--guide-r", type=int, required=False, default=10)
parser.add_argument("--guide-nfft", type=int, required=False, default=512)

parser.add_argument("--output", type=str, required=True) 
args = parser.parse_args()

if args.apply_jointb or args.apply_guide:
  if args.apply_jointb:
    import jointbilatFil
  else:
    import gf

  def magnitude(e): return np.abs(e)
  def phase(e): return np.arctan2(e.imag, e.real)
  def polar(mag, phase):
    return mag * (np.cos(phase) + np.sin(phase) * 1j)

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

os.makedirs(args.output, exist_ok=True)
train_list = [e.strip() for e in open(args.train_list).readlines()]

if args.noise_list is None:
  noise_list = [None for _ in range(len(train_list))]

else:
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

max_text_len = max([len(e.split()) for e in train_list]) - 1
print("Maximum text length : {}".format(max_text_len))

if args.text_len is not None:
  max_text_len = args.text_len
  print("Overrided text length : {}".format(max_text_len))

else:
  args.text_len = max_text_len

import warnings
import tensorflow as tf
import multiprocessing
import numpy as np
np.seterr(all='raise')
import copy
import librosa
import tqdm
import parse_data

def get_feats(pcm, txt, n): 
  if pcm.shape[0] > args.samp_len:
    if not args.crop_middle: return None
    else:
      _pad = pcm.shape[0] - args.samp_len
      _pad = _pad // 2
      pcm = pcm[_pad:_pad+args.samp_len]
      assert pcm.shape[0] == args.samp_len
  
  if n is not None:
    if len(n.split()) > 1:
      _snr_db, n = n.split()

    if not args.ignore_prev_snr:
      snr_db = float(_snr_db)

    else:
      snr_db = np.random.uniform(args.min_snr, args.max_snr)

    if args.apply_jointb or args.apply_guide:
      f_orig = librosa.stft(pcm, n_fft=args.guide_nfft, hop_length=160)
      m_orig = magnitude(f_orig)
      m_orig = librosa.amplitude_to_db(m_orig)

    noise, _ = librosa.load(n, sr=args.samp_rate)

    if pcm.shape[0] > noise.shape[0]:
      ns_repeat = pcm.shape[0] // noise.shape[0] + int(pcm.shape[0] % noise.shape[0] != 0)
      noise = np.tile(noise, ns_repeat)[:pcm.shape[0]]

    elif pcm.shape[0] < noise.shape[0]:
      noise_pos = np.random.randint(0, noise.shape[0] - pcm.shape[0] + 1)
      noise = noise[noise_pos:noise_pos + pcm.shape[0]]

    assert noise.shape[0] == pcm.shape[0]

    pcm_en = np.mean(pcm**2)
    noise_en = np.maximum(np.mean(noise**2), 1e-9)
    snr_en = 10.**(snr_db/10.)

    noise *= np.sqrt(pcm_en / (snr_en * noise_en))
    pcm += noise
    noise_pcm_en = np.maximum(np.mean(pcm**2), 1e-9)
    pcm *= np.sqrt(pcm_en / noise_pcm_en)

    if args.apply_jointb:
      f_ns = librosa.stft(pcm, n_fft=args.guide_nfft, hop_length=160)
      m_ns = magnitude(f_ns); ph_ns = phase(f_ns)
      m_ns = librosa.amplitude_to_db(m_ns)

      if not args.reverse_jointb:
        m_ns_new = jointbilatFil.jointBilateralFilter(
          np.expand_dims(m_ns, -1), np.expand_dims(m_orig, -1))
      else:
        m_ns_new = jointbilatFil.jointBilateralFilter(
          np.expand_dims(m_orig, -1), np.expand_dims(m_ns, -1))

      m_ns_new = np.squeeze(m_ns_new, -1)
      m_ns_new = librosa.db_to_amplitude(m_ns_new)

      pcm = librosa.istft(polar(m_ns_new, ph_ns), length=pcm.shape[0], 
              n_fft=args.guide_nfft, hop_length=160)
      noise_pcm_en = np.maximum(np.mean(pcm**2), 1e-9)
      pcm *= np.sqrt(pcm_en / noise_pcm_en)

    elif args.apply_guide:
      orig_pcm = pcm

      try:
        f_ns = librosa.stft(pcm, n_fft=args.guide_nfft, hop_length=160)
        m_ns = magnitude(f_ns); ph_ns = phase(f_ns)
        m_ns = librosa.amplitude_to_db(m_ns)

        pad_len = 0
        if m_ns.shape[1] < (2*args.guide_r):
          pad_len = ((2*args.guide_r) - m_ns.shape[1]) // 2 + 1
          orig_shape = m_ns.shape
          m_orig = np.pad(m_orig, pad_len)
          m_ns = np.pad(m_ns, pad_len)

        m_ns_new = gf.guided_filter(m_orig, m_ns, args.guide_r, 0.05)
        m_ns_new = np.clip(m_ns_new, np.min(m_ns), np.max(m_ns))
        m_ns_new = librosa.db_to_amplitude(m_ns_new)

        if pad_len > 0:
          m_ns_new = m_ns_new[pad_len:-pad_len, pad_len:-pad_len]
          assert m_ns_new.shape == orig_shape

        pcm = librosa.istft(polar(m_ns_new, ph_ns), length=pcm.shape[0],
                n_fft=args.guide_nfft, hop_length=160)
        noise_pcm_en = np.maximum(np.mean(pcm**2), 1e-9)
        pcm *= np.sqrt(pcm_en / noise_pcm_en)

      except:
        print("Error occurred and rewind to original")
        pcm = orig_pcm

  def pad(_in, _len, val):
    return np.concatenate([_in,
      np.ones(_len-len(_in), dtype=_in.dtype) * val], 0)

  pcm_len = len(pcm); txt_len = len(txt)
  if args.norm:
    pcm = (pcm - np.mean(pcm)) / (np.std(pcm) + 1e-9)

  _pcm = pad(pcm, args.samp_len, 0)
  _txt = pad(np.array(txt), max_text_len, 0)
    
  txt_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=_txt))
  txt_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[txt_len]))
  feats = {'txt': txt_feat, 'txt_len': txt_len_feat}

  if not args.no_spec:
    _dict = {'pcm': np.expand_dims(_pcm, 0).astype(np.float32), 'pcm_len':pcm_len}
    _spec_dict = parse_data.conv_spec(_dict)
    _spec = np.squeeze(_spec_dict['spec'], 0)
    _spec = np.reshape(_spec, [-1])
    spec_len = _spec_dict['spec_len']

    spec_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_spec))
    spec_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[spec_len]))
    feats['spec'] = spec_feat; feats['spec_len'] = spec_len_feat

  else:
    pcm_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_pcm))
    pcm_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[pcm_len]))
    feats['pcm'] = pcm_feat; feats['pcm_len'] = pcm_len_feat

  ex = tf.train.Example(features=tf.train.Features(feature=feats))
  return ex.SerializeToString()

num_chunks = min(len(train_list), args.num_chunks)
writers = [tf.io.TFRecordWriter(os.path.join(
    args.output, "train-{}.tfrecord".format(idx))) for idx in range(num_chunks)]

chunk_idx = 0; chunk_lens = [0 for _ in range(num_chunks)]
num_process = 8

for bidx in tqdm.tqdm(range(len(train_list)//num_process+1)):
  blist = train_list[bidx*num_process:(bidx+1)*num_process]
  nblist = noise_list[bidx*num_process:(bidx+1)*num_process]
  if len(blist) == 0: break
      
  def parse(e):
    pcm_path = e.split()[0] 
    pcm, _ = librosa.load(pcm_path, sr=args.samp_rate)

    txt = e.split()[1:]
    txt = [int(e) for e in txt]
    return pcm, txt

  pcms = []; txts = []; ns = []
  for e, n in zip(blist, nblist):
    pcm, txt = parse(e)
    pcms.append(pcm)
    txts.append(txt)
    ns.append(n)

  if num_process > 1:
    with multiprocessing.Pool(num_process) as pool:
      exs = pool.starmap(get_feats, zip(pcms, txts, ns))

  else:
    exs = []
    for pcm, txt, n in zip(pcms, txts, ns):
      exs.append(get_feats(pcm, txt, n))

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
