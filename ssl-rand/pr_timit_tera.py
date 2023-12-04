import os
import random
import numpy as np
import tensorflow as tf

seed = 1234
#os.environ['PYTHONHASHSEED'] = str(seed)
#os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tfrec", type=str, required=False, default="timit_pcm") 
parser.add_argument("--noise-list", type=str, required=False, default="musan_train.list") 
parser.add_argument("--batch-size", type=int, required=False, default=8) 
parser.add_argument("--accum-step", type=int, required=False, default=2)
parser.add_argument("--train-step", type=int, required=False, default=1400) 
parser.add_argument("--begin-lr", type=float, required=False, default=2e-4) 
parser.add_argument("--lr-decay-rate", type=float, required=False, default=0.96)
parser.add_argument("--lr-decay-step", type=float, required=False, default=1000)
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--warm-start", type=str, required=False, default="tera_timit.ckpt")
args = parser.parse_args()

import metric
import soundfile
import librosa
import sys
import json

tfrec_args = os.path.join(args.tfrec, "ARGS")
with open(tfrec_args, "r") as f:
  _json = json.loads(f.readlines()[-1])
  samp_len = _json["samp_len"]
  txt_len = _json["text_len"]
  spec_len = int((samp_len - 400 + 400) / 160) + 1
  no_spec = bool(_json["no_spec"])
  if not no_spec and args.noise_list is not None:
    sys.exit("when --noise-list is used, tf-record must be in form of raw wav")

import types
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR

args_file = os.path.join(args.output, "ARGS")
if os.path.isdir(args.output):
  msg = 'directory {} exists. Do you want to proceed?'.format(args.output)
  cont = input("%s (y/N) " % msg).lower() == 'y'
  if not cont: sys.exit(0)
  if os.path.isfile(args_file):
    os.chmod(args_file, S_IWUSR|S_IREAD)

os.makedirs(args.output, exist_ok=True)
with open(args_file, "w") as f:
  f.write(" ".join([sys.executable] + sys.argv))
  f.write("\n")
  f.write(json.dumps(vars(args)))
os.chmod(args_file, S_IREAD|S_IRGRP|S_IROTH)

ipaddr_file = os.path.join(args.output, "IPADDR")
with open(ipaddr_file, "w") as f:
  import socket
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.connect(("8.8.8.8", 80))
  f.write(s.getsockname()[0])

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

import parse_data
import glob

tfrec_list = glob.glob(os.path.join(args.tfrec, "train-*.tfrecord"))
noise_list = [e.strip() for e in open(args.noise_list).readlines()]

import tera
is_ctc = False

specs = [val.__spec__ for name, val in sys.modules.items() \
  if isinstance(val, types.ModuleType) and not ('_main_' in name)]
origins = [spec.origin for spec in specs if spec is not None]
origins = [e for e in origins if e is not None and os.getcwd() in e]

import shutil
for origin in origins + [os.path.abspath(__file__)]:
  shutil.copy(origin, args.output)

import datetime
logdir = os.path.join(args.output, "logs")
log_writer = tf.summary.create_file_writer(logdir)
log_writer.set_as_default()

import logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)

logfile = os.path.join(args.output, "train.log")
if os.path.isfile(logfile): os.remove(logfile)
fh = logging.FileHandler(logfile)
logger.addHandler(fh)

def sig_pow(e):
  return np.mean(e**2)

def add_noise(pcm, pcm_len, noise, snr):
  if pcm_len[0] > noise.shape[0]:
    ns_repeat = pcm_len[0] // noise.shape[0] + int(pcm_len[0] % noise.shape[0] != 0)
    noise = np.tile(noise, ns_repeat)[:pcm_len[0]]

  else:
    noise_pos = np.random.randint(0, noise.shape[0] - pcm_len[0] + 1)
    noise = noise[noise_pos:noise_pos + pcm_len[0]]
    assert noise.shape[0] == pcm_len[0]

  if noise.shape[0] < pcm.shape[0]:
    noise = np.concatenate([noise,
      np.zeros(pcm.shape[0] - noise.shape[0], dtype=noise.dtype)], -1)
      
  _snr = np.random.uniform(snr - 5, snr + 5)
  pcm_pow = sig_pow(pcm[:pcm_len[0]])
  noise_pow = sig_pow(noise[:pcm_len[0]])
  scale = np.sqrt(pcm_pow / (np.power(10, _snr / 10) * noise_pow))
  _pcm_noise = pcm + scale * noise

  return _pcm_noise

for snr in [None, 0, 10, 20, 30]:
  if snr is None:
    eval_list = "/data/hejung/timit/test.wav.phone"
  else:
    eval_list = "timit_test/snr{}/test.wav.txt".format(snr)

  evals = [e.strip() for e in open(eval_list, "r").readlines()]
  eval_pcms = []
  for idx, pcm_ref in enumerate(evals):
    _pcm = pcm_ref.split()[0]
    _pcm, _ = soundfile.read(_pcm)
    eval_pcms.append(_pcm)
  
  m = tera.tera_phone(num_class=50, use_last=True)
    
  lr = tf.Variable(args.begin_lr, trainable=False)
  opt = tf.keras.optimizers.Adam(learning_rate=args.begin_lr)

  _in = np.zeros((args.batch_size, spec_len, 80), dtype=np.float32)
  _ = m(_in, ctc = is_ctc)

  if args.accum_step > 1:
    #accum_grads = [tf.Variable(tf.zeros_like(e)) for e in m.trainable_weights \
    #        if 'tera_phone/dense' in e.name]
    accum_grads = [tf.Variable(tf.zeros_like(e)) for e in m.trainable_weights]
  
  ckpt = tf.train.Checkpoint(m)
  ckpt.read(args.warm_start).assert_consumed()

  dataset = parse_data.gen_train(tfrec_list, 
    (samp_len if no_spec else spec_len), txt_len,
    no_spec=no_spec, batch_size=args.batch_size, seed=seed)

  @tf.function
  def run_step(step, spec, txt,
               spec_len, txt_len,
               training=True, accum=False,
               stop_grad=False, ssl_fix=False):
    with tf.GradientTape(persistent=True) as tape, log_writer.as_default():
      loss = m(
        (spec, txt, spec_len, txt_len),
        training = training, 
        stop_grad = stop_grad,
        ctc = is_ctc)

      loss = tf.math.reduce_mean(loss)
      tf.summary.scalar("loss", loss, step=step)

    if training:
      weights = m.trainable_weights
      #weights = [e for e in weights if 'tera_phone/dense' in e.name]
      #assert len(weights) == 4

      grads = tape.gradient(loss, weights)
      grads, _ = tf.clip_by_global_norm(grads, 5.)

      if args.accum_step == 1:
        opt.apply_gradients(zip(grads, weights))

      else:   
        for idx, grad in enumerate(grads):
          if grad is None: continue
          accum_grads[idx].assign_add(grad)

        if not accum:
          for idx, grad in enumerate(grads):
            if grad is None: continue
            accum_grads[idx].assign(accum_grads[idx]/args.accum_step)

          opt.apply_gradients(zip(accum_grads, weights))
                
          for idx, grad in enumerate(grads):
            if grad is None: continue
            accum_grads[idx].assign(tf.zeros_like(grad))

    return loss

  noise_idx = 0
  for idx, data in enumerate(dataset):
    if idx > args.train_step: break
  
    # TODO not using (idx+1) to call apply_grads in initial run_step()
    accum = not (idx % args.accum_step == 0)

    if snr is None:
      _pcm, _pcm_len = data["pcm"], data["pcm_len"]
      
      _dict = {'pcm': _pcm, 'pcm_len': _pcm_len}
      _spec_dict = parse_data.conv_spec(_dict)
      spec = _spec_dict['spec']
      _spec_len = _spec_dict['spec_len']

      _in_arg = [spec, data["txt"], _spec_len, data["txt_len"]] 

    else:
      _pcm, _pcm_len = data["pcm"], data["pcm_len"]

      pcm_noise = []
      for pcm, pcm_len in zip(_pcm, _pcm_len):
        while True:
          noise, _ = librosa.load(noise_list[noise_idx], sr=16000)
          noise_idx = (noise_idx + 1) % len(noise_list)
          if sig_pow(noise) != 0: break
      
        _pcm_noise = add_noise(pcm, pcm_len, noise, snr)
        pcm_noise.append(_pcm_noise)

      _dict = {'pcm': np.array(pcm_noise), 'pcm_len': _pcm_len}
      _spec_dict = parse_data.conv_spec(_dict)
      spec = _spec_dict['spec']
      _spec_len = _spec_dict['spec_len']

      _in_arg = [spec, data["txt"], _spec_len, data["txt_len"]] 
  
    loss = run_step(
      tf.cast(idx, tf.int64), *_in_arg,
      accum=accum)

    if idx > 0 and idx % 100 == 0:
      logger.info("gstep[{}] loss[{:.2f}] lr[{:.2e}]".format(
        idx, loss, lr.numpy()))
  
    if args.accum_step == 1 or not accum:
      # follow tf.keras.optimizers.schedules.ExponentialDecay
      lr.assign(args.begin_lr * args.lr_decay_rate**(idx/args.lr_decay_step))
    
  m_saved = tera.tera_phone(num_class=50, use_last=True)
  
  _in = np.zeros((args.batch_size, spec_len, 80), dtype=np.float32)
  _ = m_saved(_in, ctc = is_ctc)

  for e, e_saved in zip(m.trainable_weights, m_saved.trainable_weights):
    e_saved.assign(e)

  for prune_ratio in [None, 0.1, 0.2, 0.3, 0.4, 0.5]:
    def add_rand(e, e_saved):
      w, b = e_saved.get_weights()

      if prune_ratio is not None:
        w_flat = w.flatten()
        idxs = np.argsort(np.abs(w_flat)) # ascending order [0, 1, 2]
        num_mask = int(idxs.shape[0] * prune_ratio)

        idxs_mask = idxs[:num_mask]
        idxs_mask_inv = idxs[num_mask:]

        w_mask = np.ones_like(w_flat)
        w_mask[idxs_mask] = 0
        w_mask = w_mask.reshape(w.shape)

        e.set_weights([w_mask * w, b])

      else:
        e.set_weights([w, b])

    add_rand(m.tera.tera.fe.spec_transform, m_saved.tera.tera.fe.spec_transform)
    # TODO what about layer norm weights?

    for i in range(3):
      add_rand(m.tera.tera.enc.layers[i].atten.self_attn.query,
            m_saved.tera.tera.enc.layers[i].atten.self_attn.query)
      add_rand(m.tera.tera.enc.layers[i].atten.self_attn.key,
            m_saved.tera.tera.enc.layers[i].atten.self_attn.key)
      add_rand(m.tera.tera.enc.layers[i].atten.self_attn.value,
            m_saved.tera.tera.enc.layers[i].atten.self_attn.value)
  
      add_rand(m.tera.tera.enc.layers[i].atten.out,
            m_saved.tera.tera.enc.layers[i].atten.out)
      # TODO lnorm
  
      add_rand(m.tera.tera.enc.layers[i].inter,
            m_saved.tera.tera.enc.layers[i].inter)
      add_rand(m.tera.tera.enc.layers[i].out,
            m_saved.tera.tera.enc.layers[i].out)
      # TODO lnorm

    def run_eval_step(pcm, pcm_len):
      # bulk inference
      spec_dict = parse_data.conv_spec({'pcm': pcm, 'pcm_len': pcm_len})
      _hyp = m(spec_dict['spec'], training=False)

      maxids = np.argmax(np.squeeze(_hyp, 0), -1)
      return [str(e) for e in maxids]

    pers = []
    for _pcm, pcm_ref in zip(eval_pcms, evals):
      _pcm_len = _pcm.shape[0]
      _pcm = np.expand_dims(_pcm, 0).astype(np.float32)

      _ref = [int(e) for e in pcm_ref.split()[1:]]
      hyp = run_eval_step(_pcm, _pcm_len)

      _per = metric.per([" ".join(hyp)], [" ".join([str(e) for e in _ref])])  
      pers.append(_per)

    logger.info("snr[{}] prune ratio[{}] per[{:.4f}]".format(snr, prune_ratio, np.mean(pers)))
