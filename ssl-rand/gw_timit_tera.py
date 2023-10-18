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
parser.add_argument("--tfrec", type=str, required="timit") 
parser.add_argument("--batch-size", type=int, required=False, default=8) 
parser.add_argument("--accum-step", type=int, required=False, default=2)
parser.add_argument("--train-step", type=int, required=False, default=4000) 
parser.add_argument("--eval-step", type=int, required=False, default=50) 
parser.add_argument("--valid-step", type=int, required=False, default=50) 
parser.add_argument("--valid-patience", type=int, required=False, default=5) 
parser.add_argument("--begin-lr", type=float, required=False, default=1e-4) 
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--warm-start", type=str, required=True, default=None)
args = parser.parse_args()

import sys
import json

tfrec_args = os.path.join(args.tfrec, "ARGS")
with open(tfrec_args, "r") as f:
  _json = json.loads(f.readlines()[-1])
  samp_len = _json["samp_len"]
  txt_len = _json["text_len"]
  spec_len = int((samp_len - 400 + 400) / 160) + 1
  no_spec = bool(_json["no_spec"])

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
val_idxs = [0, 1, 2]; test_idxs = [3, 4, 5]
val_tfrec_list = [os.path.join(args.tfrec, 'train-{}.tfrecord'.format(idx)) for idx in val_idxs]
test_tfrec_list = [os.path.join(args.tfrec, 'train-{}.tfrecord'.format(idx)) for idx in test_idxs]
tfrec_list = [e for e in tfrec_list if e not in val_tfrec_list and e not in test_tfrec_list]
assert len(tfrec_list) == (100 - len(val_idxs) - len(test_idxs))

data_arg = [
  (samp_len if no_spec else spec_len), txt_len,
  no_spec, args.batch_size, seed]
test_dataset = parse_data.gen_val(test_tfrec_list, *data_arg)
val_dataset = parse_data.gen_val(val_tfrec_list, *data_arg)

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

@tf.function
def run_lth_eval_step(step, m, 
                      spec, txt,
                      spec_len, txt_len):
  sloss = m(
    (spec, txt, spec_len, txt_len), ssl_loss=True)
    
  sloss = tf.math.reduce_mean(sloss)
  return sloss
      
def get_grad_mask(m, ratio):
  grad_mask_dict = {}

  def add_rand(e):
    w, b = e.get_weights()

    w_flat = w.flatten()
    idxs = np.argsort(np.abs(w_flat)) # ascending order [0, 1, 2]
    num_mask = int(idxs.shape[0] * ratio)

    idxs_mask = idxs[:num_mask]
    idxs_mask_inv = idxs[num_mask:]

    w_mask = np.ones_like(w_flat)
    w_mask[idxs_mask] = 0
    w_mask = w_mask.reshape(w.shape)

    w_name = e.trainable_weights[0].name
    if w_name not in grad_mask_dict:
      grad_mask_dict[w_name] = tf.ones_like(w)

    grad_mask_dict[w_name] *= (1 - w_mask)

  add_rand(m.tera.tera.fe.spec_transform)
  # TODO what about layer norm weights?

  for i in range(3):
    add_rand(m.tera.tera.enc.layers[i].atten.self_attn.query)
    add_rand(m.tera.tera.enc.layers[i].atten.self_attn.key)
    add_rand(m.tera.tera.enc.layers[i].atten.self_attn.value)
  
    add_rand(m.tera.tera.enc.layers[i].atten.out)
    # TODO lnorm
  
    add_rand(m.tera.tera.enc.layers[i].inter)
    add_rand(m.tera.tera.enc.layers[i].out)
    # TODO lnorm

  grad_mask = [None for e in m.trainable_weights]
  for idx, w in enumerate(grad_mask):
    w_name = m.trainable_weights[idx].name
    if w_name in grad_mask_dict:
      grad_mask[idx] = grad_mask_dict[w_name]

  return grad_mask

for prune_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
  for rewind in [-200, -100, 0, 100, 200]:
    m = tera.tera_phone(num_class=50, use_last=True)

    opt = tf.keras.optimizers.Adam(learning_rate=args.begin_lr)
    opt_lth = tf.keras.optimizers.SGD(learning_rate=args.begin_lr)

    _in = np.zeros((args.batch_size, spec_len, 80), dtype=np.float32)
    _in_len = np.ones((args.batch_size, 1), dtype=np.int32) * spec_len
    _ = m((_in, _in_len), ctc = is_ctc)
    _ = m((_in, _in_len), ctc = is_ctc, ssl_loss = True)

    if args.accum_step > 1:
      accum_grads = [tf.Variable(tf.zeros_like(e)) for e in m.trainable_weights] 
      #accum_grads_lth = [tf.Variable(tf.zeros_like(e)) for e in m.trainable_weights] 

    m_saved = [tf.Variable(tf.zeros_like(e)) for e in m.trainable_weights]
    m_saved_set = False

    ckpt = tf.train.Checkpoint(m)
    ckpt.read(args.warm_start).assert_consumed()

    @tf.function
    def run_lth_step(step, m, 
                     _opt, accum_grads_v2,
                     spec, txt,
                     spec_len, txt_len,
                     accum=False, neg=True):
      with tf.GradientTape(persistent=True) as tape, log_writer.as_default():
        sloss = m(
          (spec, txt, spec_len, txt_len), ssl_loss=True)
        
        sloss = tf.math.reduce_mean(sloss)
        if neg: neg_sloss = -sloss
        else: neg_sloss = sloss
        tf.summary.scalar("sloss", sloss, step=step)
        
      weights = m.trainable_weights
      grads = tape.gradient(neg_sloss, weights)
      grads, _ = tf.clip_by_global_norm(grads, 5.)
        
      if args.accum_step == 1:
        _opt.apply_gradients(zip(grads, weights))
    
      else:
        for idx, grad in enumerate(grads):
          if grad is None: continue
          accum_grads_v2[idx].assign_add(grad)
    
        if not accum:
          for idx, grad in enumerate(grads):
            if grad is None: continue
            accum_grads_v2[idx].assign(accum_grads_v2[idx]/args.accum_step)
    
          _opt.apply_gradients(zip(accum_grads_v2, weights))
                    
          for idx, grad in enumerate(grads):
            if grad is None: continue
            accum_grads_v2[idx].assign(tf.zeros_like(grad))
    
      return sloss
    
    @tf.function
    def run_lth_step_v2(step, m, 
                     _opt, accum_grads_v2,
                     spec, txt,
                     spec_len, txt_len,
                     accum=False, neg=True):
      with tf.GradientTape(persistent=True) as tape, log_writer.as_default():
        sloss = m(
          (spec, txt, spec_len, txt_len), ssl_loss=True)
        
        sloss = tf.math.reduce_mean(sloss)
        if neg: neg_sloss = -sloss
        else: neg_sloss = sloss
        tf.summary.scalar("sloss", sloss, step=step)
        
      weights = m.trainable_weights
      grads = tape.gradient(neg_sloss, weights)
      grads, _ = tf.clip_by_global_norm(grads, 5.)
        
      if args.accum_step == 1:
        _opt.apply_gradients(zip(grads, weights))
    
      else:
        for idx, grad in enumerate(grads):
          if grad is None: continue
          accum_grads_v2[idx].assign_add(grad)
    
        if not accum:
          for idx, grad in enumerate(grads):
            if grad is None: continue
            accum_grads_v2[idx].assign(accum_grads_v2[idx]/args.accum_step)
    
          _opt.apply_gradients(zip(accum_grads_v2, weights))
                    
          for idx, grad in enumerate(grads):
            if grad is None: continue
            accum_grads_v2[idx].assign(tf.zeros_like(grad))
    
      return sloss
    
    @tf.function
    def run_lth_step_v3(step, m, 
                     _opt, accum_grads_v2,
                     spec, txt,
                     spec_len, txt_len,
                     accum=False, neg=True):
      with tf.GradientTape(persistent=True) as tape, log_writer.as_default():
        sloss = m(
          (spec, txt, spec_len, txt_len), ssl_loss=True)
        
        sloss = tf.math.reduce_mean(sloss)
        if neg: neg_sloss = -sloss
        else: neg_sloss = sloss
        tf.summary.scalar("sloss", sloss, step=step)
        
      weights = m.trainable_weights
      grads = tape.gradient(neg_sloss, weights)
      grads, _ = tf.clip_by_global_norm(grads, 5.)
        
      if args.accum_step == 1:
        _opt.apply_gradients(zip(grads, weights))
    
      else:
        for idx, grad in enumerate(grads):
          if grad is None: continue
          accum_grads_v2[idx].assign_add(grad)
    
        if not accum:
          for idx, grad in enumerate(grads):
            if grad is None: continue
            accum_grads_v2[idx].assign(accum_grads_v2[idx]/args.accum_step)
    
          _opt.apply_gradients(zip(accum_grads_v2, weights))
                    
          for idx, grad in enumerate(grads):
            if grad is None: continue
            accum_grads_v2[idx].assign(tf.zeros_like(grad))
    
      return sloss

    def get_val_loss(m):
      val_losses = []
      for idx, data in enumerate(val_dataset):
        _in_arg = [data["spec"], data["txt"],
                   data["spec_len"], data["txt_len"]]
        
        loss = run_lth_eval_step(
          tf.cast(idx, tf.int64), m, *_in_arg)
        val_losses.append(loss)

      return np.mean(val_losses)

    train_step = args.train_step
    if rewind < 0: train_step -= 2 * rewind

    dataset = parse_data.gen_train(tfrec_list, *data_arg)
    val_cnt = 0; prev_val = None

    # 1st pass
    for idx, data in enumerate(dataset):
      if idx > train_step: break

      if (rewind < 0 and idx == (-rewind)) or idx == rewind:
        m_saved_set = True
        assert len(m_saved) == len(m.trainable_weights)

        for w_idx, w in enumerate(m.trainable_weights):
          m_saved[w_idx].assign(w)
  
      # TODO not using (idx+1) to call apply_grads in initial run_step()
      accum = not (idx % args.accum_step == 0)
    
      _in_arg = [data["spec"], data["txt"],
                 data["spec_len"], data["txt_len"]]

      if rewind < 0 and (rewind + idx) < 0:
        loss = run_lth_step(
          tf.cast(idx, tf.int64), m,
          opt_lth, accum_grads_lth,
          *_in_arg, accum=accum, neg=True)

      else:
        loss = run_lth_step_v2(
          tf.cast(idx, tf.int64), m,
          opt, accum_grads,
          *_in_arg, accum=accum, neg=False)

      quit_by_val = False; val_loss = np.inf
      if not (rewind < 0 and (rewind + idx) < 0) and idx % args.valid_step == 0:
        val_loss = get_val_loss(m)
        if prev_val is None or prev_val > val_loss:
          val_cnt = 0; prev_val = val_loss 
        else:
          val_cnt += 1

        if val_cnt >= args.valid_patience:
          quit_by_val = True

      if idx % args.eval_step == 0:
        logger.info("1st gstep[{}] prune ratio[{}] rewind[{}] loss[{:.3f}] valid loss[{:.3f}]".format(
          idx, prune_ratio, rewind, np.mean(loss), val_loss))

      if quit_by_val:
        logger.info("1st gstep[{}] prune ratio[{}] rewind[{}] quit by validation loss".format(
          idx, prune_ratio, rewind))
        break

    grad_mask = get_grad_mask(m, prune_ratio)
    assert m_saved_set
    assert len(m_saved) == len(m.trainable_weights)

    # rewind
    for w_idx, w in enumerate(m_saved):
      wmask = 1.
      if grad_mask[w_idx] is not None: wmask = 1. - grad_mask[w_idx]

      w_masked = w * wmask
      m.trainable_weights[w_idx].assign(w_masked)
    
    # reset variables
    opt = tf.keras.optimizers.Adam(learning_rate=args.begin_lr)
    if args.accum_step > 1:
      accum_grads = [tf.Variable(tf.zeros_like(e)) for e in m.trainable_weights] 

    train_step = args.train_step
    dataset = parse_data.gen_train(tfrec_list, *data_arg)
    val_cnt = 0; prev_val = None

    # 2nd pass
    for idx, data in enumerate(dataset):
      if idx > train_step: break
      
      # TODO not using (idx+1) to call apply_grads in initial run_step()
      accum = not (idx % args.accum_step == 0)
    
      _in_arg = [data["spec"], data["txt"],
                 data["spec_len"], data["txt_len"]]
        
      loss = run_lth_step_v3(
        tf.cast(idx, tf.int64), m,
        opt, accum_grads,
        *_in_arg, accum=accum, neg=False)

      for w_idx, w in enumerate(m_saved):
        wmask = 1.
        if grad_mask[w_idx] is not None: wmask = 1. - grad_mask[w_idx]

        w_masked = w * wmask
        m.trainable_weights[w_idx].assign(w_masked)

      quit_by_val = False; val_loss = np.inf
      if idx % args.valid_step == 0:
        val_loss = get_val_loss(m)
        if prev_val is None or prev_val > val_loss:
          val_cnt = 0; prev_val = val_loss 
        else:
          val_cnt += 1

        if val_cnt >= args.valid_patience:
          quit_by_val = True
      
      if idx % args.eval_step == 0:
        logger.info("2nd gstep[{}] prune ratio[{}] rewind[{}] loss[{:.3f}] valid loss[{:.3f}]".format(
          idx, prune_ratio, rewind, np.mean(loss), val_loss))

      if quit_by_val:
        logger.info("2nd gstep[{}] prune ratio[{}] rewind[{}] quit by validation loss".format(
          idx, prune_ratio, rewind))
        break
    
    test_losses = []
    for idx, data in enumerate(test_dataset):
      _in_arg = [data["spec"], data["txt"],
                 data["spec_len"], data["txt_len"]]
        
      loss = run_lth_eval_step(
        tf.cast(idx, tf.int64), m, *_in_arg)
      test_losses.append(loss)

    logger.info("prune ratio[{}] rewind[{}] test loss[{}]".format(
      prune_ratio, rewind, np.mean(test_losses)))
      
