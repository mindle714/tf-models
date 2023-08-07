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
parser.add_argument("--tfrec", type=str, required=True) 
parser.add_argument("--val-tfrec", type=str, required=False, default=None)
parser.add_argument("--ssl-tfrec", type=str, required=False, default=None)
parser.add_argument("--eval-list", type=str, required=False, default=None) 
parser.add_argument("--batch-size", type=int, required=False, default=8) 
parser.add_argument("--accum-step", type=int, required=False, default=2)
parser.add_argument("--eval-step", type=int, required=False, default=100) 
parser.add_argument("--save-step", type=int, required=False, default=None) 
parser.add_argument("--val-step", type=int, required=False, default=5000) 
parser.add_argument("--train-step", type=int, required=False, default=None) 
parser.add_argument("--begin-lr", type=float, required=False, default=2e-4) 
parser.add_argument("--lr-decay-rate", type=float, required=False, default=0.96)
parser.add_argument("--lr-decay-step", type=float, required=False, default=None)
parser.add_argument("--val-lr-update", type=float, required=False, default=3) 
parser.add_argument("--ssl-weight", type=float, required=False, default=0)
parser.add_argument("--ssl-decay", action='store_true')
parser.add_argument("--ssl-decay-schedule", type=str, required=False,
  choices=["linear", "exp", "cos", "stair", "inv-linear"], default="linear")
parser.add_argument("--ssl-margin", type=float, required=False, default=0)
parser.add_argument("--begin-ssl", type=int, required=False, default=0)
parser.add_argument("--end-ssl", type=int, required=False, default=None)
parser.add_argument("--period-ssl", type=int, required=False, default=None)
parser.add_argument("--period-ssl-decay", type=float, required=False, default=1.0)
parser.add_argument("--ssl-from-ema", action='store_true')
parser.add_argument("--ssl-ema-beta", type=float, required=False, default=0.9)
parser.add_argument("--ssl-ema-update", type=int, required=False, default=1)
parser.add_argument("--ewc", action='store_true')
parser.add_argument("--ewc-lambda", type=float, required=False, default=0.01)
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--timit", action='store_true')
parser.add_argument("--warm-start", type=str, required=False, default=None)
parser.add_argument("--stop-grad", action='store_true')
parser.add_argument("--from-init", action='store_true')
parser.add_argument("--profile", action='store_true')
args = parser.parse_args()

if args.ewc and args.ssl_from_ema:
  import sys
  sys.exit('--ewc and --ssl-from-ema cannot co-exist')

if args.timit:
  if args.save_step is None: args.save_step = 100
  if args.train_step is None: args.train_step = 2000
  if args.lr_decay_step is None: args.lr_decay_step = 300
  # different phoneme size unit; require different TIMIT segmentation
  if args.eval_list is None: args.eval_list = "/data/hejung/timit/test.wav.phone"

else:
  if args.save_step is None: args.save_step = 10000
  if args.train_step is None: args.train_step = 45000
  if args.lr_decay_step is None: args.lr_decay_step = 4000
  if args.eval_list is None: args.eval_list = "/data/hejung/librispeech/test-clean.flac.phone"
  
  from text import WordTextEncoder
  path = "/data/hejung/librispeech"
  tokenizer = WordTextEncoder.load_from_file(
    os.path.join(path, "vocab/phoneme.txt"))

import metric
import soundfile
assert os.path.isfile(args.eval_list)
evals = [e.strip() for e in open(args.eval_list, "r").readlines()]
eval_pcms = []
for idx, pcm_ref in enumerate(evals):
  _pcm = pcm_ref.split()[0]
  _pcm, _ = soundfile.read(_pcm)
  eval_pcms.append(_pcm)

if args.end_ssl is None:
  args.end_ssl = args.train_step

if args.period_ssl is None:
  args.period_ssl = args.train_step

def cos_decay(step, decay_steps, init_lr, alpha=0.):
  step = tf.math.minimum(step, decay_steps)
  cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  return init_lr * decayed

def exp_decay(step, decay_steps, init_lr, end_lr=0.01):
  if step > decay_steps: return 0.
  if init_lr <= 0.: return 0.
  decay_rate = end_lr / init_lr
  return init_lr * decay_rate ** (step / decay_steps)

def linear_decay(step, decay_steps, init_lr, end_lr = 0.):
  if step > decay_steps: return end_lr
  return ((end_lr - init_lr) / decay_steps) * step + init_lr

# TODO add optional multiplier
def inv_linear_decay(step, decay_steps, init_lr):
  end_lr = 2 * init_lr
  if step > decay_steps: return end_lr
  return ((end_lr - init_lr) / decay_steps) * step + init_lr

def stair_decay(step, decay_steps, init_lr, end_lr = 0., decay_unit = 100):
  if step > decay_steps: return end_lr
  _step = (step // decay_unit) * decay_unit
  return ((end_lr - init_lr) / decay_steps) * _step + init_lr

if args.ssl_decay_schedule == "linear":
  factor_decay = linear_decay
elif args.ssl_decay_schedule == "exp":
  factor_decay = exp_decay
elif args.ssl_decay_schedule == "cos":
  factor_decay = cos_decay
elif args.ssl_decay_schedule == "stair":
  factor_decay = stair_decay
elif args.ssl_decay_schedule == "inv-linear":
  factor_decay = inv_linear_decay
else:
  assert False, "unknown type {}".format(args.ssl_decay_schedule)

import sys
import json
tfrec_args = os.path.join(args.tfrec, "ARGS")
with open(tfrec_args, "r") as f:
  _json = json.loads(f.readlines()[-1])
  samp_len = _json["samp_len"]
  txt_len = _json["text_len"]
  spec_len = int((samp_len - 400 + 400) / 160) + 1 

if args.val_tfrec is not None:
  val_tfrec_args = os.path.join(args.val_tfrec, "ARGS")
  with open(val_tfrec_args, "r") as f:
    _json = json.loads(f.readlines()[-1])
    val_samp_len = _json["samp_len"]
    val_txt_len = _json["text_len"]
    val_spec_len = int((val_samp_len - 400 + 400) / 160) + 1 
  if val_samp_len != samp_len:
    sys.exit('validation data has sample length {}'.format(val_samp_len))

if args.ssl_tfrec is not None:
  ssl_tfrec_args = os.path.join(args.ssl_tfrec, "ARGS")
  with open(ssl_tfrec_args, "r") as f:
    _json = json.loads(f.readlines()[-1])
    ssl_samp_len = _json["samp_len"]
    ssl_spec_len = int((ssl_samp_len - 400 + 400) / 160) + 1 

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
dataset = parse_data.gen_train(tfrec_list, spec_len, txt_len,
  batch_size=args.batch_size, seed=seed)

val_dataset = None
if args.val_tfrec is not None:
  val_tfrec_list = glob.glob(os.path.join(args.val_tfrec, "train-*.tfrecord"))
  val_dataset = parse_data.gen_val(val_tfrec_list, val_spec_len,
    batch_size=args.batch_size, seed=seed)

ssl_dataset = None
if args.ssl_tfrec is not None:
  ssl_tfrec_list = glob.glob(os.path.join(args.ssl_tfrec, "train-*.tfrecord"))
  ssl_dataset = parse_data.gen_train(ssl_tfrec_list, ssl_spec_len, None,
    batch_size=args.batch_size, seed=seed)

lr = tf.Variable(args.begin_lr, trainable=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr)

if args.ewc:
  import ewc

import tera
if args.timit:
  m = tera.tera_phone(num_class=50, use_last=True)
  m_ema = tera.tera_phone(num_class=50, use_last=True)
  is_ctc = False
  if args.ewc:
    m_ewc = tera.tera_phone(num_class=50, use_last=True)
else:
  m = tera.tera_phone(use_last=False)
  m_ema = tera.tera_phone(use_last=False)
  is_ctc = True
  if args.ewc:
    m_ewc = tera.tera_phone(use_last=False)

if args.accum_step > 1:
  _in = np.zeros((args.batch_size, spec_len, 80), dtype=np.float32)
  _ = m(_in, ctc = is_ctc)
  _ = m_ema(_in, ctc = is_ctc)

  accum_grads = [tf.Variable(tf.zeros_like(e)) for e in m.trainable_weights]

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
if args.profile:
  tf.profiler.experimental.start(logdir)

@tf.function
def run_step(step, spec, ssl_spec, txt,
             spec_len, ssl_spec_len, txt_len,
             training=True, accum=False,
             stop_grad=False, update_ema=False):
  ssl_weight = 0.
  ssl_step = step % args.period_ssl
  num_period = step // args.period_ssl

  if ssl_step >= args.begin_ssl and ssl_step < args.end_ssl:
    _ssl_weight = args.ssl_weight * (args.period_ssl_decay ** tf.cast(num_period, tf.float32))

    if args.ssl_decay:
      ssl_weight = factor_decay(tf.cast(ssl_step, tf.float32),
        args.end_ssl - args.begin_ssl, _ssl_weight)
    else:
      ssl_weight = _ssl_weight

  tf.summary.scalar("ssl_weight", ssl_weight, step=step)

  if args.ssl_from_ema:
    mask_label, seq_out = m_ema((ssl_spec, ssl_spec_len), ssl_only=True)

    with tf.GradientTape(persistent=True) as tape, log_writer.as_default():
      loss, _, _seq_out = m(
        (spec, ssl_spec, txt, spec_len, ssl_spec_len, txt_len, mask_label),
        training = training, 
        ssl_loss = (not (args.ssl_weight == 0)),
        stop_grad = stop_grad,
        ctc = is_ctc)

      sloss = tf.math.abs(mask_label * (seq_out - _seq_out))
      sloss = tf.math.reduce_sum(sloss, [-1, -2])
      denom = tf.math.reduce_sum(mask_label, [-1, -2])
      sloss /= (denom + 1e-9)

      loss = tf.math.reduce_mean(loss)
      tf.summary.scalar("loss", loss, step=step)
      sloss = tf.where(sloss > args.ssl_margin, sloss, tf.zeros_like(sloss))
      sloss = tf.math.reduce_mean(sloss)
      tf.summary.scalar("sloss", sloss, step=step)

      sloss = ssl_weight * sloss
      total_loss = loss + sloss

  else:
    with tf.GradientTape(persistent=True) as tape, log_writer.as_default():
      loss, sloss, _ = m(
        (spec, ssl_spec, txt, spec_len, ssl_spec_len, txt_len),
        training = training, 
        ssl_loss = (not (args.ssl_weight == 0)),
        stop_grad = stop_grad,
        ctc = is_ctc)

      loss = tf.math.reduce_mean(loss)
      tf.summary.scalar("loss", loss, step=step)
      sloss = tf.where(sloss > args.ssl_margin, sloss, tf.zeros_like(sloss))
      sloss = tf.math.reduce_mean(sloss)
      tf.summary.scalar("sloss", sloss, step=step)

      sloss = ssl_weight * sloss
      total_loss = loss + sloss

      if args.ewc:
        ewc_loss = ewc.ewc_loss(args.ewc_lambda, m, 
          m_ewc.trainable_weights, (ssl_spec, ssl_spec_len))
        sloss = ewc_loss(m)
        total_loss += sloss

  if training:
    weights = m.trainable_weights
    grads = tape.gradient(total_loss, weights)
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

        if update_ema:
          for w, w_ema in zip(m.trainable_weights, m_ema.trainable_weights):
            delta = (1. - args.ssl_ema_beta) * (w - w_ema)
            w_ema.assign_add(delta)
                
        for idx, grad in enumerate(grads):
          if grad is None: continue
          accum_grads[idx].assign(tf.zeros_like(grad))

  return loss, sloss, ssl_weight

def run_eval_step(pcm, pcm_len):
  if not args.timit:
    # sample_len-wise inference
    hyps = []
    for idx in range(int(np.ceil(pcm_len / samp_len))):
      _pcm = pcm[:, idx * samp_len : (idx+1) * samp_len]
      _pcm_len = _pcm.shape[-1]

      if _pcm_len < samp_len:
        if _pcm_len < 200: continue # if > n_fft//2, error in reflect pad

      spec_dict = parse_data.conv_spec({'pcm': _pcm, 'pcm_len': pcm_len})
      _hyp = m(spec_dict['spec'], training=False)
      hyps.append(_hyp)

    _hyp = np.concatenate(hyps, 1)

  else:
    # bulk inference
    spec_dict = parse_data.conv_spec({'pcm': pcm, 'pcm_len': pcm_len})
    _hyp = m(spec_dict['spec'], training=False)

  maxids = np.argmax(np.squeeze(_hyp, 0), -1)

  if args.timit:
    return [str(e) for e in maxids]

  def greedy(hyp):
    truns = []; prev = 0
    for idx in hyp:
      if idx != prev:
        if prev != 0: truns.append(prev)
      prev = idx
    if prev != 0: truns.append(prev)
    return tokenizer.decode(truns)
  
  return greedy(maxids)

import logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)

logfile = os.path.join(args.output, "train.log")
if os.path.isfile(logfile): os.remove(logfile)
fh = logging.FileHandler(logfile)
logger.addHandler(fh)

ckpt = tf.train.Checkpoint(m)
ema_ckpt = tf.train.Checkpoint(m_ema)
if args.ewc:
  ewc_ckpt = tf.train.Checkpoint(m_ewc)
prev_val_loss = None; stall_cnt = 0

init_epoch = 0
if args.warm_start is not None:
  logger.info("warm start from {}".format(args.warm_start))

  expdir = os.path.abspath(os.path.dirname(args.warm_start))

  if not args.from_init:
    init_epoch = os.path.basename(args.warm_start).replace(".", "-").split("-")[1]
    try:
      init_epoch = int(init_epoch)
    except:
      init_epoch = 0

    opt_weight = os.path.join(expdir, "adam-{}-weight.npy".format(init_epoch))
    opt_cfg = os.path.join(expdir, "adam-{}-config.npy".format(init_epoch))

    if os.path.isfile(opt_weight) and os.path.isfile(opt_cfg):
      opt_weight = np.load(opt_weight, allow_pickle=True)
      opt_cfg = np.load(opt_cfg, allow_pickle=True).flatten()[0] 

  _in = np.zeros((args.batch_size, spec_len, 80), dtype=np.float32)
  _ = m(_in)
  ckpt.read(args.warm_start).assert_consumed()
  _ = m_ema(_in)
  ema_ckpt.read(args.warm_start).assert_consumed()
  if args.ewc:
    _ = m_ewc(_in)
    ewc_ckpt.read(args.warm_start).assert_consumed()

  if not args.from_init:
    if isinstance(opt_weight, np.ndarray):
      opt = tf.keras.optimizers.Adam.from_config(opt_cfg)
      lr.assign(opt_cfg["learning_rate"])

      grad_vars = m.trainable_weights
      zero_grads = [tf.zeros_like(w) for w in grad_vars]
      opt.apply_gradients(zip(zero_grads, grad_vars))
      opt.set_weights(opt_weight)

_traced_begin = 2; _traced_cnt = 10
if ssl_dataset is None:
  ssl_dataset = iter(lambda: None, 0)

for idx, (data, ssl_data) in enumerate(zip(dataset, ssl_dataset)):
  idx += init_epoch
  if idx > args.train_step: break
            
  # TODO not using (idx+1) to call apply_grads in initial run_step()
  accum = not (idx % args.accum_step == 0)
  update_ema = (idx % args.ssl_ema_update == 0)

  if ssl_data is not None:
    ssl_spec = ssl_data["spec"]
    ssl_spec_len = ssl_data["spec_len"]

  else:
    ssl_spec = data["spec"]
    ssl_spec_len = data["spec_len"]

  if args.profile:
    if idx > (init_epoch + _traced_begin) and _traced_cnt > 0:
      with tf.profiler.experimental.Trace('train', step_num=idx, _r=1):
        loss, sloss, sw = run_step(
          tf.cast(idx, tf.int64),
          data["spec"], ssl_spec, data["txt"], 
          data["spec_len"], ssl_spec_len, data["txt_len"],
          accum=accum, update_ema=update_ema, stop_grad=args.stop_grad)
      _traced_cnt -= 1

    elif idx > (init_epoch + _traced_begin) and  _traced_cnt == 0:
      tf.profiler.experimental.stop()
      _traced_cnt -= 1

    else:
        loss, sloss, sw = run_step(
          tf.cast(idx, tf.int64),
          data["spec"], ssl_spec, data["txt"], 
          data["spec_len"], ssl_spec_len, data["txt_len"],
          accum=accum, update_ema=update_ema, stop_grad=args.stop_grad)

  else:
    loss, sloss, sw = run_step(
      tf.cast(idx, tf.int64),
      data["spec"], ssl_spec, data["txt"], 
      data["spec_len"], ssl_spec_len, data["txt_len"],
      accum=accum, update_ema=update_ema, stop_grad=args.stop_grad)

  log_writer.flush()
  tf.summary.scalar("lr", lr, step=idx)

  if idx > init_epoch and idx % args.eval_step == 0:
    logger.info("gstep[{}] loss[{:.2f}] sloss[{:.2f}] lr[{:.2e}] sr[{:.2e}]".format(
      idx, loss, sloss, lr.numpy(), sw))

  if val_dataset is None:
    if args.accum_step == 1 or not accum:
      # follow tf.keras.optimizers.schedules.ExponentialDecay
      lr.assign(args.begin_lr * args.lr_decay_rate**(idx/args.lr_decay_step))

  elif idx > init_epoch and idx % args.val_step == 0:
    val_loss = 0; num_val = 0
    for val_data in val_dataset:
      val_loss += run_eval_step(val_data["spec"], val_data["ref"])
      num_val += 1
    val_loss /= num_val

    if prev_val_loss is None:
      prev_val_loss = val_loss
    else:
      if prev_val_loss < val_loss: stall_cnt += 1
      else: stall_cnt = 0

    if stall_cnt >= args.val_lr_update:
      lr.assign(lr / 2.)
      stall_cnt = 0
    
    logger.info("gstep[{}] val-loss[{:.2f}] lr[{:.2e}]".format(
      idx, val_loss, lr.numpy()))
    prev_val_loss = min(prev_val_loss, val_loss)

  if idx > init_epoch and idx % args.save_step == 0:
    pers = []
    expname = [e for e in args.output.split("/") if len(e) > 0][-1]
    resname = "{}-{}".format(expname, idx)

    with open(os.path.join("results", "{}.eval".format(resname)), "w") as f:
      for _pcm, pcm_ref in zip(eval_pcms, evals):
        _pcm_len = _pcm.shape[0]
        _pcm = np.expand_dims(_pcm, 0).astype(np.float32)

        _ref = [int(e) for e in pcm_ref.split()[1:]]
        hyp = run_eval_step(_pcm, _pcm_len)

        if args.timit:
          _per = metric.per([" ".join(hyp)], [" ".join([str(e) for e in _ref])])
        else:
          _per = metric.per([hyp], [tokenizer.decode(_ref)])
        
        pers.append(_per)
      
        f.write("{} {}\n".format(_per, " ".join(hyp)))
        f.flush()
    
      f.write("final: {}\n".format(np.mean(pers)))

    logger.info("gstep[{}] per[{:.4f}]".format(idx, np.mean(pers)))

    modelname = "model-{}.ckpt".format(idx)
    modelpath = os.path.join(args.output, modelname)
    ckpt.write(modelpath)

    optname = "adam-{}-weight".format(idx)
    optpath = os.path.join(args.output, optname)
    np.save(optpath, opt.get_weights())
    
    cfgname = "adam-{}-config".format(idx)
    cfgpath = os.path.join(args.output, cfgname)
    np.save(cfgpath, opt.get_config())

    logger.info("model is saved as {}".format(modelpath))
