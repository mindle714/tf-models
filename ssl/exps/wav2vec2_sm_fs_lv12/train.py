import os
import random
import numpy as np
import tensorflow as tf

seed = 1234
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tfrec", type=str, required=True) 
parser.add_argument("--val-tfrec", type=str, required=False, default=None)
parser.add_argument("--batch-size", type=int, required=False, default=8) 
parser.add_argument("--eval-step", type=int, required=False, default=100) 
parser.add_argument("--save-step", type=int, required=False, default=10000) 
parser.add_argument("--val-step", type=int, required=False, default=10000) 
parser.add_argument("--train-step", type=int, required=False, default=100000) 
parser.add_argument("--begin-lr", type=float, required=False, default=1e-4) 
parser.add_argument("--lr-decay-rate", type=float, required=False, default=0.96)
parser.add_argument("--lr-decay-step", type=float, required=False, default=2000.)
parser.add_argument("--val-lr-update", type=float, required=False, default=3) 
parser.add_argument("--begin-factors", type=float, nargs="+", default=[])
parser.add_argument("--beta-r", type=float, nargs="+", default=[]) 
parser.add_argument("--beta-m", type=float, nargs="+", default=[])
parser.add_argument("--fnorm-schedule", type=str, required=False,
  choices=["linear", "exp", "cos", "stair"], default="linear")
parser.add_argument("--fnorm-steps", type=float, nargs="+", default=[])
parser.add_argument("--fnorm-decay", type=float, required=False, default=100) 
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--warm-start", type=str, required=False, default=None)
parser.add_argument("--from-init", action='store_true')
args = parser.parse_args()

if len(args.beta_r) == 0:
  args.beta_r = [0.999, 0.999, 0.999, 0.999, 0.999, 0.999]
assert len(args.beta_r) == 6

if len(args.beta_m) == 0:
  args.beta_m = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
assert len(args.beta_m) == 6

if len(args.fnorm_steps) == 0:
  args.fnorm_steps = [70000, 70000, 70000, 70000, 70000, 70000]
assert len(args.fnorm_steps) == 6

if len(args.begin_factors) == 0:
  args.begin_factors = [0., 0., 0., 0., 0., 0.]
assert len(args.begin_factors) == 6

import json
tfrec_args = os.path.join(args.tfrec, "ARGS")
with open(tfrec_args, "r") as f:
  samp_len = json.loads(f.readlines()[-1])["samp_len"]

if args.val_tfrec is not None:
  val_tfrec_args = os.path.join(args.val_tfrec, "ARGS")
  with open(val_tfrec_args, "r") as f:
    val_samp_len = json.loads(f.readlines()[-1])["samp_len"]
  if val_samp_len != samp_len:
    sys.exit('validation data has sample length {}'.format(val_samp_len))

import types
import sys
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
dataset = parse_data.gen_train(tfrec_list, samp_len,
  batch_size=args.batch_size, seed=seed)

val_dataset = None
if args.val_tfrec is not None:
  val_tfrec_list = glob.glob(os.path.join(args.val_tfrec, "train-*.tfrecord"))
  val_dataset = parse_data.gen_val(val_tfrec_list, samp_len,
    batch_size=args.batch_size, seed=seed)

lr = tf.Variable(args.begin_lr, trainable=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr)

import wav2vec2
m = wav2vec2.wav2vec2_unet(beta_r=args.beta_r, beta_m=args.beta_m)

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

@tf.function
def run_step(step, pcm, ref, training=True):
  for i in range(len(m.wav2vec2.fe.factors)):
    m.wav2vec2.fe.factors[i].assign(
      factor_decay(tf.cast(step, dtype=tf.float32), 
      args.fnorm_steps[i], args.begin_factors[i]))

  with tf.GradientTape() as tape, log_writer.as_default():
    loss = m((pcm, ref), training=training)
    loss = tf.math.reduce_mean(loss)
    tf.summary.scalar("loss", loss, step=step)
    total_loss = loss

  if training:
    train_weights = [e for e in m.trainable_weights if 'frozen' not in e.name]
    assert len(m.trainable_weights) - len(train_weights) == 33
    grads = tape.gradient(total_loss, train_weights)
    grads, _ = tf.clip_by_global_norm(grads, 5.)
    opt.apply_gradients(zip(grads, train_weights))

  return loss

import logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)

logfile = os.path.join(args.output, "train.log")
if os.path.isfile(logfile): os.remove(logfile)
fh = logging.FileHandler(logfile)
logger.addHandler(fh)

ckpt = tf.train.Checkpoint(m)
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

  _in = np.zeros((args.batch_size, samp_len), dtype=np.float32)
  _ = m((_in, None))
  _ = m.wav2vec2((_in, None))
  _ = m.wav2vec2_frozen((_in, None))
  ckpt.read(args.warm_start).assert_consumed()

  if not args.from_init:
    if isinstance(opt_weight, np.ndarray):
      opt = tf.keras.optimizers.Adam.from_config(opt_cfg)
      lr.assign(opt_cfg["learning_rate"])

      grad_vars = m.trainable_weights
      zero_grads = [tf.zeros_like(w) for w in grad_vars]
      opt.apply_gradients(zip(zero_grads, grad_vars))
      opt.set_weights(opt_weight)

def cos_decay(step, decay_steps, init_lr, alpha=0.):
  step = min(step, decay_steps)
  cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
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

def stair_decay(step, decay_steps, init_lr, end_lr = 0., decay_unit = args.fnorm_decay):
  if step > decay_steps: return end_lr
  _step = (step // decay_unit) * decay_unit
  return ((end_lr - init_lr) / decay_steps) * _step + init_lr

if args.fnorm_schedule == "linear":
  factor_decay = linear_decay
elif args.fnorm_schedule == "exp":
  factor_decay = exp_decay
elif args.fnorm_schedule == "cos":
  factor_decay = cos_decay
elif args.fnorm_schedule == "stair":
  factor_decay = stair_decay
else:
  assert False, "unknown type {}".format(args.fnorm_schedule)

for idx, data in enumerate(dataset):
  idx += init_epoch
  if idx > args.train_step: break

  loss = run_step(tf.cast(idx, tf.int64), data["pcm"], data["ref"])
  log_writer.flush()

  if idx > init_epoch and idx % args.eval_step == 0:
    logger.info("gstep[{}] loss[{:.2f}] lr[{:.2e}] flr[{}]".format(
      idx, loss, lr.numpy(),
      ",".join([str("{:.2f}".format(e.numpy())) for e in m.wav2vec2.fe.factors])))

  if val_dataset is None:
    # follow tf.keras.optimizers.schedules.ExponentialDecay
    lr.assign(args.begin_lr * args.lr_decay_rate**(idx/args.lr_decay_step))

  elif idx > init_epoch and idx % args.val_step == 0:
    val_loss = 0; num_val = 0
    for val_data in val_dataset:
      val_loss += run_step(tf.cast(idx, tf.int64),
        val_data["pcm"], val_data["ref"], training=False)
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
    
    logger.info("gstep[{}] val-loss[{:.2f}] lr[{:.2e}] flr[{}]".format(
      idx, val_loss, lr.numpy(),
      ",".join([str("{:.2f}".format(e.numpy())) for e in m.wav2vec2.fe.factors])))
    prev_val_loss = min(prev_val_loss, val_loss)

  if idx > init_epoch and idx % args.save_step == 0:
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
