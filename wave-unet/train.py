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
parser.add_argument("--save-step", type=int, required=False, default=1000) 
parser.add_argument("--val-step", type=int, required=False, default=5000) 
parser.add_argument("--train-step", type=int, required=False, default=100000) 
parser.add_argument("--begin-lr", type=float, required=False, default=1e-4) 
parser.add_argument("--lr-decay-rate", type=float, required=False, default=0.96)
parser.add_argument("--lr-decay-step", type=float, required=False, default=2000.)
parser.add_argument("--val-lr-update", type=float, required=False, default=3) 
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--warm-start", type=str, required=False, default=None) 
args = parser.parse_args()

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
gen_opt = tf.keras.optimizers.Adam(learning_rate=lr)
disc_opt = tf.keras.optimizers.Adam(learning_rate=lr)

import model
m = model.wavegan()

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
  with tf.GradientTape(persistent=True) as tape, log_writer.as_default():
    _, gen_loss, disc_loss = m((pcm, ref), training=training)
    tf.summary.scalar("gen_loss", gen_loss, step=step)
    tf.summary.scalar("disc_loss", disc_loss, step=step)

  if training:
    gen_vars = [e for e in m.trainable_weights if 'waveunet' in e.name]
    disc_vars = [e for e in m.trainable_weights if 'disc' in e.name]
    assert (len(gen_vars) + len(disc_vars)) == len(m.trainable_weights)

    gen_grads = tape.gradient(gen_loss, gen_vars)
    gen_grads, _ = tf.clip_by_global_norm(gen_grads, 5.)
    gen_opt.apply_gradients(zip(gen_grads, gen_vars))

    disc_grads = tape.gradient(disc_loss, disc_vars)
    disc_grads, _ = tf.clip_by_global_norm(disc_grads, 5.)
    disc_opt.apply_gradients(zip(disc_grads, disc_vars))

  return gen_loss, disc_loss

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
  expname = expdir.split("/")[-1]
  init_epoch = os.path.basename(args.warm_start).replace(".", "-").split("-")[1]
  init_epoch = int(init_epoch)

  gen_opt_weight = os.path.join(expdir, "adam-gen-{}-weight.npy".format(init_epoch))
  gen_opt_cfg = os.path.join(expdir, "adam-gen-{}-config.npy".format(init_epoch))

  gen_opt_weight = np.load(gen_opt_weight, allow_pickle=True)
  gen_opt_cfg = np.load(gen_opt_cfg, allow_pickle=True).flatten()[0] 

  disc_opt_weight = os.path.join(expdir, "adam-disc-{}-weight.npy".format(init_epoch))
  disc_opt_cfg = os.path.join(expdir, "adam-disc-{}-config.npy".format(init_epoch))

  disc_opt_weight = np.load(disc_opt_weight, allow_pickle=True)
  disc_opt_cfg = np.load(disc_opt_cfg, allow_pickle=True).flatten()[0] 

  _in = np.zeros((args.batch_size, samp_len), dtype=np.float32)
  _ = m((_in, _in))
  ckpt.read(args.warm_start)

  gen_opt = tf.keras.optimizers.Adam.from_config(gen_opt_cfg)
  disc_opt = tf.keras.optimizers.Adam.from_config(disc_opt_cfg)

  assert gen_opt_cfg["learning_rate"] == disc_opt_cfg["learning_rate"]
  lr.assign(gen_opt_cfg["learning_rate"])

  gen_vars = [e for e in m.trainable_weights if 'waveunet' in e.name]
  disc_vars = [e for e in m.trainable_weights if 'disc' in e.name]
  
  gen_zero_grads = [tf.zeros_like(w) for w in gen_vars]
  gen_opt.apply_gradients(zip(gen_zero_grads, gen_vars))
  gen_opt.set_weights(gen_opt_weight)

  disc_zero_grads = [tf.zeros_like(w) for w in disc_vars]
  disc_opt.apply_gradients(zip(disc_zero_grads, disc_vars))
  disc_opt.set_weights(disc_opt_weight)

for idx, data in enumerate(dataset):
  idx += init_epoch
  if idx > args.train_step: break

  gen_loss, disc_loss = run_step(tf.cast(idx, tf.int64), data["pcm"], data["ref"])
  log_writer.flush()

  if idx > init_epoch and idx % args.eval_step == 0:
    logger.info("gstep[{}] gen_loss[{:.2f}] disc_loss[{:.2f}] lr[{:.2e}]".format(
      idx, gen_loss, disc_loss, lr.numpy()))

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
    
    logger.info("gstep[{}] val-loss[{:.2f}] lr[{:.2e}]".format(
      idx, val_loss, lr.numpy()))
    prev_val_loss = min(prev_val_loss, val_loss)

  if idx > init_epoch and idx % args.save_step == 0:
    modelname = "model-{}.ckpt".format(idx)
    modelpath = os.path.join(args.output, modelname)
    ckpt.write(modelpath)

    optname = "adam-gen-{}-weight".format(idx)
    optpath = os.path.join(args.output, optname)
    np.save(optpath, gen_opt.get_weights())
    
    cfgname = "adam-gen-{}-config".format(idx)
    cfgpath = os.path.join(args.output, cfgname)
    np.save(cfgpath, gen_opt.get_config())

    optname = "adam-disc-{}-weight".format(idx)
    optpath = os.path.join(args.output, optname)
    np.save(optpath, disc_opt.get_weights())
    
    cfgname = "adam-disc-{}-config".format(idx)
    cfgpath = os.path.join(args.output, cfgname)
    np.save(cfgpath, disc_opt.get_config())

    logger.info("model is saved as {}".format(modelpath))
