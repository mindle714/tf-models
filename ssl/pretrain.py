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
parser.add_argument("--batch-size", type=int, required=False, default=32) 
parser.add_argument("--epoch", type=int, required=False, default=10)
parser.add_argument("--log-step", type=int, required=False, default=100) 
parser.add_argument("--warm-start", type=str, required=True) 
parser.add_argument("--output", type=str, required=True) 
args = parser.parse_args()

import json
tfrec_args = os.path.join(args.tfrec, "ARGS")
with open(tfrec_args, "r") as f:
  samp_len = json.loads(f.readlines()[-1])["samp_len"]

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
  batch_size=args.batch_size, seed=seed, epoch=args.epoch)

import tera
m = tera.tera_unet()

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

ckpt = tf.train.Checkpoint(m)

_in = np.zeros((args.batch_size, samp_len), dtype=np.float32)
_ = m(_in) # generate tera object
fs = m.tera(_in, training=False)
ckpt.read(args.warm_start).assert_consumed()

fs_mean = []; fs_var = []
for f in fs:
  fs_mean.append(tf.Variable(tf.zeros_like(tf.math.reduce_mean(f, 0))))
  fs_var.append(tf.Variable(tf.zeros_like(tf.math.reduce_variance(f, 0))))

@tf.function
def run_step(step, ref, beta_1 = 0.999, beta_2 = 0.999):
  _fs = m.tera(ref, training=False)

  d_means = []; d_vars = []
  for _f, f_mean, f_var in zip(_fs, fs_mean, fs_var):
    d_mean = m.tera.enc.moving_avg(step, f_mean, 
      tf.math.reduce_mean(_f, 0), beta_1)
    d_var = m.tera.enc.moving_avg(step, f_var, 
      tf.math.reduce_variance(_f, 0), beta_2)

    d_means.append(tf.norm(d_mean))
    d_vars.append(tf.norm(d_var))

  for idx in range(len(_fs)):
    tf.summary.scalar("d_mean_{}".format(idx), d_means[idx], step=step)
    tf.summary.scalar("d_var_{}".format(idx), d_vars[idx], step=step)

  return d_means, d_vars

import logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)
logger.info("warm start from {}".format(args.warm_start))

logfile = os.path.join(args.output, "pretrain.log")
if os.path.isfile(logfile): os.remove(logfile)
fh = logging.FileHandler(logfile)
logger.addHandler(fh)

for idx, data in enumerate(dataset):
  d_means, d_vars = run_step(tf.cast(idx, tf.int64), data["ref"])
  log_writer.flush()

  if idx > 0 and idx % args.log_step == 0:
    d_mean = " ".join(["{:.5f}".format(e.numpy()) for e in d_means])
    d_var = " ".join(["{:.5f}".format(e.numpy()) for e in d_vars])
    logger.info("gstep[{}] mean-delta[{}] var-delta[{}]".format(
        idx, d_mean, d_var))

for idx, (f_mean, f_var) in enumerate(zip(fs_mean, fs_var)):
  if idx < len(m.tera.enc.layers):
    m.tera.enc.ref_means[idx].assign(f_mean)
    m.tera.enc.ref_vars[idx].assign(f_var)
    
modelname = "model.ckpt"
modelpath = os.path.join(args.output, modelname)
ckpt.write(modelpath)

fsname = "mean-feat"
fspath = os.path.join(args.output, fsname)
np.save(fspath, [e.numpy() for e in fs_mean])

fsname = "var-feat"
fspath = os.path.join(args.output, fsname)
np.save(fspath, [e.numpy() for e in fs_var])

logger.info("model is saved as {}".format(args.output))
