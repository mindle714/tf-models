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
parser.add_argument("--tfrec2", type=str, required=False, default=None) 
parser.add_argument("--val-tfrec", type=str, required=False, default=None)
parser.add_argument("--batch-size", type=int, required=False, default=4) 
parser.add_argument("--accum-step", type=int, required=False, default=4)
parser.add_argument("--eval-step", type=int, required=False, default=100) 
parser.add_argument("--save-step", type=int, required=False, default=None) 
parser.add_argument("--val-step", type=int, required=False, default=5000) 
parser.add_argument("--train-step", type=int, required=False, default=None) 
parser.add_argument("--begin-lr", type=float, required=False, default=2e-4) 
parser.add_argument("--lr-decay-rate", type=float, required=False, default=0.96)
parser.add_argument("--lr-decay-step", type=float, required=False, default=None)
parser.add_argument("--val-lr-update", type=float, required=False, default=3) 
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--warm-start", type=str, required=False, default=None)
parser.add_argument("--stop-grad", action='store_true')
parser.add_argument("--from-init", action='store_true')
parser.add_argument("--profile", action='store_true')
args = parser.parse_args()

if args.save_step is None: args.save_step = 1000
if args.train_step is None: args.train_step = 10000
if args.lr_decay_step is None: args.lr_decay_step = 1000
eval_lists = [
  ("/data/hejung/timit/test_w2v.wav.phone", "none"),
  ("timit_test_w2v/snr0/test.wav.txt", "snr0"),
  ("timit_test_w2v/snr10/test.wav.txt", "snr10"),
  ("timit_test_w2v/snr20/test.wav.txt", "snr20")
]

import metric
import soundfile
import librosa

eval_pcms_list =[]
for eval_list, eval_name in eval_lists:
  assert os.path.isfile(eval_list)
  evals = [e.strip() for e in open(eval_list, "r").readlines()]
  eval_pcms = []
  for idx, pcm_ref in enumerate(evals):
    _pcm = pcm_ref.split()[0]
    _pcm, _ = soundfile.read(_pcm)
    eval_pcms.append(_pcm)

  eval_pcms_list.append((eval_pcms, eval_name))

import sys
import json

tfrec_args = os.path.join(args.tfrec, "ARGS")
with open(tfrec_args, "r") as f:
  _json = json.loads(f.readlines()[-1])
  samp_len = _json["samp_len"]
  txt_len = _json["text_len"]

if args.val_tfrec is not None:
  val_tfrec_args = os.path.join(args.val_tfrec, "ARGS")
  with open(val_tfrec_args, "r") as f:
    _json = json.loads(f.readlines()[-1])
    val_samp_len = _json["samp_len"]
    val_txt_len = _json["text_len"]
    val_spec_len = int((val_samp_len - 400 + 400) / 160) + 1 
  if val_samp_len != samp_len:
    sys.exit('validation data has sample length {}'.format(val_samp_len))

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
if args.tfrec2 is not None:
  tfrec_list += glob.glob(os.path.join(args.tfrec2, "train-*.tfrecord"))

dataset = parse_data.gen_train(tfrec_list, 
  samp_len, txt_len, no_spec=True,
  batch_size=args.batch_size, seed=seed)

val_dataset = None
if args.val_tfrec is not None:
  val_tfrec_list = glob.glob(os.path.join(args.val_tfrec, "train-*.tfrecord"))
  val_dataset = parse_data.gen_val(val_tfrec_list,
    val_samp_len, no_spec=True,
    batch_size=args.batch_size, seed=seed)

lr = tf.Variable(args.begin_lr, trainable=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr)

import wav2vec2
m = wav2vec2.wav2vec2_phone(num_class=50, use_last=False, use_layers=12)
is_ctc = False

_in = np.zeros((args.batch_size, samp_len), dtype=np.float32)
_ref = np.zeros((args.batch_size, txt_len), dtype=np.int32)
_in_len = np.ones((args.batch_size, 1), dtype=np.int32) * samp_len
_ref_len = np.ones((args.batch_size, 1), dtype=np.int32) * txt_len

_ = m((_in, _ref, _in_len, _ref_len),
  training = True, ctc = True)

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
def run_step(step, pcm, txt,
             samp_len, txt_len,
             training=True, accum=False,
             stop_grad=False):
  with tf.GradientTape(persistent=True) as tape, log_writer.as_default():
    loss = m(
      (pcm, txt, samp_len, txt_len),
      training = training, 
      stop_grad = stop_grad,
      ctc = is_ctc)

    loss = tf.math.reduce_mean(loss)
    tf.summary.scalar("loss", loss, step=step)

  if training:
    weights = m.trainable_weights
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

def run_eval_step(pcm, pcm_len):
  # bulk inference
  _hyp = m(pcm, training=False)

  maxids = np.argmax(np.squeeze(_hyp, 0), -1)
  return [str(e) for e in maxids]

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

  ckpt.read(args.warm_start)#.assert_consumed()

  if not args.from_init:
    if isinstance(opt_weight, np.ndarray):
      opt = tf.keras.optimizers.Adam.from_config(opt_cfg)
      lr.assign(opt_cfg["learning_rate"])

      grad_vars = m.trainable_weights
      zero_grads = [tf.zeros_like(w) for w in grad_vars]
      opt.apply_gradients(zip(zero_grads, grad_vars))
      opt.set_weights(opt_weight)

_traced_begin = 2; _traced_cnt = 10

for idx, data in enumerate(dataset):
  idx += init_epoch
  if idx > args.train_step: break
            
  # TODO not using (idx+1) to call apply_grads in initial run_step()
  accum = not (idx % args.accum_step == 0)
  
  _in_arg = [data["pcm"], data["txt"],
             data["pcm_len"], data["txt_len"]]

  def do_step():
    loss = run_step(
      tf.cast(idx, tf.int64), *_in_arg,
      accum=accum, stop_grad=args.stop_grad)
    return loss

  if args.profile:
    if idx > (init_epoch + _traced_begin) and _traced_cnt > 0:
      with tf.profiler.experimental.Trace('train', step_num=idx, _r=1):
        loss = do_step()
      _traced_cnt -= 1

    elif idx > (init_epoch + _traced_begin) and  _traced_cnt == 0:
      tf.profiler.experimental.stop()
      _traced_cnt -= 1

    else:
      loss = do_step()

  else:
    loss = do_step()

  log_writer.flush()
  tf.summary.scalar("lr", lr, step=idx)

  if idx > init_epoch and idx % args.eval_step == 0:
    logger.info("gstep[{}] loss[{:.2f}] lr[{:.2e}]".format(
      idx, loss, lr.numpy()))

  if val_dataset is None:
    if args.accum_step == 1 or not accum:
      # follow tf.keras.optimizers.schedules.ExponentialDecay
      lr.assign(args.begin_lr * args.lr_decay_rate**(idx/args.lr_decay_step))

  elif idx > init_epoch and idx % args.val_step == 0:
    val_loss = 0; num_val = 0
    for val_data in val_dataset:
      val_loss += run_eval_step(val_data["pcm"], val_data["ref"])
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

    for eval_pcms, eval_name in eval_pcms_list:
      with open(os.path.join("results", "{}_{}.eval".format(resname, eval_name)), "w") as f:
        for _pcm, pcm_ref in zip(eval_pcms, evals):
          _pcm_len = _pcm.shape[0]
          _pcm = np.expand_dims(_pcm, 0).astype(np.float32)

          _ref = [int(e) for e in pcm_ref.split()[1:]]
          hyp = run_eval_step(_pcm, _pcm_len)

          _per = metric.per([" ".join(hyp)], [" ".join([str(e) for e in _ref])])
        
          pers.append(_per)
       
          f.write("{} {}\n".format(_per, " ".join(hyp)))
          f.flush()
    
        f.write("final: {}\n".format(np.mean(pers)))

      logger.info("gstep[{}] type[{}] per[{:.4f}]".format(idx, eval_name, np.mean(pers)))

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
