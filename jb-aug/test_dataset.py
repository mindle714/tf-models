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
parser.add_argument("--batch-size", type=int, required=False, default=8) 
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

import parse_data
import glob

tfrec_list = glob.glob(os.path.join(args.tfrec, "train-*.tfrecord"))
adapt_tfrec_list = [e for e in tfrec_list if 'train-0' in e]
tfrec_list = [e for e in tfrec_list if 'train-0' not in e]

for _ in range(3):
  dataset = parse_data.gen_train(tfrec_list, 
    (samp_len if no_spec else spec_len), txt_len,
    no_spec=no_spec, batch_size=args.batch_size, seed=seed)

  for idx, data in enumerate(dataset):
    _in_arg = [data["spec"], data["txt"],
               data["spec_len"], data["txt_len"]]
    print(data["spec"][0][0])
    break
