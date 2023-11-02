import os
import random
import numpy as np
import tensorflow as tf
import sys
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--warm-start", type=str, required=False, default=None)
args = parser.parse_args()

import model_v2
m = model_v2.wav2vec2_t5_phone(num_class=50)

samp_len = 128000; txt_len = 399
_in = np.zeros((1, samp_len), dtype=np.float32)
_ref = np.zeros((1, txt_len), dtype=np.int32)
_in_len = np.ones((1, 1), dtype=np.int32) * samp_len
_ref_len = np.ones((1, 1), dtype=np.int32) * txt_len

_ = m((_in, _ref, _in_len, _ref_len),
  training = True, ctc = True)

ckpt = tf.train.Checkpoint(m)
ckpt.read(args.warm_start)

from matplotlib import pyplot as plt
fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
w = m.wav2vec2_t5.encoder_1w.numpy()
ax.imshow(w, interpolation='nearest')
print(w.mean(), w.std())

ax2 = fig.add_subplot(1, 2, 2)
w = m.wav2vec2_t5.encoder_2w.numpy()
ax2.imshow(w, interpolation='nearest')
print(w.mean(), w.std())

plt.savefig('viz_v2.png')
