import tensorflow as tf
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import soundfile
import numpy as np

tfrecs = glob.glob(os.path.join("timit_w2v_snr0_v3", "train-0.tfrecord"))
data = tf.data.TFRecordDataset(tfrecs)

idx = 0
for rec in data.take(2):
  ex = tf.train.Example()
  ex.ParseFromString(rec.numpy())
  soundfile.write("snr0_v3_{}.wav".format(idx), ex.features.feature['pcm'].float_list.value, 16000)
  print(ex.features.feature['pcm_len'])
  idx += 1

tfrecs = glob.glob(os.path.join("timit_w2v_snr0_v3_jb", "train-0.tfrecord"))
data = tf.data.TFRecordDataset(tfrecs)

idx = 0
for rec in data.take(2):
  ex = tf.train.Example()
  ex.ParseFromString(rec.numpy())
  soundfile.write("snr0_v3_jb_{}.wav".format(idx), ex.features.feature['pcm'].float_list.value, 16000)
  print(ex.features.feature['pcm_len'])
  idx += 1
