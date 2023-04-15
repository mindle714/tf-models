import tensorflow as tf
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tfrecs = glob.glob(os.path.join("libriphone_v4", "train-*.tfrecord"))
data = tf.data.TFRecordDataset(tfrecs)

import soundfile
import numpy as np

i = 0
for rec in data.take(3):
  ex = tf.train.Example()
  ex.ParseFromString(rec.numpy())
  print(len(ex.features.feature['spec'].float_list.value))
  #pcm = np.array(list(ex.features.feature['pcm'].float_list.value))
  #ref = np.array(list(ex.features.feature['ref'].float_list.value))
  i += 1
