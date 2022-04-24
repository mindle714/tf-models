import tensorflow as tf
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tfrecs = glob.glob(os.path.join("wsj0", "train-*.tfrecord"))
data = tf.data.TFRecordDataset(tfrecs)

import soundfile
import numpy as np

i = 0
for rec in data.take(7):
  ex = tf.train.Example()
  ex.ParseFromString(rec.numpy())
  #print(ex.features.feature['samp_len'].int64_list.value)
  pcm = np.array(list(ex.features.feature['pcm'].float_list.value))
  ref = np.array(list(ex.features.feature['ref'].float_list.value))
  soundfile.write("pcm_{}.wav".format(i), pcm, 16000)
  soundfile.write("ref_{}.wav".format(i), ref, 16000)
  i += 1
