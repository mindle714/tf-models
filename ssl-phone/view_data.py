import tensorflow as tf
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tfrecs = glob.glob(os.path.join("libriphone", "train-*.tfrecord"))
data = tf.data.TFRecordDataset(tfrecs)

import soundfile
import numpy as np

i = 0
for rec in data.take(3):
  ex = tf.train.Example()
  ex.ParseFromString(rec.numpy())
  pcm = np.array(list(ex.features.feature['pcm'].float_list.value))
  txt = np.array(list(ex.features.feature['txt'].int64_list.value))
  soundfile.write("pcm_{}.wav".format(i), pcm, 16000)
  print(txt)
  pcm_len = np.array(list(ex.features.feature['pcm_len'].int64_list.value))
  txt_len = np.array(list(ex.features.feature['txt_len'].int64_list.value))
  print(pcm_len, txt_len)
  i += 1
