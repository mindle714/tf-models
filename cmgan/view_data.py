import tensorflow as tf
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tfrecs = glob.glob(os.path.join("vbank_demand_28", "train-*.tfrecord"))
data = tf.data.TFRecordDataset(tfrecs)

import soundfile
import numpy as np
import viz

i = 0
for rec in data.take(3):
  ex = tf.train.Example()
  ex.ParseFromString(rec.numpy())
  #print(ex.features.feature['samp_len'].int64_list.value)
  pcm = np.array(list(ex.features.feature['pcm'].float_list.value))
  ref = np.array(list(ex.features.feature['ref'].float_list.value))
  soundfile.write("pcm_lpf_{}.wav".format(i), pcm, 16000)
  soundfile.write("ref_lpf_{}.wav".format(i), ref, 16000)
  viz.plot_spec(pcm, "pcm_1pf_{}".format(i), 16000)
  viz.plot_spec(ref, "ref_1pf_{}".format(i), 16000)
  i += 1
