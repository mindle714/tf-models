import tensorflow as tf
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tfrecs = glob.glob(os.path.join("8k_tr_min_rem_v2_prebatch", "train-*.tfrecord"))
data = tf.data.TFRecordDataset(tfrecs)

for rec in data.take(7):
  ex = tf.train.Example()
  ex.ParseFromString(rec.numpy())
  #print(ex.features.feature['samp_len'].int64_list.value)
  print(len(ex.features.feature['s1'].float_list.value))
