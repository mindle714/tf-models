import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

dataset = tf.data.Dataset.range(5)
#dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
#dataset = dataset.repeat(2)
#dataset = dataset.batch(3)
for e in dataset:
  print(e)
for e in dataset:
  print(e)
