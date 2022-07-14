import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import model
import tensorflow as tf

inputs = tf.keras.Input(shape=(16384,))
m = model.waveunet()
km = tf.keras.Model(inputs=inputs, outputs=m((inputs, inputs)))
print(km.summary())