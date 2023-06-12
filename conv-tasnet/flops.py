import model
import numpy as np
import tensorflow as tf

session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

with graph.as_default():
  with session.as_default():
    inputs = tf.keras.Input(shape=(32000,))
    m = model.convtas()
    km = tf.keras.Model(inputs=inputs, outputs=m((inputs, inputs, inputs)))
    print(km.summary())
#    _in = np.zeros((1,32000), dtype=np.float32)
#    _ = m(_in)

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(tf.compat.v1.get_default_graph(), options=opts).total_float_ops
    print(flops)
