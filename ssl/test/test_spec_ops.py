import tensorflow as tf
import spec_ops
import numpy as np

p = np.random.uniform(low=-np.pi, high=np.pi, size=(100,))
p = tf.constant(p)
up = spec_ops.unwrap(p)
pp = tf.math.floormod(up + np.pi, 2.0 * np.pi) - np.pi

print(p - pp)
assert np.allclose(p, pp)
