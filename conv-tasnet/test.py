import model
import tensorflow as tf
import numpy as np

shape=[3,7,3]
shapesize=shape[0]*shape[1]*shape[2]

orig_np = np.random.uniform(size=shapesize).astype(np.float32)
est_np = np.random.uniform(size=shapesize).astype(np.float32)

#orig = tf.reshape(tf.range(shapesize, dtype=tf.float32), shape)/float(shapesize)
#est = tf.reshape(tf.range(shapesize, dtype=tf.float32), shape)/float(shapesize)+1.
orig = tf.reshape(tf.convert_to_tensor(orig_np), shape)
est = tf.reshape(tf.convert_to_tensor(est_np), shape)

res, _ = model.si_snr(orig, est)
print(-np.mean(res))

import torch

#orig = torch.arange(shapesize).reshape(shape).transpose(1,2)/float(shapesize)
#est = torch.arange(shapesize).reshape(shape).transpose(1,2)/float(shapesize)+1.
orig = torch.from_numpy(orig_np).reshape(shape).transpose(1,2)
est = torch.from_numpy(est_np).reshape(shape).transpose(1,2)

import pit_criterion
print(-torch.mean(pit_criterion.cal_si_snr_with_pit(orig, est)[0]))
