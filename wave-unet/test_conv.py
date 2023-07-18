import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(1234)

'''
a = tf.keras.layers.DepthwiseConv1D(3)
_in = tf.reshape(tf.range(64, dtype=tf.float32), [4,4,4])
print(a(_in))
'''

#############################################
#             depthwiseconv2d               #
#############################################
#x = np.arange(12, dtype=np.float32).reshape((1, 3, 2, 2))
#kernel = np.arange(8, dtype=np.float32).reshape((2, 1, 2, 2))
x = np.random.uniform(size=(2, 4, 3, 3))
kernel = np.random.uniform(size=(2, 1, 3, 2))
res = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
                       padding='SAME').numpy()

#print(res)

kernel = np.expand_dims(kernel[:,:,:,0], -1)
res = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
                       padding='SAME').numpy()
print(res)
#print(kernel)

x_chs = []
for idx in range(kernel.shape[2]):
  x_ch = tf.slice(x, [0, 0, 0, idx], [-1, -1, -1, 1])
  kernel_ch = tf.slice(kernel, [0, 0, idx, 0], [-1, -1, 1, -1])
  x_chs.append(tf.nn.conv2d(x_ch, kernel_ch, [1,1,1,1], 'SAME').numpy())

print("x_ch")
x_ch = np.concatenate(x_chs, -1)
print(x_ch)

print(">>>")
print(res - x_ch)
print("<<<")

#############################################
#             depthwiseconv1d               #
#############################################
x = np.random.uniform(size=(2, 4, 3))
kernel = np.random.uniform(size=(2, 3, 1))
res = tf.nn.depthwise_conv2d(np.expand_dims(x, 1), 
        np.expand_dims(kernel, 0), strides=[1, 1, 1, 1],
        padding='SAME').numpy()

print(res)

x_chs = []
for idx in range(kernel.shape[1]):
  x_ch = tf.slice(x, [0, 0, idx], [-1, -1, 1])
  kernel_ch = tf.slice(kernel, [0, idx, 0], [-1, 1, -1])
  x_chs.append(tf.nn.conv2d(np.expand_dims(x_ch, 1), 
      np.expand_dims(kernel_ch, 0), [1,1,1,1], 'SAME').numpy())

print("x_ch")
x_ch = np.concatenate(x_chs, -1)
print(x_ch)

print(">>>")
print(res - x_ch)
print("<<<")

res = tf.nn.depthwise_conv2d(np.expand_dims(x, 1), 
        np.expand_dims(kernel, 0), strides=[1, 2, 2, 1],
        padding='SAME').numpy()

print(res)

x_chs = []
for idx in range(kernel.shape[1]):
  x_ch = tf.slice(x, [0, 0, idx], [-1, -1, 1])
  kernel_ch = tf.slice(kernel, [0, idx, 0], [-1, 1, -1])
#  x_chs.append(tf.nn.conv2d(tf.expand_dims(x_ch, 1), 
#      tf.expand_dims(kernel_ch, 0), 2, 'SAME').numpy())
  x_chs.append(np.expand_dims(tf.nn.conv1d(x_ch, 
      kernel_ch, 2, 'SAME').numpy(), 1))

print("x_ch")
x_ch = np.concatenate(x_chs, -1)
print(x_ch)

print(">>>")
print(res - x_ch)
print("<<<")
