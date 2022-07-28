import tffts
import numpy as np
import tensorflow as tf

N = 16
a = np.random.uniform(size=N)

m = tffts.tfft(N)
rxs, ixs = m(np.reshape(a, [1, 1, N]))
rx = rxs[-1].numpy(); ix = ixs[-1].numpy()
assert np.allclose(np.fft.fft(a), rx + 1j * ix)

im = tffts.tfft(N)
x = im((rx, ix), inverse=True)
assert np.allclose(x, a)

'''
a = np.random.uniform(size=16000)

ms = tffts.tffts([16, 32, 64], [8, 16, 32])
x = ms(a)
print(x.shape)
#assert np.allclose(x, a)

ckpt = tf.train.Checkpoint(ms)
ckpt.write("tffts.ckpt")
'''
