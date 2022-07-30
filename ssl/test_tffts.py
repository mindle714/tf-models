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

hlen = 8
a = np.random.uniform(size=96)
frame = tf.signal.frame(a, frame_length=N, frame_step=hlen, pad_end=True)
frame = np.expand_dims(frame, 0)

from scipy import signal
win = signal.get_window('hann', N)
m.window.assign(win.reshape([1, 1, N]))
rx, ix = m(frame)

win_sq = tf.reshape(tf.tile(win, [frame.shape[1]]), [-1, win.shape[-1]])
win_sq = win_sq ** 2
win_sq = tf.signal.overlap_and_add(win_sq, hlen)[:96]
win_sq = tf.cast(win_sq, tf.float32)

__a = m((rx, ix), inverse=True)
#__a = __a * win
_a = tf.signal.overlap_and_add(__a, hlen)[:,:96]
_a = tf.where(win_sq > 1e-10, _a/win_sq, _a)

print(a)
print(_a)
#assert np.allclose(a, _a)

m = tffts.tstft(N, hlen)
_ = m(a)

m.tfft.window.assign(win.reshape([1, 1, N]))
rx, ix = m(a.reshape([1, -1]))
_a = m((rx, ix), inverse=True)
assert np.allclose(a, _a)

'''
a = np.random.uniform(size=16000)

ms = tffts.tffts([16, 32, 64], [8, 16, 32])
x = ms(a)
print(x.shape)
#assert np.allclose(x, a)

ckpt = tf.train.Checkpoint(ms)
ckpt.write("tffts.ckpt")
'''
