import librosa
import numpy as np
import cv2

def magnitude(e): return np.abs(e)
def phase(e): return np.arctan2(e.imag, e.real)
def polar(mag, phase):
  return mag * (np.cos(phase) + np.sin(phase) * 1j)

def pattern_mask(pcm, ratio=0.25, fmin=0., fmax=1.):
  f = librosa.stft(pcm)

  #interp = np.ones((f.shape[0]//128, f.shape[1]//16), dtype=np.float32)
  interp = np.ones((6, 6), dtype=np.float32)
  for i in range(int(interp.size*ratio)):
    x = np.random.randint(interp.shape[0])
    y = np.random.randint(interp.shape[1])
    interp[x][y] = 0.

  interp2 = cv2.resize(interp, dsize=f.shape, interpolation=cv2.INTER_CUBIC).T
  fmin = int(f.shape[0] * fmin); fmax = int(f.shape[0] * fmax)
  interp2 = np.concatenate([
    np.ones((fmin, f.shape[1])),
    interp2[fmin:fmax, :],
    np.ones((f.shape[0]-fmax, f.shape[1]))
  ])

  mag = magnitude(f) * interp2
  ph = phase(f)
  f2 = polar(mag, ph)

  pcm2 = librosa.istft(f2, length=pcm.shape[0])
  return pcm2
