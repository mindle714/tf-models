import librosa
import numpy as np

def magnitude(e): return np.abs(e)
def phase(e): return np.arctan2(e.imag, e.real)
def polar(mag, phase):
  return mag * (np.cos(phase) + np.sin(phase) * 1j)

def lowpass(pcm, thres):
  f = librosa.stft(pcm)
  f_thres = int(f.shape[0] * thres)
  mask = np.concatenate([
    np.ones(f_thres), np.zeros(f.shape[0]-f_thres)])

  mag = magnitude(f) * np.expand_dims(mask, -1)
  ph = phase(f)
  f2 = polar(mag, ph)

  pcm2 = librosa.istft(f2, length=pcm.shape[0])
  return pcm2
