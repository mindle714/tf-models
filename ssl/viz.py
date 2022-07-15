import librosa
import librosa.display
import numpy as np

def plot_spec(pcm, fname, sr, frame_sec=0.02):
  f = librosa.stft(pcm, n_fft=int(sr*frame_sec))
  db = librosa.amplitude_to_db(np.abs(f), ref=np.max)

  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(19.2, 4.8))
  librosa.display.specshow(db, x_axis='time', y_axis='linear',
    sr=sr, hop_length=int(sr*frame_sec)//4)
  plt.colorbar()
  plt.savefig("{}.png".format(fname))

