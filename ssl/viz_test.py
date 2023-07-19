import librosa
import librosa.display
import numpy as np

import cv2
import soundfile
pcm, sr = soundfile.read("clean_16k.wav")

def norm(e):
  e_norm = e - (np.max(e) + np.min(e)) / 2#np.mean(e)
  e_norm /= 2 * np.max(e_norm) #np.std(e_norm)
  e_norm += 0.5
  return e_norm

frame_sec=0.02; n_fft=400; hop_len=80 #n_fft=int(sr*frame_sec)
f = librosa.stft(pcm, n_fft=n_fft, hop_length=hop_len)
mag = np.abs(f)
power = mag**2
db = librosa.power_to_db(power) #librosa.amplitude_to_db(mag)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(19.2, 4.8))
mesh = librosa.display.specshow(db, #x_axis='time', y_axis='linear',
  sr=sr, hop_length=int(sr*frame_sec)//4, cmap='jet')

rgba = mesh.to_rgba(mesh.get_array().reshape(db.shape))
print(rgba.shape)
print(db.shape)

#fig.tight_layout()
plt.savefig("clean.png", pad_inches=0)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(19.2, 4.8))
ax = fig.add_subplot(1, 1, 1)

rgba_resize = cv2.resize(rgba, dsize=list(db.shape)[::-1], interpolation=cv2.INTER_CUBIC)

ax.imshow(rgba_resize, origin='lower')
plt.savefig("clean_resize.png")

'''
from imageio import imread
image = imread("clean.png")
print(image)
'''

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(19.2, 4.8))
ax = fig.add_subplot(1, 1, 1)

db_norm = norm(db)
print(db_norm)

db_rgb = np.zeros(list(db_norm.shape) + [3])
db_rgb[:,:,0] = db_norm

ax.imshow(db_rgb, origin='lower')
plt.savefig("clean_v2.png")

'''
mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft)
print(mel_fb.shape)
mel_power = np.einsum("...ft,mf->...mt", power, mel_fb)
'''
mel_power = librosa.feature.melspectrogram(pcm, sr=sr, n_fft=n_fft, hop_length=hop_len)
mel_db = librosa.power_to_db(mel_power) #librosa.amplitude_to_db(mel_mag)
print(mel_db.shape)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(19.2, 4.8))
ax = fig.add_subplot(1, 1, 1)

mel_db_norm = cv2.resize(mel_db, dsize=list(db.shape)[::-1], interpolation=cv2.INTER_CUBIC)
mel_db_norm = norm(mel_db_norm)

mel_db_rgb = np.zeros(list(mel_db_norm.shape) + [3])
mel_db_rgb[:,:,1] = mel_db_norm

ax.imshow(mel_db_rgb, origin='lower')
plt.savefig("clean_v3.png")

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(19.2, 4.8))
ax = fig.add_subplot(1, 1, 1)

import scipy.fftpack
mfcc = scipy.fftpack.dct(mel_db, axis=-2, type=2, norm='ortho')[..., :40, :]
mfcc_norm = cv2.resize(mfcc, dsize=list(db.shape)[::-1], interpolation=cv2.INTER_CUBIC)
mfcc_norm = norm(mfcc_norm)

mfcc_rgb = np.zeros(list(mfcc_norm.shape) + [3])
mfcc_rgb[:,:,2] = mfcc_norm

ax.imshow(mfcc_rgb, origin='lower')
plt.savefig("clean_v4.png")

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(19.2, 4.8))
ax = fig.add_subplot(1, 1, 1)

import gammatone
gm_fb = gammatone.gammatone(sr=sr, n_fft=n_fft, n_bins=128)
print(gm_fb.shape)
gm_power = np.einsum("...ft,mf->...mt", power, gm_fb)
gm_db = librosa.power_to_db(gm_power) #librosa.amplitude_to_db(mel_mag)
print(gm_db.shape)

gm_db_norm = cv2.resize(gm_db, dsize=list(db.shape)[::-1], interpolation=cv2.INTER_CUBIC)
gm_db_norm = norm(gm_db_norm)

gm_db_rgb = np.zeros(list(gm_db_norm.shape) + [3])
gm_db_rgb[:,:,2] = gm_db_norm

ax.imshow(gm_db_rgb, origin='lower')
plt.savefig("clean_v5.png")

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(19.2, 4.8))
ax = fig.add_subplot(1, 1, 1)

import cqt
cqt_power = cqt.pcqt(pcm, sr=sr, hop_length=80, n_bins=128, bins_per_octave=20).numpy()
cqt_power = cqt_power.transpose((1, 0))
cqt_db = librosa.power_to_db(cqt_power)

cqt_db_norm = cv2.resize(cqt_db, dsize=list(db.shape)[::-1], interpolation=cv2.INTER_CUBIC)
cqt_db_norm = norm(cqt_db_norm)

cqt_db_rgb = np.zeros(list(cqt_db_norm.shape) + [3])
cqt_db_rgb[:,:,2] = cqt_db_norm

ax.imshow(cqt_db_rgb, origin='lower')
plt.savefig("clean_v6.png")

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(19.2, 4.8))
ax = fig.add_subplot(1, 1, 1)

mix_db_rgb = np.zeros(list(db_norm.shape) + [3])
mix_db_rgb[:,:,0] = db_norm
mix_db_rgb[:,:,1] = mel_db_norm
mix_db_rgb[:,:,2] = gm_db_norm#mfcc_norm

ax.imshow(mix_db_rgb, origin='lower')
plt.savefig("clean_v7.png")
