# copied from librosa/core/constantq.py
import librosa
import numpy as np
import tensorflow as tf

def __bpo_to_alpha(bins_per_octave):
  """Compute the alpha coefficient for a given number of bins per octave

  Parameters
  ----------
  bins_per_octave : int

  Returns
  -------
  alpha : number > 0
  """

  r = 2 ** (1 / bins_per_octave)
  return (r ** 2 - 1) / (r ** 2 + 1)

def __vqt_filter_fft(
  sr,
  freqs,
  filter_scale,
  norm,
  sparsity,
  hop_length=None,
  window="hann",
  gamma=0.0,
  dtype=np.complex64,
  alpha=None,
):
  """Generate the frequency domain variable-Q filter basis."""

  basis, lengths = librosa.filters.wavelet(
    freqs=freqs,
    sr=sr,
    filter_scale=filter_scale,
    norm=norm,
    pad_fft=True,
    window=window,
    gamma=gamma,
    alpha=alpha,
  )

  # Filters are padded up to the nearest integral power of 2
  n_fft = basis.shape[1]

  if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):
    n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

  # re-normalize bases with respect to the FFT window length
  basis *= lengths[:, np.newaxis] / float(n_fft)

  # FFT and retain only the non-negative frequencies
  fft_basis = np.fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]

  # sparsify the basis
  fft_basis = librosa.util.sparsify_rows(fft_basis, quantile=sparsity, dtype=dtype)

  return fft_basis, n_fft, lengths

def pcqt(x, sr, hop_length=512, 
         n_bins=128, bins_per_octave=20, 
         filter_scale=1, norm=1, sparsity=0.01, power=2):

  # C1 by default
  #fmin = librosa.note_to_hz("C1")
  fmin = librosa.note_to_hz("C2")#("G2")

  freqs = librosa.cqt_frequencies(fmin=fmin, 
    n_bins=n_bins, bins_per_octave=bins_per_octave)

  alpha = __bpo_to_alpha(bins_per_octave)

  fft_basis, n_fft, _ = __vqt_filter_fft(
    sr,
    freqs,
    filter_scale,
    norm,
    sparsity,
    hop_length=hop_length,
    window="hann",
    alpha=alpha,
  )
  
  fft_basis = np.abs(fft_basis).todense()

  x = tf.signal.stft(x,
    frame_length=n_fft, frame_step=hop_length, fft_length=n_fft, pad_end=True)

  x_pow = tf.abs(x) ** power
  x = tf.einsum("...tf,mf->...tm", x_pow, fft_basis)
  x /= np.sqrt(n_fft)

  return x
