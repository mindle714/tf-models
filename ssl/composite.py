import soundfile
import numpy as np
import math

eps = 2.2204e-16

cent_freq = [
  50.0000, 120.000, 190.000, 260.000, 330.000, 
  400.000, 470.000, 540.000, 617.372, 703.378, 
  798.717, 904.128, 1020.38, 1148.30, 1288.72, 
  1442.54, 1610.70, 1794.16, 1993.93, 2211.08,
  2446.71, 2701.97, 2978.04, 3276.17, 3597.63]

bandwidth = [
  70.0000, 70.0000, 70.0000, 70.0000, 70.0000,
  70.0000, 70.0000, 77.3724, 86.0056, 95.3398,
  105.411, 116.256, 127.914, 140.423, 153.823,
  168.154, 183.457, 199.776, 217.153, 235.631,
  255.255, 276.072, 298.126, 321.465, 346.136]

def nextpow2(x):
  return 1 if x == 0 else 2**math.ceil(math.log2(x))

def wss(clean_speech, processed_speech, sr):
  assert len(clean_speech) == len(processed_speech)

  winlength = round(30 * sr / 1000)
  skiprate = math.floor(winlength / 4)
  max_freq = sr / 2
  num_crit = 25

  USE_FFT_SPECTRUM = True
  n_fft = nextpow2(2*winlength)
  n_fftby2 = n_fft//2
  Kmax = 20
  Klocmax = 1
  bw_min = bandwidth[0]
  min_factor = math.exp(-30.0 / (2.0 * 2.303))

  crit_filter = np.zeros((num_crit, n_fftby2))
  for i in range(num_crit):
    f0 = (cent_freq[i] / max_freq) * n_fftby2
    bw = (bandwidth[i] / max_freq) * n_fftby2
    norm_factor = math.log(bw_min) - math.log(bandwidth[i])
    crit_filter[i,:] = np.exp(-11 * (((np.arange(n_fftby2)-math.floor(f0))/bw)**2) + norm_factor)
    crit_filter[i,:] = crit_filter[i,:] * (crit_filter[i,:] > min_factor)

  num_frames = int(len(clean_speech) / skiprate - (winlength / skiprate))
  start = 0
  window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1,winlength+1)/(winlength+1)))

  distortion = np.zeros(num_frames)
  for frame_count in range(num_frames):

    def wss_frame(frame, window, crit_filter):
      frame = frame * window
    
      # compute power spectrum of clean/processed
      if USE_FFT_SPECTRUM:
        spec = np.abs(np.fft.fft(frame, n_fft))**2
      else:
        a_vec = np.zeros(n_fft)
        a_vec[:11] = lpc(frame, 10)
        spec = 1.0 / (np.abs(np.fft.fft(a_vec, n_fft))**2)
    
      # compute filterbank output energy (db)
      energy = np.zeros(num_crit)
      for i in range(num_crit):
        energy[i] = np.sum(spec[:n_fftby2] * crit_filter[i,:])

      energy = 10 * np.log10(np.maximum(energy, 1e-10))
    
      # compute spectral slope
      slope = energy[1:] - energy[:-1]

      # nearest peak location in the spectra to each critical band
      loc_peak = np.zeros(num_crit-1)
      for i in range(num_crit-1):
        if slope[i] > 0:
          n = i
          while (n < (num_crit-1)) and (slope[n] > 0):
            n += 1
          loc_peak[i] = energy[n-1]

        else:
          n = i
          while (n>=0) and (slope[n] <= 0):
            n -= 1
          loc_peak[i] = energy[n+1]
    
      # compute wss measure for this frame
      dBMax = np.max(energy)
      Wmax = Kmax / (Kmax + dBMax - energy[:num_crit-1])
      Wlocmax = Klocmax / (Klocmax + loc_peak - energy[:num_crit-1])
      W = Wmax * Wlocmax

      return W, slope

    clean_frame = clean_speech[start:start+winlength]
    processed_frame = processed_speech[start:start+winlength]

    W_clean, clean_slope = wss_frame(clean_frame, window, crit_filter)
    W_processed, processed_slope = wss_frame(processed_frame, window, crit_filter)

    W = (W_clean + W_processed) / 2.
    distortion[frame_count] = np.sum(W * (clean_slope[:num_crit-1] - processed_slope[:num_crit-1])**2)
    distortion[frame_count] /= np.sum(W)

    start += skiprate

  return distortion

def lpcoeff(speech_frame, order):
  winlength = speech_frame.shape[0]

  R = np.zeros(order+1)
  for k in range(order+1):
    R[k] = np.sum(speech_frame[:winlength-k] * speech_frame[k:winlength])

  a = np.ones(order)
  E = R[0]
  rcoeff = np.zeros(order)

  for i in range(order):
    a_past = a[:i]
    sum_term = np.sum(a_past * R[1:i+1][::-1])
    rcoeff[i] = (R[i+1] - sum_term) / E
    a[i] = rcoeff[i]
    a[:i] = a_past - rcoeff[i] * a_past[::-1];
    E = (1-rcoeff[i]*rcoeff[i]) * E;

  return R, rcoeff, np.concatenate([[1], -a])

from scipy.linalg import toeplitz

def llr(clean_speech, processed_speech, sr):
  assert len(clean_speech) == len(processed_speech)
  
  winlength = round(30 * sr / 1000)
  skiprate = math.floor(winlength / 4)
  if sr < 10000: P = 10
  else: P = 16
  
  num_frames = int(len(clean_speech) / skiprate - (winlength / skiprate))
  start = 0
  window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1,winlength+1)/(winlength+1)))
  
  distortion = np.zeros(num_frames)
  for frame_count in range(num_frames):
    
    clean_frame = clean_speech[start:start+winlength]
    processed_frame = processed_speech[start:start+winlength]

    clean_frame = clean_frame * window
    processed_frame = processed_frame * window

    R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
    R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)

    numerator = np.matmul(np.matmul(A_processed, 
      toeplitz(R_clean)), 
      np.transpose(A_processed))
    denominator = np.matmul(np.matmul(A_clean, 
      toeplitz(R_clean)), 
      np.transpose(A_clean))

    distortion[frame_count] = np.log(numerator / denominator)
    start += skiprate

  return distortion

def snr(clean_speech, processed_speech, sr):
  assert len(clean_speech) == len(processed_speech)

  winlength = round(30 * sr / 1000)
  skiprate = math.floor(winlength / 4)
  MIN_SNR = -10
  MAX_SNR = 35
  
  num_frames = int(len(clean_speech) / skiprate - (winlength / skiprate))
  start = 0
  window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1,winlength+1)/(winlength+1)))

  segmental_snr = np.zeros(num_frames)
  for frame_count in range(num_frames):

    clean_frame = clean_speech[start:start+winlength]
    processed_frame = processed_speech[start:start+winlength]
    
    clean_frame = clean_frame * window
    processed_frame = processed_frame * window

    signal_energy = np.sum(clean_frame**2)
    noisy_energy = np.sum((clean_frame - processed_frame)**2)

    segmental_snr[frame_count] = 10 * np.log10(signal_energy / (noisy_energy + eps) + eps)
    segmental_snr[frame_count] = np.maximum(segmental_snr[frame_count], MIN_SNR)
    segmental_snr[frame_count] = np.minimum(segmental_snr[frame_count], MAX_SNR)

    start += skiprate

  return segmental_snr 

def composite(data1, data2, sr, alpha=0.95):
  _len = min(data1.shape[0], data2.shape[0])
  data1 = data1[:_len] + eps
  data2 = data2[:_len] + eps

  wss_dist_vec = wss(data1, data2, sr)
  wss_dist_vec = sorted(wss_dist_vec)
  wss_dist = np.mean(wss_dist_vec[:round(len(wss_dist_vec)*alpha)])

  LLR_dist = llr(data1, data2, sr)
  LLRs = sorted(LLR_dist)
  llr_mean = np.mean(LLRs[:round(len(LLR_dist) * alpha)])

  segsnr_dist = snr(data1, data2, sr)
  segSNR = np.mean(segsnr_dist)

  pesq_mos = 0

  Csig = 3.093 - 1.029*llr_mean + 0.603*pesq_mos-0.009*wss_dist
  Cbak = 1.634 + 0.478 *pesq_mos - 0.007*wss_dist + 0.063*segSNR
  Covl = 1.594 + 0.805*pesq_mos - 0.512*llr_mean - 0.007*wss_dist

  return Csig, Cbak, Covl, segSNR
