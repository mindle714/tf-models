import soundfile
import numpy as np
import viz
import paug
import lpf

pcm, sr = soundfile.read("SA1_norm.wav")
viz.plot_spec(pcm, "SA1_norm", sr)

pcm2 = lpf.lowpass(pcm, 0.5)
#pcm2, sr = soundfile.read("/home/speech/tmp/SA1_norm_paug.wav")
soundfile.write("SA1_norm_lpf.wav", pcm2, sr)
viz.plot_spec(pcm2, "SA1_norm_lpf", sr)
#soundfile.write("/home/speech/tmp/SA1_norm_paug_ch2.wav", 
#  np.concatenate([np.expand_dims(pcm,-1), np.expand_dims(pcm2,-1)], -1), 16000)
