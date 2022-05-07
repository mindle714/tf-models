import soundfile
import numpy as np
import viz
import paug

pcm, sr = soundfile.read("SA1_norm.wav")
viz.plot_spec(pcm, "SA1_norm", sr)

pcm2 = paug.pattern_mask(pcm)
soundfile.write("SA1_norm_paug.wav", pcm2, sr)
viz.plot_spec(pcm2, "SA1_norm_paug", sr)

pcm3 = paug.pattern_mask(pcm, fmin=.5)
soundfile.write("SA1_norm_paug2.wav", pcm3, sr)
viz.plot_spec(pcm3, "SA1_norm_paug2", sr)

pcm4 = paug.pattern_mask(pcm, fmax=.5)
soundfile.write("SA1_norm_paug3.wav", pcm4, sr)
viz.plot_spec(pcm4, "SA1_norm_paug3", sr)
