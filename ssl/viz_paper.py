import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import cm
import soundfile
import numpy as np

pcm1, _ = soundfile.read("/home/hejung/tf-models/ssl/eval_output/tera_pre3/tera_pre3-20000_orig_0.wav")
pcm2, _ = soundfile.read("/home/hejung/tf-models/ssl/eval_output/tera_pre3/tera_pre3-20000_hyp_0.wav")
pcm3, _ = soundfile.read("/home/hejung/tf-models/ssl/eval_output/tera_fs_lv1_50k_v2_0.8/tera_fs_lv1_50k_v2_0.8-20000_hyp_0.wav")

fig = plt.figure(figsize=[9.6, 7.2])

ax_2d = fig.add_subplot(3, 1, 1)
ax_2d.set_title('(a)', y=0, pad=-15)
d = librosa.stft(pcm1, win_length=512)
s_db = librosa.amplitude_to_db(np.abs(d))
spec = librosa.display.specshow(s_db, sr=16000, ax=ax_2d, cmap=cm.jet, y_axis='linear')

ax_2d = fig.add_subplot(3, 1, 2)
ax_2d.set_title('(b)', y=0, pad=-15)
d = librosa.stft(pcm2, win_length=512)
s_db = librosa.amplitude_to_db(np.abs(d))
spec = librosa.display.specshow(s_db, sr=16000, ax=ax_2d, cmap=cm.jet, y_axis='linear')

ax_2d = fig.add_subplot(3, 1, 3)
ax_2d.set_title('(c)', y=0, pad=-15)
d = librosa.stft(pcm3, win_length=512)
s_db = librosa.amplitude_to_db(np.abs(d))
spec = librosa.display.specshow(s_db, sr=16000, ax=ax_2d, cmap=cm.jet, y_axis='linear')

plt.tight_layout()
plt.savefig('compare_0.png')
