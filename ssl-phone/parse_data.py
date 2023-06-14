import tensorflow as tf

def parse_func(pcm_len, txt_len):
  def _parse_func(ex):
    desc = {
      'pcm': tf.io.FixedLenFeature([pcm_len], tf.float32),
      'txt': tf.io.FixedLenFeature([txt_len], tf.int64),
      'pcm_len': tf.io.FixedLenFeature([1], tf.int64),
      'txt_len': tf.io.FixedLenFeature([1], tf.int64)
    }

    e = tf.io.parse_single_example(ex, desc)
    return e
  return _parse_func

from util import melspec

def _normalize_wav_decibel(wav, target_level=-25):
  rms = (tf.math.reduce_mean(wav**2))**0.5
  scalar = (10 ** (target_level / 20)) / (rms + 1e-10)
  wav = wav * scalar
  return wav

def conv_spec(e, n_fft=400, hop_len=160):
  x = e['pcm']
  del e['pcm']

  x = _normalize_wav_decibel(x)
  x = melspec(x, num_mel_bins=80,
    frame_length=n_fft, frame_step=hop_len, fft_length=n_fft,
    lower_edge_hertz=0., upper_edge_hertz=8000.)
  e['spec'] = x

  x_len = e['pcm_len']
  del e['pcm_len']
  e['spec_len'] = int((x_len - n_fft) / hop_len) + 1

  return e

def gen_train(tfrec_list, pcm_len, txt_len, batch_size=16, seed=1234, epoch=None):
  dataset = tf.data.TFRecordDataset(tfrec_list)
  dataset = dataset.shuffle(batch_size*100, seed=seed, reshuffle_each_iteration=True)
  if isinstance(epoch, int):
    dataset = dataset.repeat(count=epoch)
  else:
    dataset = dataset.repeat()

  dataset = dataset.map(parse_func(pcm_len, txt_len),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if isinstance(epoch, int):
    dataset = dataset.batch(batch_size, drop_remainder=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    dataset = dataset.batch(batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset

def gen_val(tfrec_list, pcm_len, batch_size=16, seed=1234):
  dataset = tf.data.TFRecordDataset(tfrec_list)

  dataset = dataset.map(parse_func(pcm_len, False))
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset
