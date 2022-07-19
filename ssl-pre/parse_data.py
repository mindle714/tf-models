import tensorflow as tf

def parse_func(pcm_len, samp_len=True):
  def _parse_func(ex):
    desc = {
      'pcm': tf.io.FixedLenFeature([pcm_len], tf.float32)
    }
    if samp_len:
      desc['samp_len'] = tf.io.FixedLenFeature([1], tf.int64)

    return tf.io.parse_single_example(ex, desc)
  return _parse_func

def add_mask(pcm_len):
  def _add_mask(ex):
    if 'samp_len' not in ex:
      mask = tf.ones(pcm_len, dtype=tf.float32)
      ex['mask'] = mask

      return ex

    samp_len = ex['samp_len']
    mask = tf.concat([tf.ones(samp_len, dtype=tf.float32),
      tf.zeros(pcm_len-samp_len, dtype=tf.float32)], 0)
    ex['mask'] = mask

    return ex
  return _add_mask

def gen_train(tfrec_list, pcm_len, batch_size=16, seed=1234):
  dataset = tf.data.TFRecordDataset(tfrec_list)
  dataset = dataset.shuffle(batch_size*100, seed=seed, reshuffle_each_iteration=True)
  dataset = dataset.repeat()

  dataset = dataset.map(parse_func(pcm_len))
  dataset = dataset.map(add_mask(pcm_len))
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset

def gen_val(tfrec_list, pcm_len, batch_size=16, seed=1234):
  dataset = tf.data.TFRecordDataset(tfrec_list)

  dataset = dataset.map(parse_func(pcm_len, False))
  dataset = dataset.map(add_mask(pcm_len))
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset
