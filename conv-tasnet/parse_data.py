import tensorflow as tf

def parse_func(pcm_len):
  def _parse_func(ex):
    desc = {
      's1': tf.io.FixedLenFeature([pcm_len], tf.float32),
      's2': tf.io.FixedLenFeature([pcm_len], tf.float32),
      'mix': tf.io.FixedLenFeature([pcm_len], tf.float32)
    }
    return tf.io.parse_single_example(ex, desc)
  return _parse_func

def gen_train(tfrec_list, pcm_len, batch_size=16, seed=1234):
  dataset = tf.data.TFRecordDataset(tfrec_list)
  dataset = dataset.shuffle(batch_size, seed=seed, reshuffle_each_iteration=True)
  dataset = dataset.repeat()

  dataset = dataset.map(parse_func(pcm_len))
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset

def gen_val(tfrec_list, pcm_len, batch_size=16, seed=1234):
  dataset = tf.data.TFRecordDataset(tfrec_list)

  dataset = dataset.map(parse_func(pcm_len))
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset
