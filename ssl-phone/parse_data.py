import tensorflow as tf

def parse_func(pcm_len, txt_len):
  def _parse_func(ex):
    desc = {
      'pcm': tf.io.FixedLenFeature([pcm_len], tf.float32),
      'txt': tf.io.FixedLenFeature([txt_len], tf.int64),
      'pcm_len': tf.io.FixedLenFeature([1], tf.int64),
      'txt_len': tf.io.FixedLenFeature([1], tf.int64)
    }

    return tf.io.parse_single_example(ex, desc)
  return _parse_func

def gen_train(tfrec_list, pcm_len, txt_len, batch_size=16, seed=1234, epoch=None):
  dataset = tf.data.TFRecordDataset(tfrec_list)
  dataset = dataset.shuffle(batch_size*100, seed=seed, reshuffle_each_iteration=True)
  if isinstance(epoch, int):
    dataset = dataset.repeat(count=epoch)
  else:
    dataset = dataset.repeat()

  dataset = dataset.map(parse_func(pcm_len, txt_len))
  if isinstance(epoch, int):
    dataset = dataset.batch(batch_size, drop_remainder=True)
  else:
    dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset

def gen_val(tfrec_list, pcm_len, batch_size=16, seed=1234):
  dataset = tf.data.TFRecordDataset(tfrec_list)

  dataset = dataset.map(parse_func(pcm_len, False))
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset
