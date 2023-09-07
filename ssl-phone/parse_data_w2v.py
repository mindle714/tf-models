import tensorflow as tf

def parse_func(pcm_len, txt_len):
  desc = {
    'pcm': tf.io.FixedLenFeature([pcm_len], tf.float32),
    #'spec': tf.io.FixedLenFeature([spec_len*80], tf.float32),
    'pcm_len': tf.io.FixedLenFeature([1], tf.int64),
    #'spec_len': tf.io.FixedLenFeature([1], tf.int64),
  }
  if txt_len is not None:
    desc['txt'] = tf.io.FixedLenFeature([txt_len], tf.int64)
    desc['txt_len'] = tf.io.FixedLenFeature([1], tf.int64)

  def _parse_func(ex):
    e = tf.io.parse_single_example(ex, desc)
    #e['spec'] = tf.reshape(e['spec'], [spec_len, 80]) # TODO
    return e
  return _parse_func

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
