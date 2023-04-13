# TODO if being inside graph, first call and first random value will be fixed!
import tensorflow as tf
import random

tf.random.set_seed(1234)
random.seed(1234)

def mask_tera(x, mask_ratio=0.15, 
              min_len=7, max_len=7, mask_freq=0.2):
  batch_size = x.shape[0]
  spec_len = tf.cast(tf.shape(x)[1], tf.float32)
  mask_label = tf.zeros_like(x, tf.uint8)
    
  def _starts_to_intervals(starts, mask_len):
    tiled = tf.tile(tf.expand_dims(starts, -1), [1, mask_len])
    offset = tf.tile(tf.expand_dims(tf.range(mask_len, dtype=tiled.dtype), 0), [tf.shape(starts)[0], 1])
    intervals = tiled + offset
    intervals = tf.cast(intervals, tf.int32)
    return tf.reshape(intervals, [-1])        

  for idx in range(batch_size):
    # time masking
    if mask_ratio > 0:
      mask_len = random.randint(min_len, max_len)
      valid_start_max = tf.math.maximum(spec_len - mask_len - 1, 0)
      ratio = tf.math.round(spec_len * mask_ratio / mask_len)
      ratio = tf.cast(ratio, tf.int32)

      chosen_starts = tf.random.shuffle(tf.range(valid_start_max + 1), seed=1234)[:ratio]
      chosen_intervals = _starts_to_intervals(chosen_starts, mask_len)
                
      # determine whether to mask / random / or do nothing to the frame
      dice = tf.random.uniform(shape=()) #random.random()
      # mask to zero
      if dice < 0.8:
        indices = tf.concat([
          tf.expand_dims(tf.ones_like(chosen_intervals) * idx, -1),
          tf.expand_dims(chosen_intervals, -1)
        ], -1)
        x = tf.tensor_scatter_nd_update(x, indices, 
          tf.tile(tf.expand_dims(tf.zeros_like(chosen_intervals, dtype=tf.float32), -1), [1,tf.shape(x)[-1]]))

        # the gradients will be calculated on chosen frames
        mask_label = tf.tensor_scatter_nd_update(mask_label, indices,
          tf.tile(tf.expand_dims(tf.ones_like(chosen_intervals, dtype=mask_label.dtype), -1), [1,tf.shape(mask_label)[-1]]))

      # replace to random frames
      elif dice >= 0.8 and dice < 0.9:
        random_starts = tf.random.shuffle(tf.range(valid_start_max + 1))[:ratio]
        random_intervals = _starts_to_intervals(random_starts, mask_len)
        
        random_indices = tf.concat([
          tf.expand_dims(tf.ones_like(random_intervals) * idx, -1),
          tf.expand_dims(random_intervals, -1)
        ], -1)
        random_spec = tf.gather_nd(x, random_indices)

        indices = tf.concat([
          tf.expand_dims(tf.ones_like(chosen_intervals) * idx, -1),
          tf.expand_dims(chosen_intervals, -1)
        ], -1)
        x = tf.tensor_scatter_nd_update(x, indices, random_spec)

        # the gradients will be calculated on chosen frames
        mask_label = tf.tensor_scatter_nd_update(mask_label, indices,
          tf.tile(tf.expand_dims(tf.ones_like(chosen_intervals, dtype=mask_label.dtype), -1), [1,tf.shape(mask_label)[-1]]))

    # frequency masking
    if mask_freq > 0:
      max_width = int(x.shape[2] * mask_freq)
      rand_bandwidth = tf.random.uniform(shape=(), minval=0, maxval=max_width, dtype=tf.int32)#random.randint(0, max_width)
      chosen_starts = tf.random.shuffle(tf.range(x.shape[2] - rand_bandwidth))[:1]
      chosen_intervals = _starts_to_intervals(chosen_starts, rand_bandwidth)
        
      x = tf.transpose(x, [0, 2, 1])
      indices = tf.concat([
        tf.expand_dims(tf.ones_like(chosen_intervals) * idx, -1),
        tf.expand_dims(chosen_intervals, -1)
      ], -1)
      x = tf.tensor_scatter_nd_update(x, indices,
        tf.tile(tf.expand_dims(tf.zeros_like(chosen_intervals, dtype=tf.float32), -1), [1,tf.shape(x)[-1]]))
      x = tf.transpose(x, [0, 2, 1])
                
      # the gradients will be calculated on chosen frames
      mask_label = tf.transpose(mask_label, [0, 2, 1])
      mask_label = tf.tensor_scatter_nd_update(mask_label, indices,
        tf.tile(tf.expand_dims(tf.ones_like(chosen_intervals, dtype=mask_label.dtype), -1), [1,tf.shape(mask_label)[-1]]))
      mask_label = tf.transpose(mask_label, [0, 2, 1])

  return x, mask_label
