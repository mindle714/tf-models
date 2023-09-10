# TODO if being inside graph, first call and first random value will be fixed!
import tensorflow as tf
import random

tf.random.set_seed(1234)
random.seed(1234)

def mask_tera(x, x_len=None,
              mask_ratio=0.15, 
              min_len=7, max_len=7, mask_freq=0.2):
  batch_size = x.shape[0]
  mask_label = tf.zeros_like(x, tf.uint8)

  if x_len is None:
    x_len = tf.cast(tf.shape(x)[1], tf.float32)
    x_len = tf.tile(tf.expand_dims(x_len, 0), [batch_size])
  else:
    x_len = tf.cast(tf.squeeze(x_len, -1), tf.float32)
    
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
      valid_start_max = tf.math.maximum(x_len[idx] - mask_len - 1, 0)
      ratio = tf.math.round(x_len[idx] * mask_ratio / mask_len)
      ratio = tf.cast(ratio, tf.int32)

      chosen_starts = tf.random.shuffle(tf.range(valid_start_max + 1), seed=1234)[:ratio]
      chosen_intervals = _starts_to_intervals(chosen_starts, mask_len)
                
      # determine whether to mask / random / or do nothing to the frame
      dice = tf.random.uniform(shape=())
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

def get_feat_extract_output_length(input_len):
  def _conv_out_length(input_len, ksize, stride):
    return ((input_len - ksize) // stride) + 1

  for ksize, stride in zip(
    [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
    input_len = _conv_out_length(input_len, ksize, stride)

  return input_len

def get_feat_extract_ksize_stride(input_len):
  ksizes = [10, 3, 3, 3, 3, 2, 2]
  strides = [5, 2, 2, 2, 2, 2, 2]
  assert len(ksizes) == len(strides)

  _ksize = None; _stride = 1

  for idx in range(len(ksizes)):
    if _ksize is None:
      _ksize = ksizes[idx]
    else:
      _ksize = (ksizes[idx] - 1) * _stride + _ksize
    
    _stride *= strides[idx]

  return _ksize, _stride

assert get_feat_extract_output_length(16000) == 49 

def compute_mask_indices(batch_size, seq_len, attention_mask = None,
                         mask_prob = 0.05, mask_len = 10, min_masks = 2):
  epsilon = tf.random.uniform(())

  def compute_num_span(input_len):
    num_span = mask_prob * tf.cast(input_len, tf.float32) / tf.cast(mask_len, tf.float32) + epsilon
    num_span = tf.cast(num_span, tf.int32)
    num_span = tf.math.maximum(num_span, min_masks)

    num_span = tf.math.minimum(num_span, seq_len // mask_len)
    num_span = tf.math.minimum(num_span, input_len - (mask_len - 1))
    num_span = tf.math.maximum(num_span, 0)

    return num_span

  if attention_mask is not None:
    input_lens = tf.math.reduce_sum(attention_mask, -1)
    input_lens = tf.cast(input_lens, dtype=tf.int32)
  else:
    input_lens = tf.ones(batch_size, dtype=tf.int32) * seq_len

  mask = tf.zeros((batch_size, seq_len))
  mask_idxs = []

  max_num_span = compute_num_span(seq_len)
  if max_num_span == 0:
    return mask

  for idx in range(batch_size):
    input_len = input_lens[idx]

    num_span = compute_num_span(input_len)
    mask_idx = tf.random.shuffle(tf.range(input_len - (mask_len - 1)))[:num_span]

    if num_span == 0:
      dummy_mask_idx = seq_len - 1
    else:
      dummy_mask_idx = mask_idx[0]

    pad = tf.ones(max_num_span - num_span, dtype=tf.int32)
    mask_idx = tf.concat([
      mask_idx, pad * dummy_mask_idx], 0)
    mask_idxs.append(mask_idx)
  
  mask_idxs = tf.concat([tf.expand_dims(e, 0) for e in mask_idxs], 0)
  mask_idxs = tf.tile(tf.expand_dims(mask_idxs, -1), [1, 1, mask_len])
  mask_idxs = tf.reshape(mask_idxs, [batch_size, -1])

  offsets = tf.reshape(tf.range(mask_len), [1, 1, -1])
  offsets = tf.tile(offsets, [batch_size, max_num_span, 1])
  offsets = tf.reshape(offsets, [batch_size, -1])

  mask_idxs += offsets
  mask_idxs = tf.math.minimum(mask_idxs, seq_len - 1)

  # TODO naive! need to fully utilize scatter_nd
  mask = [
    tf.scatter_nd(tf.expand_dims(mask_idxs[_idx], -1), 
      tf.ones(max_num_span*mask_len), [seq_len]) \
    for _idx in range(batch_size)
  ]
  mask = tf.concat([tf.expand_dims(e, 0) for e in mask], 0)
  mask = tf.cast(mask > 0, tf.float32)

  return mask

