import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def gumbel_softmax(x, tau, hard = False, dim = -1):
  '''
  gumbels = tfp.distributions.Exponential(1.).sample(tf.shape(x))
  gumbels = -tf.math.log(gumbels) # Gumbel(0, 1)
  '''
  gumbels = np.load('gumbels.npy')
  gumbels = (x + gumbels) / tau # Gumbel(logits, tau)
  y_soft = tf.nn.softmax(gumbels, dim)

  if hard:
    index = tf.math.argmax(y_soft, axis=dim)
    y_hard = tf.one_hot(index, tf.shape(x)[dim])
    ret = y_hard - tf.stop_gradient(y_soft) + y_soft

  else:
    ret = y_soft

  return ret

def compute_perplexity(probs, mask = None):
  if mask is not None:
    mask_extended = tf.tile(tf.reshape(mask, [-1, 1, 1]), 
      [1, tf.shape(probs)[1], tf.shape(probs)[2]])
    probs = tf.where(tf.cast(mask_extended, tf.bool), probs, tf.zeros_like(probs))
    marg_probs = tf.math.reduce_sum(probs, 0) / tf.math.reduce_sum(mask)

  else:
    marg_probs = tf.math.reduce_mean(probs, 0)

  perp = tf.math.exp(-tf.math.reduce_sum(
    marg_probs * tf.math.log(marg_probs + 1e-7), -1))
  perp = tf.math.reduce_sum(perp)
  return perp

class gumbelq(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    self.num_groups = 2
    self.num_vars = 320
    self.codevector_dim = 256
    self.temperature = 2

    super(gumbelq, self).__init__(*args, **kwargs)
  
  def build(self, input_shape):
    self.codevectors = self.add_weight(
      shape=(1, self.num_groups * self.num_vars, self.codevector_dim // self.num_groups),
      name="codevectors")

    self.weight_proj = tf.keras.layers.Dense(self.num_groups * self.num_vars, use_bias=True)
  
  def call(self, inputs, training=None):
    mask_time_indices = None

    if isinstance(inputs, tuple):
      x, mask_time_indices = inputs

    else:
      x = inputs

    batch_size, seq_len, hdim = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

    x = self.weight_proj(x)
    x = tf.reshape(x, [batch_size * seq_len * self.num_groups, -1])

    if training:
      codevector_probs = gumbel_softmax(x, self.temperature, hard=True)
      codevector_soft_dist = tf.nn.softmax(
        tf.reshape(x, [batch_size * seq_len, self.num_groups, -1]), -1)

      perp = compute_perplexity(codevector_soft_dist, mask_time_indices)

    else:
      codevector_idx = tf.math.argmax(x, -1)
      codevector_probs = tf.one_hot(codevector_idx, tf.shape(x)[-1])
      codevector_probs = tf.reshape(codevector_probs,
        [batch_size * seq_len, self.num_groups, -1])

      perp = compute_perplexity(codevector_probs, mask_time_indices)

    codevector_probs = tf.reshape(codevector_probs,
      [batch_size * seq_len, -1])
    codevectors_per_group = tf.expand_dims(codevector_probs, -1) * self.codevectors
    codevectors = tf.reshape(codevectors_per_group,
      [batch_size * seq_len, self.num_groups, self.num_vars, -1])

    codevectors = tf.math.reduce_sum(codevectors, -2)
    codevectors = tf.reshape(codevectors, [batch_size, seq_len, -1])

    return codevectors, perp
