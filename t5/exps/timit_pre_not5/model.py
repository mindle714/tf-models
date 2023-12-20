import tensorflow as tf
from util import *
import gumbel
import mask
from tf_seq2seq_losses import classic_ctc_loss as _ctc_loss

from wav2vec2 import featencoder, featproj
from tfive import t5_encoder 

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims
gelu = tf.keras.activations.gelu

class wav2vec2_t5(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(wav2vec2_t5, self).__init__(*args, **kwargs)
   
  def build(self, input_shape):
    self.fe = featencoder()
    self.fp = featproj()
    
    self.encoder = t5_encoder()

  def call(self, inputs, training=None):
    x, _x_len = inputs

    if _x_len is None:
      x_len = None

    else:
#      x_len = mask.get_feat_extract_output_length(_x_len)
      x_len = tf.squeeze(_x_len, -1)

    x = self.fe(tf_expd(x, -1))
    _, x_feat = self.fp(x)
    
    x = self.encoder((x_feat, x_len))
    return x

class wav2vec2_t5_phone(tf.keras.layers.Layer):
  def __init__(self, num_class=74, *args, **kwargs):
    super(wav2vec2_t5_phone, self).__init__(*args, **kwargs)
    self.num_class = num_class

  def build(self, input_shape):
    conv_opt = dict(padding='same', use_bias=False)

    self.wav2vec2_t5 = wav2vec2_t5()
    
    self.proj = tf.keras.layers.Dense(256, use_bias=True)
    self.linear = tf.keras.layers.Dense(self.num_class, use_bias=True)
    
  def call(self, inputs, training=None, ctc=True):
    if isinstance(inputs, tuple) and len(inputs) == 4:
      x, ref, _x_len, ref_len = inputs
      x_len = mask.get_feat_extract_output_length(_x_len)

    else:
      x = inputs
      x_len = None
      ref = None; ref_len = None

    x = self.wav2vec2_t5((x, x_len), training=training)

    x = gelu(x)

    x = self.proj(x)

    # TODO in s3prl, no activation between two linear layers
    x = self.linear(x)

    if ref is not None:
      if ctc:
        ctc_loss = _ctc_loss(
          tf.cast(ref, tf.int32), x, 
          tf.squeeze(tf.cast(ref_len, tf.int32), -1), 
          tf.squeeze(tf.cast(x_len, tf.int32), -1), 
          blank_index = 0)

        return ctc_loss

      else:
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          tf.cast(ref, tf.int32), x)

        _ref_len = tf.squeeze(ref_len, -1)
        ce_mask = tf.sequence_mask(_ref_len, tf.shape(x)[1])
        ce_mask = tf.cast(ce_mask, x.dtype)

        ce_loss = ce_loss * ce_mask
        # instead of sample-wise masking, do batch-wise
        ce_loss = tf.math.reduce_sum(ce_loss)
        ce_loss /= (tf.math.reduce_sum(ce_mask) + 1e-9)

        return ce_loss

    return x
