'''
import tensorflow as tf

for e in tf.compat.v1.train.summary_iterator("exps/tera_dbg5/logs/events.out.tfevents.1666381340.speech.725467.0.v2"):
  for v in e.summary.value:
    if v.tag == 'sim':
      print(v)
'''

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_acc = EventAccumulator("exps/tera_dbg5/logs/events.out.tfevents.1666381340.speech.725467.0.v2")
event_acc.Reload()
print(event_acc)
