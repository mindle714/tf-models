import tensorflow as tf
from tensorflow.python.training import py_checkpoint_reader

reader     = py_checkpoint_reader.NewCheckpointReader("pre-exps/tera_fshift/model.ckpt")

dtype_map  = reader.get_variable_to_dtype_map()
shape_map  = reader.get_variable_to_shape_map()

state_dict = { v: reader.get_tensor(v) for v in shape_map}
print(state_dict)
