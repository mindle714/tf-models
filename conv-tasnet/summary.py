import model
import tensorflow as tf

inputs = tf.keras.Input(shape=(32000,))
m = model.convtas()
km = tf.keras.Model(inputs=inputs, outputs=m((inputs, inputs, inputs)))
print(km.summary())
