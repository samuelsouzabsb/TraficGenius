import tensorflow as tf

print(tf.__version__)

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

print(a + b)