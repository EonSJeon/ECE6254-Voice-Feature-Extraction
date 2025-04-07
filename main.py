import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Device Name: ", tf.config.list_physical_devices('GPU'))
print("Is Metal available:", tf.config.list_physical_devices('GPU') != []) 