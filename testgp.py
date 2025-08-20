import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)
print("Built with CUDA:", tf.test.is_built_with_cuda())